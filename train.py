from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import WeatherSequenceDataset
from .evaluation import regression_metrics
from .losses import FixedExtremeWeatherLoss
from .utils import count_parameters, get_device, set_seed


def train_torch_model(
    model: nn.Module,
    model_data: Dict[str, object],
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience: int = 15,
    loss_type: str = "extreme",
    force_cpu: bool = False,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Train a PyTorch sequence model with early stopping."""
    set_seed(seed)
    device = get_device(force_cpu=force_cpu)
    model = model.to(device)

    train_ds = WeatherSequenceDataset(model_data["X_train"], model_data["y_train"])
    val_ds = WeatherSequenceDataset(model_data["X_val"], model_data["y_val"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    y_train_scaled = np.asarray(model_data["y_train"]).ravel()
    low_q = float(np.percentile(y_train_scaled, 5))
    high_q = float(np.percentile(y_train_scaled, 95))
    if loss_type == "extreme":
        criterion = FixedExtremeWeatherLoss(low_threshold=low_q, high_threshold=high_q, alpha_low=2.0, alpha_high=2.0, beta=1.0)
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}
    start = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out, yb)
                val_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    training_time = time.time() - start
    info = {
        "seed": seed,
        "best_val_loss": best_val,
        "epochs_ran": len(history["train_loss"]),
        "training_time": training_time,
        "history": history,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "loss_type": loss_type,
    }
    return model, info


def predict_torch_model(model: nn.Module, model_data: Dict[str, object], batch_size: int = 256, force_cpu: bool = False):
    """Predict on the held-out test set and return original-scale predictions."""
    device = get_device(force_cpu=force_cpu)
    model = model.to(device)
    model.eval()
    ds = WeatherSequenceDataset(model_data["X_test"], model_data["y_test"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    attn = []
    clusters = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            if isinstance(out, tuple):
                pred, cluster_probs, attention = out
                clusters.append(cluster_probs.detach().cpu().numpy())
                attn.append(attention.detach().cpu().numpy())
            else:
                pred = out
            preds.append(pred.detach().cpu().numpy())
    preds_scaled = np.vstack(preds)
    preds_orig = model_data["scaler_y"].inverse_transform(preds_scaled).ravel()
    extra = {}
    if clusters:
        extra["cluster_probs"] = np.concatenate(clusters, axis=0)
    if attn:
        extra["attention_weights"] = np.concatenate(attn, axis=0)
    return preds_orig, extra


def evaluate_torch_model(
    name: str,
    model: nn.Module,
    model_data: Dict[str, object],
    train_high_threshold_original: float,
    train_low_threshold_original: float,
    training_info: Optional[Dict[str, object]] = None,
    force_cpu: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, object]]:
    """Evaluate a trained PyTorch model on the test period."""
    preds, extra = predict_torch_model(model, model_data, force_cpu=force_cpu)
    y_true = model_data["y_test_original"]
    metrics = regression_metrics(y_true, preds, train_high_threshold_original, train_low_threshold_original)
    if training_info:
        metrics["training_time"] = float(training_info.get("training_time", np.nan))
        metrics["parameter_count"] = int(training_info.get("parameter_count", 0))
        metrics["epochs_ran"] = int(training_info.get("epochs_ran", 0))
    metrics["model"] = name
    return metrics, preds, extra
