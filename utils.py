from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Set seeds for reproducible experiments across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as formatted JSON."""
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def count_parameters(model: Any) -> int:
    """Count trainable parameters for a PyTorch model."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_device(force_cpu: bool = False):
    """Return the available torch device."""
    import torch

    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
