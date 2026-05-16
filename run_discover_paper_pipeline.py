#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Allow running without package installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoclimate_fusionllm.baselines import run_simple_baselines
from geoclimate_fusionllm.data import load_weather_data
from geoclimate_fusionllm.evaluation import skill_against_persistence
from geoclimate_fusionllm.features import (
    SplitConfig,
    add_causal_features,
    choose_feature_columns,
    create_sequence_arrays,
    make_supervised_table,
    split_supervised,
)
from geoclimate_fusionllm.figures import create_all_static_figures
from geoclimate_fusionllm.llm_report import make_explainable_warning_report
from geoclimate_fusionllm.models import MMWSTM_ADRAN_Plus, NBEATSLike, TCNRegressor, TemporalTransformer
from geoclimate_fusionllm.station_validation import validate_against_station
from geoclimate_fusionllm.train import evaluate_torch_model, train_torch_model
from geoclimate_fusionllm.utils import count_parameters, ensure_dir, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run GeoClimate-FusionLLM Discover Geoscience pipeline.")
    parser.add_argument("--raw-data", required=True, help="Path to raw Baghdad Excel/CSV weather file.")
    parser.add_argument("--processed-data", default=None, help="Optional processed CSV file.")
    parser.add_argument("--station-data", default=None, help="Optional independent station observation CSV with datetime,tempmax_obs.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--run-neural-baselines", action="store_true", help="Train Transformer, TCN, and N-BEATS baselines.")
    parser.add_argument("--run-ablations", action="store_true", help="Run ablation variants across all seeds.")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--fast-test", action="store_true", help="Run a very small smoke test for code checking.")
    return parser.parse_args()


def prepare_output_dirs(output_dir: Path):
    for sub in ["figures", "tables", "models", "reports"]:
        ensure_dir(output_dir / sub)


def train_named_model(name, model, model_data, args, seed, loss_type="extreme"):
    model, info = train_torch_model(
        model,
        model_data,
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        loss_type=loss_type,
        force_cpu=args.force_cpu,
    )
    return model, info


def main():
    args = parse_args()
    if args.fast_test:
        args.epochs = min(args.epochs, 1)
        args.patience = 1
        args.seeds = args.seeds[:1]
        args.sequence_length = min(args.sequence_length, 5)
        args.hidden_dim = min(args.hidden_dim, 8)
        args.batch_size = min(args.batch_size, 32)
        args.run_neural_baselines = False
        args.run_ablations = False

    output_dir = Path(args.output_dir)
    prepare_output_dirs(output_dir)
    set_seed(args.seeds[0])

    # 1. Load and engineer features.
    split = SplitConfig()
    raw = load_weather_data(args.raw_data, args.processed_data, prefer_processed=False)
    if args.fast_test:
        # Keep the full 2019-2024 time span but reduce the number of samples for a quick CI/smoke run.
        raw = raw.iloc[::10].reset_index(drop=True)
    df = add_causal_features(raw, split)
    feature_cols = choose_feature_columns(df, split, target_col="tempmax", max_features=40)

    # 2. Create simple supervised table for classical baselines.
    supervised = make_supervised_table(df, feature_cols, target_col="tempmax")
    train_df, val_df, test_df = split_supervised(supervised, split)
    train_high = float(np.percentile(train_df["target"].values, 95))
    train_low = float(np.percentile(train_df["target"].values, 5))
    p99 = float(np.percentile(train_df["target"].values, 99))

    # 3. Simple baselines: persistence, climatology, AR(1), MLR.
    baseline_table, baseline_preds = run_simple_baselines(train_df, val_df, test_df, feature_cols, train_high, train_low)
    baseline_table = skill_against_persistence(baseline_table)
    baseline_table.to_csv(output_dir / "tables" / "baseline_metrics.csv", index=False)

    # 4. Neural sequence data.
    model_data = create_sequence_arrays(df, feature_cols, split, target_col="tempmax", sequence_length=args.sequence_length)
    y_test = model_data["y_test_original"]

    # 5. Proposed model, first seed.
    neural_rows = []
    prediction_bank = dict(baseline_preds)
    extra_for_best = {}

    full_model = MMWSTM_ADRAN_Plus(
        input_dim=model_data["input_dim"],
        hidden_dim=args.hidden_dim,
        sequence_length=args.sequence_length,
        use_transition=True,
        use_attention=True,
        use_anomaly=True,
    )
    full_model, full_info = train_named_model("MMWSTM-ADRAN+", full_model, model_data, args, args.seeds[0], loss_type="extreme")
    full_metrics, full_pred, full_extra = evaluate_torch_model("MMWSTM-ADRAN+", full_model, model_data, train_high, train_low, full_info, force_cpu=args.force_cpu)
    neural_rows.append(full_metrics)
    prediction_bank["MMWSTM-ADRAN+"] = full_pred
    extra_for_best = full_extra
    torch.save(full_model.state_dict(), output_dir / "models" / "mmwstm_adran_plus.pt")

    # 6. Optional neural baselines.
    if args.run_neural_baselines:
        neural_defs = {
            "Temporal Transformer": TemporalTransformer(input_dim=model_data["input_dim"], hidden_dim=64),
            "TCN": TCNRegressor(input_dim=model_data["input_dim"]),
            "N-BEATS": NBEATSLike(input_dim=model_data["input_dim"], sequence_length=args.sequence_length),
        }
        for name, model in neural_defs.items():
            trained, info = train_named_model(name, model, model_data, args, args.seeds[0], loss_type="mse")
            metrics, preds, _ = evaluate_torch_model(name, trained, model_data, train_high, train_low, info, force_cpu=args.force_cpu)
            neural_rows.append(metrics)
            prediction_bank[name] = preds

    neural_table = pd.DataFrame(neural_rows)
    if not neural_table.empty:
        neural_table = skill_against_persistence(neural_table)
    neural_table.to_csv(output_dir / "tables" / "neural_metrics.csv", index=False)

    # 7. Combined main metrics table.
    combined_metrics = pd.concat([baseline_table, neural_table], ignore_index=True, sort=False)
    combined_metrics = skill_against_persistence(combined_metrics)
    combined_metrics.to_csv(output_dir / "tables" / "combined_metrics.csv", index=False)

    # 8. Ablations and seed robustness.
    ablation_rows = []
    ablation_variants = {
        "Full MMWSTM-ADRAN+": dict(use_transition=True, use_attention=True, use_anomaly=True, loss_type="extreme"),
        "No transition matrix": dict(use_transition=False, use_attention=True, use_anomaly=True, loss_type="extreme"),
        "No self-attention": dict(use_transition=True, use_attention=False, use_anomaly=True, loss_type="extreme"),
        "No anomaly amplification": dict(use_transition=True, use_attention=True, use_anomaly=False, loss_type="extreme"),
        "Standard MSE loss": dict(use_transition=True, use_attention=True, use_anomaly=True, loss_type="mse"),
    }
    parameter_rows = []
    if args.run_ablations:
        for variant, cfg in ablation_variants.items():
            for seed in args.seeds:
                model = MMWSTM_ADRAN_Plus(
                    input_dim=model_data["input_dim"],
                    hidden_dim=args.hidden_dim,
                    sequence_length=args.sequence_length,
                    use_transition=cfg["use_transition"],
                    use_attention=cfg["use_attention"],
                    use_anomaly=cfg["use_anomaly"],
                )
                parameter_rows.append({"variant": variant, "parameter_count": count_parameters(model)})
                trained, info = train_named_model(variant, model, model_data, args, seed, loss_type=cfg["loss_type"])
                metrics, preds, _ = evaluate_torch_model(variant, trained, model_data, train_high, train_low, info, force_cpu=args.force_cpu)
                metrics["variant"] = variant
                metrics["seed"] = seed
                metrics["loss_type"] = cfg["loss_type"]
                ablation_rows.append(metrics)
    else:
        # Still write parameter-count table for review transparency.
        for variant, cfg in ablation_variants.items():
            model = MMWSTM_ADRAN_Plus(
                input_dim=model_data["input_dim"],
                hidden_dim=args.hidden_dim,
                sequence_length=args.sequence_length,
                use_transition=cfg["use_transition"],
                use_attention=cfg["use_attention"],
                use_anomaly=cfg["use_anomaly"],
            )
            parameter_rows.append({"variant": variant, "parameter_count": count_parameters(model)})

    ablation_table = pd.DataFrame(ablation_rows)
    ablation_table.to_csv(output_dir / "tables" / "ablation_seed_metrics.csv", index=False)
    parameter_table = pd.DataFrame(parameter_rows).drop_duplicates().sort_values("variant")
    parameter_table.to_csv(output_dir / "tables" / "parameter_counts.csv", index=False)
    if not ablation_table.empty:
        ablation_summary = ablation_table.groupby("variant").agg(
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            extreme_high_rmse_mean=("extreme_high_rmse", "mean"),
            extreme_low_rmse_mean=("extreme_low_rmse", "mean"),
            parameter_count=("parameter_count", "mean"),
        ).reset_index()
        ablation_summary.to_csv(output_dir / "tables" / "ablation_summary.csv", index=False)

    # 9. Optional station validation.
    validate_against_station(df, args.station_data, output_dir)

    # 10. Figures and reports.
    best_model_name = combined_metrics.sort_values("rmse").iloc[0]["model"]
    best_pred = prediction_bank.get(best_model_name, full_pred)
    create_all_static_figures(
        output_dir=output_dir,
        df=df,
        feature_cols=feature_cols,
        metrics=combined_metrics,
        test_meta=model_data["test_meta"],
        y_true=y_test,
        predictions=prediction_bank,
        best_pred=best_pred,
        ablation_metrics=ablation_table,
    )

    make_explainable_warning_report(
        metrics=combined_metrics,
        feature_cols=feature_cols,
        test_meta=model_data["test_meta"],
        y_true=y_test,
        best_predictions=best_pred,
        output_path=output_dir / "reports" / "llm_explainable_warning_report.md",
        high_threshold=train_high,
        p99=p99,
    )

    # 11. Save run summary.
    summary = {
        "raw_data": args.raw_data,
        "processed_data": args.processed_data,
        "station_data": args.station_data,
        "n_rows_raw": int(len(raw)),
        "n_features_selected": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "split": split.__dict__,
        "train_high_threshold": train_high,
        "train_low_threshold": train_low,
        "train_p99_threshold": p99,
        "best_model_by_rmse": str(best_model_name),
        "outputs": str(output_dir.resolve()),
    }
    save_json(summary, output_dir / "reports" / "run_summary.json")
    print("\nPipeline completed successfully.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("Main metrics:")
    print(combined_metrics[["model", "rmse", "mae", "r2", "extreme_high_rmse", "extreme_low_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
