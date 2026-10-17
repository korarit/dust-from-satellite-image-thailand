"""
Phase 4: Hyperparameter Fine-Tuning CLI (Optuna) for the Dust Model Suite
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Thin CLI wrapper over ``src.models.hyperparameter_tuner.run_optuna_tuning``:
loads the unified feature matrix, uses the temporal TRAIN split (2 km reference),
and writes the best parameters to ``{outputs}/<pollutant>/tuned_params_<pollutant>.yaml``
for pipeline reuse (configs/ holds only a bundled default fallback).

Usage:
    python tune_hyperparameters.py --pollutant pm25 --trials 30 --subsample-frac 0.3
    python tune_hyperparameters.py --providers Combined --feature-sets full \
        --models lightgbm,catboost,xgboost
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.config.paths import (
    FEATURES_DIR,
    MODEL_CONFIG_PATH,
    get_tuned_params_path,
    load_yaml_config,
)
from src.models.ensemble_model import MODEL_KEYS
from src.models.hyperparameter_tuner import DEFAULT_FEATURE_SETS, run_optuna_tuning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for _noisy in ("interpret", "interpret.utils._native", "interpret.utils._compressed_dataset", "interpret.glassbox._ebm._boost"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("tune_hyperparameters")


def _parse_list(value: str, allowed: Optional[List[str]] = None) -> List[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if allowed is not None:
        bad = [i for i in items if i not in allowed]
        if bad:
            raise ValueError(f"Unknown entries {bad}; allowed: {allowed}")
    return items


def load_training_matrix(pollutant: str, matrix_path: Optional[str], split_date: str) -> Tuple[pd.DataFrame, Path]:
    """Loads the unified feature matrix and returns the temporal TRAIN split."""
    path = Path(matrix_path) if matrix_path else FEATURES_DIR / f"unified_{pollutant}_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df[ts < pd.to_datetime(split_date, utc=True)].copy()
    return df, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for the Dust Model Suite (2 km reuse reference).")
    parser.add_argument("--pollutant", default="pm25", choices=["pm25", "pm10"])
    parser.add_argument("--matrix", default=None, help="Optional path to the unified feature matrix parquet.")
    parser.add_argument("--split-date", default="2026-01-01", help="Temporal train/test cutoff (tuning uses TRAIN only).")
    parser.add_argument("--resolution-km", type=float, default=2.0)
    parser.add_argument("--feature-sets", default="full,pure_satellite", help="Comma list: full,pure_satellite")
    parser.add_argument("--providers", default="Air4Thai", help="Comma list: Air4Thai,AirGradient,Combined (default: Air4Thai reference only)")
    parser.add_argument("--models", default=",".join(MODEL_KEYS), help=f"Comma list from: {MODEL_KEYS}")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--subsample-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output YAML path (default: {outputs}/<pollutant>/tuned_params_<pollutant>.yaml)")
    args = parser.parse_args()

    pol = args.pollutant.lower()
    feature_sets = {
        k: v for k, v in DEFAULT_FEATURE_SETS.items()
        if k in _parse_list(args.feature_sets, allowed=list(DEFAULT_FEATURE_SETS.keys()))
    }
    providers = _parse_list(args.providers, allowed=["Air4Thai", "AirGradient", "Combined"])
    models = _parse_list(args.models, allowed=list(MODEL_KEYS))

    base_config = load_yaml_config(MODEL_CONFIG_PATH) if MODEL_CONFIG_PATH.exists() else {}
    df_train, matrix_path = load_training_matrix(pol, args.matrix, args.split_date)
    logger.info("Tuning train matrix: %s (%d rows)", matrix_path.name, len(df_train))

    run_optuna_tuning(
        train_df=df_train,
        pollutant=pol,
        base_config=base_config,
        feature_sets=feature_sets,
        providers=providers,
        models=models,
        trials=args.trials,
        subsample_frac=args.subsample_frac,
        seed=args.seed,
        resolution_km=args.resolution_km,
        output_yaml=Path(args.output) if args.output else get_tuned_params_path(pol),
    )


if __name__ == "__main__":
    main()
