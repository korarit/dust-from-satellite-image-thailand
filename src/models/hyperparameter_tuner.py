"""
Optuna Hyperparameter Tuning Engine (Phase 4)
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Shared, pipeline-callable tuning core:
- Tunes base learners on the 2 km reference grid per feature set x provider.
- Objective: hold-out MAE (single seeded 85/15 split -- NO K-Fold).
- Persists full trial history (params + metrics) as CSV and writes the best
  parameters to ``configs/tuned_params_<pollutant>.yaml`` for pipeline reuse.

Optuna is imported lazily so the main pipeline keeps working without it
(unless tuning is explicitly requested).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.config.paths import (
    get_reports_dir,
    get_tuned_params_path,
)
from src.features import (
    FULL_MULTIMODAL_COLUMNS,
    NO_WEATHER_FEATURE_COLUMNS,
    resolve_dataset_providers,
)
from src.models.lgbm_model import LGBMDustModel
from src.models.catboost_model import CatBoostDustModel
from src.models.xgboost_model import XGBoostDustModel
from src.models.random_forest_model import RandomForestDustModel
from src.models.extra_trees_model import ExtraTreesDustModel
from src.models.figs_model import FIGSDustModel
from src.models.nystroem_model import NystroemRidgeDustModel
from src.models.base_model import compute_dust_metrics
from src.models.ensemble_model import MODEL_KEYS

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_SETS: Dict[str, List[str]] = {
    "full": list(FULL_MULTIMODAL_COLUMNS),
    "pure_satellite": list(NO_WEATHER_FEATURE_COLUMNS),
}

AUTHORIZED_PROVIDERS = ["Air4Thai", "AirGradient"]


def build_model(model_key: str, params: Dict[str, Any], monotonic_constraints: Dict[str, int]):
    """Instantiates a base learner wrapper with the given hyper-parameters."""
    if model_key == "lightgbm":
        return LGBMDustModel(model_params=params, monotonic_constraints=monotonic_constraints)
    if model_key == "catboost":
        return CatBoostDustModel(model_params=params, monotonic_constraints=monotonic_constraints)
    if model_key == "xgboost":
        return XGBoostDustModel(model_params=params, monotonic_constraints=monotonic_constraints)
    if model_key == "extra_trees":
        return ExtraTreesDustModel(model_params=params)
    if model_key == "figs":
        return FIGSDustModel(model_params=params)
    if model_key == "nystroem":
        return NystroemRidgeDustModel(model_params=params)
    if model_key == "random_forest":
        return RandomForestDustModel(model_params=params)
    raise ValueError(f"Unknown model_key: {model_key}")


def suggest_params(trial: Any, model_key: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Defines the Optuna search space per model and returns the merged params."""
    params = dict(base_params)

    if model_key == "lightgbm":
        params.update({
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
        })
    elif model_key == "catboost":
        tvar = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
        params.update({
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "tweedie_variance_power": tvar,
            "loss_function": f"Tweedie:variance_power={tvar:.3f}",
        })
    elif model_key == "xgboost":
        params.update({
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
        })
    elif model_key in ("extra_trees", "random_forest"):
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 28),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        })
    elif model_key == "figs":
        params.update({
            "max_rules": trial.suggest_int("max_rules", 10, 40),
            "max_trees": trial.suggest_int("max_trees", 3, 10),
        })
    elif model_key == "nystroem":
        params.update({
            "n_components": trial.suggest_int("n_components", 50, 300, step=10),
            "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
        })

    return params


def split_holdout(df: pd.DataFrame, seed: int, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single seeded 85/15 hold-out split (matches pipeline fast direct mode)."""
    n = len(df)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _to_native(obj: Any) -> Any:
    """Recursively converts numpy scalars/containers to native Python types for YAML dump."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_optuna_tuning(
    train_df: pd.DataFrame,
    pollutant: str = "pm25",
    base_config: Optional[Dict[str, Any]] = None,
    feature_sets: Optional[Dict[str, List[str]]] = None,
    providers: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    trials: int = 25,
    subsample_frac: float = 0.3,
    seed: int = 42,
    resolution_km: float = 2.0,
    output_yaml: Optional[Path] = None,
    save_history: bool = True,
) -> Dict[str, Any]:
    """
    Tunes base learners on the supplied 2 km training dataframe and writes the best
    parameters to ``configs/tuned_params_<pollutant>.yaml`` (returned as a dict).

    Returns a mapping ``{"<feature_set>__<provider>": {model_key: params}}``.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Optuna is required for tuning. Install it with: pip install optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pol = pollutant.lower()
    tgt_col = "observed_pm10" if pol == "pm10" else "observed_pm25"
    feature_sets = feature_sets or DEFAULT_FEATURE_SETS
    providers = providers or ["Air4Thai", "AirGradient", "Combined"]
    models = models or list(MODEL_KEYS)
    out_yaml = Path(output_yaml) if output_yaml else get_tuned_params_path(pol)

    base_config = base_config or {}
    base_models_cfg = base_config.get("models", {})
    mono = base_config.get("features", {}).get("monotonic_constraints", {})

    df_train = train_df.dropna(subset=[tgt_col]).copy() if tgt_col in train_df.columns else train_df.copy()
    if df_train.empty:
        logger.warning("No usable rows for tuning; skipping.")
        return {}
    df_train = resolve_dataset_providers(df_train, allowed_providers=tuple(AUTHORIZED_PROVIDERS))

    reports_dir = get_reports_dir(pol) / "tuning"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("🎛️  OPTUNA HYPERPARAMETER TUNING (%s) @ %.1f km", pol.upper(), resolution_km)
    logger.info("Rows: %d | Feature sets: %s | Providers: %s",
                len(df_train), list(feature_sets.keys()), providers)
    logger.info("Models: %s | Trials/model: %d | Subsample: %.0f%%", models, trials, subsample_frac * 100)
    logger.info("=" * 80)

    tuned: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    master_history: List[pd.DataFrame] = []

    for fs_label, fs_cols_all in feature_sets.items():
        fs_cols = [c for c in fs_cols_all if c in df_train.columns]
        if not fs_cols:
            logger.warning("Feature set '%s' has no matching columns; skipping.", fs_label)
            continue

        for prov in providers:
            if prov == "Combined":
                subset = df_train[df_train["provider"].isin(AUTHORIZED_PROVIDERS)].copy()
            else:
                subset = df_train[df_train["provider"] == prov].copy()
            if len(subset) < 20:
                logger.warning("Provider '%s' (%s) has too few rows (N=%d); skipping.", prov, fs_label, len(subset))
                continue

            tr_full, ho = split_holdout(subset, seed)
            if subsample_frac < 1.0:
                tr_full = tr_full.sample(frac=max(0.05, subsample_frac), random_state=seed).reset_index(drop=True)

            X_tr, y_tr = tr_full[fs_cols], tr_full[tgt_col].astype(float)
            X_ho, y_ho = ho[fs_cols], ho[tgt_col].astype(float)

            key = f"{fs_label}__{prov.lower()}"
            tuned[key] = {}
            logger.info("\n" + "-" * 80)
            logger.info("▶ [%s] N_train=%d (subsampled) | N_holdout=%d | features=%d",
                        key, len(X_tr), len(X_ho), len(fs_cols))
            logger.info("-" * 80)

            for model_key in models:
                base_params = dict(base_models_cfg.get(model_key, {}))

                def objective(trial, _mk=model_key, _bp=base_params) -> float:
                    params = suggest_params(trial, _mk, _bp)
                    model = build_model(_mk, params, mono)
                    model.fit(X_tr, y_tr, feature_names=fs_cols)
                    preds = model.predict(X_ho)
                    m = compute_dust_metrics(y_ho, preds, pollutant=pol)
                    trial.set_user_attr("rmse", float(m["rmse"]))
                    trial.set_user_attr("r2", float(m["r2"]))
                    trial.set_user_attr("pearson_r", float(m["pearson_r"]))
                    trial.set_user_attr("aqi_acc", float(m["aqi_acc"]))
                    return float(m["mae"])

                study = optuna.create_study(
                    study_name=f"{key}__{model_key}",
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=seed),
                )
                study.optimize(objective, n_trials=trials, show_progress_bar=False)

                best_params = dict(base_params)
                best_params.update(study.best_params)
                if model_key == "catboost" and "tweedie_variance_power" in study.best_params:
                    best_params["loss_function"] = f"Tweedie:variance_power={study.best_params['tweedie_variance_power']:.3f}"
                tuned[key][model_key] = best_params

                if save_history:
                    hist = study.trials_dataframe()
                    hist.insert(0, "feature_set", fs_label)
                    hist.insert(1, "provider", prov)
                    hist.insert(2, "model", model_key)
                    master_history.append(hist)
                    hist.to_csv(reports_dir / f"{key}__{model_key}_trials.csv", index=False)

                bt = study.best_trial
                summary_rows.append({
                    "feature_set": fs_label,
                    "provider": prov,
                    "model": model_key,
                    "best_mae": float(bt.value),
                    "best_rmse": float(bt.user_attrs.get("rmse", 0.0)),
                    "best_r2": float(bt.user_attrs.get("r2", 0.0)),
                    "best_pearson_r": float(bt.user_attrs.get("pearson_r", 0.0)),
                    "best_aqi_acc": float(bt.user_attrs.get("aqi_acc", 0.0)),
                    "n_train": len(X_tr),
                    "n_holdout": len(X_ho),
                    "trials": trials,
                    "params": json.dumps(best_params),
                })
                logger.info("  ✓ %-13s best MAE: %6.3f | R²: %6.4f | (trial %d)",
                            model_key, float(bt.value), float(bt.user_attrs.get("r2", 0.0)), bt.number)

    if save_history and master_history:
        master_df = pd.concat(master_history, ignore_index=True)
        master_csv = reports_dir / f"tuning_history_{pol}.csv"
        master_df.to_csv(master_csv, index=False)
        logger.info("\nSaved full tuning history (%d trials) -> %s", len(master_df), master_csv)

    if save_history and summary_rows:
        summary_csv = reports_dir / f"tuning_summary_{pol}.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        logger.info("Saved tuning summary -> %s", summary_csv)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pollutant": pol,
        "resolution_km": resolution_km,
        "subsample_frac": subsample_frac,
        "trials": trials,
        "seed": seed,
        "feature_sets": list(feature_sets.keys()),
        "providers": providers,
        "models": models,
        "tuned": _to_native(tuned),
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    logger.info("Saved tuned parameters -> %s", out_yaml)
    logger.info("\n🎉 HYPERPARAMETER TUNING COMPLETED (%s)!", pol.upper())

    return tuned
