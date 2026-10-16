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
import time
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
from src.models.hist_gbr_model import HistGBRDustModel
from src.models.ebm_model import EBMDustModel
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
    if model_key == "hist_gbr":
        return HistGBRDustModel(model_params=params, monotonic_constraints=monotonic_constraints)
    if model_key == "ebm":
        return EBMDustModel(model_params=params)
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
        # NOTE: CatBoostRegressor has NO `tweedie_variance_power` kwarg — the power
        # is encoded inside `loss_function="Tweedie:variance_power=<p>"` only.
        # Suggesting it as an Optuna param is fine, but it must NOT be passed to
        # the model constructor (TypeError otherwise).
        tvar = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
        params.update({
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "loss_function": f"Tweedie:variance_power={tvar:.3f}",
        })
        params.pop("tweedie_variance_power", None)
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
    elif model_key == "hist_gbr":
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, step=2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "loss": trial.suggest_categorical("loss", ["poisson", "absolute_error", "squared_error"]),
        })
    elif model_key == "ebm":
        params.update({
            "interactions": trial.suggest_int("interactions", 0, 30),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_rounds": trial.suggest_int("max_rounds", 500, 8000, step=250),
            "outer_bags": trial.suggest_int("outer_bags", 4, 14, step=2),
        })

    return params


def split_holdout(
    df: pd.DataFrame,
    seed: int,
    val_frac: float = 0.15,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold-out split (matches pipeline fast direct mode).

    P0-1 fix: when group_col (station/grid/province) is supplied, splits by
    GROUP (GroupShuffleSplit) so no station appears on both sides. Falls back
    to the legacy seeded row shuffle only when no group column exists.
    """
    n = len(df)
    if group_col and group_col in df.columns:
        from sklearn.model_selection import GroupShuffleSplit
        groups = df[group_col].astype(str).to_numpy()
        if len(np.unique(groups)) >= 5:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr_idx, val_idx = next(gss.split(df, groups=groups))
            return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
        logger.warning("Too few groups for group holdout; using row shuffle.")
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def split_out_of_time(
    df: pd.DataFrame,
    split_date: str = "2026-01-01",
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    P0-1 fix: Out-of-Time split (train 2024-25 / test 2026) as the primary
    generalisation number alongside GroupKFold. No shuffling — strict past->future.
    """
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    cutoff = pd.to_datetime(split_date, utc=True)
    return (
        df[ts < cutoff].reset_index(drop=True),
        df[ts >= cutoff].reset_index(drop=True),
    )


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
    log_every: int = 1,
) -> Dict[str, Any]:
    """
    Tunes base learners on the supplied 2 km training dataframe and writes the best
    parameters to ``configs/tuned_params_<pollutant>.yaml`` (returned as a dict).

    Returns a mapping ``{"<feature_set>__<provider>": {model_key: params}}``.

    Live logging (per --tune pipeline runs):
    - header per model: ``=== [key|model] (i/N)`` with trial budget,
    - one line per trial (or every ``log_every`` trials): current MAE/RMSE/R2 +
      running best MAE, with ``*`` on improvement,
    - footer per model: best MAE/R2/elapsed + best-trial number.
    Per-model trial CSVs (``{key}__{model}_trials.csv``) are still written at
    the end of each model — the console log is the realtime channel.

    Crash-safety: best-so-far YAML + ``tuning_summary`` are checkpointed after
    EVERY model (not once at the end), so a later crash loses nothing.
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
    summary_csv = reports_dir / f"tuning_summary_{pol}.csv"

    # --------------------------------------------------------------------------
    # Resume support: if the checkpoint YAML already contains tuned params for a
    # (feature_set, provider, model) combo, skip it instead of re-tuning. Only
    # combos missing from the checkpoint are tuned fresh.
    # --------------------------------------------------------------------------
    tuned_prev: Dict[str, Dict[str, Any]] = {}
    if out_yaml.exists():
        try:
            with open(out_yaml, "r", encoding="utf-8") as f:
                _prev_payload = yaml.safe_load(f) or {}
            tuned_prev = _prev_payload.get("tuned") or {}
        except Exception as exc:
            logger.warning("Could not read existing checkpoint %s (%s); tuning from scratch.", out_yaml, exc)
            tuned_prev = {}
        if tuned_prev:
            _prev_total = sum(len(v) for v in tuned_prev.values())
            logger.info("⏩ Resume: %d combos already tuned in %s -> will be skipped",
                        _prev_total, out_yaml)
    tuned = {k: dict(v) for k, v in tuned_prev.items()}
    # Preserve summary rows (metrics) of previously completed combos so the CSV
    # checkpoint stays complete across resumed runs.
    if summary_csv.exists():
        try:
            _prev_summary = pd.read_csv(summary_csv)
            _prev_bys = {
                (str(r["feature_set"]), str(r["provider"]), str(r["model"])): dict(r)
                for _, r in _prev_summary.iterrows()
            }
            for (fs_, prov_, mk_), row in _prev_bys.items():
                if mk_ in tuned_prev.get(f"{fs_}__{prov_.lower()}", {}):
                    summary_rows.append(row)
        except Exception as exc:
            logger.warning("Could not inherit prior tuning summary (%s); it will be rebuilt.", exc)

    def _write_checkpoint() -> None:
        """Persists best-so-far YAML + summary immediately (crash-safe).

        Called after EVERY model finishes, so completed models are never lost
        if a later model crashes the run. Best-effort: never raises.
        """
        try:
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
                "completed_models": sum(len(v) for v in tuned.values()),
                "tuned": _to_native(tuned),
            }
            out_yaml.parent.mkdir(parents=True, exist_ok=True)
            with open(out_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        except Exception as exc:  # checkpoint must never break tuning
            logger.warning("Checkpoint write failed (tuning continues): %s", exc)

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

            key = f"{fs_label}__{prov.lower()}"
            _remaining = [m for m in models if m not in tuned_prev.get(key, {})]
            if not _remaining:
                logger.info("⏩ [%s] all %d models already tuned -> SKIP (resume)", key, len(models))
                continue

            # P0-1 fix: group-aware holdout (station/cell) + P2-2: subsample warning.
            _gcol = next((c for c in ("station_code", "grid_id", "provider") if c in subset.columns), None)
            tr_full, ho = split_holdout(subset, seed, group_col=_gcol)
            if subsample_frac < 1.0:
                logger.warning(
                    "Tuner subsample %.0f%% is NOT representative of full-train distribution; "
                    "tuned params are 2km-reference only — re-validate on full data.",
                    subsample_frac * 100,
                )
                tr_full = tr_full.sample(frac=max(0.05, subsample_frac), random_state=seed).reset_index(drop=True)

            X_tr, y_tr = tr_full[fs_cols], tr_full[tgt_col].astype(float)
            X_ho, y_ho = ho[fs_cols], ho[tgt_col].astype(float)

            tuned[key] = tuned.get(key, {})
            logger.info("\n" + "-" * 80)
            logger.info("▶ [%s] N_train=%d (subsampled) | N_holdout=%d | features=%d | to tune: %s",
                        key, len(X_tr), len(X_ho), len(fs_cols), _remaining)
            logger.info("-" * 80)

            for mi, model_key in enumerate(models, start=1):
                if model_key in tuned_prev.get(key, {}):
                    logger.info("  === [%s|%s] (%d/%d) already tuned -> SKIP", key, model_key, mi, len(models))
                    continue
                base_params = dict(base_models_cfg.get(model_key, {}))
                logger.info("  === [%s|%s] (%d/%d) trials=%d -- start",
                            key, model_key, mi, len(models), trials)
                t_model_start = time.perf_counter()

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

                def _trial_log_callback(study, trial) -> None:
                    try:
                        n_done = trial.number + 1
                        is_last = (trial.number == trials - 1)
                        if (n_done % max(1, log_every) != 0) and not is_last:
                            return
                        val = trial.value if trial.value is not None else float("nan")
                        best = study.best_value if study.best_value is not None else float("nan")
                        rmse = float(trial.user_attrs.get("rmse", float("nan")))
                        r2 = float(trial.user_attrs.get("r2", float("nan")))
                        star = " * new best" if (trial.value is not None
                                                 and study.best_trial is not None
                                                 and trial.number == study.best_trial.number) else ""
                        elapsed = time.perf_counter() - t_model_start
                        logger.info("    - [%s|%s] trial %2d/%d MAE=%.3f RMSE=%.3f R2=%.4f | best=%.3f (+%ds)%s",
                                    key, model_key, n_done, trials, val, rmse, r2, best,
                                    int(elapsed), star)
                    except Exception:  # logging must never break tuning
                        pass

                study = optuna.create_study(
                    study_name=f"{key}__{model_key}",
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=seed),
                )
                study.optimize(objective, n_trials=trials, show_progress_bar=False,
                               callbacks=[_trial_log_callback], catch=(Exception,))

                best_params = dict(base_params)
                best_params.update(study.best_params)
                if model_key == "catboost":
                    # Drop the Optuna-only search key; the wrapper + YAML must only
                    # carry `loss_function="Tweedie:variance_power=<p>"`.
                    best_params.pop("tweedie_variance_power", None)
                    if "tweedie_variance_power" in study.best_params:
                        best_params["loss_function"] = f"Tweedie:variance_power={study.best_params['tweedie_variance_power']:.3f}"
                tuned[key][model_key] = best_params

                if save_history:
                    hist = study.trials_dataframe()
                    hist.insert(0, "feature_set", fs_label)
                    hist.insert(1, "provider", prov)
                    hist.insert(2, "model", model_key)
                    master_history.append(hist)
                    hist.to_csv(reports_dir / f"{key}__{model_key}_trials.csv", index=False)

                bt = None
                try:
                    bt = study.best_trial
                except Exception:
                    pass
                if bt is None or bt.value is None:
                    logger.warning("  [SKIP] %-13s all %d trials failed; no checkpoint for this model",
                                   model_key, trials)
                    continue
                elapsed_model = time.perf_counter() - t_model_start
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
                logger.info("  [OK] %-13s best MAE: %6.3f | R2: %6.4f | (trial %d/%d, %ds)",
                            model_key, float(bt.value), float(bt.user_attrs.get("r2", 0.0)),
                            bt.number + 1, trials, int(elapsed_model))
                # Incremental checkpoint: best-so-far YAML + summary are durable
                # NOW — a later crash (or Colab preemption) loses nothing.
                _write_checkpoint()
                logger.info("  checkpoint saved -> %s", out_yaml)

    if save_history and master_history:
        master_df = pd.concat(master_history, ignore_index=True)
        master_csv = reports_dir / f"tuning_history_{pol}.csv"
        master_df.to_csv(master_csv, index=False)
        logger.info("\nSaved full tuning history (%d trials) -> %s", len(master_df), master_csv)

    if save_history and summary_rows:
        # Already checkpointed per-model; this is just the final confirmation.
        logger.info("Saved tuning summary -> %s", summary_csv)

    _write_checkpoint()
    logger.info("Saved tuned parameters -> %s", out_yaml)
    logger.info("\n🎉 HYPERPARAMETER TUNING COMPLETED (%s)!", pol.upper())

    return tuned
