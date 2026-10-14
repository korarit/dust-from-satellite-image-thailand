"""
The Quartet Stacking Ensemble & Cross-Model Feature Attribution Engine (Phase 4)
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Integrates:
- 4 Base Learners: LightGBM, CatBoost, XGBoost, Random Forest
- K-Fold Out-Of-Fold (OOF) Cross-Validation Matrix
- Non-Negative Least Squares (NNLS) Meta-Regressor (w_i >= 0, sum(w) = 1.0)
- Cross-Model Feature Attribution Engine (Multi-model Consensus)
- Deterministic Persistence & End-to-End Inference
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize
from sklearn.model_selection import KFold

from src.models.base_model import BaseDustModel, compute_dust_metrics
from src.models.lgbm_model import LGBMDustModel
from src.models.catboost_model import CatBoostDustModel
from src.models.xgboost_model import XGBoostDustModel
from src.models.random_forest_model import RandomForestDustModel

logger = logging.getLogger(__name__)

MODEL_KEYS = ["lightgbm", "catboost", "xgboost", "random_forest"]


class DustStackingEnsemble(BaseDustModel):
    """
    Stacking & Blending Ensemble combining the 4-Model Quartet:
    1. LightGBM (Leaf-wise speed & Huber loss)
    2. CatBoost (Oblivious trees & directional interactions)
    3. XGBoost (Depth-wise growth & monotonic physics constraints)
    4. Random Forest (Bagging variance reduction & outlier damping)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: str = "StackingEnsemble",
        pollutant: str = "pm25",
        target_col: Optional[str] = None,
    ) -> None:
        super().__init__(model_params=config, name=name)
        self.config: Dict[str, Any] = config or {}
        self.pollutant = pollutant.lower()
        self.target_col = target_col or ("observed_pm10" if self.pollutant == "pm10" else "observed_pm25")

        # Extract sub-configs
        models_cfg = self.config.get("models", {})
        feat_cfg = self.config.get("features", {})
        self.mono_constraints: Dict[str, int] = feat_cfg.get("monotonic_constraints", {})

        # Instantiate 4 base learners
        self.base_models: Dict[str, BaseDustModel] = {
            "lightgbm": LGBMDustModel(
                model_params=models_cfg.get("lightgbm", {}),
                monotonic_constraints=self.mono_constraints,
            ),
            "catboost": CatBoostDustModel(
                model_params=models_cfg.get("catboost", {}),
                monotonic_constraints=self.mono_constraints,
            ),
            "xgboost": XGBoostDustModel(
                model_params=models_cfg.get("xgboost", {}),
                monotonic_constraints=self.mono_constraints,
            ),
            "random_forest": RandomForestDustModel(
                model_params=models_cfg.get("random_forest", {}),
            ),
        }

        # Meta-learner weights: w_lgb, w_cat, w_xgb, w_rf
        self.weights: Dict[str, float] = {k: 0.25 for k in MODEL_KEYS}
        self.oof_predictions_: Optional[pd.DataFrame] = None
        self.oof_metrics_: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
        cv: int = 5,
        random_state: int = 42,
    ) -> "DustStackingEnsemble":
        """
        Fits the ensemble using K-Fold Out-Of-Fold (OOF) stacking:
        1. Generates out-of-fold prediction matrix S (N x 4)
        2. Optimizes meta-learner non-negative weights (w_i >= 0, sum(w) = 1.0)
        3. Trains full base learners on 100% of the training dataset
        """
        X_arr, f_names = self._prepare_inputs(X, feature_names)
        y_arr = self._prepare_target(y)
        self.feature_names_ = f_names

        n_samples = len(y_arr)
        effective_cv = min(cv, n_samples)
        if effective_cv < 2:
            logger.info("Fast Direct Training Mode (No K-Fold): Training Quartet on %d samples...", n_samples)
            val_size = min(int(n_samples * 0.15), 50000)
            train_size = n_samples - val_size
            indices = np.arange(n_samples)
            np.random.RandomState(random_state).shuffle(indices)
            tr_idx, val_idx = indices[:train_size], indices[train_size:]

            X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
            X_vl, y_vl = X_arr[val_idx], y_arr[val_idx]

            logger.info("  • Training partition  : %d samples", len(X_tr))
            logger.info("  • Validation partition: %d samples", len(X_vl))

            val_preds_matrix = np.zeros((len(X_vl), len(MODEL_KEYS)), dtype=np.float32)

            for m_idx, m_key in enumerate(MODEL_KEYS, 1):
                model = self.base_models[m_key]
                logger.info("  [%d/%d] Training %s on %d samples...", m_idx, len(MODEL_KEYS), model.name, len(X_tr))
                t_start = time.perf_counter()
                model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), feature_names=self.feature_names_)
                t_elapsed = time.perf_counter() - t_start

                preds = model.predict(X_vl)
                val_preds_matrix[:, m_idx - 1] = preds
                m_metrics = compute_dust_metrics(y_vl, preds, pollutant=self.pollutant)
                self.oof_metrics_[m_key] = m_metrics
                logger.info("  ✓ [%d/%d] %-14s completed in %6.1fs | Val MAE: %6.3f | RMSE: %6.3f | R²: %6.4f",
                            m_idx, len(MODEL_KEYS), model.name, t_elapsed,
                            m_metrics["mae"], m_metrics["rmse"], m_metrics["r2"])

            loss_metric = self.config.get("general", {}).get("eval_metric", "mae").lower()

            def objective(w: np.ndarray) -> float:
                pred = val_preds_matrix @ w
                diff = pred - y_vl
                if loss_metric == "mae":
                    return float(np.mean(np.sqrt(diff ** 2 + 1e-6)))
                return float(np.mean(diff ** 2))

            init_w = np.ones(len(MODEL_KEYS), dtype=np.float64) / len(MODEL_KEYS)
            bounds = [(0.0, 1.0) for _ in range(len(MODEL_KEYS))]
            constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

            opt_res = minimize(objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
            if opt_res.success and np.sum(opt_res.x) > 0:
                normalized_weights = np.clip(opt_res.x, 0.0, 1.0)
                normalized_weights = normalized_weights / np.sum(normalized_weights)
            else:
                raw_weights, _ = nnls(val_preds_matrix, y_vl)
                sum_w = np.sum(raw_weights)
                normalized_weights = raw_weights / sum_w if sum_w > 1e-7 else init_w

            self.weights = {m_key: float(normalized_weights[i]) for i, m_key in enumerate(MODEL_KEYS)}
            logger.info("Optimal Non-Negative Meta-Weights: %s",
                        {k: round(v, 4) for k, v in self.weights.items()})

            ens_val_preds = np.sum(val_preds_matrix * normalized_weights, axis=1)
            self.oof_metrics_["ensemble"] = compute_dust_metrics(y_vl, ens_val_preds, pollutant=self.pollutant)
            logger.info("  ⭐ VAL ENSEMBLE  MAE: %.3f µg/m³, RMSE: %.3f, R2: %.4f",
                        self.oof_metrics_["ensemble"]["mae"],
                        self.oof_metrics_["ensemble"]["rmse"],
                        self.oof_metrics_["ensemble"]["r2"])

            self.oof_predictions_ = pd.DataFrame(
                val_preds_matrix,
                columns=[f"oof_{k}" for k in MODEL_KEYS]
            )
            self.oof_predictions_["oof_ensemble"] = ens_val_preds
            self.oof_predictions_[self.target_col] = y_vl
            if self.target_col != "observed_pm25":
                self.oof_predictions_["observed_pm25"] = y_vl

            self.is_fitted_ = True
            return self

        # ----------------------------------------------------------------------
        # Step 1: Out-Of-Fold (OOF) Prediction Matrix
        # ----------------------------------------------------------------------
        oof_matrix = np.zeros((n_samples, len(MODEL_KEYS)), dtype=np.float32)
        kf = KFold(n_splits=effective_cv, shuffle=True, random_state=random_state)

        logger.info("Executing %d-Fold Cross-Validation for OOF Stacking Matrix...", effective_cv)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_vl, y_vl = X_arr[val_idx], y_arr[val_idx]

            logger.info(">>> [Fold %d/%d] Training base models on %d samples (validation: %d samples)...",
                        fold_idx + 1, effective_cv, len(X_tr), len(X_vl))
            for m_idx, m_key in enumerate(MODEL_KEYS, 1):
                fold_model = self._create_base_model_instance(m_key)
                logger.info("  [Fold %d/%d] [%d/4] Training %s...", fold_idx + 1, effective_cv, m_idx, fold_model.name)
                t_fstart = time.perf_counter()
                fold_model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), feature_names=self.feature_names_)
                t_felapsed = time.perf_counter() - t_fstart
                oof_matrix[val_idx, m_idx - 1] = fold_model.predict(X_vl)
                logger.info("  ✓ [Fold %d/%d] [%d/4] %s completed in %.1fs",
                            fold_idx + 1, effective_cv, m_idx, fold_model.name, t_felapsed)

        # Store OOF DataFrame
        self.oof_predictions_ = pd.DataFrame(
            oof_matrix,
            columns=[f"oof_{k}" for k in MODEL_KEYS]
        )
        self.oof_predictions_[self.target_col] = y_arr
        if self.target_col != "observed_pm25":
            self.oof_predictions_["observed_pm25"] = y_arr

        # Compute individual base model OOF performance
        for m_idx, m_key in enumerate(MODEL_KEYS):
            self.oof_metrics_[m_key] = compute_dust_metrics(y_arr, oof_matrix[:, m_idx], pollutant=self.pollutant)
            logger.info("  • OOF %-14s MAE: %.3f µg/m³, RMSE: %.3f, R2: %.4f",
                        m_key.upper(),
                        self.oof_metrics_[m_key]["mae"],
                        self.oof_metrics_[m_key]["rmse"],
                        self.oof_metrics_[m_key]["r2"])

        # ----------------------------------------------------------------------
        # Step 2: Meta-Learner Optimization (Constrained Convex Optimization)
        # ----------------------------------------------------------------------
        loss_metric = self.config.get("general", {}).get("eval_metric", "mae").lower()

        def objective(w: np.ndarray) -> float:
            pred = oof_matrix @ w
            diff = pred - y_arr
            if loss_metric == "mae":
                # Smooth Charbonnier / Pseudo-Huber loss approximating L1
                return float(np.mean(np.sqrt(diff ** 2 + 1e-6)))
            return float(np.mean(diff ** 2))

        init_w = np.ones(len(MODEL_KEYS), dtype=np.float64) / len(MODEL_KEYS)
        bounds = [(0.0, 1.0) for _ in range(len(MODEL_KEYS))]
        constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

        opt_res = minimize(objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        if opt_res.success and np.sum(opt_res.x) > 0:
            normalized_weights = np.clip(opt_res.x, 0.0, 1.0)
            normalized_weights = normalized_weights / np.sum(normalized_weights)
        else:
            # Fallback to Simplex-projected NNLS if SLSQP fails
            raw_weights, _ = nnls(oof_matrix, y_arr)
            sum_w = np.sum(raw_weights)
            if sum_w > 1e-7:
                normalized_weights = raw_weights / sum_w
            else:
                normalized_weights = init_w

        self.weights = {m_key: float(normalized_weights[i]) for i, m_key in enumerate(MODEL_KEYS)}
        logger.info("Optimal Non-Negative Meta-Weights: %s",
                    {k: round(v, 4) for k, v in self.weights.items()})

        # Calculate Ensemble OOF Performance
        oof_ensemble = np.sum(oof_matrix * normalized_weights, axis=1)
        self.oof_predictions_["oof_ensemble"] = oof_ensemble
        self.oof_metrics_["ensemble"] = compute_dust_metrics(y_arr, oof_ensemble, pollutant=self.pollutant)
        logger.info("  ⭐ OOF ENSEMBLE    MAE: %.3f µg/m³, RMSE: %.3f, R2: %.4f",
                    self.oof_metrics_["ensemble"]["mae"],
                    self.oof_metrics_["ensemble"]["rmse"],
                    self.oof_metrics_["ensemble"]["r2"])

        # ----------------------------------------------------------------------
        # Step 3: Full Base Model Training on 100% of Data
        # ----------------------------------------------------------------------
        logger.info("Training Quartet Base Learners on 100%% of training dataset...")
        for m_idx, (m_key, m) in enumerate(self.base_models.items(), 1):
            logger.info("  [100%% Fit %d/4] Training %s on %d samples...", m_idx, m.name, len(X_arr))
            t_mstart = time.perf_counter()
            m.fit(X_arr, y_arr, eval_set=eval_set, feature_names=self.feature_names_)
            logger.info("  ✓ [100%% Fit %d/4] %s completed in %.1fs", m_idx, m.name, time.perf_counter() - t_mstart)

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Blended PM2.5 Prediction:
        y_final = sum(w_i * y_hat_i) clamped to >= 0.0 µg/m³.
        """
        if not self.is_fitted_:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        X_arr, _ = self._prepare_inputs(X, feature_names=self.feature_names_)
        blended = np.zeros(len(X_arr), dtype=np.float64)

        for m_key, m in self.base_models.items():
            w = self.weights.get(m_key, 0.0)
            if w > 0.0:
                blended += w * m.predict(X_arr)

        return np.maximum(blended, 0.0)

    def predict_components(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """Generates individual predictions for all 4 models alongside final blend."""
        if not self.is_fitted_:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        X_arr, _ = self._prepare_inputs(X, feature_names=self.feature_names_)
        comp_dict = {}

        for m_key, m in self.base_models.items():
            comp_dict[f"pred_{m_key}"] = m.predict(X_arr)

        # Final blend
        blended = np.zeros(len(X_arr), dtype=np.float64)
        for m_key in MODEL_KEYS:
            w = self.weights.get(m_key, 0.0)
            blended += w * comp_dict[f"pred_{m_key}"]

        comp_dict["pred_ensemble"] = np.maximum(blended, 0.0)
        return pd.DataFrame(comp_dict)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Returns ensemble-weighted feature importance table."""
        attrib_df = self.get_cross_model_feature_attribution()
        res = attrib_df[["feature", "ensemble_importance"]].copy()
        res.columns = ["feature", "importance"]
        res["normalized_importance"] = res["importance"]
        return res

    def get_cross_model_feature_attribution(self) -> pd.DataFrame:
        """
        Cross-Model Feature Attribution Engine:
        Gathers normalized feature importance from all 4 models and
        computes consensus weighted importance based on meta-learner weights.
        """
        if not self.is_fitted_:
            raise RuntimeError(f"{self.name} is not fitted.")

        records = {fn: {} for fn in self.feature_names_}

        for m_key, m in self.base_models.items():
            imp_df = m.get_feature_importance()
            # Map normalized importance
            imp_map = dict(zip(imp_df["feature"], imp_df["normalized_importance"]))
            for fn in self.feature_names_:
                records[fn][f"{m_key}_importance"] = float(imp_map.get(fn, 0.0))

        rows = []
        for fn, vals in records.items():
            row = {"feature": fn}
            row.update(vals)
            # Weighted importance
            w_imp = sum(self.weights.get(k, 0.0) * row[f"{k}_importance"] for k in MODEL_KEYS)
            row["ensemble_importance"] = float(w_imp)
            rows.append(row)

        df = pd.DataFrame(rows)
        # Normalize ensemble importance to sum to 1.0
        total_w = df["ensemble_importance"].sum()
        if total_w > 0:
            df["ensemble_importance"] = df["ensemble_importance"] / total_w

        df = df.sort_values(by="ensemble_importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def save(self, dir_path: Union[str, Path]) -> None:
        """Persists all 4 base models and ensemble metadata into directory with pollutant suffix."""
        save_dir = Path(dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        pol = self.pollutant.lower()
        # Save base models with pollutant in filename: e.g. lightgbm_model_pm25.joblib
        for m_key, m in self.base_models.items():
            m_path_pol = save_dir / f"{m_key}_model_{pol}.joblib"
            m.save(m_path_pol)
            # Also save standard filename for backward compatibility
            m_path_std = save_dir / f"{m_key}_model.joblib"
            m.save(m_path_std)

        # Save metadata and weights
        meta = {
            "name": self.name,
            "pollutant": self.pollutant,
            "target_col": self.target_col,
            "weights": self.weights,
            "feature_names_": self.feature_names_,
            "is_fitted_": self.is_fitted_,
            "oof_metrics": self.oof_metrics_,
            "monotonic_constraints": self.mono_constraints,
        }
        with open(save_dir / f"ensemble_meta_{pol}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(save_dir / "ensemble_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save OOF predictions if present
        if self.oof_predictions_ is not None:
            self.oof_predictions_.to_parquet(save_dir / f"oof_predictions_{pol}.parquet", index=False)
            self.oof_predictions_.to_parquet(save_dir / "oof_predictions.parquet", index=False)

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "DustStackingEnsemble":
        """Loads ensemble checkpoint and all 4 constituent base models."""
        load_dir = Path(dir_path)
        meta_path = load_dir / "ensemble_meta.json"
        if not meta_path.exists():
            candidates = list(load_dir.glob("ensemble_meta*.json"))
            if candidates:
                meta_path = candidates[0]
            else:
                raise FileNotFoundError(f"Missing ensemble metadata at {load_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        pol = meta.get("pollutant", "pm25")
        ensemble = cls(
            config={"features": {"monotonic_constraints": meta.get("monotonic_constraints", {})}},
            name=meta.get("name", "StackingEnsemble"),
            pollutant=pol,
            target_col=meta.get("target_col"),
        )
        ensemble.feature_names_ = meta["feature_names_"]
        ensemble.weights = meta["weights"]
        ensemble.is_fitted_ = meta["is_fitted_"]
        ensemble.oof_metrics_ = meta.get("oof_metrics", {})

        # Load each base model
        model_classes = {
            "lightgbm": LGBMDustModel,
            "catboost": CatBoostDustModel,
            "xgboost": XGBoostDustModel,
            "random_forest": RandomForestDustModel,
        }
        for m_key, cls_type in model_classes.items():
            m_path_pol = load_dir / f"{m_key}_model_{pol}.joblib"
            m_path_std = load_dir / f"{m_key}_model.joblib"
            if m_path_pol.exists():
                ensemble.base_models[m_key] = cls_type.load(m_path_pol)
            elif m_path_std.exists():
                ensemble.base_models[m_key] = cls_type.load(m_path_std)
            else:
                logger.warning("Base model checkpoint for %s not found in %s", m_key, load_dir)

        # Load OOF predictions if present
        oof_path_pol = load_dir / f"oof_predictions_{pol}.parquet"
        oof_path_std = load_dir / "oof_predictions.parquet"
        if oof_path_pol.exists():
            ensemble.oof_predictions_ = pd.read_parquet(oof_path_pol)
        elif oof_path_std.exists():
            ensemble.oof_predictions_ = pd.read_parquet(oof_path_std)

        return ensemble

    def _create_base_model_instance(self, model_key: str) -> BaseDustModel:
        """Instantiates a fresh base learner copy for fold training."""
        models_cfg = self.config.get("models", {})
        if model_key == "lightgbm":
            return LGBMDustModel(
                model_params=models_cfg.get("lightgbm", {}),
                monotonic_constraints=self.mono_constraints,
            )
        elif model_key == "catboost":
            return CatBoostDustModel(
                model_params=models_cfg.get("catboost", {}),
                monotonic_constraints=self.mono_constraints,
            )
        elif model_key == "xgboost":
            return XGBoostDustModel(
                model_params=models_cfg.get("xgboost", {}),
                monotonic_constraints=self.mono_constraints,
            )
        elif model_key == "random_forest":
            return RandomForestDustModel(
                model_params=models_cfg.get("random_forest", {}),
            )
        else:
            raise ValueError(f"Unknown model key: {model_key}")
