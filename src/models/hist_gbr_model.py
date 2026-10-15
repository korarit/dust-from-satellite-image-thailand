"""
Histogram Gradient Boosting Regressor (sklearn) Wrapper for Thailand Dust Prediction Satellite (Phase 4)
Native-NaN, monotonic-constraint GBDT with non-Tweedie losses (poisson / absolute_error)
diversifying the ensemble for NNLS stacking.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.base_model import BaseDustModel

logger = logging.getLogger(__name__)


class HistGBRDustModel(BaseDustModel):
    """
    scikit-learn Histogram Gradient Boosting Regressor Wrapper.
    Features:
    - Native NaN support (no imputation pass required)
    - Manual monotonic constraints via `monotonic_cst` (calm air ||∇P||, split-window BTD)
    - poisson / absolute_error loss family (diversity vs LGBM/XGB/CatBoost Tweedie)
    - Built-in early stopping on internal validation split
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "HistGBR",
        monotonic_constraints: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(model_params=model_params, name=name)
        self.monotonic_constraints: Dict[str, int] = monotonic_constraints or {}

        # Default high-performance parameters (poisson keeps targets non-negative
        # with a right-skewed distribution — complements the Tweedie trio)
        self.params: Dict[str, Any] = {
            "loss": "poisson",
            "learning_rate": 0.05,
            "max_iter": 500,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "early_stopping": "auto",
            "random_state": 42,
        }
        if model_params:
            self.params.update(model_params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "HistGBRDustModel":
        """Fits HistGradientBoostingRegressor with optional monotonic constraints."""
        X_arr, f_names = self._prepare_inputs(X, feature_names)
        y_arr = self._prepare_target(y).astype(np.float64)
        self.feature_names_ = f_names

        fit_params = dict(self.params)
        if self.monotonic_constraints and len(self.feature_names_) > 0:
            m_list = [self.monotonic_constraints.get(fn, 0) for fn in self.feature_names_]
            if any(m != 0 for m in m_list):
                fit_params["monotonic_cst"] = m_list

        eval_data = None
        if eval_set is not None:
            # external validation split requires early stopping to be enabled
            if fit_params.get("early_stopping") in (False, None):
                fit_params["early_stopping"] = True
            X_val, y_val = eval_set
            X_val_arr, _ = self._prepare_inputs(X_val, feature_names=self.feature_names_)
            y_val_arr = self._prepare_target(y_val).astype(np.float64)
            eval_data = (X_val_arr, y_val_arr)

        self.estimator = HistGradientBoostingRegressor(**fit_params)

        try:
            # sklearn >= 1.3: explicit external validation split for early stopping
            self.estimator.fit(
                X_arr,
                y_arr,
                X_val=eval_data[0] if eval_data else None,
                y_val=eval_data[1] if eval_data else None,
            )
        except TypeError:
            self.estimator.fit(X_arr, y_arr)

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts PM concentrations clamped to >= 0.0 (poisson log-link already >= 0)."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        X_arr, _ = self._prepare_inputs(X, feature_names=self.feature_names_)
        preds = self.estimator.predict(X_arr)
        return np.maximum(preds, 0.0)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Returns feature importance table. HistGradientBoostingRegressor does not
        expose native gain importances; returns uniform zeros (importance for this
        model is excluded from cross-model consensus attribution).
        """
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} is not fitted.")

        raw_imp = np.zeros(len(self.feature_names_), dtype=np.float64)
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": raw_imp,
            "normalized_importance": raw_imp,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes model checkpoint to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "name": self.name,
            "params": self.params,
            "monotonic_constraints": self.monotonic_constraints,
            "feature_names_": self.feature_names_,
            "is_fitted_": self.is_fitted_,
            "estimator": self.estimator,
        }
        joblib.dump(checkpoint, path)
        logger.info("Saved %s checkpoint to: %s", self.name, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HistGBRDustModel":
        """Deserializes model checkpoint from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = joblib.load(path)
        instance = cls(
            model_params=checkpoint.get("params", {}),
            name=checkpoint.get("name", "HistGBR"),
            monotonic_constraints=checkpoint.get("monotonic_constraints", {}),
        )
        instance.feature_names_ = checkpoint.get("feature_names_", [])
        instance.is_fitted_ = checkpoint.get("is_fitted_", False)
        instance.estimator = checkpoint.get("estimator", None)
        logger.info("Loaded %s checkpoint from: %s", instance.name, path)
        return instance
