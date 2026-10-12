"""
XGBoost Regressor Wrapper for Thailand Dust Prediction Satellite (Phase 4)
Depth-wise gradient boosted trees with physics-based monotonicity constraints.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base_model import BaseDustModel

logger = logging.getLogger(__name__)


class XGBoostDustModel(BaseDustModel):
    """
    XGBoost Model Wrapper for Multi-Resolution Satellite Dust Prediction.
    Features:
    - Depth-wise tree growth with exact/hist split evaluation
    - Strict physics monotonic constraints support:
      * Calm air (||∇P|| -> 0) increases dust accumulation
      * Negative split-window (BTD_15_13) increases dust loading
    - Native feature importance extraction
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "XGBoost",
        monotonic_constraints: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(model_params=model_params, name=name)
        self.monotonic_constraints: Dict[str, int] = monotonic_constraints or {}

        # Default high-performance parameters
        self.params: Dict[str, Any] = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_estimators": 800,
            "random_state": 42,
            "n_jobs": -1,
        }
        if model_params:
            self.params.update(model_params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "XGBoostDustModel":
        """Fits XGBoost regressor with optional monotonic constraints."""
        X_arr, f_names = self._prepare_inputs(X, feature_names)
        y_arr = self._prepare_target(y)
        self.feature_names_ = f_names

        fit_params = dict(self.params)

        # Build monotone constraints tuple
        if self.monotonic_constraints and len(self.feature_names_) > 0:
            m_tuple = tuple(self.monotonic_constraints.get(fn, 0) for fn in self.feature_names_)
            if any(m != 0 for m in m_tuple):
                fit_params["monotone_constraints"] = m_tuple

        self.estimator = xgb.XGBRegressor(**fit_params)

        eval_data = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_arr, _ = self._prepare_inputs(X_val, feature_names=self.feature_names_)
            y_val_arr = self._prepare_target(y_val)
            eval_data = [(X_val_arr, y_val_arr)]

        self.estimator.fit(
            X_arr,
            y_arr,
            eval_set=eval_data,
            verbose=False,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts PM2.5 concentrations clamped to >= 0.0."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        X_arr, _ = self._prepare_inputs(X, feature_names=self.feature_names_)
        preds = self.estimator.predict(X_arr)
        return np.maximum(preds, 0.0)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Returns sorted feature importance table."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} is not fitted.")

        raw_imp = np.asarray(self.estimator.feature_importances_, dtype=np.float64)
        total_imp = np.sum(raw_imp)
        norm_imp = raw_imp / total_imp if total_imp > 0 else np.zeros_like(raw_imp)

        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": raw_imp,
            "normalized_importance": norm_imp,
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

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "XGBoostDustModel":
        """Loads serialized model checkpoint."""
        checkpoint = joblib.load(filepath)
        model = cls(
            model_params=checkpoint["params"],
            name=checkpoint["name"],
            monotonic_constraints=checkpoint.get("monotonic_constraints", {}),
        )
        model.estimator = checkpoint["estimator"]
        model.feature_names_ = checkpoint["feature_names_"]
        model.is_fitted_ = checkpoint["is_fitted_"]
        return model
