"""
Extra Trees (Extremely Randomized Trees) Regressor Wrapper for Thailand Dust Prediction Satellite (Phase 4)
Randomized cut-point bagging ensemble providing extreme variance reduction and noise damping.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from src.models.base_model import BaseDustModel

logger = logging.getLogger(__name__)


class ExtraTreesDustModel(BaseDustModel):
    """
    Extra Trees Model Wrapper for Multi-Resolution Satellite Dust Prediction.
    Features:
    - Extreme Randomization of cut-points yields superior variance reduction
    - Resilient to sensor measurement anomalies and satellite radiance noise
    - High computational speed with full multi-core parallelization
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "ExtraTrees",
    ) -> None:
        super().__init__(model_params=model_params, name=name)

        # Default parameters
        self.params: Dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 16,
            "max_features": "sqrt",
            "min_samples_split": 5,
            "min_samples_leaf": 2,
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
    ) -> "ExtraTreesDustModel":
        """Fits Extra Trees regressor on feature matrix X and target y."""
        X_arr, f_names = self._prepare_inputs(X, feature_names)
        y_arr = self._prepare_target(y)
        self.feature_names_ = f_names

        self.estimator = ExtraTreesRegressor(**self.params)
        self.estimator.fit(X_arr, y_arr)
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
            "feature_names_": self.feature_names_,
            "is_fitted_": self.is_fitted_,
            "estimator": self.estimator,
        }
        joblib.dump(checkpoint, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExtraTreesDustModel":
        """Loads serialized model checkpoint."""
        checkpoint = joblib.load(filepath)
        model = cls(
            model_params=checkpoint["params"],
            name=checkpoint["name"],
        )
        model.estimator = checkpoint["estimator"]
        model.feature_names_ = checkpoint["feature_names_"]
        model.is_fitted_ = checkpoint["is_fitted_"]
        return model
