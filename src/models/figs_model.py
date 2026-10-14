"""
Fast Interpretable Greedy-Tree Sums (FIGS) Regressor Wrapper for Thailand Dust Prediction Satellite (Phase 4)
Lightweight sum-of-small-trees ensemble (UC Berkeley / imodels) providing extreme transparency and instant inference.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imodels import FIGSRegressor

from src.models.base_model import BaseDustModel

logger = logging.getLogger(__name__)


class FIGSDustModel(BaseDustModel):
    """
    FIGS (Fast Interpretable Greedy-Tree Sums) Model Wrapper.
    Features:
    - Disentangles additive patterns via a concurrent sum of shallow greedy trees
    - Strict global split/rules budget (e.g., max_rules=20) preventing overfitting
    - Ultra-lightweight footprint (< 50 KB) and microsecond tabular inference
    - Clean native feature importance
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "FIGS",
    ) -> None:
        super().__init__(model_params=model_params, name=name)

        # Default parameters
        self.params: Dict[str, Any] = {
            "max_rules": 20,
            "max_trees": 5,
            "random_state": 42,
        }
        if model_params:
            self.params.update(model_params)

        self.imputer: Optional[SimpleImputer] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "FIGSDustModel":
        """Fits FIGS regressor with internal median imputation."""
        X_arr, f_names = self._prepare_inputs(X, feature_names)
        y_arr = self._prepare_target(y)
        self.feature_names_ = f_names

        self.imputer = SimpleImputer(strategy="median")
        X_clean = self.imputer.fit_transform(X_arr)

        self.estimator = FIGSRegressor(**self.params)
        self.estimator.fit(X_clean, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts PM concentrations clamped to >= 0.0."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        X_arr, _ = self._prepare_inputs(X, feature_names=self.feature_names_)
        if self.imputer is not None:
            X_clean = self.imputer.transform(X_arr)
        else:
            X_clean = X_arr

        preds = self.estimator.predict(X_clean)
        return np.maximum(preds, 0.0)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Returns sorted feature importance table."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} is not fitted.")

        if hasattr(self.estimator, "feature_importances_") and self.estimator.feature_importances_ is not None:
            raw_imp = np.asarray(self.estimator.feature_importances_, dtype=np.float64)
        else:
            raw_imp = np.zeros(len(self.feature_names_), dtype=np.float64)

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
            "imputer": self.imputer,
            "estimator": self.estimator,
        }
        joblib.dump(checkpoint, path)
        logger.info("Saved %s checkpoint to: %s", self.name, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FIGSDustModel":
        """Deserializes model checkpoint from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = joblib.load(path)
        instance = cls(model_params=checkpoint.get("params", {}), name=checkpoint.get("name", "FIGS"))
        instance.feature_names_ = checkpoint.get("feature_names_", [])
        instance.is_fitted_ = checkpoint.get("is_fitted_", False)
        instance.imputer = checkpoint.get("imputer", None)
        instance.estimator = checkpoint.get("estimator", None)
        logger.info("Loaded %s checkpoint from: %s", instance.name, path)
        return instance
