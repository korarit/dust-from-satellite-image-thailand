"""
LightGBM Regressor Wrapper for Thailand Dust Prediction Satellite (Phase 4)
Fast leaf-wise tree growth with Huber / L1 (MAE) loss optimization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.base_model import BaseDustModel

logger = logging.getLogger(__name__)


class LGBMDustModel(BaseDustModel):
    """
    LightGBM Model Wrapper for Multi-Resolution Satellite Dust Prediction.
    Features:
    - Leaf-wise tree growth with Huber / L1 loss
    - Fast histogram-based split finding across nationwide grid points
    - Built-in monotonicity constraints support
    - Feature importance extraction (Gain & Split)
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "LightGBM",
        monotonic_constraints: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(model_params=model_params, name=name)
        self.monotonic_constraints: Dict[str, int] = monotonic_constraints or {}

        # Default high-performance parameters
        self.params: Dict[str, Any] = {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_estimators": 800,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if model_params:
            self.params.update(model_params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LGBMDustModel":
        """Fits LightGBM regressor on feature matrix X and target y."""
        X_arr, f_names = self._prepare_inputs(X, feature_names)
        y_arr = self._prepare_target(y)
        self.feature_names_ = f_names

        fit_params = dict(self.params)

        # Build monotone constraints sequence if specified
        # LightGBM requires smooth objective (such as huber) for monotone_constraints
        obj = fit_params.get("objective", "regression_l1")
        if self.monotonic_constraints and len(self.feature_names_) > 0:
            m_list = [self.monotonic_constraints.get(fn, 0) for fn in self.feature_names_]
            if any(m != 0 for m in m_list):
                if obj in ("regression_l1", "mae", "l1"):
                    logger.info("LightGBM requires smooth loss for monotone_constraints: adapting objective from '%s' to 'huber'.", obj)
                    fit_params["objective"] = "huber"
                fit_params["monotone_constraints"] = m_list


        self.estimator = lgb.LGBMRegressor(**fit_params)

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_arr, _ = self._prepare_inputs(X_val, feature_names=self.feature_names_)
            y_val_arr = self._prepare_target(y_val)
            try:
                self.estimator.fit(
                    X_arr,
                    y_arr,
                    eval_X=X_val_arr,
                    eval_y=y_val_arr,
                    feature_name=self.feature_names_,
                )
            except TypeError:
                self.estimator.fit(
                    X_arr,
                    y_arr,
                    eval_set=[(X_val_arr, y_val_arr)],
                    feature_name=self.feature_names_,
                )
        else:
            self.estimator.fit(
                X_arr,
                y_arr,
                feature_name=self.feature_names_,
            )
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts PM2.5 concentrations clamped to >= 0.0."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        X_arr, _ = self._prepare_inputs(X, feature_names=self.feature_names_)
        if self.feature_names_ is not None and len(self.feature_names_) == X_arr.shape[1]:
            X_in = pd.DataFrame(X_arr, columns=self.feature_names_)
        else:
            X_in = X_arr
        preds = self.estimator.predict(X_in)
        return np.maximum(preds, 0.0)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Returns sorted feature importance table."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} is not fitted.")

        lgb_imp_type = "gain" if importance_type.lower() == "gain" else "split"
        raw_imp = self.estimator.booster_.feature_importance(importance_type=lgb_imp_type)
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
    def load(cls, filepath: Union[str, Path]) -> "LGBMDustModel":
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
