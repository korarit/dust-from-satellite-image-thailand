"""
Nyström Fast Kernel Ridge Regressor Wrapper for Thailand Dust Prediction Satellite (Phase 4)
Approximate infinite-dimensional Hilbert space kernel regression via Nyström sampling and Ridge regularization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge

from src.models.base_model import BaseDustModel

logger = logging.getLogger(__name__)


class NystroemRidgeDustModel(BaseDustModel):
    """
    Nyström Fast Kernel Ridge Regressor.
    Features:
    - Approximates RBF non-linear kernel mappings without O(N^2) Gram matrix explosion
    - Ultra-fast closed-form solution with L2 regularized Ridge
    - Native feature scaling and median imputation
    - Extreme training speed on CPU (< 1 sec) with low memory footprint
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "NystroemRidge",
    ) -> None:
        super().__init__(model_params=model_params, name=name)

        # Default parameters
        self.params: Dict[str, Any] = {
            "n_components": 120,
            "kernel": "rbf",
            "alpha": 10.0,
            "random_state": 42,
        }
        if model_params:
            self.params.update(model_params)

        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.nystroem: Optional[Nystroem] = None
        self.ridge: Optional[Ridge] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "NystroemRidgeDustModel":
        """Fits Nystroem Kernel Ridge pipeline on feature matrix X and target y."""
        X_arr, f_names = self._prepare_inputs(X, feature_names)
        y_arr = self._prepare_target(y)
        self.feature_names_ = f_names

        # 1. Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X_arr)

        # 2. Scale features for isotropic kernel distance
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imp)

        # 3. Nyström kernel approximation
        n_comp = int(self.params.get("n_components", 120))
        kernel = str(self.params.get("kernel", "rbf"))
        r_state = int(self.params.get("random_state", 42))
        alpha = float(self.params.get("alpha", 10.0))

        self.nystroem = Nystroem(
            kernel=kernel,
            n_components=min(n_comp, len(X_scaled)),
            random_state=r_state,
        )
        X_kernel = self.nystroem.fit_transform(X_scaled)

        # 4. Ridge regression
        self.ridge = Ridge(alpha=alpha, random_state=r_state)
        self.ridge.fit(X_kernel, y_arr)

        self.estimator = self.ridge
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts PM concentrations clamped to >= 0.0."""
        if not self.is_fitted_ or self.ridge is None or self.scaler is None or self.nystroem is None:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        X_arr, _ = self._prepare_inputs(X, feature_names=self.feature_names_)
        if self.imputer is not None:
            X_imp = self.imputer.transform(X_arr)
        else:
            X_imp = X_arr

        X_scaled = self.scaler.transform(X_imp)
        X_kernel = self.nystroem.transform(X_scaled)
        preds = self.ridge.predict(X_kernel)
        return np.maximum(preds, 0.0)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Computes projected feature importance from absolute Ridge weights mapped back
        through Nystroem support components: imp_j = sum_k (|w_k| * |C_kj|).
        """
        if not self.is_fitted_ or self.ridge is None or self.nystroem is None:
            raise RuntimeError(f"{self.name} is not fitted.")

        # components_ has shape (n_components, n_features)
        # ridge.coef_ has shape (n_components,)
        comps = np.abs(self.nystroem.components_)
        coefs = np.abs(self.ridge.coef_)

        raw_imp = coefs @ comps
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
            "scaler": self.scaler,
            "nystroem": self.nystroem,
            "ridge": self.ridge,
            "estimator": self.estimator,
        }
        joblib.dump(checkpoint, path)
        logger.info("Saved %s checkpoint to: %s", self.name, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NystroemRidgeDustModel":
        """Deserializes model checkpoint from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = joblib.load(path)
        instance = cls(model_params=checkpoint.get("params", {}), name=checkpoint.get("name", "NystroemRidge"))
        instance.feature_names_ = checkpoint.get("feature_names_", [])
        instance.is_fitted_ = checkpoint.get("is_fitted_", False)
        instance.imputer = checkpoint.get("imputer", None)
        instance.scaler = checkpoint.get("scaler", None)
        instance.nystroem = checkpoint.get("nystroem", None)
        instance.ridge = checkpoint.get("ridge", None)
        instance.estimator = checkpoint.get("estimator", None)
        logger.info("Loaded %s checkpoint from: %s", instance.name, path)
        return instance
