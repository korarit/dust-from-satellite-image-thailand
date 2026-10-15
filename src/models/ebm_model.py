"""
Explainable Boosting Machine (EBM / GA2M) Wrapper for Thailand Dust Prediction Satellite (Phase 4)
Glass-box GAM + pairwise interactions (InterpretML): fast inference, native NaN,
and built-in global explanation (replaces/expands the FIGS interpretability role).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd

from src.models.base_model import BaseDustModel

logger = logging.getLogger(__name__)

try:
    from interpret.glassbox import ExplainableBoostingRegressor
    INTERPRET_AVAILABLE = True
except ImportError as _exc:  # lazy-guard: EBM member is optional in the suite
    INTERPRET_AVAILABLE = False
    _INTERPRET_IMPORT_ERROR = _exc


class EBMDustModel(BaseDustModel):
    """
    Explainable Boosting Machine (EBM) Wrapper.
    Features:
    - Glass-box GA2M: additive main effects + pairwise interactions, fully human-readable
    - Native NaN support
    - Ultra-cheap inference (bin lookups + shallow node evaluation) — smallest RAM/latency in the suite
    - Native global feature attribution via explain_global()
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "EBM",
    ) -> None:
        super().__init__(model_params=model_params, name=name)

        # Default parameters (EBM does its own internal validation-based early
        # stopping per outer bag; conservative defaults for ambient PM modelling)
        self.params: Dict[str, Any] = {
            "interactions": 10,
            "outer_bags": 8,
            "learning_rate": 0.03,
            "max_rounds": 1500,
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
    ) -> "EBMDustModel":
        """Fits ExplainableBoostingRegressor (no external eval_set used; EBM self-validates)."""
        if not INTERPRET_AVAILABLE:
            raise ImportError(
                "interpret is required for the EBM model member. "
                "Install it with: pip install interpret"
            ) from _INTERPRET_IMPORT_ERROR

        X_arr, f_names = self._prepare_inputs(X, feature_names)
        y_arr = self._prepare_target(y).astype(np.float64)
        self.feature_names_ = f_names

        if eval_set is not None:
            logger.info("EBM uses internal per-bag early stopping; external eval_set is ignored.")

        self.estimator = ExplainableBoostingRegressor(**self.params)
        self.estimator.fit(X_arr.astype(np.float64), y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts PM concentrations clamped to >= 0.0."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predict.")

        X_arr, _ = self._prepare_inputs(X, feature_names=self.feature_names_)
        preds = self.estimator.predict(X_arr.astype(np.float64))
        return np.maximum(preds, 0.0)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Returns EBM global explanation scores as the feature importance table."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError(f"{self.name} is not fitted.")

        raw_imp = np.zeros(len(self.feature_names_), dtype=np.float64)
        try:
            glob_data = self.estimator.explain_global().data()
            names = list(glob_data.get("names", []))
            scores = list(glob_data.get("scores", []))
            # Main effects come first in explain_global(); interactions ("a & b")
            # follow after. Map positionally because fitted ndarray columns are
            # auto-named feature_0000.. and do not match the wrapper's feature names.
            n_main = min(len(self.feature_names_), len(scores))
            raw_imp[:n_main] = np.abs(np.asarray(scores[:n_main], dtype=np.float64))
        except Exception as exc:
            logger.warning("EBM global explanation unavailable (%s); using zeros.", exc)

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
        logger.info("Saved %s checkpoint to: %s", self.name, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EBMDustModel":
        """Deserializes model checkpoint from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = joblib.load(path)
        instance = cls(model_params=checkpoint.get("params", {}), name=checkpoint.get("name", "EBM"))
        instance.feature_names_ = checkpoint.get("feature_names_", [])
        instance.is_fitted_ = checkpoint.get("is_fitted_", False)
        instance.estimator = checkpoint.get("estimator", None)
        logger.info("Loaded %s checkpoint from: %s", instance.name, path)
        return instance
