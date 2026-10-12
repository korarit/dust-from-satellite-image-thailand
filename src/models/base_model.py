"""
Base Model Interface & Statistical Metric Engine (Phase 4)
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Defines:
- BaseDustModel: Abstract base class for all 4 quartet learners and ensemble.
- compute_dust_metrics: Calculation of MAE, RMSE, MAPE, R2, Pearson-r, and AQI accuracy.
- THAILAND_AQI_THRESHOLDS: PCD Thailand PM2.5 health standard levels.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Thailand PCD AQI Bands (µg/m³)
# PM2.5:
# 0 - 15.0: Very Good (Blue)
# 15.1 - 25.0: Good (Green)
# 25.1 - 37.5: Moderate (Yellow)
# 37.6 - 75.0: Orange (Unhealthy for sensitive groups)
# > 75.0: Red (Hazardous)
THAILAND_AQI_THRESHOLDS_PM25 = [0.0, 15.0, 25.0, 37.5, 75.0, 1000.0]

# PM10:
# 0 - 50.0: Very Good (Blue)
# 50.1 - 80.0: Good (Green)
# 80.1 - 120.0: Moderate (Yellow)
# 120.1 - 180.0: Orange (Unhealthy for sensitive groups)
# > 180.0: Red (Hazardous)
THAILAND_AQI_THRESHOLDS_PM10 = [0.0, 50.0, 80.0, 120.0, 180.0, 1000.0]

THAILAND_AQI_THRESHOLDS = THAILAND_AQI_THRESHOLDS_PM25


def assign_aqi_category(pm_values: np.ndarray, pollutant: str = "pm25") -> np.ndarray:
    """Discretizes continuous PM concentration into PCD AQI color category indices (0 to 4)."""
    bins = THAILAND_AQI_THRESHOLDS_PM10[1:-1] if "pm10" in pollutant.lower() else THAILAND_AQI_THRESHOLDS_PM25[1:-1]
    return np.digitize(pm_values, bins=bins, right=True)


def compute_dust_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    pollutant: str = "pm25",
) -> Dict[str, float]:
    """
    Computes statistical and domain-specific air quality performance metrics.

    Metrics:
    - MAE: Mean Absolute Error (µg/m³)
    - RMSE: Root Mean Squared Error (µg/m³)
    - MAPE: Mean Absolute Percentage Error (%)
    - R2: Coefficient of Determination
    - Pearson_r: Pearson Linear Correlation Coefficient
    - AQI_Acc: PCD Thailand 5-tier Color Band Classification Accuracy (%)
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    # Drop NaNs if any
    valid_mask = (~np.isnan(y_t)) & (~np.isnan(y_p))
    if np.sum(valid_mask) < 2:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "aqi_acc": float("nan"),
            "count": int(np.sum(valid_mask)),
        }

    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # Safe MAPE: avoid division by zero when PM is near zero
    denom = np.maximum(y_t, 1.0)
    mape = float(np.mean(np.abs(y_t - y_p) / denom) * 100.0)

    # R2 Score
    if np.var(y_t) > 1e-9:
        r2 = float(r2_score(y_t, y_p))
    else:
        r2 = 0.0

    # Pearson r
    if np.std(y_t) > 1e-9 and np.std(y_p) > 1e-9:
        r, _ = pearsonr(y_t, y_p)
        pearson_val = float(r)
    else:
        pearson_val = 0.0

    # AQI Categorical Accuracy
    cat_true = assign_aqi_category(y_t, pollutant=pollutant)
    cat_pred = assign_aqi_category(y_p, pollutant=pollutant)
    aqi_acc = float(np.mean(cat_true == cat_pred) * 100.0)

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "pearson_r": round(pearson_val, 4),
        "aqi_acc": round(aqi_acc, 2),
        "count": int(len(y_t)),
    }


class BaseDustModel(ABC):
    """
    Abstract Base Class for Dust Prediction Machine Learning Models.
    Enforces standardized interfaces across LightGBM, CatBoost, XGBoost,
    Random Forest, and Ensemble Stacking Meta-Learners.
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> None:
        self.model_params: Dict[str, Any] = model_params or {}
        self.name: str = name or self.__class__.__name__
        self.estimator: Any = None
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "BaseDustModel":
        """Fits the underlying model with feature validation."""
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generates continuous PM2.5 predictions clamped to >= 0.0 µg/m³."""
        pass

    @abstractmethod
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Returns a sorted pandas DataFrame with feature importance scores:
        Columns: ['feature', 'importance', 'normalized_importance']
        """
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Persists model weights and configuration to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseDustModel":
        """Loads serialized model weights and metadata from disk."""
        pass

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluates prediction performance against ground-truth labels."""
        preds = self.predict(X)
        return compute_dust_metrics(y, preds)

    def _prepare_inputs(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Validates and extracts matrix values and column identifiers."""
        if isinstance(X, pd.DataFrame):
            f_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            if feature_names is not None:
                f_names = list(feature_names)
            elif self.feature_names_:
                f_names = list(self.feature_names_)
            else:
                f_names = [f"f_{i}" for i in range(X_arr.shape[1])]

        return X_arr, f_names

    def _prepare_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Validates and extracts target float array."""
        if isinstance(y, pd.Series):
            return y.to_numpy(dtype=np.float32)
        return np.asarray(y, dtype=np.float32)
