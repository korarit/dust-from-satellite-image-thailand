"""
Phase 5: Evaluation & Statistical / Domain AQI Metric Engine
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Implements:
1. Continuous Regression Metrics:
   - MAE: Mean Absolute Error (µg/m³)
   - RMSE: Root Mean Squared Error (µg/m³)
   - MAPE: Mean Absolute Percentage Error (%)
   - R²: Coefficient of Determination
   - Mean Bias: mean(y_pred - y_true) (Direction of error: over vs under-prediction)
   - Pearson r: Linear correlation coefficient
   - Spearman rho: Monotonic rank correlation coefficient
2. Domain AQI Classification (PCD Thailand 5-tier Health Warning Bands):
   - Blue (0 - 15.0 µg/m³): Very Good (ดีมาก)
   - Green (15.1 - 25.0 µg/m³): Good (ดี)
   - Yellow (25.1 - 37.5 µg/m³): Moderate (ปานกลาง)
   - Orange (37.6 - 75.0 µg/m³): Unhealthy for Sensitive Groups (เริ่มมีผลกระทบต่อสุขภาพ)
   - Red (> 75.0 µg/m³): Hazardous / High Impact (มีผลกระทบต่อสุขภาพ)
3. Categorical Classification Metrics:
   - 5x5 Confusion Matrix
   - Per-tier Precision, Recall, F1-Score, and Sample Support
   - Macro-averaged F1 Score & Weighted F1 Score
   - Public Health High-Risk Recall (Orange + Red combined alert sensitivity)
4. Strict Zero-Mock: 100% deterministic mathematical formulas, zero random calls.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

logger = logging.getLogger(__name__)

# Thailand Pollution Control Department (PCD) PM2.5 AQI Thresholds (µg/m³)
THAILAND_AQI_THRESHOLDS = [0.0, 15.0, 25.0, 37.5, 75.0, 1000.0]

AQI_CATEGORY_NAMES_EN = [
    "Very Good",
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Hazardous",
]

AQI_CATEGORY_NAMES_TH = [
    "ดีมาก",
    "ดี",
    "ปานกลาง",
    "เริ่มมีผลกระทบต่อสุขภาพ",
    "มีผลกระทบต่อสุขภาพ",
]

AQI_COLOR_HEX = [
    "#00b0f0",  # Blue
    "#92d050",  # Green
    "#ffff00",  # Yellow
    "#ff7c00",  # Orange
    "#ff0000",  # Red
]

AQI_COLOR_NAMES = ["Blue", "Green", "Yellow", "Orange", "Red"]


def assign_aqi_category(pm25_values: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """
    Discretizes continuous PM2.5 concentrations (µg/m³) into Thailand PCD
    AQI category indices (0 to 4).
    """
    arr = np.asarray(pm25_values, dtype=np.float64)
    # Clip negative values to zero
    arr = np.maximum(arr, 0.0)
    # Bins: (0, 15], (15, 25], (25, 37.5], (37.5, 75], (75, inf)
    cats = np.digitize(arr, bins=THAILAND_AQI_THRESHOLDS[1:-1], right=True)
    return np.clip(cats, 0, 4).astype(np.int32)


def get_aqi_metadata(cat_idx: int) -> Dict[str, str]:
    """Returns color hex, English label, and Thai label for an AQI category index."""
    idx = max(0, min(int(cat_idx), 4))
    return {
        "index": idx,
        "color_hex": AQI_COLOR_HEX[idx],
        "color_name": AQI_COLOR_NAMES[idx],
        "label_en": AQI_CATEGORY_NAMES_EN[idx],
        "label_th": AQI_CATEGORY_NAMES_TH[idx],
    }


def compute_continuous_metrics(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> Dict[str, float]:
    """
    Computes standard physical and statistical regression performance metrics.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    valid_mask = (~np.isnan(y_t)) & (~np.isnan(y_p))
    n_valid = int(np.sum(valid_mask))
    if n_valid < 2:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "r2": float("nan"),
            "mean_bias": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "n_samples": n_valid,
        }

    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # Mean Bias: positive = over-predict, negative = under-predict
    mean_bias = float(np.mean(y_p - y_t))

    # Safe MAPE: denom floor at 1.0 µg/m³
    denom = np.maximum(y_t, 1.0)
    mape = float(np.mean(np.abs(y_t - y_p) / denom) * 100.0)

    # R2 Score
    if np.var(y_t) > 1e-9:
        r2 = float(r2_score(y_t, y_p))
    else:
        r2 = 0.0

    # Pearson r
    if np.std(y_t) > 1e-9 and np.std(y_p) > 1e-9:
        pr, _ = pearsonr(y_t, y_p)
        pearson_val = float(pr)
    else:
        pearson_val = 0.0

    # Spearman rho
    if np.std(y_t) > 1e-9 and np.std(y_p) > 1e-9:
        sr, _ = spearmanr(y_t, y_p)
        spearman_val = float(sr)
    else:
        spearman_val = 0.0

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "mean_bias": round(mean_bias, 3),
        "pearson_r": round(pearson_val, 4),
        "spearman_rho": round(spearman_val, 4),
        "n_samples": n_valid,
    }


def compute_aqi_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> Dict[str, Any]:
    """
    Evaluates predictions against Thailand PCD 5-tier health warning categories.
    Computes Confusion Matrix, Precision, Recall, Macro F1, and High-Risk Alert Recall.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    valid_mask = (~np.isnan(y_t)) & (~np.isnan(y_p))
    if np.sum(valid_mask) < 2:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "high_risk_recall": float("nan"),
            "per_tier": {},
            "confusion_matrix": [],
        }

    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]

    cat_t = assign_aqi_category(y_t)
    cat_p = assign_aqi_category(y_p)

    acc = float(accuracy_score(cat_t, cat_p) * 100.0)

    # 5-tier labels: [0, 1, 2, 3, 4]
    all_labels = list(range(5))
    cm = confusion_matrix(cat_t, cat_p, labels=all_labels)

    prec, rec, f1, sup = precision_recall_fscore_support(
        cat_t, cat_p, labels=all_labels, zero_division=0
    )

    per_tier = {}
    for i in range(5):
        per_tier[f"tier_{i}_{AQI_COLOR_NAMES[i].lower()}"] = {
            "index": i,
            "name_en": AQI_CATEGORY_NAMES_EN[i],
            "name_th": AQI_CATEGORY_NAMES_TH[i],
            "color_hex": AQI_COLOR_HEX[i],
            "precision": round(float(prec[i]) * 100.0, 2),
            "recall": round(float(rec[i]) * 100.0, 2),
            "f1": round(float(f1[i]) * 100.0, 2),
            "support": int(sup[i]),
        }

    macro_f1 = float(np.mean(f1) * 100.0)
    total_sup = np.sum(sup)
    weighted_f1 = float((np.sum(f1 * sup) / total_sup) * 100.0) if total_sup > 0 else 0.0

    # High-Risk Alert Recall: Orange (3) and Red (4) tiers
    high_risk_mask = (cat_t >= 3)
    if np.sum(high_risk_mask) > 0:
        high_risk_pred_correct = np.sum((cat_p >= 3) & high_risk_mask)
        high_risk_recall = float((high_risk_pred_correct / np.sum(high_risk_mask)) * 100.0)
    else:
        high_risk_recall = 100.0

    return {
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "high_risk_recall": round(high_risk_recall, 2),
        "per_tier": per_tier,
        "confusion_matrix": cm.tolist(),
    }


def compute_comprehensive_metrics(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> Dict[str, Any]:
    """
    Combines continuous regression metrics and PCD AQI classification metrics
    into a unified evaluation dictionary.
    """
    cont_metrics = compute_continuous_metrics(y_true, y_pred)
    aqi_metrics = compute_aqi_classification_metrics(y_true, y_pred)

    return {
        **cont_metrics,
        "aqi_accuracy": aqi_metrics["accuracy"],
        "aqi_macro_f1": aqi_metrics["macro_f1"],
        "aqi_weighted_f1": aqi_metrics["weighted_f1"],
        "high_risk_recall": aqi_metrics["high_risk_recall"],
        "per_tier_metrics": aqi_metrics["per_tier"],
        "confusion_matrix": aqi_metrics["confusion_matrix"],
    }


def format_evaluation_summary_table(metrics: Dict[str, Any], model_name: str = "Model") -> str:
    """Generates a clean Markdown formatted summary report."""
    lines = [
        f"### Performance Summary: {model_name}",
        "",
        "| Metric Type | Metric Name | Value | Physical Unit / Scale |",
        "| :--- | :--- | :---: | :--- |",
        f"| Continuous | **MAE** | `{metrics.get('mae', 0.0):.3f}` | µg/m³ |",
        f"| Continuous | **RMSE** | `{metrics.get('rmse', 0.0):.3f}` | µg/m³ |",
        f"| Continuous | **Mean Bias** | `{metrics.get('mean_bias', 0.0):.3f}` | µg/m³ (+over / -under) |",
        f"| Continuous | **MAPE** | `{metrics.get('mape', 0.0):.2f}%` | % |",
        f"| Continuous | **R² Score** | `{metrics.get('r2', 0.0):.4f}` | [-inf, 1.0] |",
        f"| Continuous | **Pearson r** | `{metrics.get('pearson_r', 0.0):.4f}` | [-1.0, 1.0] |",
        f"| Continuous | **Spearman ρ** | `{metrics.get('spearman_rho', 0.0):.4f}` | [-1.0, 1.0] |",
        f"| AQI Tier | **Color Accuracy** | `{metrics.get('aqi_accuracy', 0.0):.2f}%` | 5 PCD Color Tiers |",
        f"| AQI Tier | **Macro F1 Score** | `{metrics.get('aqi_macro_f1', 0.0):.2f}%` | Unweighted average |",
        f"| AQI Tier | **High-Risk Alert Recall** | `{metrics.get('high_risk_recall', 0.0):.2f}%` | Orange & Red Warning Recall |",
        f"| Samples | **Sample Count** | `{metrics.get('n_samples', 0)}` | Valid pairs |",
    ]
    return "\n".join(lines)
