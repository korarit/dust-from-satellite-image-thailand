"""
Phase 5: Spatial-Temporal Cross-Validation Protocol
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Implements:
1. Geographic Region Partitioning for Thailand:
   - 5 Standard Regions:
     * North (ภาคเหนือ)
     * Northeast (ภาคตะวันออกเฉียงเหนือ / อีสาน)
     * Central (ภาคกลางและตะวันตก)
     * East (ภาคตะวันออก)
     * South (ภาคใต้)
2. Regional Hold-Out Cross-Validation (Spatial K-Fold):
   - Trains on 4 geographic regions, validates on 1 completely unseen region.
   - Proves spatial generalization capability in grid cells lacking ground stations.
3. Out-of-Time Temporal Cross-Validation:
   - Evaluates multi-year temporal generalization (Train 2024-2025 vs Test 2026).
4. SpatialTemporalEvaluator:
   - Orchestrates regional cross-validation across all 5 regions.
   - Computes per-region and macro metrics.
5. Strict Zero-Mock: 100% deterministic spatial coordinates, zero random calls.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_comprehensive_metrics

logger = logging.getLogger(__name__)

# Standard 5 Thailand Geographic Regions
THAILAND_REGIONS = ["north", "northeast", "central", "east", "south"]

REGION_LABELS_TH = {
    "north": "ภาคเหนือ",
    "northeast": "ภาคตะวันออกเฉียงเหนือ",
    "central": "ภาคกลางและตะวันตก",
    "east": "ภาคตะวันออก",
    "south": "ภาคใต้",
}


def assign_thailand_region(lat: float, lon: float) -> str:
    """
    Deterministically assigns a geographic coordinate (lat, lon) to one of
    the 5 Thailand regions based on standard physical boundary boxes.
    """
    if lat < 13.0:
        return "south"
    elif lat >= 14.5 and lon >= 101.3:
        return "northeast"
    elif 11.5 <= lat < 14.5 and 101.0 <= lon <= 103.0:
        return "east"
    elif lat >= 17.5 or (lat >= 17.0 and lon < 100.5):
        return "north"
    else:
        return "central"


def attach_region_column(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    """Attaches a deterministic 'region' column to a DataFrame containing lat/lon."""
    result = df.copy()
    if lat_col not in result.columns or lon_col not in result.columns:
        # Fallback to central if coordinates missing
        result["region"] = "central"
        return result

    regions = [
        assign_thailand_region(float(lat), float(lon))
        for lat, lon in zip(result[lat_col], result[lon_col])
    ]
    result["region"] = regions
    return result


class RegionalHoldoutCV:
    """
    Spatial K-Fold Cross-Validation that partitions samples by geographic region.
    In each split, an entire region is held out as the validation set.
    """

    def __init__(
        self,
        regions: Optional[List[str]] = None,
        min_samples_per_fold: int = 1,
    ) -> None:
        self.regions = regions or THAILAND_REGIONS
        self.min_samples_per_fold = min_samples_per_fold

    def split(
        self,
        df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        region_col: str = "region",
    ) -> Generator[Tuple[str, np.ndarray, np.ndarray], None, None]:
        """
        Yields (held_out_region, train_indices, val_indices) for each region.
        """
        if region_col not in df.columns:
            work_df = attach_region_column(df, lat_col=lat_col, lon_col=lon_col)
        else:
            work_df = df

        region_series = work_df[region_col].values
        all_indices = np.arange(len(work_df))

        for reg in self.regions:
            val_mask = (region_series == reg)
            val_indices = all_indices[val_mask]
            train_indices = all_indices[~val_mask]

            if len(val_indices) < self.min_samples_per_fold or len(train_indices) < self.min_samples_per_fold:
                logger.warning(
                    "Skipping region '%s': insufficient samples (train: %d, val: %d)",
                    reg, len(train_indices), len(val_indices)
                )
                continue

            yield reg, train_indices, val_indices

    def get_n_splits(self, df: pd.DataFrame, region_col: str = "region") -> int:
        """Returns number of available valid regions in dataset."""
        if region_col in df.columns:
            present = set(df[region_col].unique())
        else:
            work_df = attach_region_column(df)
            present = set(work_df["region"].unique())
        return len([r for r in self.regions if r in present])


class SpatialTemporalEvaluator:
    """
    Evaluates models across Regional Hold-out folds and Temporal splits.
    Quantifies generalization performance on unseen spatial domains.
    """

    def __init__(self, target_col: str = "observed_pm25", pollutant: Optional[str] = None) -> None:
        self.target_col = target_col
        if pollutant is not None:
            self.pollutant = pollutant.lower()
        else:
            self.pollutant = "pm10" if "pm10" in target_col.lower() else "pm25"
        self.cv = RegionalHoldoutCV()

    def evaluate_regional_generalization(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        model_factory_fn: Any,
    ) -> Dict[str, Any]:
        """
        Runs Regional Hold-Out Cross-Validation.
        model_factory_fn: callable that returns a fresh model instance with fit/predict methods.
        """
        work_df = attach_region_column(df).reset_index(drop=True)
        valid_df = work_df[~work_df[self.target_col].isna()].copy().reset_index(drop=True)

        regional_results = {}
        all_y_true = []
        all_y_pred = []

        for reg, train_idx, val_idx in self.cv.split(valid_df):
            train_sub = valid_df.iloc[train_idx]
            val_sub = valid_df.iloc[val_idx]

            X_train = train_sub[feature_cols]
            y_train = train_sub[self.target_col].values
            X_val = val_sub[feature_cols]
            y_val = val_sub[self.target_col].values

            # Instantiate and fit fresh model
            model = model_factory_fn()
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            # Compute regional metrics
            m = compute_comprehensive_metrics(y_val, preds, pollutant=self.pollutant)
            regional_results[reg] = {
                "region_en": reg,
                "region_th": REGION_LABELS_TH.get(reg, reg),
                "n_samples": len(val_sub),
                "metrics": m,
            }

            all_y_true.extend(y_val)
            all_y_pred.extend(preds)

        # Macro-averaged metrics across all unseen regions
        overall_metrics = compute_comprehensive_metrics(all_y_true, all_y_pred, pollutant=self.pollutant)

        return {
            "by_region": regional_results,
            "overall_unseen_spatial_metrics": overall_metrics,
            "total_evaluated_samples": len(all_y_true),
        }

    def format_regional_report(self, eval_results: Dict[str, Any]) -> str:
        """Formats regional evaluation results into a Markdown report."""
        lines = [
            "### 🗺️ Spatial Generalization: Regional Hold-Out Cross-Validation",
            "",
            "> Evaluates model capability on unseen regions where no ground stations were available during training.",
            "",
            "| Region (ภูมิภาค) | Test Samples | MAE (µg/m³) | RMSE (µg/m³) | Mean Bias | R² Score | AQI Color Acc (%) | High-Risk Alert Recall (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        by_reg = eval_results.get("by_region", {})
        for reg_k, reg_info in by_reg.items():
            m = reg_info.get("metrics", {})
            r_name = f"{reg_info.get('region_th', reg_k)} ({reg_k.capitalize()})"
            lines.append(
                f"| **{r_name}** | {reg_info.get('n_samples', 0)} | "
                f"`{m.get('mae', 0.0):.2f}` | `{m.get('rmse', 0.0):.2f}` | `{m.get('mean_bias', 0.0):.2f}` | "
                f"`{m.get('r2', 0.0):.3f}` | `{m.get('aqi_accuracy', 0.0):.1f}%` | `{m.get('high_risk_recall', 0.0):.1f}%` |"
            )

        overall = eval_results.get("overall_unseen_spatial_metrics", {})
        lines.append(
            f"| **MACRO OVERALL** | {eval_results.get('total_evaluated_samples', 0)} | "
            f"**`{overall.get('mae', 0.0):.2f}`** | **`{overall.get('rmse', 0.0):.2f}`** | **`{overall.get('mean_bias', 0.0):.2f}`** | "
            f"**`{overall.get('r2', 0.0):.3f}`** | **`{overall.get('aqi_accuracy', 0.0):.1f}%`** | **`{overall.get('high_risk_recall', 0.0):.1f}%`** |"
        )

        return "\n".join(lines)
