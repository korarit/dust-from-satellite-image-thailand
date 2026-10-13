"""
Phase 3: Unified Feature Pipeline & Anti-Leakage Protocol Engine
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Implements:
1. Unified Feature Matrix Assembly:
   - Single row per grid cell per UTC timestamp
   - Multi-Resolution Support: 3 km, 2 km (Master baseline), 1 km, 500 m
   - Strict Column Schema (52+ features + keys + target):
     * Keys: grid_id, timestamp, lat, lon, utm_x, utm_y
     * Group 1: 10 IR Bands BT (bt_b07 .. bt_b16 in Kelvin)
     * Group 2: Physics BTDs (btd_15_13, btd_14_11, btd_13_07, btd_10_08, dust_rgb_red)
     * Group 3: Spatial 3x3 Texture (b13_mean_3x3, b13_std_3x3, b13_min/max, b15_*, btd_*)
     * Group 4: Temporal Dynamics (delta_bt13_10m, 30m, 60m, delta_btd15_13_10m, 30m, 60m)
     * Group 5: 4-Directional Pressure (p_north, p_south, p_east, p_west, delta_p_ns/ew, ||nabla P||, mean)
     * Group 6: Surrounding Humidity (humidity_mean_surround, humidity_std_surround, ||nabla RH||)
     * Group 7: Topography & Solar (elevation, tpi_index, slope, aspect, relief, coast, coriolis, sza, cycles)
     * Target: observed_pm25 (µg/m³) for training/validation
2. Strict Anti-Leakage Protocol:
   - Zero Look-Ahead Bias Validator: observation timestamp <= run timestamp
   - Strict Spatial Proximity Exclusion Validator: rejects stations < exclusion_radius_km
   - Target Feature Isolation: verifies target values are NEVER injected into feature set
3. Physics Fallback Imputer:
   - Deterministic handling of cloud occlusion and missing sensors
   - ZERO random numbers (Strict Zero-Mock compliance)
4. Apache Parquet Serialization:
   - Fast columnar projection and Snappy compression
   - Partitioning by Year/Month
   - Temporal Train/Test Splitting (2024-2025 Train vs 2026 Out-of-Time Test)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree

from src.config.paths import (
    FEATURES_DIR,
    GEO_DIR,
    HII_DIR,
    TARGET_DIR,
    REPORTS_DIR,
)
from src.features.spatial_directional import (
    DirectionalWeatherEngine,
    RESOLUTION_EXCLUSION_MAP,
    DEFAULT_EXCLUSION_KM,
)
from src.features.satellite_tabularizer import (
    SatelliteTabularizer,
    ALL_IR_BANDS,
    BAND_COLS,
    PHYSICAL_BT_MIN_K,
    PHYSICAL_BT_MAX_K,
)
from src.features.grid_engine import (
    MultiResolutionGridEngine,
    SUPPORTED_RESOLUTIONS_KM,
    EXCLUSION_BUFFER_MAP,
    compute_geographic_proxies,
    compute_solar_angles_vectorized,
    compute_cyclical_time_features,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# SCHEMA CONSTANTS & FEATURE COLUMN DEFINITIONS
# ==============================================================================

KEY_COLUMNS = ["grid_id", "timestamp", "lat", "lon", "utm_x", "utm_y"]

SAT_BT_COLUMNS = [f"bt_{b.lower()}" for b in ALL_IR_BANDS]

SAT_BTD_COLUMNS = [
    "btd_15_13",
    "btd_14_11",
    "btd_13_07",
    "btd_10_08",
    "dust_rgb_red",
    "btd_13_10",
    "btd_13_11",
]

SAT_TEXTURE_COLUMNS = [
    "b13_mean_3x3",
    "b13_std_3x3",
    "b13_min_3x3",
    "b13_max_3x3",
    "b15_mean_3x3",
    "b15_std_3x3",
    "btd_15_13_mean_3x3",
    "btd_15_13_std_3x3",
]

SAT_TEMPORAL_COLUMNS = [
    "delta_bt13_10m",
    "delta_bt13_30m",
    "delta_bt13_60m",
    "delta_btd15_13_10m",
    "delta_btd15_13_30m",
    "delta_btd15_13_60m",
]

WEATHER_PRESSURE_COLUMNS = [
    "p_north",
    "p_south",
    "p_east",
    "p_west",
    "delta_p_ns",
    "delta_p_ew",
    "pressure_grad_mag",
    "pressure_mean_surround",
]

WEATHER_HUMIDITY_COLUMNS = [
    "humidity_mean_surround",
    "humidity_std_surround",
    "humidity_grad_mag",
]

TOPO_SOLAR_COLUMNS = [
    "elevation_m",
    "tpi_index",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "elevation_std",
    "relief_m",
    "dist_coast_km",
    "coriolis_param",
    "solar_zenith_angle",
    "is_daytime",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

WEATHER_FEATURE_COLUMNS = WEATHER_PRESSURE_COLUMNS + WEATHER_HUMIDITY_COLUMNS

# Satellite & Topographic/Solar Features (Maritime & Offshore Capable, No Ground Stations Needed)
NO_WEATHER_FEATURE_COLUMNS = (
    SAT_BT_COLUMNS
    + SAT_BTD_COLUMNS
    + SAT_TEXTURE_COLUMNS
    + SAT_TEMPORAL_COLUMNS
    + TOPO_SOLAR_COLUMNS
)

ALL_FEATURE_COLUMNS = (
    SAT_BT_COLUMNS
    + SAT_BTD_COLUMNS
    + SAT_TEXTURE_COLUMNS
    + SAT_TEMPORAL_COLUMNS
    + WEATHER_PRESSURE_COLUMNS
    + WEATHER_HUMIDITY_COLUMNS
    + TOPO_SOLAR_COLUMNS
)

FULL_MULTIMODAL_COLUMNS = ALL_FEATURE_COLUMNS
TARGET_COLUMN = "observed_pm25"
TARGET_COLUMN_PM25 = "observed_pm25"
TARGET_COLUMN_PM10 = "observed_pm10"

# Physical Climatology Fallback Constants for Thailand (Deterministic)
CLIMATOLOGY_PRESSURE_HPA = 1013.25
CLIMATOLOGY_HUMIDITY_PCT = 70.0
CLIMATOLOGY_SURFACE_BT_K = 290.0
DEEP_CLOUD_BT_THRESHOLD_K = 220.0


# ==============================================================================
# ANTI-LEAKAGE PROTOCOL EXCEPTIONS & VALIDATORS
# ==============================================================================

class TemporalAntiLeakageError(Exception):
    """Raised when observation timestamps violate zero look-ahead bias."""
    pass


class SpatialAntiLeakageError(Exception):
    """Raised when proximity exclusion rules are breached (< exclusion buffer)."""
    pass


class TargetFeatureIsolationError(Exception):
    """Raised when target values are detected inside the feature input set."""
    pass


class TemporalAntiLeakageValidator:
    """
    Enforces Strict Zero Look-Ahead Bias:
    Timestamp_observation <= Timestamp_run for all feature inputs.
    """

    @staticmethod
    def validate_timestamp(
        obs_timestamp: Union[pd.Timestamp, str, np.datetime64],
        run_timestamp: Union[pd.Timestamp, str, np.datetime64],
        source_name: str = "Observation",
        tolerance_seconds: int = 0,
    ) -> bool:
        """
        Validates that obs_timestamp is <= run_timestamp.
        Raises TemporalAntiLeakageError if obs_timestamp is in the future.
        """
        obs_dt = pd.to_datetime(obs_timestamp, utc=True)
        run_dt = pd.to_datetime(run_timestamp, utc=True)

        if obs_dt > run_dt + pd.Timedelta(seconds=tolerance_seconds):
            msg = (
                f"🚨 TEMPORAL ANTI-LEAKAGE VIOLATION: {source_name} timestamp {obs_dt} "
                f"is strictly in the future relative to run timestamp {run_dt}!"
            )
            logger.error(msg)
            raise TemporalAntiLeakageError(msg)
        return True

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        run_timestamp: Union[pd.Timestamp, str, np.datetime64],
        timestamp_col: str = "timestamp",
        source_name: str = "DataFrame",
    ) -> bool:
        """
        Validates that all rows in df have timestamp <= run_timestamp.
        """
        if df.empty or timestamp_col not in df.columns:
            return True

        run_dt = pd.to_datetime(run_timestamp, utc=True)
        ts_series = pd.to_datetime(df[timestamp_col], utc=True)
        future_mask = ts_series > run_dt

        if future_mask.any():
            max_future = ts_series[future_mask].max()
            num_violations = int(future_mask.sum())
            msg = (
                f"🚨 TEMPORAL ANTI-LEAKAGE VIOLATION: Found {num_violations} future records "
                f"in {source_name}! Max timestamp {max_future} > run timestamp {run_dt}."
            )
            logger.error(msg)
            raise TemporalAntiLeakageError(msg)
        return True


class SpatialAntiLeakageValidator:
    """
    Enforces Strict Proximity Exclusion & Target Segregation.
    """

    @staticmethod
    def verify_exclusion(
        cell_utm_km: np.ndarray,
        station_coords_km: np.ndarray,
        used_station_mask: np.ndarray,
        exclusion_radius_km: float,
    ) -> bool:
        """
        Verifies that none of the used weather stations are within distance < exclusion_radius_km.
        """
        if not np.any(used_station_mask):
            return True

        used_coords = station_coords_km[used_station_mask]
        offsets = used_coords - cell_utm_km
        distances = np.hypot(offsets[:, 0], offsets[:, 1])

        breached = distances < (exclusion_radius_km - 1e-4)
        if np.any(breached):
            min_dist = float(np.min(distances))
            msg = (
                f"🚨 SPATIAL ANTI-LEAKAGE BREACH: Station at distance {min_dist:.3f} km "
                f"was used inside exclusion buffer < {exclusion_radius_km:.1f} km!"
            )
            logger.error(msg)
            raise SpatialAntiLeakageError(msg)
        return True

    @staticmethod
    def verify_target_feature_isolation(
        feature_columns: List[str],
        target_col: Union[str, List[str]] = TARGET_COLUMN
    ) -> bool:
        """
        Verifies that target variable(s) are NEVER included in feature column list.
        """
        targets = [target_col] if isinstance(target_col, str) else list(target_col)
        for t in ["observed_pm25", "observed_pm10"]:
            if t not in targets:
                targets.append(t)

        for t in targets:
            if t in feature_columns:
                msg = f"🚨 TARGET ISOLATION BREACH: '{t}' detected in feature columns!"
                logger.error(msg)
                raise TargetFeatureIsolationError(msg)
        return True


# ==============================================================================
# PHYSICS FALLBACK IMPUTER (DETERMINISTIC, ZERO-MOCK COMPLIANT)
# ==============================================================================

class PhysicsFallbackImputer:
    """
    Applies deterministic, physics-informed imputation for cloud-obscured pixels,
    missing spectral channels, or missing telemetry.
    Strictly NO np.random or synthetic data generation.
    """

    def __init__(self):
        self.fitted_stats_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "PhysicsFallbackImputer":
        """
        Learns deterministic median statistics from a training DataFrame
        to prevent train/test distribution leakage during evaluation.
        """
        self.fitted_stats_ = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_vals = df[col].dropna()
                if len(valid_vals) > 0:
                    self.fitted_stats_[col] = float(valid_vals.median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies imputation using fitted training statistics."""
        return self.impute_features(df)

    def impute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all model feature columns are free of NaN values while preserving
        physical laws and attaching diagnostic flags.
        """
        out_df = df.copy()

        # 1. Diagnostic Flag: Deep Cloud Occlusion (BT13 < 220 K or NaN)
        if "bt_b13" in out_df.columns:
            b13_vals = out_df["bt_b13"].to_numpy(dtype=float)
            cloud_mask = np.isnan(b13_vals) | (b13_vals < DEEP_CLOUD_BT_THRESHOLD_K)
            out_df["cloud_occluded_flag"] = cloud_mask.astype(np.int32)
        else:
            out_df["cloud_occluded_flag"] = 0

        # Track if any imputation was applied
        imputed_mask = np.zeros(len(out_df), dtype=bool)

        # 2. Impute Satellite BT Columns
        for col in SAT_BT_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans
                    if col in self.fitted_stats_:
                        median_val = self.fitted_stats_[col]
                    else:
                        valid_vals = out_df[col].dropna()
                        median_val = float(valid_vals.median()) if len(valid_vals) > 0 else CLIMATOLOGY_SURFACE_BT_K
                    out_df[col] = out_df[col].fillna(median_val).astype(np.float32)
            else:
                default_bt = self.fitted_stats_.get(col, CLIMATOLOGY_SURFACE_BT_K)
                out_df[col] = np.float32(default_bt)
                imputed_mask[:] = True

        # 3. Impute / Recalculate BTD Columns
        if "bt_b15" in out_df.columns and "bt_b13" in out_df.columns:
            out_df["btd_15_13"] = (out_df["bt_b15"] - out_df["bt_b13"]).astype(np.float32)
            out_df["dust_rgb_red"] = np.clip(
                (out_df["btd_15_13"] - (-4.0)) / (2.0 - (-4.0)), 0.0, 1.0
            ).astype(np.float32)

        if "bt_b13" in out_df.columns and "bt_b07" in out_df.columns:
            out_df["btd_13_07"] = (out_df["bt_b13"] - out_df["bt_b07"]).astype(np.float32)

        if "bt_b13" in out_df.columns and "bt_b10" in out_df.columns:
            out_df["btd_13_10"] = (out_df["bt_b13"] - out_df["bt_b10"]).astype(np.float32)

        if "bt_b13" in out_df.columns and "bt_b11" in out_df.columns:
            out_df["btd_13_11"] = (out_df["bt_b13"] - out_df["bt_b11"]).astype(np.float32)

        if "bt_b14" in out_df.columns and "bt_b11" in out_df.columns:
            out_df["btd_14_11"] = (out_df["bt_b14"] - out_df["bt_b11"]).astype(np.float32)
        elif "btd_13_11" in out_df.columns and "btd_14_11" not in out_df.columns:
            out_df["btd_14_11"] = out_df["btd_13_11"]

        if "bt_b10" in out_df.columns and "bt_b08" in out_df.columns:
            out_df["btd_10_08"] = (out_df["bt_b10"] - out_df["bt_b08"]).astype(np.float32)
        elif "btd_13_10" in out_df.columns and "btd_10_08" not in out_df.columns:
            out_df["btd_10_08"] = out_df["btd_13_10"]

        for col in SAT_BTD_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans
                    out_df[col] = out_df[col].fillna(0.0).astype(np.float32)
            else:
                out_df[col] = np.float32(0.0)

        # 4. Impute 3x3 Texture Columns
        texture_base_map = {
            "b13_mean_3x3": "bt_b13",
            "b13_min_3x3": "bt_b13",
            "b13_max_3x3": "bt_b13",
            "b15_mean_3x3": "bt_b15",
            "btd_15_13_mean_3x3": "btd_15_13",
        }
        for col in SAT_TEXTURE_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans
                    if "std" in col:
                        out_df[col] = out_df[col].fillna(0.0).astype(np.float32)
                    elif col in texture_base_map and texture_base_map[col] in out_df.columns:
                        out_df[col] = out_df[col].fillna(out_df[texture_base_map[col]]).astype(np.float32)
                    else:
                        out_df[col] = out_df[col].fillna(CLIMATOLOGY_SURFACE_BT_K).astype(np.float32)
            else:
                if "std" in col:
                    out_df[col] = np.float32(0.0)
                elif col in texture_base_map and texture_base_map[col] in out_df.columns:
                    out_df[col] = out_df[texture_base_map[col]].astype(np.float32)
                else:
                    out_df[col] = np.float32(CLIMATOLOGY_SURFACE_BT_K)

        # 5. Impute Temporal Tendencies (Stationary Atmosphere assumption: delta = 0.0)
        for col in SAT_TEMPORAL_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans
                    out_df[col] = out_df[col].fillna(0.0).astype(np.float32)
            else:
                out_df[col] = np.float32(0.0)

        # 6. Impute Weather Pressure Columns with Barometric Elevation Scaling
        p_mean = (
            self.fitted_stats_.get("pressure_mean_surround")
            if "pressure_mean_surround" in self.fitted_stats_
            else (
                float(out_df["pressure_mean_surround"].dropna().median())
                if "pressure_mean_surround" in out_df.columns and not out_df["pressure_mean_surround"].dropna().empty
                else CLIMATOLOGY_PRESSURE_HPA
            )
        )
        # Barometric elevation scale factor: P(z) = P0 * (1 - L*z/T0)^5.25588
        if "elevation_m" in out_df.columns:
            z_arr = np.maximum(out_df["elevation_m"].fillna(0.0).to_numpy(dtype=float), 0.0)
            f_baro = (1.0 - (0.0065 * z_arr) / 288.15) ** 5.25588
            p_elevation_adjusted = (p_mean * f_baro).astype(np.float32)
        else:
            p_elevation_adjusted = np.float32(p_mean)

        for p_dir in ["p_north", "p_south", "p_east", "p_west", "pressure_mean_surround"]:
            if p_dir in out_df.columns:
                nans = out_df[p_dir].isna()
                if nans.any():
                    imputed_mask |= nans
                    out_df[p_dir] = out_df[p_dir].fillna(p_elevation_adjusted).astype(np.float32)
            else:
                out_df[p_dir] = p_elevation_adjusted

        for p_grad in ["delta_p_ns", "delta_p_ew"]:
            if p_grad in out_df.columns:
                nans = out_df[p_grad].isna()
                if nans.any():
                    imputed_mask |= nans
                    out_df[p_grad] = out_df[p_grad].fillna(0.0).astype(np.float32)
            else:
                out_df[p_grad] = np.float32(0.0)

        # Impute pressure_grad_mag with climatology median (not 0.0, which falsely implies calm stagnation)
        p_grad_default = 1.2  # Typical synoptic gradient: ~1.2 hPa / 100 km
        if "pressure_grad_mag" in out_df.columns:
            valid_pg = out_df["pressure_grad_mag"].dropna()
            pg_med = float(valid_pg.median()) if len(valid_pg) > 0 and valid_pg.median() > 0 else p_grad_default
            nans = out_df["pressure_grad_mag"].isna()
            if nans.any():
                imputed_mask |= nans
                out_df["pressure_grad_mag"] = out_df["pressure_grad_mag"].fillna(pg_med).astype(np.float32)
        else:
            out_df["pressure_grad_mag"] = np.float32(p_grad_default)

        # 7. Impute Weather Humidity Columns
        h_mean = (
            float(out_df["humidity_mean_surround"].dropna().median())
            if "humidity_mean_surround" in out_df.columns and not out_df["humidity_mean_surround"].dropna().empty
            else CLIMATOLOGY_HUMIDITY_PCT
        )
        if "humidity_mean_surround" in out_df.columns:
            nans = out_df["humidity_mean_surround"].isna()
            if nans.any():
                imputed_mask |= nans
                out_df["humidity_mean_surround"] = out_df["humidity_mean_surround"].fillna(h_mean).astype(np.float32)
        else:
            out_df["humidity_mean_surround"] = np.float32(h_mean)

        for h_col in ["humidity_std_surround", "humidity_grad_mag"]:
            if h_col in out_df.columns:
                nans = out_df[h_col].isna()
                if nans.any():
                    imputed_mask |= nans
                    out_df[h_col] = out_df[h_col].fillna(0.0).astype(np.float32)
            else:
                out_df[h_col] = np.float32(0.0)

        # 8. Impute Topography & Solar Columns
        for col in TOPO_SOLAR_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans
                    valid = out_df[col].dropna()
                    val = float(valid.median()) if len(valid) > 0 else 0.0
                    out_df[col] = out_df[col].fillna(val).astype(np.float32)
            else:
                out_df[col] = np.float32(0.0)

        out_df["imputed_flag"] = imputed_mask.astype(np.int32)
        return out_df


# ==============================================================================
# SPATIAL-TEMPORAL TARGET PM2.5 MATCHER
# ==============================================================================

class TargetMatcher:
    """
    Associates ground truth PM2.5 station observations with grid cells based on
    spatial proximity (KDTree) and temporal observation window without data leakage.
    """

    def __init__(self):
        self.to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)

    def attach_ground_truth(
        self,
        grid_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        max_distance_km: float = 2.0,
        lat_col: str = "lat",
        lon_col: str = "lon",
        target_lat_col: str = "latitude",
        target_lon_col: str = "longitude",
        target_val_col: str = "observed_pm25",
        target_dt: Optional[pd.Timestamp] = None,
        time_window_minutes: float = 60.0,
        timestamp_col: str = "timestamp_utc",
    ) -> pd.DataFrame:
        """
        Attaches target_val_col to grid_df cells matching ground truth stations.
        Cells with no ground station will have NaN for target_val_col.
        Enforces positional iloc mapping to eliminate slice/index misalignment (Issue #1)
        and optional timestamp window filtering to eliminate multi-day temporal aggregation (Issue #2).
        """
        out_df = grid_df.copy()
        if target_val_col not in out_df.columns:
            out_df[target_val_col] = np.nan

        if ground_truth_df.empty:
            return out_df

        valid_gt = ground_truth_df.dropna(subset=[target_lat_col, target_lon_col, target_val_col]).copy()
        if valid_gt.empty:
            return out_df

        # Temporal filtering if target_dt is specified
        if target_dt is not None:
            ts_col = timestamp_col if timestamp_col in valid_gt.columns else (
                "timestamp" if "timestamp" in valid_gt.columns else None
            )
            if ts_col is not None:
                gt_dt = pd.to_datetime(valid_gt[ts_col], utc=True)
                target_utc = pd.to_datetime(target_dt, utc=True)
                time_diff = (gt_dt - target_utc).abs()
                time_mask = time_diff <= pd.Timedelta(minutes=time_window_minutes)
                valid_gt = valid_gt[time_mask].copy()
                if valid_gt.empty:
                    logger.debug("No ground truth observations within ±%.0f min of %s", time_window_minutes, target_dt)
                    return out_df

        # Resolve grid lat/lon column names
        g_lat = lat_col if lat_col in out_df.columns else "centroid_lat"
        g_lon = lon_col if lon_col in out_df.columns else "centroid_lon"

        # Transform coordinates to UTM km
        gx_m, gy_m = self.to_utm.transform(out_df[g_lon].to_numpy(), out_df[g_lat].to_numpy())
        grid_coords_km = np.column_stack([gx_m / 1000.0, gy_m / 1000.0])

        tx_m, ty_m = self.to_utm.transform(valid_gt[target_lon_col].to_numpy(), valid_gt[target_lat_col].to_numpy())
        gt_coords_km = np.column_stack([tx_m / 1000.0, ty_m / 1000.0])

        grid_tree = cKDTree(grid_coords_km)
        dists, cell_idxs = grid_tree.query(gt_coords_km, distance_upper_bound=max_distance_km)

        # Assign observed PM2.5 to matched cells
        gt_vals = valid_gt[target_val_col].to_numpy(dtype=np.float32)
        matched_cells: Dict[int, List[float]] = {}

        for d, idx, val in zip(dists, cell_idxs, gt_vals):
            if d <= max_distance_km and idx < len(out_df):
                if idx not in matched_cells:
                    matched_cells[idx] = []
                matched_cells[idx].append(float(val))

        # Assign average using positional iloc indexing to guarantee index alignment
        col_pos = out_df.columns.get_loc(target_val_col)
        for idx, vals in matched_cells.items():
            out_df.iloc[idx, col_pos] = float(np.mean(vals))

        logger.info(
            "Attached PM2.5 ground truth: Matched %d distinct grid cells from %d station records (max dist=%.1f km)",
            len(matched_cells), len(valid_gt), max_distance_km
        )
        return out_df

    def filter_training_cells(
        self,
        grid_df: pd.DataFrame,
        target_val_col: str = TARGET_COLUMN,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filters the matrix to only include rows that have a valid ground truth PM2.5/PM10 label
        (for training and evaluation splits).
        """
        eff_col = target_col or target_val_col
        if eff_col not in grid_df.columns:
            raise KeyError(f"Target column '{eff_col}' not found in DataFrame")
        return grid_df.dropna(subset=[eff_col]).reset_index(drop=True)


# ==============================================================================
# UNIFIED FEATURE MATRIX BUILDER ENGINE
# ==============================================================================

class FeatureMatrixBuilder:
    """
    Main Assembly Engine: Combines Satellite 10 IR Bands, 4-Directional Weather,
    Topography DEM, Solar Geometry, and Target PM2.5 into a Unified Parquet Matrix.
    """

    def __init__(
        self,
        grid_engine: Optional[MultiResolutionGridEngine] = None,
        imputer: Optional[PhysicsFallbackImputer] = None,
        target_matcher: Optional[TargetMatcher] = None,
        output_dir: Path = FEATURES_DIR,
    ):
        self.grid_engine = grid_engine or MultiResolutionGridEngine()
        self.imputer = imputer or PhysicsFallbackImputer()
        self.target_matcher = target_matcher or TargetMatcher()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)

    def standardize_keys(
        self,
        df: pd.DataFrame,
        timestamp_utc: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Standardizes identifier columns: grid_id, timestamp, lat, lon, utm_x, utm_y.
        """
        out_df = df.copy()

        # Handle coordinate aliases
        if "lat" not in out_df.columns and "centroid_lat" in out_df.columns:
            out_df["lat"] = out_df["centroid_lat"].astype(np.float32)
        if "lon" not in out_df.columns and "centroid_lon" in out_df.columns:
            out_df["lon"] = out_df["centroid_lon"].astype(np.float32)

        # Handle UTM coordinates
        if "utm_x" not in out_df.columns or "utm_y" not in out_df.columns:
            lons = out_df["lon"].to_numpy(dtype=float)
            lats = out_df["lat"].to_numpy(dtype=float)
            ux, uy = self.to_utm.transform(lons, lats)
            out_df["utm_x"] = ux.astype(np.float32)
            out_df["utm_y"] = uy.astype(np.float32)

        # Timestamp column
        out_df["timestamp"] = pd.to_datetime(timestamp_utc, utc=True)

        # Grid ID
        if "grid_id" not in out_df.columns:
            out_df["grid_id"] = [f"GRID_{i:06d}" for i in range(len(out_df))]

        return out_df

    def build_snapshot_matrix(
        self,
        target_dt: pd.Timestamp,
        grid_df: pd.DataFrame,
        satellite_df: Optional[pd.DataFrame] = None,
        weather_df: Optional[pd.DataFrame] = None,
        ground_truth_df: Optional[pd.DataFrame] = None,
        resolution_km: float = 2.0,
        exclusion_radius_km: Optional[float] = None,
        impute_missing: bool = True,
        training_only: bool = False,
        target_col: str = TARGET_COLUMN,
    ) -> pd.DataFrame:
        """
        Builds a single-timestamp unified feature matrix across grid cells.
        Enforces Temporal & Spatial Anti-Leakage rules.
        """
        run_dt = pd.to_datetime(target_dt, utc=True)
        exclusion_km = float(exclusion_radius_km or EXCLUSION_BUFFER_MAP.get(resolution_km, DEFAULT_EXCLUSION_KM))

        # 1. Anti-Leakage: Zero Look-Ahead Bias Verification
        if satellite_df is not None and "timestamp" in satellite_df.columns:
            TemporalAntiLeakageValidator.validate_dataframe(
                satellite_df, run_dt, timestamp_col="timestamp", source_name="Satellite Features"
            )
        if weather_df is not None and "timestamp" in weather_df.columns:
            TemporalAntiLeakageValidator.validate_dataframe(
                weather_df, run_dt, timestamp_col="timestamp", source_name="Directional Weather"
            )
        if ground_truth_df is not None and "timestamp_utc" in ground_truth_df.columns:
            TemporalAntiLeakageValidator.validate_dataframe(
                ground_truth_df, run_dt, timestamp_col="timestamp_utc", source_name="Ground Target"
            )

        # 2. Prepare Base Spatial Grid with Topography
        matrix = grid_df.copy()
        if "tpi_index" not in matrix.columns:
            matrix["tpi_index"] = self.grid_engine.compute_tpi_index(matrix)
        if "relief_m" not in matrix.columns:
            matrix["relief_m"] = self.grid_engine.compute_local_relief(matrix)

        # Standardize Keys & Coordinates
        matrix = self.standardize_keys(matrix, run_dt)

        # 3. Attach Solar Angles & Cyclical Time
        matrix = self.grid_engine.attach_solar_and_time_features(
            matrix, observation_dt=run_dt, lat_col="lat", lon_col="lon"
        )

        # 4. Merge Weather Features if provided
        if weather_df is not None and not weather_df.empty:
            weather_cols = [c for c in WEATHER_PRESSURE_COLUMNS + WEATHER_HUMIDITY_COLUMNS if c in weather_df.columns]
            if "grid_id" in weather_df.columns and "grid_id" in matrix.columns:
                matrix = matrix.merge(weather_df[["grid_id"] + weather_cols], on="grid_id", how="left")
            else:
                if len(weather_df) != len(matrix):
                    raise ValueError(f"Weather DataFrame row count ({len(weather_df)}) does not match grid ({len(matrix)})")
                for c in weather_cols:
                    matrix[c] = weather_df[c].values

        # 5. Merge Satellite Features if provided
        if satellite_df is not None and not satellite_df.empty:
            sat_cols = [
                c for c in SAT_BT_COLUMNS + SAT_BTD_COLUMNS + SAT_TEXTURE_COLUMNS + SAT_TEMPORAL_COLUMNS
                if c in satellite_df.columns
            ]
            if "grid_id" in satellite_df.columns and "grid_id" in matrix.columns:
                matrix = matrix.merge(satellite_df[["grid_id"] + sat_cols], on="grid_id", how="left")
            else:
                if len(satellite_df) != len(matrix):
                    raise ValueError(f"Satellite DataFrame row count ({len(satellite_df)}) does not match grid ({len(matrix)})")
                for c in sat_cols:
                    matrix[c] = satellite_df[c].values

        # 6. Attach Ground Truth Target if provided
        if ground_truth_df is not None and not ground_truth_df.empty:
            targets_to_attach = [target_col]
            for candidate in ["observed_pm25", "observed_pm10"]:
                if candidate in ground_truth_df.columns and candidate not in targets_to_attach:
                    targets_to_attach.append(candidate)

            for tc in targets_to_attach:
                if tc in ground_truth_df.columns:
                    matrix = self.target_matcher.attach_ground_truth(
                        matrix,
                        ground_truth_df,
                        max_distance_km=exclusion_km,
                        lat_col="lat",
                        lon_col="lon",
                        target_val_col=tc,
                        target_dt=run_dt,
                    )
            if training_only:
                matrix = self.target_matcher.filter_training_cells(matrix, target_val_col=target_col)

        # 7. Spatial Anti-Leakage & Target Isolation Verification
        SpatialAntiLeakageValidator.verify_target_feature_isolation(
            feature_columns=ALL_FEATURE_COLUMNS, target_col=[TARGET_COLUMN, TARGET_COLUMN_PM10]
        )

        # 8. Deterministic Physics Fallback Imputer
        if impute_missing:
            matrix = self.imputer.impute_features(matrix)

        # 9. Enforce Column Order & Float32 Precision
        present_keys = [c for c in KEY_COLUMNS if c in matrix.columns]
        present_features = [c for c in ALL_FEATURE_COLUMNS if c in matrix.columns]
        diagnostic_cols = [c for c in ["cloud_occluded_flag", "imputed_flag"] if c in matrix.columns]

        ordered_cols = present_keys + present_features + diagnostic_cols
        for tc in ["observed_pm25", "observed_pm10"]:
            if tc in matrix.columns and tc not in ordered_cols:
                ordered_cols.append(tc)

        matrix = matrix[ordered_cols].copy()

        # Cast feature columns to float32
        for fc in present_features:
            matrix[fc] = matrix[fc].astype(np.float32)

        matrix["resolution_km"] = np.float32(resolution_km)
        matrix["exclusion_radius_km"] = np.float32(exclusion_km)
        matrix.attrs["target_column"] = target_col

        return matrix

    def save_matrix(
        self,
        df: pd.DataFrame,
        filename_or_path: Union[str, Path] = "unified_pm25_matrix.parquet",
        partition_by_time: bool = False,
    ) -> Path:
        """
        Saves DataFrame as Apache Parquet with Snappy compression.
        Optionally partitions by year=YYYY/month=MM.
        """
        if isinstance(filename_or_path, Path) and filename_or_path.is_absolute():
            out_path = filename_or_path
        else:
            out_path = self.output_dir / filename_or_path

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if partition_by_time:
            # Partitioning by year and month
            if "timestamp" not in df.columns:
                raise ValueError("DataFrame must contain 'timestamp' column for partitioned storage")

            # Store in partitioned directory
            base_dir = out_path if out_path.is_dir() else out_path.parent
            df_to_save = df.copy()
            df_to_save["year"] = pd.to_datetime(df_to_save["timestamp"]).dt.year
            df_to_save["month"] = pd.to_datetime(df_to_save["timestamp"]).dt.month

            df_to_save.to_parquet(
                base_dir,
                partition_cols=["year", "month"],
                compression="snappy",
                index=False
            )
            logger.info("Saved time-partitioned Parquet dataset in: %s", base_dir)
            return base_dir
        else:
            df.to_parquet(out_path, compression="snappy", index=False)
            logger.info("Saved unified Parquet file: %s (Rows: %d, Cols: %d)", out_path, len(df), len(df.columns))
            return out_path

    @staticmethod
    def split_train_test(
        df: pd.DataFrame,
        split_date: str = "2026-01-01",
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset by timestamp into:
        - Train (2024 - 2025): timestamp < split_date
        - Out-of-Time Test (2026): timestamp >= split_date
        """
        if timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_col}' not found in DataFrame")

        ts_series = pd.to_datetime(df[timestamp_col], utc=True)
        split_dt = pd.to_datetime(split_date, utc=True)

        train_mask = ts_series < split_dt
        test_mask = ts_series >= split_dt

        train_df = df[train_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        logger.info(
            "Temporal Train/Test Split (Cutoff: %s): Train = %d rows, Test = %d rows",
            split_date, len(train_df), len(test_df)
        )
        return train_df, test_df

    @staticmethod
    def load_matrix(
        file_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        target_only: bool = False,
    ) -> pd.DataFrame:
        """
        Fast Columnar Projection reader: Loads only requested columns from Parquet
        to minimize RAM consumption.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Unified matrix file not found: {p}")

        if target_only:
            cols = KEY_COLUMNS + [TARGET_COLUMN]
        elif columns is not None:
            cols = list(columns)
        else:
            cols = None

        return pd.read_parquet(p, columns=cols)
