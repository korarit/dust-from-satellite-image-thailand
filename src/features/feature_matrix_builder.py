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
     * Group 4: Temporal Dynamics (delta_bt13_cadence, delta_btd15_13_cadence + rate_per_hour, temporal_delta_valid)
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
    METADATA_DIR,
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

# Physics BTD engineers actually fed to the models.
# Removed duplicates:
#   - btd_14_11 / btd_10_08 -> aliases of btd_13_11 / btd_13_10 (B14/B08 are never ingested)
#   - dust_rgb_red         -> deterministic monotonic transform of btd_15_13
SAT_BTD_COLUMNS = [
    "btd_15_13",
    "btd_13_07",
    "btd_13_10",
    "btd_13_11",
]

# Spatial texture statistics fed to the models.
# The four "*_diff_mean_*" defects are exact linear combinations of the
# center BT and the corresponding window mean (center - mean), so they are
# not emitted as features to avoid perfect collinearity.
SAT_TEXTURE_COLUMNS = [
    "b13_mean_3x3",
    "b13_std_3x3",
    "b13_min_3x3",
    "b13_max_3x3",
    "b15_mean_3x3",
    "b15_std_3x3",
    "btd_15_13_mean_3x3",
    "btd_15_13_std_3x3",
    "b13_mean_rad3",
    "b13_std_rad3",
    "btd_15_13_mean_rad3",
    "btd_15_13_std_rad3",
]

# Temporal tendencies fed to the models.
# Only the per-hour rate is kept: "*_cadence" is the same delta divided by a
# station-specific dt, i.e. perfectly collinear with the rate for valid rows.
# A companion "temporal_delta_valid" mask (see AUXILIARY_FEATURE_COLUMNS)
# disambiguates "no previous observation" from "genuinely unchanged".
SAT_TEMPORAL_COLUMNS = [
    "delta_bt13_rate_per_hour",
    "delta_btd15_13_rate_per_hour",
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
    # A1+A2 fix (15-10-2026): time/solar + coriolis removed from training features.
    # Only static topography + coastline remain. Solar/time are meta-only (see
    # SOLAR_META_COLUMNS) used for stratify/eval and B07 physics correction.
    "elevation_m",
    "tpi_index",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "elevation_std",
    "relief_m",
    "dist_coast_km",
]

# Solar geometry + cyclical time — META ONLY (never fed to models).
# Attached by grid_engine.attach_solar_meta() / attach_solar_and_time_features()
# for day-night stratification, B07 physics correction and reporting.
SOLAR_META_COLUMNS = [
    "solar_zenith_angle",
    "is_daytime",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

# Back-compat alias: old code imported TOPO_SOLAR_COLUMNS expecting 15 cols.
# Kept for reference only — DO NOT use for training.
TOPO_SOLAR_ALL_COLUMNS = TOPO_SOLAR_COLUMNS + SOLAR_META_COLUMNS + ["coriolis_param"]

WEATHER_FEATURE_COLUMNS = WEATHER_PRESSURE_COLUMNS + WEATHER_HUMIDITY_COLUMNS

# Data-quality / provenance masks exposed to the models so that an imputed or
# cloud-occluded value is never mistaken for a genuine measurement:
#   - temporal_delta_valid : 1 when a valid previous satellite observation exists
#   - imputed_flag         : 1 when any physical feature was physics-imputed
#   - cloud_occluded_flag  : 1 when deep cloud prevented an authentic retrieval
AUXILIARY_FEATURE_COLUMNS = [
    "temporal_delta_valid",
    "imputed_flag",
    "cloud_occluded_flag",
]

# Satellite & Topographic/Solar Features (Maritime & Offshore Capable, No Ground Stations Needed)
NO_WEATHER_FEATURE_COLUMNS = (
    SAT_BT_COLUMNS
    + SAT_BTD_COLUMNS
    + SAT_TEXTURE_COLUMNS
    + SAT_TEMPORAL_COLUMNS
    + TOPO_SOLAR_COLUMNS
    + AUXILIARY_FEATURE_COLUMNS
)

# Physical feature schema (no quality masks) - used for dtype/audit checks.
ALL_FEATURE_COLUMNS = (
    SAT_BT_COLUMNS
    + SAT_BTD_COLUMNS
    + SAT_TEXTURE_COLUMNS
    + SAT_TEMPORAL_COLUMNS
    + WEATHER_PRESSURE_COLUMNS
    + WEATHER_HUMIDITY_COLUMNS
    + TOPO_SOLAR_COLUMNS
)

# Full model input set = physical features + quality masks.
FULL_MULTIMODAL_COLUMNS = ALL_FEATURE_COLUMNS + AUXILIARY_FEATURE_COLUMNS
TARGET_COLUMN = "observed_pm25"
TARGET_COLUMN_PM25 = "observed_pm25"
TARGET_COLUMN_PM10 = "observed_pm10"

# Physical Climatology Fallback Constants for Thailand (Deterministic)
CLIMATOLOGY_PRESSURE_HPA = 1013.25
CLIMATOLOGY_HUMIDITY_PCT = 70.0
CLIMATOLOGY_SURFACE_BT_K = 290.0
DEEP_CLOUD_BT_THRESHOLD_K = 220.0
# P0-4 fix: cloud-occluded NaN BT must NOT be filled with clear-sky ~290K.
# Use a physically cold cloud-top default so flag+BTD keep the cloud signal.
CLOUD_BT_FILL_K = 210.0
# P1-1 fix: offshore / out-of-domain threshold (no ground stations beyond this).
OFFSHORE_DIST_COAST_KM = 50.0


def is_offshore_mask(df: pd.DataFrame) -> pd.Series:
    """P1-1 router helper: True for maritime/out-of-domain cells (pure-sat only)."""
    if "dist_coast_km" in df.columns:
        return df["dist_coast_km"].astype(float) > OFFSHORE_DIST_COAST_KM
    return pd.Series(False, index=df.index)


def route_offshore_to_pure_sat(
    df: pd.DataFrame,
    weather_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    P1-1 fix: offshore cells must not use imputed land climatology as if it were
    a measurement. Blank out ground-weather features offshore so downstream code
    is forced onto the pure-satellite path (or climatology flag).
    """
    out = df.copy()
    mask = is_offshore_mask(out)
    if not bool(mask.any()):
        return out
    wcols = weather_cols or (WEATHER_PRESSURE_COLUMNS + WEATHER_HUMIDITY_COLUMNS)
    for c in wcols:
        if c in out.columns:
            out.loc[mask, c] = np.nan
    out["offshore_pure_sat_only"] = mask.astype(np.int32)
    return out


def add_target_anomaly(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    group_col: str = "grid_id",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    A2 fix: target-anomaly to stop the model memorising station x season x hour
    means via static fingerprints. anomaly = y - monthly_station_climatology.
    Train on the anomaly; add climatology back at inference.
    """
    out = df.copy()
    if target_col not in out.columns:
        return out
    ts = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce") if timestamp_col in out.columns else None
    month = ts.dt.month if ts is not None else 1
    grp = out[group_col] if group_col in out.columns else "all"
    clim_key = pd.DataFrame({"_g": grp, "_m": month, "_y": out[target_col].astype(float)})
    climatology = clim_key.groupby(["_g", "_m"])["_y"].transform("mean")
    out[f"{target_col}_climatology"] = climatology.astype(np.float32)
    out[f"{target_col}_anomaly"] = (out[target_col].astype(float) - climatology).astype(np.float32)
    return out


def get_group_labels(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
) -> np.ndarray:
    """
    P0-1 fix: group labels for GroupKFold (station/province/cell).
    Priority: explicit group_col > station_code > grid_id > provider > region.
    """
    if group_col and group_col in df.columns:
        return df[group_col].astype(str).to_numpy()
    for cand in ("station_code", "grid_id", "provider", "region"):
        if cand in df.columns:
            return df[cand].astype(str).to_numpy()
    return np.zeros(len(df), dtype=str)


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

    def _fitted_or_default(self, col: str, default: float) -> float:
        """P0-2 fix: fitted train median first, else deterministic climatology default.

        Never falls back to the current batch median here (that would be
        test-imputing-test + train-serve skew). Batch medians are only learned
        in fit().
        """
        if col in self.fitted_stats_:
            return float(self.fitted_stats_[col])
        if not self.fitted_stats_:
            logger.warning(
                "PhysicsFallbackImputer not fitted; using climatology default "
                "%.3f for '%s'. Call fit(train_df) once before transform().",
                default, col,
            )
        return float(default)

    def impute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all model feature columns are free of NaN values while preserving
        physical laws and attaching diagnostic flags.

        P0-2: transform path uses ONLY fitted_stats_ + climatology defaults.
        P0-4: cloud-occluded BT NaNs use CLOUD_BT_FILL_K (210K), not clear-sky 290K.
        P1-2: no silent BTD aliases, no dust_rgb_red (removed from schema).
        """
        out_df = df.copy()

        # 1. Diagnostic Flag: Deep Cloud Occlusion (BT13 < 220 K or NaN)
        if "bt_b13" in out_df.columns:
            b13_vals = out_df["bt_b13"].to_numpy(dtype=float)
            cloud_mask = np.isnan(b13_vals) | (b13_vals < DEEP_CLOUD_BT_THRESHOLD_K)
            out_df["cloud_occluded_flag"] = cloud_mask.astype(np.int32)
        else:
            cloud_mask = np.zeros(len(out_df), dtype=bool)
            out_df["cloud_occluded_flag"] = 0

        # Track if any imputation was applied
        imputed_mask = np.zeros(len(out_df), dtype=bool)

        # 2. Impute Satellite BT Columns (P0-4: cloud pixels keep cold signal)
        cloud_series = pd.Series(cloud_mask, index=out_df.index)
        for col in SAT_BT_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans.to_numpy()
                    clear_fill = self._fitted_or_default(col, CLIMATOLOGY_SURFACE_BT_K)
                    fill_vals = pd.Series(np.float32(clear_fill), index=out_df.index)
                    # Cloud-occluded rows: cold cloud-top default, never clear-sky median
                    fill_vals = fill_vals.where(~(nans & (cloud_series == True)), np.float32(CLOUD_BT_FILL_K))  # noqa: E712
                    out_df[col] = out_df[col].fillna(fill_vals).astype(np.float32)
            else:
                default_bt = self._fitted_or_default(col, CLIMATOLOGY_SURFACE_BT_K)
                out_df[col] = np.float32(default_bt)
                imputed_mask[:] = True

        # 3. Impute / Recalculate BTD Columns (P1-2: genuine only, no silent aliases;
        # dust_rgb_red kept as diagnostic only, never a training feature)
        if "bt_b15" in out_df.columns and "bt_b13" in out_df.columns:
            out_df["btd_15_13"] = (out_df["bt_b15"] - out_df["bt_b13"]).astype(np.float32)
            # Diagnostic only (not in SAT_BTD_COLUMNS / not a training feature).
            out_df["dust_rgb_red"] = np.clip(
                (out_df["btd_15_13"] - (-4.0)) / (2.0 - (-4.0)), 0.0, 1.0
            ).astype(np.float32)

        if "bt_b13" in out_df.columns and "bt_b07" in out_df.columns:
            out_df["btd_13_07"] = (out_df["bt_b13"] - out_df["bt_b07"]).astype(np.float32)

        if "bt_b13" in out_df.columns and "bt_b10" in out_df.columns:
            out_df["btd_13_10"] = (out_df["bt_b13"] - out_df["bt_b10"]).astype(np.float32)

        if "bt_b13" in out_df.columns and "bt_b11" in out_df.columns:
            out_df["btd_13_11"] = (out_df["bt_b13"] - out_df["bt_b11"]).astype(np.float32)

        # NOTE: btd_14_11 / btd_10_08 / dust_rgb_red intentionally NOT created
        # (P1-2: B14/B08 never ingested; dust_rgb_red is a monotonic transform of
        # btd_15_13). Downstream prune in trainer drops exact duplicates if legacy
        # files still contain them.

        for col in SAT_BTD_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans.to_numpy() if hasattr(nans, "to_numpy") else nans
                    out_df[col] = out_df[col].fillna(0.0).astype(np.float32)
            else:
                out_df[col] = np.float32(0.0)

        # 4. Impute Spatial Texture & Defect Columns
        if "b13_diff_mean_3x3" not in out_df.columns and "bt_b13" in out_df.columns and "b13_mean_3x3" in out_df.columns:
            out_df["b13_diff_mean_3x3"] = (out_df["bt_b13"] - out_df["b13_mean_3x3"]).astype(np.float32)
        if "btd_15_13_diff_mean_3x3" not in out_df.columns and "btd_15_13" in out_df.columns and "btd_15_13_mean_3x3" in out_df.columns:
            out_df["btd_15_13_diff_mean_3x3"] = (out_df["btd_15_13"] - out_df["btd_15_13_mean_3x3"]).astype(np.float32)
        if "b13_diff_mean_rad3" not in out_df.columns and "bt_b13" in out_df.columns and "b13_mean_rad3" in out_df.columns:
            out_df["b13_diff_mean_rad3"] = (out_df["bt_b13"] - out_df["b13_mean_rad3"]).astype(np.float32)
        if "btd_15_13_diff_mean_rad3" not in out_df.columns and "btd_15_13" in out_df.columns and "btd_15_13_mean_rad3" in out_df.columns:
            out_df["btd_15_13_diff_mean_rad3"] = (out_df["btd_15_13"] - out_df["btd_15_13_mean_rad3"]).astype(np.float32)

        texture_base_map = {
            "b13_mean_3x3": "bt_b13",
            "b13_min_3x3": "bt_b13",
            "b13_max_3x3": "bt_b13",
            "b15_mean_3x3": "bt_b15",
            "btd_15_13_mean_3x3": "btd_15_13",
            "b13_mean_rad3": "b13_mean_3x3",
            "b13_std_rad3": "b13_std_3x3",
            "btd_15_13_mean_rad3": "btd_15_13_mean_3x3",
            "btd_15_13_std_rad3": "btd_15_13_std_3x3",
        }
        for col in SAT_TEXTURE_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans
                    if "std" in col or "diff" in col:
                        out_df[col] = out_df[col].fillna(0.0).astype(np.float32)
                    elif col in texture_base_map and texture_base_map[col] in out_df.columns:
                        out_df[col] = out_df[col].fillna(out_df[texture_base_map[col]]).astype(np.float32)
                    else:
                        out_df[col] = out_df[col].fillna(CLIMATOLOGY_SURFACE_BT_K).astype(np.float32)
            else:
                if "std" in col or "diff" in col:
                    out_df[col] = np.float32(0.0)
                elif col in texture_base_map and texture_base_map[col] in out_df.columns:
                    out_df[col] = out_df[texture_base_map[col]].astype(np.float32)
                else:
                    out_df[col] = np.float32(CLIMATOLOGY_SURFACE_BT_K)

        # 5. Impute Temporal Tendencies (Stationary Atmosphere assumption: delta = 0.0).
        #    A missing tendency is NOT the same as a genuinely flat one, so we first
        #    record a validity mask before back-filling 0.0 for numerical continuity.
        temporal_valid = pd.Series(1.0, index=out_df.index, dtype=np.float32)
        for col in SAT_TEMPORAL_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    temporal_valid = temporal_valid.where(~nans, 0.0)
                    imputed_mask |= nans
                    out_df[col] = out_df[col].fillna(0.0).astype(np.float32)
            else:
                out_df[col] = np.float32(0.0)
                temporal_valid = pd.Series(0.0, index=out_df.index, dtype=np.float32)
        out_df["temporal_delta_valid"] = temporal_valid.astype(np.float32)

        # 6. Impute Weather Pressure Columns with Barometric Elevation Scaling
        # P0-2 fix: fitted median first, else deterministic climatology (no batch leak).
        p_mean = self._fitted_or_default("pressure_mean_surround", CLIMATOLOGY_PRESSURE_HPA)
        # Barometric elevation scale factor: P(z) = P0 * (1 - L*z/T0)^5.25588
        if "elevation_m" in out_df.columns:
            _elev_fill = self._fitted_or_default("elevation_m", 0.0)
            z_s = out_df["elevation_m"].fillna(_elev_fill).clip(lower=0.0).astype(float)
            f_baro = (1.0 - (0.0065 * z_s) / 288.15) ** 5.25588
            p_elevation_adjusted = (p_mean * f_baro).astype(np.float32)
        else:
            p_elevation_adjusted = pd.Series(np.float32(p_mean), index=out_df.index, dtype=np.float32)

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

        # Impute pressure_grad_mag with fitted median (not 0.0, which falsely implies calm stagnation)
        p_grad_default = self._fitted_or_default("pressure_grad_mag", 1.2)
        if "pressure_grad_mag" in out_df.columns:
            nans = out_df["pressure_grad_mag"].isna()
            if nans.any():
                imputed_mask |= nans.to_numpy() if hasattr(nans, "to_numpy") else nans
                out_df["pressure_grad_mag"] = out_df["pressure_grad_mag"].fillna(p_grad_default).astype(np.float32)
        else:
            out_df["pressure_grad_mag"] = np.float32(p_grad_default)

        # 7. Impute Weather Humidity Columns (P0-2: fitted or climatology, no batch leak)
        h_mean = self._fitted_or_default("humidity_mean_surround", CLIMATOLOGY_HUMIDITY_PCT)
        if "humidity_mean_surround" in out_df.columns:
            nans = out_df["humidity_mean_surround"].isna()
            if nans.any():
                imputed_mask |= nans
                out_df["humidity_mean_surround"] = out_df["humidity_mean_surround"].fillna(h_mean).astype(np.float32)
        else:
            out_df["humidity_mean_surround"] = np.float32(h_mean)

        # humidity_grad_mag uses a fitted/climatology default (0.0 would falsely imply a
        # perfectly uniform humidity field); humidity_std_surround keeps 0.0
        # (zero spatial spread is the neutral value).
        h_grad_default = self._fitted_or_default("humidity_grad_mag", 0.0)
        for h_col in ["humidity_std_surround", "humidity_grad_mag"]:
            if h_col in out_df.columns:
                nans = out_df[h_col].isna()
                if nans.any():
                    imputed_mask |= nans.to_numpy() if hasattr(nans, "to_numpy") else nans
                    if h_col == "humidity_grad_mag":
                        out_df[h_col] = out_df[h_col].fillna(h_grad_default).astype(np.float32)
                    else:
                        out_df[h_col] = out_df[h_col].fillna(0.0).astype(np.float32)
            else:
                out_df[h_col] = np.float32(0.0)

        # 8. Impute Topography Columns (A1/A2: topo only — solar/time are meta, never imputed as features)
        for col in TOPO_SOLAR_COLUMNS:
            if col in out_df.columns:
                nans = out_df[col].isna()
                if nans.any():
                    imputed_mask |= nans.to_numpy() if hasattr(nans, "to_numpy") else nans
                    val = self._fitted_or_default(col, 0.0)
                    out_df[col] = out_df[col].fillna(val).astype(np.float32)
            else:
                out_df[col] = np.float32(0.0)
        # Legacy coriolis column: drop if present (A2 — must not enter the model).
        if "coriolis_param" in out_df.columns:
            out_df = out_df.drop(columns=["coriolis_param"])

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

        # Extract or infer providers from valid_gt
        gt_providers: List[Optional[str]] = []
        has_prov_col = "provider" in valid_gt.columns
        for _, r in valid_gt.iterrows():
            if has_prov_col and pd.notna(r["provider"]):
                gt_providers.append(str(r["provider"]).strip())
            else:
                stn = str(r.get("station_code", ""))
                src = str(r.get("source", ""))
                if stn.startswith("PCD_") or "Air4Thai" in src:
                    gt_providers.append("Air4Thai")
                elif "AirGradient" in src:
                    gt_providers.append("AirGradient")
                else:
                    gt_providers.append(None)

        # Assign observed PM2.5 and provider to matched cells
        gt_vals = valid_gt[target_val_col].to_numpy(dtype=np.float32)
        matched_cells: Dict[int, List[float]] = {}
        matched_providers: Dict[int, List[str]] = {}

        for d, idx, val, prov in zip(dists, cell_idxs, gt_vals, gt_providers):
            if d <= max_distance_km and idx < len(out_df):
                if idx not in matched_cells:
                    matched_cells[idx] = []
                    matched_providers[idx] = []
                matched_cells[idx].append(float(val))
                if prov is not None:
                    matched_providers[idx].append(prov)

        # Assign average using positional iloc indexing to guarantee index alignment
        col_pos = out_df.columns.get_loc(target_val_col)
        if "provider" not in out_df.columns:
            out_df["provider"] = None
        prov_col_pos = out_df.columns.get_loc("provider")

        for idx, vals in matched_cells.items():
            out_df.iloc[idx, col_pos] = float(np.mean(vals))
            p_list = matched_providers.get(idx, [])
            if p_list:
                out_df.iloc[idx, prov_col_pos] = p_list[0]

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

        # 3. Attach Solar Angles & Cyclical Time as META ONLY (A1 fix).
        # Kept in output for day/night stratify + B07 correction, but excluded
        # from ALL_FEATURE_COLUMNS so they never enter the model.
        # Late import to avoid circular dependency.
        try:
            from src.features.satellite_tabularizer import apply_solar_correction_to_btd13_07
            _has_solar_corr = True
        except Exception:
            _has_solar_corr = False
        matrix = self.grid_engine.attach_solar_meta(
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
        # A1 guard: solar/time meta must never be in the training schema.
        for _leak in ("solar_zenith_angle", "is_daytime", "hour_sin", "hour_cos",
                      "month_sin", "month_cos", "coriolis_param"):
            if _leak in ALL_FEATURE_COLUMNS:
                raise TargetFeatureIsolationError(f"Leakage: '{_leak}' must not be in ALL_FEATURE_COLUMNS")

        # P0-5: physics-correct B07 (correct-then-drop) once SZA meta exists.
        if _has_solar_corr and "btd_13_07" in matrix.columns and "solar_zenith_angle" in matrix.columns:
            try:
                matrix = apply_solar_correction_to_btd13_07(matrix)
            except Exception as exc:
                logger.warning("B07 solar correction skipped: %s", exc)

        # P1-1: offshore router — flag maritime out-of-domain cells BEFORE imputation
        # so they can be forced onto the pure-sat path downstream.
        try:
            matrix = route_offshore_to_pure_sat(matrix)
        except Exception as exc:
            logger.warning("Offshore routing skipped: %s", exc)

        # 8. Deterministic Physics Fallback Imputer (P0-2: fit-once, transform-many)
        if impute_missing:
            if not self.imputer.fitted_stats_:
                if training_only:
                    # First training snapshot defines the imputation climatology.
                    self.imputer.fit(matrix)
                else:
                    logger.warning(
                        "Imputer not fitted and training_only=False; using "
                        "climatology defaults only (no batch-median leak)."
                    )
            matrix = self.imputer.transform(matrix)

        # 9. Enforce Column Order & Float32 Precision
        # P0-3 fix: temporal_delta_valid restored to diagnostics (was silently dropped).
        # Solar meta retained in output for day/night eval, but NOT as features.
        present_keys = [c for c in KEY_COLUMNS if c in matrix.columns]
        present_features = [c for c in ALL_FEATURE_COLUMNS if c in matrix.columns]
        diagnostic_cols = [c for c in (list(AUXILIARY_FEATURE_COLUMNS) + ["offshore_pure_sat_only"]) if c in matrix.columns]
        meta_cols = [c for c in SOLAR_META_COLUMNS if c in matrix.columns and c not in present_features]

        ordered_cols = present_keys + present_features + diagnostic_cols + meta_cols
        for tc in ["observed_pm25", "observed_pm10"]:
            if tc in matrix.columns and tc not in ordered_cols:
                ordered_cols.append(tc)
        if "provider" in matrix.columns and "provider" not in ordered_cols:
            ordered_cols.append("provider")

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


def resolve_dataset_providers(
    df: pd.DataFrame,
    metadata_path: Optional[Path] = None,
    allowed_providers: Tuple[str, ...] = ("Air4Thai", "AirGradient"),
) -> pd.DataFrame:
    """
    Ensures that every training sample has an authenticated ground data provider label.
    Resolves providers from:
    1. Existing 'provider' column if valid
    2. 'station_code' / 'source' mapping (PCD_* -> Air4Thai, OPENAQ_* -> openaq metadata)
    3. Spatial KDTree nearest neighbor matching against openaq_thailand_stations.csv

    Strictly filters out sparse, non-representative providers (e.g. Clarity, AirNow, HabitatMap)
    and retains only authorized representative providers (default: Air4Thai and AirGradient).
    """
    if df.empty:
        return df

    out_df = df.copy()
    meta_p = metadata_path or (METADATA_DIR / "openaq_thailand_stations.csv")
    meta_df = pd.DataFrame()
    if meta_p.exists():
        meta_df = pd.read_csv(meta_p) if meta_p.suffix == ".csv" else pd.read_parquet(meta_p)

    code_map: Dict[str, str] = {}
    if not meta_df.empty and "station_code" in meta_df.columns and "provider" in meta_df.columns:
        code_map = dict(zip(meta_df["station_code"].astype(str), meta_df["provider"].astype(str)))

    providers: List[Optional[str]] = []
    need_kdtree = False

    lat_col = "lat" if "lat" in out_df.columns else "centroid_lat"
    lon_col = "lon" if "lon" in out_df.columns else "centroid_lon"
    has_coords = lat_col in out_df.columns and lon_col in out_df.columns

    has_prov_col = "provider" in out_df.columns

    for i in range(len(out_df)):
        row = out_df.iloc[i]
        if has_prov_col:
            p_val = row.get("provider")
            if pd.notna(p_val) and str(p_val).strip() and str(p_val).strip().lower() not in ("none", "nan", "unknown"):
                providers.append(str(p_val).strip())
            else:
                providers.append(None)
            continue

        sc = str(row.get("station_code", ""))
        src = str(row.get("source", ""))
        if sc.startswith("PCD_") or "Air4Thai" in src:
            providers.append("Air4Thai")
        elif "AirGradient" in src:
            providers.append("AirGradient")
        elif sc in code_map and pd.notna(code_map[sc]):
            providers.append(code_map[sc].strip())
        else:
            providers.append(None)
            need_kdtree = True

    # If some rows still lack provider and provider column wasn't explicitly given, use KDTree
    if need_kdtree and has_coords and not meta_df.empty:
        to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
        valid_meta = meta_df.dropna(subset=["longitude", "latitude", "provider"]).copy()
        sx, sy = to_utm.transform(
            valid_meta["longitude"].to_numpy(dtype=float),
            valid_meta["latitude"].to_numpy(dtype=float)
        )
        tree = cKDTree(np.column_stack([sx / 1000.0, sy / 1000.0]))

        mx, my = to_utm.transform(
            out_df[lon_col].to_numpy(dtype=float),
            out_df[lat_col].to_numpy(dtype=float)
        )
        dists, idxs = tree.query(np.column_stack([mx / 1000.0, my / 1000.0]), distance_upper_bound=100.0)

        meta_provs = valid_meta["provider"].tolist()
        for idx_row, (p, d, m_idx) in enumerate(zip(providers, dists, idxs)):
            if p is None and m_idx < len(meta_provs):
                providers[idx_row] = str(meta_provs[m_idx]).strip()

    out_df["provider"] = providers

    # Strict Filtering: Exclude sparse providers (Clarity, AirNow, HabitatMap, etc.) and None
    norm_allowed = {p.strip().lower() for p in allowed_providers}
    mask = out_df["provider"].notna() & out_df["provider"].astype(str).str.strip().str.lower().isin(norm_allowed)
    filtered = out_df[mask].reset_index(drop=True)
    return filtered

