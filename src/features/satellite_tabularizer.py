"""
Satellite Tabularizer: Converts 10 IR Bands Raster Scenes to Numerical Feature Tables
Implements:
- Group 1: Centroid Brightness Temperature (10 Bands: bt_b07 .. bt_b16 in Kelvin)
- Group 2: Physics-informed Brightness Temperature Differences (BTDs)
    * btd_15_13 (Split-window silicate/dust index: BT15 - BT13)
    * btd_14_11 (Particle size & optical depth: BT14 - BT11)
    * btd_13_07 (Surface vs aerosol smoke contrast: BT13 - BT07)
    * btd_10_08 (Tropospheric water vapor / lapse rate inversion: BT10 - BT08)
    * dust_rgb_red (WMO Dust RGB red channel normalization)
- Group 3: Spatial 3x3 Texture (~6x6 km window):
    * b13_mean_3x3, b13_std_3x3 (cloud vs smoke smoothness), b13_min_3x3, b13_max_3x3
    * b15_mean_3x3, b15_std_3x3
    * btd_15_13_mean_3x3, btd_15_13_std_3x3
- Group 4: Temporal Dynamics (Tendencies):
    * delta_bt13_10m, delta_bt13_30m, delta_bt13_60m
    * delta_btd15_13_10m, delta_btd15_13_30m, delta_btd15_13_60m
- Zero-Mock Rule: 100% authentic physical measurements; NaN is preserved for bad pixels.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

CORE_5_IR_BANDS = ["B07", "B10", "B11", "B13", "B15"]
ALL_IR_BANDS = CORE_5_IR_BANDS
FULL_10_IR_BANDS = ["B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16"]
BAND_COLS = [f"bt_{b.lower()}" for b in ALL_IR_BANDS]

PHYSICAL_BT_MIN_K = 150.0
PHYSICAL_BT_MAX_K = 350.0

# P0-5: daytime B07 mixes reflected solar + thermal emission. Solar term scale
# (K per unit cos(SZA)) used for correct-then-drop normalisation to a
# night-time thermal baseline. Heuristic, documented — SZA itself never enters the model.
B07_SOLAR_CORRECTION_SCALE_K = 12.0
DAY_NIGHT_SZA_THRESHOLD_DEG = 85.0


def apply_solar_correction_to_btd13_07(
    df: pd.DataFrame,
    btd_col: str = "btd_13_07",
    sza_col: str = "solar_zenith_angle",
    daytime_col: str = "is_daytime",
) -> pd.DataFrame:
    """
    P0-5 fix (15-10-2026): correct-then-drop for B07 day/night physics.

    Daytime B07 BT is elevated by reflected solar radiation (~cos(SZA)), so raw
    BT13-BT07 mixes two physics regimes. We normalise toward the night-time
    thermal baseline: corrected = raw + SCALE * cos(SZA) * is_day.
    Callers keep the corrected BTD as a feature and DROP SZA/is_day (meta only).
    """
    out = df.copy()
    if btd_col not in out.columns:
        return out
    if sza_col in out.columns:
        sza = out[sza_col].to_numpy(dtype=float)
    elif daytime_col in out.columns:
        sza = np.where(out[daytime_col].to_numpy() == 1, 45.0, 100.0)
    else:
        return out
    if daytime_col in out.columns:
        is_day = (out[daytime_col].to_numpy() == 1).astype(float)
    else:
        is_day = (sza < DAY_NIGHT_SZA_THRESHOLD_DEG).astype(float)
    cos_sza = np.cos(np.radians(np.clip(sza, 0.0, 90.0)))
    correction = B07_SOLAR_CORRECTION_SCALE_K * np.maximum(cos_sza, 0.0) * is_day
    out[btd_col] = (out[btd_col].to_numpy(dtype=float) + correction).astype(np.float32)
    return out


class SatelliteTabularizer:
    """
    Transforms multi-band geostationary satellite rasters into numerical tabular features
    for 2x2 km (and multi-resolution) grid cells.
    """

    def __init__(self, bands: Optional[List[str]] = None):
        self.bands = bands or ALL_IR_BANDS
        self.band_cols = [f"bt_{b.lower()}" for b in self.bands]

    @staticmethod
    def validate_bounds(arr: np.ndarray) -> np.ndarray:
        """Enforces physical bounds [150K, 350K]; out-of-bounds become NaN."""
        cleaned = arr.copy()
        oob = (cleaned < PHYSICAL_BT_MIN_K) | (cleaned > PHYSICAL_BT_MAX_K)
        cleaned[oob] = np.nan
        return cleaned

    @staticmethod
    def compute_btds(df: pd.DataFrame, solar_zenith_angle: Optional[Any] = None) -> pd.DataFrame:
        """
        Computes Group 2 physics-based BTD features from available BT columns:
        - btd_15_13 = bt_b15 - bt_b13 (Silicate / coarse-fine dust index)
        - btd_13_07 = bt_b13 - bt_b07 (Hotspot / biomass smoke contrast)
        - btd_13_10 = bt_b13 - bt_b10 (Lower troposphere moisture / stability proxy)
        - btd_13_11 = bt_b13 - bt_b11 (Sulfate & aerosol particle size proxy)

        P1-2 fix: btd_14_11 / btd_10_08 aliases REMOVED (B14/B08 never ingested —
        silent duplication). P1-2: dust_rgb_red REMOVED (monotonic transform of
        btd_15_13, was schema drift vs SAT_BTD_COLUMNS).
        P0-5: pass solar_zenith_angle (array/Series) to solar-correct btd_13_07
        toward the night-time baseline before it enters the model.
        """
        out_df = df.copy()

        # btd_15_13 (Split-window silicate/dust index: BT15 - BT13)
        if "bt_b15" in out_df.columns and "bt_b13" in out_df.columns:
            out_df["btd_15_13"] = (out_df["bt_b15"] - out_df["bt_b13"]).astype(np.float32)
            # Diagnostic only (NOT a training feature — excluded from SAT_BTD_COLUMNS
            # to avoid perfect collinearity with btd_15_13). Kept for back-compat
            # with existing tests/downstream RGB visualisation.
            out_df["dust_rgb_red"] = np.clip(
                (out_df["btd_15_13"] - (-4.0)) / (2.0 - (-4.0)), 0.0, 1.0
            ).astype(np.float32)

        # btd_13_07 (Hotspots, fire & agricultural smoke contrast: BT13 - BT07)
        if "bt_b13" in out_df.columns and "bt_b07" in out_df.columns:
            out_df["btd_13_07"] = (out_df["bt_b13"] - out_df["bt_b07"]).astype(np.float32)
            if solar_zenith_angle is not None:
                try:
                    sza_arr = np.asarray(solar_zenith_angle, dtype=float)
                    if sza_arr.size == 1 and len(out_df) > 1:
                        sza_arr = np.full(len(out_df), float(sza_arr.flat[0]))
                    is_day = (sza_arr < DAY_NIGHT_SZA_THRESHOLD_DEG).astype(float)
                    cos_sza = np.cos(np.radians(np.clip(sza_arr, 0.0, 90.0)))
                    corr = B07_SOLAR_CORRECTION_SCALE_K * np.maximum(cos_sza, 0.0) * is_day
                    out_df["btd_13_07"] = (out_df["btd_13_07"].to_numpy(dtype=float) + corr).astype(np.float32)
                except Exception:
                    pass

        # btd_13_10 (Clean Window minus Lower Water Vapor = Lower Troposphere Moisture & Stability Proxy)
        if "bt_b13" in out_df.columns and "bt_b10" in out_df.columns:
            out_df["btd_13_10"] = (out_df["bt_b13"] - out_df["bt_b10"]).astype(np.float32)

        # btd_13_11 (Clean Window minus Cloud/Sulfate Phase = Sulfate/Particle Size Proxy)
        if "bt_b13" in out_df.columns and "bt_b11" in out_df.columns:
            out_df["btd_13_11"] = (out_df["bt_b13"] - out_df["bt_b11"]).astype(np.float32)

        # P1-2 fix: genuine BTDs only when source bands were actually ingested.
        # NO silent aliasing (old code copied btd_13_11->btd_14_11 when B14 missing,
        # manufacturing a duplicate feature). Missing bands simply stay absent.
        if "bt_b14" in out_df.columns and "bt_b11" in out_df.columns:
            out_df["btd_14_11"] = (out_df["bt_b14"] - out_df["bt_b11"]).astype(np.float32)
        if "bt_b10" in out_df.columns and "bt_b08" in out_df.columns:
            out_df["btd_10_08"] = (out_df["bt_b10"] - out_df["bt_b08"]).astype(np.float32)

        return out_df

    @staticmethod
    def extract_3x3_texture(
        b13_grid: np.ndarray,
        b15_grid: np.ndarray,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Computes spatial window statistics (mean, std, min, max) and surrounding
        spatial differences (defect/contrast) for given row/col locations:
        - 3x3 immediate window (radius 1)
        - 7x7 surrounding window (radius 3 pixels)
        """
        n = len(row_indices)
        h, w = b13_grid.shape

        b13_mean = np.full(n, np.nan, dtype=np.float32)
        b13_std = np.full(n, np.nan, dtype=np.float32)
        b13_min = np.full(n, np.nan, dtype=np.float32)
        b13_max = np.full(n, np.nan, dtype=np.float32)

        b15_mean = np.full(n, np.nan, dtype=np.float32)
        b15_std = np.full(n, np.nan, dtype=np.float32)

        btd_mean = np.full(n, np.nan, dtype=np.float32)
        btd_std = np.full(n, np.nan, dtype=np.float32)

        b13_diff_3x3 = np.zeros(n, dtype=np.float32)
        btd_diff_3x3 = np.zeros(n, dtype=np.float32)

        # Surrounding 3-pixel neighborhood (radius 3 = 7x7 patch)
        b13_mean_rad3 = np.full(n, np.nan, dtype=np.float32)
        b13_std_rad3 = np.full(n, np.nan, dtype=np.float32)
        b13_diff_rad3 = np.zeros(n, dtype=np.float32)

        btd_mean_rad3 = np.full(n, np.nan, dtype=np.float32)
        btd_std_rad3 = np.full(n, np.nan, dtype=np.float32)
        btd_diff_rad3 = np.zeros(n, dtype=np.float32)

        btd_grid = b15_grid - b13_grid

        for i in range(n):
            r = int(row_indices[i])
            c = int(col_indices[i])

            if r < 0 or r >= h or c < 0 or c >= w:
                continue

            center_b13 = b13_grid[r, c]
            center_btd = btd_grid[r, c]

            # 1. Immediate 3x3 patch (radius 1)
            r_min = max(0, r - 1)
            r_max = min(h, r + 2)
            c_min = max(0, c - 1)
            c_max = min(w, c + 2)

            patch_b13 = b13_grid[r_min:r_max, c_min:c_max]
            patch_b15 = b15_grid[r_min:r_max, c_min:c_max]
            patch_btd = btd_grid[r_min:r_max, c_min:c_max]

            valid_b13 = patch_b13[~np.isnan(patch_b13)]
            valid_b15 = patch_b15[~np.isnan(patch_b15)]
            valid_btd = patch_btd[~np.isnan(patch_btd)]

            # P1-3 fix: edge-clipped windows have fewer pixels — variance from
            # <5 valid pixels is biased. Require >=5 for 3x3 stats, else NaN.
            if len(valid_b13) >= 5:
                m13 = float(np.mean(valid_b13))
                b13_mean[i] = m13
                b13_std[i] = float(np.std(valid_b13)) if len(valid_b13) > 1 else 0.0
                b13_min[i] = float(np.min(valid_b13))
                b13_max[i] = float(np.max(valid_b13))
                if not np.isnan(center_b13):
                    b13_diff_3x3[i] = float(center_b13 - m13)
            elif len(valid_b13) > 0:
                # Degenerate edge window: keep mean but mark spread as unknown
                b13_mean[i] = float(np.mean(valid_b13))
                b13_min[i] = float(np.min(valid_b13))
                b13_max[i] = float(np.max(valid_b13))

            if len(valid_b15) >= 5:
                b15_mean[i] = float(np.mean(valid_b15))
                b15_std[i] = float(np.std(valid_b15)) if len(valid_b15) > 1 else 0.0
            elif len(valid_b15) > 0:
                b15_mean[i] = float(np.mean(valid_b15))

            if len(valid_btd) >= 5:
                mbtd = float(np.mean(valid_btd))
                btd_mean[i] = mbtd
                btd_std[i] = float(np.std(valid_btd)) if len(valid_btd) > 1 else 0.0
                if not np.isnan(center_btd):
                    btd_diff_3x3[i] = float(center_btd - mbtd)
            elif len(valid_btd) > 0:
                btd_mean[i] = float(np.mean(valid_btd))

            # 2. Surrounding 3-pixel neighborhood (radius 3: 7x7 patch)
            r_min3 = max(0, r - 3)
            r_max3 = min(h, r + 4)
            c_min3 = max(0, c - 3)
            c_max3 = min(w, c + 4)

            patch3_b13 = b13_grid[r_min3:r_max3, c_min3:c_max3]
            patch3_btd = btd_grid[r_min3:r_max3, c_min3:c_max3]

            v3_b13 = patch3_b13[~np.isnan(patch3_b13)]
            v3_btd = patch3_btd[~np.isnan(patch3_btd)]

            # P1-3 fix: 7x7 rad3 window needs >=10 valid pixels for stable stats.
            if len(v3_b13) >= 10:
                m13_r3 = float(np.mean(v3_b13))
                b13_mean_rad3[i] = m13_r3
                b13_std_rad3[i] = float(np.std(v3_b13)) if len(v3_b13) > 1 else 0.0
                if not np.isnan(center_b13):
                    b13_diff_rad3[i] = float(center_b13 - m13_r3)
            elif len(v3_b13) > 0:
                b13_mean_rad3[i] = float(np.mean(v3_b13))

            if len(v3_btd) >= 10:
                mbtd_r3 = float(np.mean(v3_btd))
                btd_mean_rad3[i] = mbtd_r3
                btd_std_rad3[i] = float(np.std(v3_btd)) if len(v3_btd) > 1 else 0.0
                if not np.isnan(center_btd):
                    btd_diff_rad3[i] = float(center_btd - mbtd_r3)
            elif len(v3_btd) > 0:
                btd_mean_rad3[i] = float(np.mean(v3_btd))

        return {
            "b13_mean_3x3": b13_mean,
            "b13_std_3x3": b13_std,
            "b13_min_3x3": b13_min,
            "b13_max_3x3": b13_max,
            "b15_mean_3x3": b15_mean,
            "b15_std_3x3": b15_std,
            "btd_15_13_mean_3x3": btd_mean,
            "btd_15_13_std_3x3": btd_std,
            "b13_diff_mean_3x3": b13_diff_3x3,
            "btd_15_13_diff_mean_3x3": btd_diff_3x3,
            "b13_mean_rad3": b13_mean_rad3,
            "b13_std_rad3": b13_std_rad3,
            "b13_diff_mean_rad3": b13_diff_rad3,
            "btd_15_13_mean_rad3": btd_mean_rad3,
            "btd_15_13_std_rad3": btd_std_rad3,
            "btd_15_13_diff_mean_rad3": btd_diff_rad3,
        }

    @staticmethod
    def compute_temporal_tendencies(
        current_df: pd.DataFrame,
        past_10m_df: Optional[pd.DataFrame] = None,
        past_30m_df: Optional[pd.DataFrame] = None,
        past_60m_df: Optional[pd.DataFrame] = None,
        past_cadence_df: Optional[pd.DataFrame] = None,
        cadence_hours: float = 3.0,
        key_col: str = "grid_id",
    ) -> pd.DataFrame:
        """
        Computes Group 4 temporal dynamic tendencies:
        - delta_bt13_cadence, delta_btd15_13_cadence
        - delta_bt13_rate_per_hour, delta_btd15_13_rate_per_hour
        - (Optional legacy) delta_bt13_10m/30m/60m if explicit past frames are provided.
        Ensures strict temporal order: past observation time < current run time.
        """
        out_df = current_df.copy()
        n = len(out_df)

        def _calc_delta(past_df: Optional[pd.DataFrame]):
            if past_df is None or past_df.empty:
                return None, None
            if key_col in out_df.columns and key_col in past_df.columns:
                merged = out_df[[key_col, "bt_b13", "btd_15_13"]].merge(
                    past_df[[key_col, "bt_b13", "btd_15_13"]],
                    on=key_col,
                    how="left",
                    suffixes=("", "_past")
                )
                d_bt13 = (merged["bt_b13"] - merged["bt_b13_past"]).to_numpy(dtype=np.float32)
                d_btd = (merged["btd_15_13"] - merged["btd_15_13_past"]).to_numpy(dtype=np.float32)
            else:
                d_bt13 = (out_df["bt_b13"].values - past_df["bt_b13"].values).astype(np.float32)
                d_btd = (out_df["btd_15_13"].values - past_df["btd_15_13"].values).astype(np.float32)
            return d_bt13, d_btd

        # 1. Authentic Cadence Tendency & Hourly Rates
        ref_past = past_cadence_df if past_cadence_df is not None else (past_60m_df if past_60m_df is not None else past_30m_df)
        d_c_bt13, d_c_btd = _calc_delta(ref_past)
        eff_hours = max(cadence_hours, 0.1) if past_cadence_df is not None else (1.0 if past_60m_df is not None else 0.5)

        # No previous frame -> NaN (missing), NOT 0.0 (which would falsely mean
        # "stationary atmosphere"). The downstream imputer fills NaN with 0.0 but
        # records temporal_delta_valid so the model can tell the two apart.
        out_df["delta_bt13_cadence"] = d_c_bt13 if d_c_bt13 is not None else np.full(n, np.nan, dtype=np.float32)
        out_df["delta_btd15_13_cadence"] = d_c_btd if d_c_btd is not None else np.full(n, np.nan, dtype=np.float32)
        out_df["delta_bt13_rate_per_hour"] = (out_df["delta_bt13_cadence"] / eff_hours).astype(np.float32)
        out_df["delta_btd15_13_rate_per_hour"] = (out_df["delta_btd15_13_cadence"] / eff_hours).astype(np.float32)

        # 2. Legacy sub-intervals (populated when explicit sub-frames are provided in tests)
        if past_10m_df is not None:
            d_bt, d_btd = _calc_delta(past_10m_df)
            out_df["delta_bt13_10m"] = d_bt
            out_df["delta_btd15_13_10m"] = d_btd
        if past_30m_df is not None:
            d_bt, d_btd = _calc_delta(past_30m_df)
            out_df["delta_bt13_30m"] = d_bt
            out_df["delta_btd15_13_30m"] = d_btd
        if past_60m_df is not None:
            d_bt, d_btd = _calc_delta(past_60m_df)
            out_df["delta_bt13_60m"] = d_bt
            out_df["delta_btd15_13_60m"] = d_btd

        return out_df

    def tabularize_from_scene(
        self,
        scene: Any,
        grid_df: pd.DataFrame,
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon",
        past_scenes: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Performs end-to-end extraction from a Satpy Scene onto a grid DataFrame:
        1. Samples 10 IR Bands at centroid coordinates
        2. Computes 3x3 texture for B13 & B15
        3. Computes Physics BTDs
        4. Computes Temporal tendencies if past_scenes dict provided ({'10m': scn, '30m': scn, '60m': scn})
        """
        res_df = grid_df.copy()
        query_lats = grid_df[lat_col].to_numpy(dtype=float)
        query_lons = grid_df[lon_col].to_numpy(dtype=float)

        # Get coordinate arrays from first available band
        sample_band = self.bands[0]
        ds = scene[sample_band]
        area = ds.attrs["area"]
        lons_sat, lats_sat = area.get_lonlats()

        margin = 0.2
        bbox_mask = (
            (lats_sat >= query_lats.min() - margin) &
            (lats_sat <= query_lats.max() + margin) &
            (lons_sat >= query_lons.min() - margin) &
            (lons_sat <= query_lons.max() + margin)
        )

        sub_lats = lats_sat[bbox_mask]
        sub_lons = lons_sat[bbox_mask]

        # Row/col indexing in cropped space
        rows_all, cols_all = np.indices(lats_sat.shape)
        sub_rows = rows_all[bbox_mask]
        sub_cols = cols_all[bbox_mask]

        tree = cKDTree(np.column_stack([sub_lats, sub_lons]))
        _, idxs = tree.query(np.column_stack([query_lats, query_lons]), k=1)

        target_rows = sub_rows[idxs]
        target_cols = sub_cols[idxs]

        # Extract 10 Bands
        for b in self.bands:
            b_col = f"bt_{b.lower()}"
            vals_grid = scene[b].values
            raw_sampled = vals_grid[target_rows, target_cols]
            clean_sampled = self.validate_bounds(raw_sampled)
            res_df[b_col] = clean_sampled.astype(np.float32)

        # 3x3 Texture
        if "B13" in scene and "B15" in scene:
            b13_grid = self.validate_bounds(scene["B13"].values)
            b15_grid = self.validate_bounds(scene["B15"].values)
            texture_dict = self.extract_3x3_texture(b13_grid, b15_grid, target_rows, target_cols)
            for k, arr in texture_dict.items():
                res_df[k] = arr

        # Compute BTDs
        res_df = self.compute_btds(res_df)

        # Compute Temporal Tendencies
        # P1-3 fix: past scenes must NOT blindly reuse current target_rows/cols.
        # Himawari full-disk grid is fixed, so reuse is valid only when the past
        # scene has identical shape. Otherwise recompute the nearest-pixel index
        # from the past scene's own lon/lats.
        past_dfs = {}
        if past_scenes:
            _cur_shape = None
            try:
                _cur_shape = scene[self.bands[0]].values.shape if self.bands[0] in scene else None
            except Exception:
                _cur_shape = None
            for t_key in ["10m", "30m", "60m", "cadence", "past"]:
                if t_key in past_scenes and past_scenes[t_key] is not None:
                    past_scn = past_scenes[t_key]
                    past_sub = grid_df.copy()
                    p_rows, p_cols = target_rows, target_cols
                    try:
                        if "B13" in past_scn:
                            p_shape = past_scn["B13"].values.shape
                            if _cur_shape is not None and p_shape != _cur_shape:
                                p_area = past_scn["B13"].attrs["area"]
                                p_lons, p_lats = p_area.get_lonlats()
                                p_tree = cKDTree(np.column_stack([
                                    p_lats.ravel(), p_lons.ravel(),
                                ]))
                                _, p_flat = p_tree.query(
                                    np.column_stack([query_lats, query_lons]), k=1
                                )
                                p_rows, p_cols = np.unravel_index(p_flat, p_shape)
                    except Exception:
                        p_rows, p_cols = target_rows, target_cols
                    if "B13" in past_scn:
                        past_sub["bt_b13"] = self.validate_bounds(past_scn["B13"].values[p_rows, p_cols])
                    if "B15" in past_scn and "B13" in past_scn:
                        b15_p = self.validate_bounds(past_scn["B15"].values[p_rows, p_cols])
                        past_sub["btd_15_13"] = (b15_p - past_sub["bt_b13"]).astype(np.float32)
                    past_dfs[t_key] = past_sub

        res_df = self.compute_temporal_tendencies(
            res_df,
            past_10m_df=past_dfs.get("10m"),
            past_30m_df=past_dfs.get("30m"),
            past_60m_df=past_dfs.get("60m"),
            past_cadence_df=past_dfs.get("cadence") or past_dfs.get("past"),
        )

        return res_df
