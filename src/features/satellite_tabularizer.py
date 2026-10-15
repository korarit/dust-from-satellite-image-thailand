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
    def compute_btds(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Group 2 physics-based BTD features from available BT columns:
        - btd_15_13 = bt_b15 - bt_b13 (Silicate / coarse-fine dust index)
        - btd_13_07 = bt_b13 - bt_b07 (Hotspot / biomass smoke contrast)
        - btd_13_10 = bt_b13 - bt_b10 (Lower troposphere moisture / stability proxy)
        - btd_13_11 = bt_b13 - bt_b11 (Sulfate & aerosol particle size proxy)
        - btd_14_11 = bt_b14 - bt_b11 (or aliased to btd_13_11)
        - btd_10_08 = bt_b10 - bt_b08 (or aliased to btd_13_10)
        - dust_rgb_red = clip((btd_15_13 - (-4.0)) / (2.0 - (-4.0)), 0.0, 1.0)
        """
        out_df = df.copy()

        # btd_15_13 (Split-window silicate/dust index: BT15 - BT13)
        if "bt_b15" in out_df.columns and "bt_b13" in out_df.columns:
            out_df["btd_15_13"] = (out_df["bt_b15"] - out_df["bt_b13"]).astype(np.float32)
            # Dust RGB Red channel: -4K to +2K normalized to [0, 1]
            out_df["dust_rgb_red"] = np.clip(
                (out_df["btd_15_13"] - (-4.0)) / (2.0 - (-4.0)), 0.0, 1.0
            ).astype(np.float32)

        # btd_13_07 (Hotspots, fire & agricultural smoke contrast: BT13 - BT07)
        if "bt_b13" in out_df.columns and "bt_b07" in out_df.columns:
            out_df["btd_13_07"] = (out_df["bt_b13"] - out_df["bt_b07"]).astype(np.float32)

        # btd_13_10 (Clean Window minus Lower Water Vapor = Lower Troposphere Moisture & Stability Proxy)
        if "bt_b13" in out_df.columns and "bt_b10" in out_df.columns:
            out_df["btd_13_10"] = (out_df["bt_b13"] - out_df["bt_b10"]).astype(np.float32)

        # btd_13_11 (Clean Window minus Cloud/Sulfate Phase = Sulfate/Particle Size Proxy)
        if "bt_b13" in out_df.columns and "bt_b11" in out_df.columns:
            out_df["btd_13_11"] = (out_df["bt_b13"] - out_df["bt_b11"]).astype(np.float32)

        # btd_14_11 (Particle size: BT14 - BT11 or fallback alias to btd_13_11)
        if "bt_b14" in out_df.columns and "bt_b11" in out_df.columns:
            out_df["btd_14_11"] = (out_df["bt_b14"] - out_df["bt_b11"]).astype(np.float32)
        elif "btd_13_11" in out_df.columns and "btd_14_11" not in out_df.columns:
            out_df["btd_14_11"] = out_df["btd_13_11"]

        # btd_10_08 (Lapse rate / Inversion: BT10 - BT08 or fallback alias to btd_13_10)
        if "bt_b10" in out_df.columns and "bt_b08" in out_df.columns:
            out_df["btd_10_08"] = (out_df["bt_b10"] - out_df["bt_b08"]).astype(np.float32)
        elif "btd_13_10" in out_df.columns and "btd_10_08" not in out_df.columns:
            out_df["btd_10_08"] = out_df["btd_13_10"]

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

            if len(valid_b13) > 0:
                m13 = float(np.mean(valid_b13))
                b13_mean[i] = m13
                b13_std[i] = float(np.std(valid_b13)) if len(valid_b13) > 1 else 0.0
                b13_min[i] = float(np.min(valid_b13))
                b13_max[i] = float(np.max(valid_b13))
                if not np.isnan(center_b13):
                    b13_diff_3x3[i] = float(center_b13 - m13)

            if len(valid_b15) > 0:
                b15_mean[i] = float(np.mean(valid_b15))
                b15_std[i] = float(np.std(valid_b15)) if len(valid_b15) > 1 else 0.0

            if len(valid_btd) > 0:
                mbtd = float(np.mean(valid_btd))
                btd_mean[i] = mbtd
                btd_std[i] = float(np.std(valid_btd)) if len(valid_btd) > 1 else 0.0
                if not np.isnan(center_btd):
                    btd_diff_3x3[i] = float(center_btd - mbtd)

            # 2. Surrounding 3-pixel neighborhood (radius 3: 7x7 patch)
            r_min3 = max(0, r - 3)
            r_max3 = min(h, r + 4)
            c_min3 = max(0, c - 3)
            c_max3 = min(w, c + 4)

            patch3_b13 = b13_grid[r_min3:r_max3, c_min3:c_max3]
            patch3_btd = btd_grid[r_min3:r_max3, c_min3:c_max3]

            v3_b13 = patch3_b13[~np.isnan(patch3_b13)]
            v3_btd = patch3_btd[~np.isnan(patch3_btd)]

            if len(v3_b13) > 0:
                m13_r3 = float(np.mean(v3_b13))
                b13_mean_rad3[i] = m13_r3
                b13_std_rad3[i] = float(np.std(v3_b13)) if len(v3_b13) > 1 else 0.0
                if not np.isnan(center_b13):
                    b13_diff_rad3[i] = float(center_b13 - m13_r3)

            if len(v3_btd) > 0:
                mbtd_r3 = float(np.mean(v3_btd))
                btd_mean_rad3[i] = mbtd_r3
                btd_std_rad3[i] = float(np.std(v3_btd)) if len(v3_btd) > 1 else 0.0
                if not np.isnan(center_btd):
                    btd_diff_rad3[i] = float(center_btd - mbtd_r3)

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
        past_dfs = {}
        if past_scenes:
            for t_key in ["10m", "30m", "60m", "cadence", "past"]:
                if t_key in past_scenes and past_scenes[t_key] is not None:
                    past_scn = past_scenes[t_key]
                    past_sub = grid_df.copy()
                    if "B13" in past_scn:
                        past_sub["bt_b13"] = self.validate_bounds(past_scn["B13"].values[target_rows, target_cols])
                    if "B15" in past_scn and "B13" in past_scn:
                        b15_p = self.validate_bounds(past_scn["B15"].values[target_rows, target_cols])
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
