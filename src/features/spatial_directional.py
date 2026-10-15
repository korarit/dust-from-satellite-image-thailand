"""
Directional Atmospheric Pressure & Surrounding Weather Feature Engine
Implements:
- Strict Proximity Exclusion: Rejects any station with distance < exclusion_radius_km
  (default 2.0 km for 2x2 km grid) to prevent label/spatial leakage.
- KDTree Spatial Indexing for fast neighbor searches across Thailand stations.
- 4 Cardinal Quadrants: North (N), South (S), East (E), West (W).
- Inverse Distance Weighting (IDW) interpolation per quadrant.
- Atmospheric Gradients:
    * Delta P_ns (North - South pressure differential)
    * Delta P_ew (East - West pressure differential)
    * ||nabla P|| (Pressure gradient magnitude / Stagnation index, hPa / 100km)
    * ||nabla RH|| (Humidity gradient magnitude, % / 100km)
- Surrounding Mean & Standard Deviation for Pressure and Relative Humidity.
- Multi-resolution support (3km, 2km, 1km, 500m) with adaptive exclusion radii.
- Zero-Mock Rule: 100% deterministic physical equations, no synthetic or random data.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import pyproj

logger = logging.getLogger(__name__)

# Multi-resolution exclusion defaults (km)
RESOLUTION_EXCLUSION_MAP = {
    3.0: 3.0,
    2.0: 2.0,
    1.0: 1.0,
    0.5: 0.5,
}

DEFAULT_EXCLUSION_KM = 2.0
MAX_SEARCH_RADIUS_KM = 50.0
MIN_VALID_QUADRANT_STATIONS = 1


class DirectionalWeatherEngine:
    """
    Computes 4-directional atmospheric pressure and surrounding weather features
    relative to target grid cells with strict proximity exclusion.
    """

    def __init__(
        self,
        stations_df: pd.DataFrame,
        default_exclusion_km: float = DEFAULT_EXCLUSION_KM,
        max_search_radius_km: float = MAX_SEARCH_RADIUS_KM,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        code_col: str = "station_code",
    ):
        """
        Initializes spatial engine from station master metadata.
        stations_df must contain station_code, latitude, longitude.
        """
        self.default_exclusion_km = float(default_exclusion_km)
        self.max_search_radius_km = float(max_search_radius_km)
        self.to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)

        clean_stn = stations_df.dropna(subset=[lat_col, lon_col]).copy()
        if clean_stn.empty:
            raise ValueError("Stations dataframe contains no valid coordinates")

        lons = clean_stn[lon_col].to_numpy(dtype=float)
        lats = clean_stn[lat_col].to_numpy(dtype=float)

        utm_x, utm_y = self.to_utm.transform(lons, lats)
        clean_stn["utm_x_km"] = utm_x / 1000.0
        clean_stn["utm_y_km"] = utm_y / 1000.0

        self.stations_df = clean_stn.reset_index(drop=True)
        self.code_col = code_col
        self.coords_km = np.column_stack([self.stations_df["utm_x_km"], self.stations_df["utm_y_km"]])
        self.kdtree = cKDTree(self.coords_km)
        self.station_codes = self.stations_df[code_col].tolist()
        self.code_to_idx = {c: i for i, c in enumerate(self.station_codes)}

        logger.info(
            "Initialized DirectionalWeatherEngine with %d stations. Default exclusion=%.1f km, max radius=%.1f km",
            len(self.stations_df), self.default_exclusion_km, self.max_search_radius_km
        )

    def extract_cell_directional_features(
        self,
        target_lat: float,
        target_lon: float,
        obs_dict: Dict[str, float],
        variable: str = "pressure",
        exclusion_radius_km: Optional[float] = None,
        idw_power: float = 1.0,
        cell_elevation_m: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes 4-directional and gradient features for a single target point.
        obs_dict: mapping of station_code -> observed value (e.g. pressure in hPa or humidity in %).
        cell_elevation_m: elevation of the target cell in meters (used in Barometric Radial Gradient).
        """
        exclusion_km = float(exclusion_radius_km if exclusion_radius_km is not None else self.default_exclusion_km)

        # Convert target coordinate to UTM km
        x_m, y_m = self.to_utm.transform(target_lon, target_lat)
        t_x = x_m / 1000.0
        t_y = y_m / 1000.0

        # Query stations within max radius
        nearby_indices = self.kdtree.query_ball_point([t_x, t_y], r=self.max_search_radius_km)
        if not nearby_indices:
            return self._empty_directional_dict(variable)

        stn_coords = self.coords_km[nearby_indices]
        offsets = stn_coords - np.array([t_x, t_y])
        dists = np.hypot(offsets[:, 0], offsets[:, 1])

        # STRICT PROXIMITY EXCLUSION: drop all stations within exclusion buffer
        keep_mask = dists >= exclusion_km
        if not np.any(keep_mask):
            return self._empty_directional_dict(variable)

        valid_indices = [nearby_indices[i] for i in range(len(dists)) if keep_mask[i]]
        valid_offsets = offsets[keep_mask]
        valid_dists = dists[keep_mask]

        # Filter to stations with non-NaN observations
        values = []
        filtered_offsets = []
        filtered_dists = []

        for idx, off, d in zip(valid_indices, valid_offsets, valid_dists):
            code = self.station_codes[idx]
            v = obs_dict.get(code, np.nan)
            if v is not None and not np.isnan(v):
                values.append(float(v))
                filtered_offsets.append(off)
                filtered_dists.append(d)

        if not values:
            return self._empty_directional_dict(variable)

        val_arr = np.array(values, dtype=np.float32)
        dist_arr = np.array(filtered_dists, dtype=np.float32)
        off_arr = np.array(filtered_offsets, dtype=np.float32)

        # 4 Cardinal Quadrant classification:
        # dx = x - x0, dy = y - y0
        dx = off_arr[:, 0]
        dy = off_arr[:, 1]
        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)

        # North: dy > 0 and |dy| >= |dx|
        mask_n = (dy > 0.0) & (abs_dy >= abs_dx)
        # South: dy < 0 and |dy| >= |dx|
        mask_s = (dy < 0.0) & (abs_dy >= abs_dx)
        # East: dx > 0 and |dx| > |dy|
        mask_e = (dx > 0.0) & (abs_dx > abs_dy)
        # West: dx < 0 and |dx| > |dy|
        mask_w = (dx < 0.0) & (abs_dx > abs_dy)

        # Inverse Distance Weighting helper
        weights = 1.0 / np.maximum(dist_arr, 0.1) ** idw_power
        overall_mean = float(np.sum(weights * val_arr) / np.sum(weights))

        def _compute_quadrant(mask: np.ndarray) -> Tuple[float, float, float]:
            """Returns (quadrant_weighted_val, weighted_dx, weighted_dy)."""
            if np.any(mask):
                w_sub = weights[mask]
                v_sub = val_arr[mask]
                dx_sub = dx[mask]
                dy_sub = dy[mask]
                w_sum = np.sum(w_sub)
                q_val = float(np.sum(w_sub * v_sub) / w_sum)
                q_dx = float(np.sum(w_sub * dx_sub) / w_sum)
                q_dy = float(np.sum(w_sub * dy_sub) / w_sum)
                return q_val, q_dx, q_dy
            return overall_mean, 0.0, 0.0

        p_n, dx_n, dy_n = _compute_quadrant(mask_n)
        p_s, dx_s, dy_s = _compute_quadrant(mask_s)
        p_e, dx_e, dy_e = _compute_quadrant(mask_e)
        p_w, dx_w, dy_w = _compute_quadrant(mask_w)

        has_n = bool(np.any(mask_n))
        has_s = bool(np.any(mask_s))
        has_e = bool(np.any(mask_e))
        has_w = bool(np.any(mask_w))
        valid_quadrants = sum([has_n, has_s, has_e, has_w])

        # ISA Barometric Reference and Elevation Scale Factor
        # P_ref(z) = P0 * (1 - L * z / T0)^5.25588
        z_m = float(cell_elevation_m) if (cell_elevation_m is not None and not np.isnan(cell_elevation_m)) else 0.0
        z_clamped = max(z_m, 0.0)
        f_baro = (1.0 - (0.0065 * z_clamped) / 288.15) ** 5.25588
        p_ref_cell = 1013.25 * f_baro

        if valid_quadrants == 1 and variable == "pressure":
            # Approach B: Barometric Radial Gradient relative to grid elevation
            # When observations are available in only 1 quadrant, compute radial gradient
            # against the standard barometric elevation reference rather than yielding NaN/0.0
            if has_n:
                eff_dist = max(float(np.hypot(dx_n, dy_n)), exclusion_km, 5.0)
                delta_ns = (p_n - 1013.25) * f_baro if p_n >= 980.0 else (p_n - p_ref_cell)
                grad_y = delta_ns / eff_dist
                grad_x = 0.0
                delta_ew = 0.0
                p_s = p_n - delta_ns
            elif has_s:
                eff_dist = max(float(np.hypot(dx_s, dy_s)), exclusion_km, 5.0)
                delta_ns = (1013.25 - p_s) * f_baro if p_s >= 980.0 else (p_ref_cell - p_s)
                grad_y = delta_ns / eff_dist
                grad_x = 0.0
                delta_ew = 0.0
                p_n = p_s + delta_ns
            elif has_e:
                eff_dist = max(float(np.hypot(dx_e, dy_e)), exclusion_km, 5.0)
                delta_ew = (p_e - 1013.25) * f_baro if p_e >= 980.0 else (p_e - p_ref_cell)
                grad_x = delta_ew / eff_dist
                grad_y = 0.0
                delta_ns = 0.0
                p_w = p_e - delta_ew
            elif has_w:
                eff_dist = max(float(np.hypot(dx_w, dy_w)), exclusion_km, 5.0)
                delta_ew = (1013.25 - p_w) * f_baro if p_w >= 980.0 else (p_ref_cell - p_w)
                grad_x = delta_ew / eff_dist
                grad_y = 0.0
                delta_ns = 0.0
                p_e = p_w + delta_ew
            else:
                delta_ns = 0.0
                delta_ew = 0.0
                grad_x = 0.0
                grad_y = 0.0

            raw_grad_mag = float(np.hypot(grad_x, grad_y) * 100.0)
            grad_mag = float(np.clip(raw_grad_mag, 0.2, 5.0))

        elif valid_quadrants >= 2:
            # Multi-directional gradient with 2, 3, or 4 quadrants
            # Compute North-South gradient component
            if has_n and has_s:
                delta_ns = p_n - p_s
                eff_dy = max(abs(dy_n - dy_s), 2.0 * exclusion_km, 10.0)
                grad_y = delta_ns / eff_dy
            elif has_n:
                delta_ns = p_n - overall_mean
                eff_dy = max(abs(dy_n), exclusion_km, 5.0)
                grad_y = delta_ns / eff_dy
            elif has_s:
                delta_ns = overall_mean - p_s
                eff_dy = max(abs(dy_s), exclusion_km, 5.0)
                grad_y = delta_ns / eff_dy
            else:
                delta_ns = 0.0
                grad_y = 0.0

            # Compute East-West gradient component
            if has_e and has_w:
                delta_ew = p_e - p_w
                eff_dx = max(abs(dx_e - dx_w), 2.0 * exclusion_km, 10.0)
                grad_x = delta_ew / eff_dx
            elif has_e:
                delta_ew = p_e - overall_mean
                eff_dx = max(abs(dx_e), exclusion_km, 5.0)
                grad_x = delta_ew / eff_dx
            elif has_w:
                delta_ew = overall_mean - p_w
                eff_dx = max(abs(dx_w), exclusion_km, 5.0)
                grad_x = delta_ew / eff_dx
            else:
                delta_ew = 0.0
                grad_x = 0.0

            # P1-1 fix: clip multi-quadrant grad identically to single-quadrant
            # ([0.2, 5.0] hPa/100km). Unclipped IDW noise previously leaked through.
            raw_grad_mag = float(np.hypot(grad_x, grad_y) * 100.0)
            if variable == "pressure":
                # Allow true 0 (uniform stagnant field) but cap IDW noise at 5.0.
                # (Single-quadrant barometric estimate keeps its [0.2, 5.0] floor
                # because 0 there would mean "no gradient measured", not calm.)
                grad_mag = float(np.clip(raw_grad_mag, 0.0, 5.0))
            else:
                grad_mag = float(max(raw_grad_mag, 0.0))
        elif valid_quadrants == 1 and variable == "humidity":
            # P1-1 fix: single-quadrant humidity previously fell through to NaN.
            # One quadrant still constrains the local mean; gradient unknown -> 0.
            delta_ns = 0.0
            delta_ew = 0.0
            grad_x = 0.0
            grad_y = 0.0
            grad_mag = 0.0
        else:
            delta_ns = 0.0
            delta_ew = 0.0
            grad_x = 0.0
            grad_y = 0.0
            grad_mag = np.nan

        # Overall surrounding statistics
        val_mean = overall_mean
        val_std = float(np.std(val_arr)) if len(val_arr) > 1 else 0.0

        if variable == "pressure":
            return {
                "p_north": np.float32(p_n),
                "p_south": np.float32(p_s),
                "p_east": np.float32(p_e),
                "p_west": np.float32(p_w),
                "delta_p_ns": np.float32(delta_ns),
                "delta_p_ew": np.float32(delta_ew),
                "pressure_grad_mag": np.float32(grad_mag),
                "pressure_mean_surround": np.float32(val_mean),
            }
        elif variable == "humidity":
            return {
                "humidity_mean_surround": np.float32(val_mean),
                "humidity_std_surround": np.float32(val_std),
                "humidity_grad_mag": np.float32(grad_mag),
            }
        else:
            return {
                f"{variable}_north": np.float32(p_n),
                f"{variable}_south": np.float32(p_s),
                f"{variable}_east": np.float32(p_e),
                f"{variable}_west": np.float32(p_w),
                f"delta_{variable}_ns": np.float32(delta_ns),
                f"delta_{variable}_ew": np.float32(delta_ew),
                f"{variable}_grad_mag": np.float32(grad_mag),
                f"{variable}_mean_surround": np.float32(val_mean),
            }

    def _empty_directional_dict(self, variable: str) -> Dict[str, float]:
        """Provides neutral NaN or zero default dictionary when no stations exist."""
        if variable == "pressure":
            return {
                "p_north": np.nan,
                "p_south": np.nan,
                "p_east": np.nan,
                "p_west": np.nan,
                "delta_p_ns": 0.0,
                "delta_p_ew": 0.0,
                "pressure_grad_mag": np.nan,
                "pressure_mean_surround": np.nan,
            }
        elif variable == "humidity":
            return {
                "humidity_mean_surround": np.nan,
                "humidity_std_surround": 0.0,
                "humidity_grad_mag": 0.0,
            }
        else:
            return {
                f"{variable}_north": np.nan,
                f"{variable}_south": np.nan,
                f"{variable}_east": np.nan,
                f"{variable}_west": np.nan,
                f"delta_{variable}_ns": 0.0,
                f"delta_{variable}_ew": 0.0,
                f"{variable}_grad_mag": 0.0,
                f"{variable}_mean_surround": np.nan,
            }

    def extract_grid_weather(
        self,
        grid_df: pd.DataFrame,
        pressure_obs: Optional[Dict[str, float]] = None,
        humidity_obs: Optional[Dict[str, float]] = None,
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon",
        exclusion_radius_km: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Vectorized batch feature extractor for all cells in grid_df.
        Extracts:
        - 8 Directional Pressure features (Group 5)
        - 3 Surrounding Humidity features (Group 6)
        """
        res_df = grid_df.copy()
        lats = grid_df[lat_col].to_numpy()
        lons = grid_df[lon_col].to_numpy()
        n_cells = len(grid_df)

        elev_col = None
        for cand in ["elevation_m", "dem_elevation", "elevation"]:
            if cand in grid_df.columns:
                elev_col = cand
                break
        elevs = grid_df[elev_col].to_numpy() if elev_col else None

        if pressure_obs:
            p_n = np.zeros(n_cells, dtype=np.float32)
            p_s = np.zeros(n_cells, dtype=np.float32)
            p_e = np.zeros(n_cells, dtype=np.float32)
            p_w = np.zeros(n_cells, dtype=np.float32)
            dp_ns = np.zeros(n_cells, dtype=np.float32)
            dp_ew = np.zeros(n_cells, dtype=np.float32)
            p_grad = np.zeros(n_cells, dtype=np.float32)
            p_mean = np.zeros(n_cells, dtype=np.float32)

            for i in range(n_cells):
                cell_elev = float(elevs[i]) if (elevs is not None and not np.isnan(elevs[i])) else None
                feat = self.extract_cell_directional_features(
                    lats[i], lons[i], pressure_obs, variable="pressure",
                    exclusion_radius_km=exclusion_radius_km,
                    cell_elevation_m=cell_elev,
                )
                p_n[i] = feat["p_north"]
                p_s[i] = feat["p_south"]
                p_e[i] = feat["p_east"]
                p_w[i] = feat["p_west"]
                dp_ns[i] = feat["delta_p_ns"]
                dp_ew[i] = feat["delta_p_ew"]
                p_grad[i] = feat["pressure_grad_mag"]
                p_mean[i] = feat["pressure_mean_surround"]

            res_df["p_north"] = p_n
            res_df["p_south"] = p_s
            res_df["p_east"] = p_e
            res_df["p_west"] = p_w
            res_df["delta_p_ns"] = dp_ns
            res_df["delta_p_ew"] = dp_ew
            res_df["pressure_grad_mag"] = p_grad
            res_df["pressure_mean_surround"] = p_mean

        if humidity_obs:
            h_mean = np.zeros(n_cells, dtype=np.float32)
            h_std = np.zeros(n_cells, dtype=np.float32)
            h_grad = np.zeros(n_cells, dtype=np.float32)

            for i in range(n_cells):
                cell_elev = float(elevs[i]) if (elevs is not None and not np.isnan(elevs[i])) else None
                feat = self.extract_cell_directional_features(
                    lats[i], lons[i], humidity_obs, variable="humidity",
                    exclusion_radius_km=exclusion_radius_km,
                    cell_elevation_m=cell_elev,
                )
                h_mean[i] = feat["humidity_mean_surround"]
                h_std[i] = feat["humidity_std_surround"]
                h_grad[i] = feat["humidity_grad_mag"]

            res_df["humidity_mean_surround"] = h_mean
            res_df["humidity_std_surround"] = h_std
            res_df["humidity_grad_mag"] = h_grad

        return res_df
