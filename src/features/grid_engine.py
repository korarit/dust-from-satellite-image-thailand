"""
Multi-Resolution Spatial Grid & Topographic Feature Engine
Implements:
- 4 Resolution Grids: 3 km (3000m), 2 km (2000m), 1 km (1000m), 500m (500m)
- Resolution-dependent Proximity Exclusion Buffer:
    * 3 km  -> d < 3.0 km
    * 2 km  -> d < 2.0 km (Master Baseline)
    * 1 km  -> d < 1.0 km
    * 500 m -> d < 0.5 km
- Topographic Feature Extraction from Copernicus DEM 30m & Master Grid:
    * elevation_m (Zonal mean elevation)
    * elevation_std (Terrain roughness from 30m DEM pixels)
    * relief_m (Topographic relief: max - min in cell)
    * slope_deg (Terrain slope in degrees from 30m spatial gradient)
    * aspect_sin, aspect_cos (Vector decomposition of aspect direction)
    * tpi_index (Topographic Position Index / Inversion Basin Index: elev - mean(elev_6km))
    * dist_coast_km (Distance to nearest coastline)
    * coriolis_param (Coriolis parameter 2*Omega*sin(lat))
- Astronomical Solar Position Engine:
    * solar_zenith_angle (NOAA Solar Model)
    * is_daytime (SZA < 85 degrees)
    * hour_sin, hour_cos (24h cyclical time)
    * month_sin, month_cos (12m cyclical seasonality)
- Strict Zero-Mock: 100% deterministic physics, zero random numbers.
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree

try:
    import rasterio
    import rasterio.windows
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from src.config.paths import GEO_DIR

logger = logging.getLogger(__name__)

OMEGA_EARTH = 7.2921159e-5  # rad/s

# 4 Supported Benchmark Resolutions
SUPPORTED_RESOLUTIONS_KM = [3.0, 2.0, 1.0, 0.5]
RESOLUTION_STEP_MAP = {
    3.0: 3000.0,
    2.0: 2000.0,
    1.0: 1000.0,
    0.5: 500.0,
}
EXCLUSION_BUFFER_MAP = {
    3.0: 3.0,
    2.0: 2.0,
    1.0: 1.0,
    0.5: 0.5,
}

# DEM search paths
COMMON_DEM_DIRS = [
    GEO_DIR / "dem",
    Path("e:/water-analysis-project/weather-forcast-enhance/data/geo/dem"),
    Path("data/geo/dem"),
]


_COASTLINE_TREE: Optional[cKDTree] = None
_COASTLINE_TRANSFORMER: Optional[pyproj.Transformer] = None

# Comprehensive anchor chains along the entire Thailand coastline (Gulf of Thailand & Andaman Sea)
THAILAND_COASTLINE_CHAINS = [
    # 1. Eastern Gulf: Trat -> Chanthaburi -> Rayong -> Sattahip -> Chonburi
    [
        (11.66, 102.89), (11.83, 102.80), (12.08, 102.58), (12.44, 102.32),
        (12.55, 102.08), (12.63, 101.78), (12.66, 101.40), (12.67, 101.18),
        (12.60, 100.90), (12.80, 100.89), (13.10, 100.90), (13.36, 100.95),
    ],
    # 2. Upper Gulf (Bight of Bangkok): Chonburi -> Samut Prakan -> BKK -> Samut Sakhon -> Samut Songkhram -> Phetchaburi
    [
        (13.48, 100.80), (13.54, 100.60), (13.51, 100.35), (13.40, 100.02),
        (13.15, 100.02), (12.80, 100.00),
    ],
    # 3. Western & Southern Gulf: Hua Hin -> Prachuap -> Chumphon -> Surat Thani -> Nakhon Si Thammarat -> Songkhla -> Pattani -> Narathiwat
    [
        (12.57, 99.96), (12.25, 100.00), (11.80, 99.80), (11.20, 99.50),
        (10.85, 99.45), (10.50, 99.20), (9.80, 99.15), (9.20, 99.35),
        (9.15, 99.65), (9.15, 99.90), (8.70, 100.15), (8.25, 100.30),
        (7.85, 100.35), (7.50, 100.45), (7.20, 100.60), (6.90, 101.25),
        (6.85, 101.55), (6.45, 101.85), (6.22, 102.10),
    ],
    # 4. Andaman Coast: Ranong -> Phang Nga -> Phuket -> Krabi -> Trang -> Satun
    [
        (10.00, 98.60), (9.50, 98.45), (9.00, 98.30), (8.50, 98.25),
        (8.20, 98.30), (7.80, 98.30), (8.00, 98.60), (8.05, 98.90),
        (7.70, 99.05), (7.35, 99.30), (7.05, 99.70), (6.70, 99.95),
        (6.45, 100.10),
    ],
]


def _get_coastline_kdtree() -> Tuple[cKDTree, pyproj.Transformer]:
    """Builds and caches a high-density 5km-interpolated KDTree of Thailand's coastline in UTM km."""
    global _COASTLINE_TREE, _COASTLINE_TRANSFORMER
    if _COASTLINE_TREE is not None and _COASTLINE_TRANSFORMER is not None:
        return _COASTLINE_TREE, _COASTLINE_TRANSFORMER

    _COASTLINE_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    points = []
    for chain in THAILAND_COASTLINE_CHAINS:
        for i in range(len(chain) - 1):
            lat1, lon1 = chain[i]
            lat2, lon2 = chain[i + 1]
            seg_dist = np.hypot((lat2 - lat1) * 111.0, (lon2 - lon1) * 105.0)
            n_steps = max(3, int(seg_dist / 5.0))
            for t in np.linspace(0.0, 1.0, n_steps):
                points.append((lat1 + t * (lat2 - lat1), lon1 + t * (lon2 - lon1)))

    pts_arr = np.array(points, dtype=np.float64)
    ux, uy = _COASTLINE_TRANSFORMER.transform(pts_arr[:, 1], pts_arr[:, 0])
    _COASTLINE_TREE = cKDTree(np.column_stack([ux / 1000.0, uy / 1000.0]))
    return _COASTLINE_TREE, _COASTLINE_TRANSFORMER


def compute_geographic_proxies(
    lats: np.ndarray,
    lons: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes distance to coast (km) and Coriolis parameter (s^-1) for Thailand coordinates.
    Uses high-resolution polyline KDTree of Thai Andaman and Gulf coastlines in UTM Zone 47N.
    """
    lat_arr = np.asarray(lats, dtype=np.float64)
    lon_arr = np.asarray(lons, dtype=np.float64)

    tree, transformer = _get_coastline_kdtree()
    ux, uy = transformer.transform(lon_arr, lat_arr)
    coords_km = np.column_stack([ux / 1000.0, uy / 1000.0])

    dists, _ = tree.query(coords_km)
    dist_coast_km = np.clip(dists, 0.1, 850.0).astype(np.float32)

    # Coriolis parameter: f = 2 * Omega * sin(lat_rad)
    lat_rad = np.radians(lat_arr)
    coriolis_param = (2.0 * OMEGA_EARTH * np.sin(lat_rad)).astype(np.float32)

    return dist_coast_km, coriolis_param


def compute_solar_zenith_angle(
    timestamp_utc: pd.Timestamp,
    latitude: float,
    longitude: float
) -> Tuple[float, int]:
    """
    Computes Solar Zenith Angle (degrees) and daytime flag using NOAA Solar Astronomical Model.
    Deterministic, physics-based, zero random.
    """
    ts = pd.to_datetime(timestamp_utc)
    day_of_year = ts.dayofyear
    hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

    # Fractional year in radians
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour - 12.0) / 24.0)

    # Equation of time in minutes
    eqtime = 229.18 * (
        0.000075 + 0.01868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma) - 0.040849 * math.sin(2.0 * gamma)
    )

    # Solar declination angle in radians
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma) + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma) + 0.00148 * math.sin(3.0 * gamma)
    )

    # True solar time in minutes
    time_offset = eqtime + 4.0 * longitude
    tst = hour * 60.0 + time_offset

    # Solar hour angle in degrees
    ha = (tst / 4.0) - 180.0
    ha_rad = math.radians(ha)

    lat_rad = math.radians(latitude)

    # Cosine of zenith angle
    cos_sza = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad)
    cos_sza = max(-1.0, min(1.0, cos_sza))
    sza_deg = math.degrees(math.acos(cos_sza))

    is_daytime = 1 if sza_deg < 85.0 else 0
    return float(sza_deg), is_daytime


def compute_solar_angles_vectorized(
    timestamps: pd.Series,
    lats: np.ndarray,
    lons: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized calculation of Solar Zenith Angles and daytime flags."""
    n = len(lats)
    sza_arr = np.zeros(n, dtype=np.float32)
    day_arr = np.zeros(n, dtype=np.int32)

    ts_list = pd.to_datetime(timestamps).tolist() if isinstance(timestamps, (pd.Series, list)) else [pd.to_datetime(timestamps)] * n

    for i in range(n):
        sza, day = compute_solar_zenith_angle(ts_list[i], float(lats[i]), float(lons[i]))
        sza_arr[i] = sza
        day_arr[i] = day

    return sza_arr, day_arr


def compute_cyclical_time_features(timestamps: Union[pd.Series, pd.DatetimeIndex, List[Any]]) -> pd.DataFrame:
    """Computes cyclical hour and month sin/cos representations."""
    dt_idx = pd.DatetimeIndex(timestamps)
    hours = dt_idx.hour + dt_idx.minute / 60.0
    months = dt_idx.month - 1  # 0 to 11

    hour_sin = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)
    month_sin = np.sin(2.0 * np.pi * months / 12.0).astype(np.float32)
    month_cos = np.cos(2.0 * np.pi * months / 12.0).astype(np.float32)

    return pd.DataFrame({
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    })


class MultiResolutionGridEngine:
    """
    Manages multi-resolution spatial grids (3km, 2km, 1km, 500m) and calculates
    Copernicus DEM 30m topography, TPI basin index, and solar angles.
    """

    def __init__(
        self,
        base_grid_path: Optional[Path] = None,
        dem_dirs: Optional[List[Path]] = None,
    ):
        self.base_grid_path = base_grid_path or (GEO_DIR / "thailand_2km_master_grid.parquet")
        self.dem_dirs = dem_dirs or COMMON_DEM_DIRS
        self.to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
        self.to_wgs = pyproj.Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)

        self._master_2km_df: Optional[pd.DataFrame] = None
        self._kdtree_2km: Optional[cKDTree] = None

    def load_master_grid_2km(self) -> pd.DataFrame:
        """Loads and caches Thailand 2km Master Grid."""
        if self._master_2km_df is None:
            if not self.base_grid_path.exists():
                raise FileNotFoundError(f"Master Grid 2km file not found at {self.base_grid_path}")
            df = pd.read_parquet(self.base_grid_path)
            self._master_2km_df = df
            coords = np.column_stack([df["centroid_lat"], df["centroid_lon"]])
            self._kdtree_2km = cKDTree(coords)
            logger.info("Loaded Thailand 2km Master Grid with %d cells", len(df))
        return self._master_2km_df

    def compute_tpi_index(
        self,
        grid_df: pd.DataFrame,
        radius_km: float = 6.0,
        elev_col: str = "elevation_m",
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon"
    ) -> np.ndarray:
        """
        Computes Topographic Position Index (TPI):
            TPI = elevation_m - Mean(elevation within radius_km)
        Negative values indicate valley basins / cold air pooling traps (PM2.5 accumulation).
        Positive values indicate ridge crests.
        """
        lats = grid_df[lat_col].to_numpy(dtype=float)
        lons = grid_df[lon_col].to_numpy(dtype=float)
        elevs = grid_df[elev_col].to_numpy(dtype=float)

        # Build KDTree in UTM km
        x_m, y_m = self.to_utm.transform(lons, lats)
        coords_km = np.column_stack([x_m / 1000.0, y_m / 1000.0])
        tree = cKDTree(coords_km)

        n = len(grid_df)
        tpi = np.zeros(n, dtype=np.float32)

        # Batch query within radius
        idx_lists = tree.query_ball_point(coords_km, r=radius_km)
        for i in range(n):
            nbrs = idx_lists[i]
            if len(nbrs) > 1:
                nbr_elevs = elevs[nbrs]
                valid = nbr_elevs[~np.isnan(nbr_elevs)]
                mean_surround = float(np.mean(valid)) if len(valid) > 0 else elevs[i]
            else:
                mean_surround = elevs[i]

            tpi[i] = float(elevs[i] - mean_surround)

        return tpi

    def compute_local_relief(
        self,
        grid_df: pd.DataFrame,
        radius_km: float = 2.0,
        elev_col: str = "elevation_m",
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon"
    ) -> np.ndarray:
        """
        Computes local topographic relief (max elevation - min elevation within cell radius).
        """
        lats = grid_df[lat_col].to_numpy(dtype=float)
        lons = grid_df[lon_col].to_numpy(dtype=float)
        elevs = grid_df[elev_col].to_numpy(dtype=float)

        x_m, y_m = self.to_utm.transform(lons, lats)
        coords_km = np.column_stack([x_m / 1000.0, y_m / 1000.0])
        tree = cKDTree(coords_km)

        n = len(grid_df)
        relief = np.zeros(n, dtype=np.float32)
        idx_lists = tree.query_ball_point(coords_km, r=radius_km)

        for i in range(n):
            nbrs = idx_lists[i]
            if len(nbrs) > 1:
                nbr_elevs = elevs[nbrs]
                valid = nbr_elevs[~np.isnan(nbr_elevs)]
                if len(valid) > 1:
                    relief[i] = float(np.max(valid) - np.min(valid))
                else:
                    relief[i] = 0.0
            else:
                relief[i] = 0.0

        return relief

    def get_grid_by_resolution(
        self,
        resolution_km: float = 2.0,
        subsample_step: Optional[int] = None,
        max_cells: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieves or generates spatial grid for the requested resolution:
        - 3.0 km: Coarse Synoptic Benchmark
        - 2.0 km: Native Himawari IR Baseline (from Master Grid)
        - 1.0 km: Topographic Downscaling Sweet Spot
        - 0.5 km: Ultra-High Micro Scale
        """
        if resolution_km not in SUPPORTED_RESOLUTIONS_KM:
            raise ValueError(f"Resolution {resolution_km} km not supported. Choose from {SUPPORTED_RESOLUTIONS_KM}")

        master_df = self.load_master_grid_2km()

        if resolution_km == 2.0:
            grid_df = master_df.copy()
        elif resolution_km == 3.0:
            # Subsample 2km grid with stride 2 or query regular grid
            grid_df = master_df.iloc[::2].copy().reset_index(drop=True)
        elif resolution_km == 1.0:
            # Subdivide 2km cells into 1km UTM sub-cells (4 sub-cells per 2km anchor: offsets [-500m, +500m])
            n_anchors = (max_cells // 4 + 1) if max_cells else len(master_df)
            base_subset = master_df.iloc[:n_anchors]
            offsets = [(-500.0, -500.0), (-500.0, 500.0), (500.0, -500.0), (500.0, 500.0)]
            records = []
            for _, row in base_subset.iterrows():
                ux, uy = row["utm_x"], row["utm_y"]
                gid = row["grid_id"]
                for si, (dx, dy) in enumerate(offsets):
                    r_dict = dict(row)
                    r_dict["grid_id"] = f"{gid}_1k_{si}"
                    r_dict["utm_x"] = ux + dx
                    r_dict["utm_y"] = uy + dy
                    records.append(r_dict)
            grid_df = pd.DataFrame(records)
            lons, lats = self.to_wgs.transform(grid_df["utm_x"].values, grid_df["utm_y"].values)
            grid_df["centroid_lon"] = lons.astype(np.float32)
            grid_df["centroid_lat"] = lats.astype(np.float32)
            grid_df["coriolis_param"] = (2.0 * 7.2921159e-5 * np.sin(np.radians(lats))).astype(np.float32)
        elif resolution_km == 0.5:
            # Subdivide 2km cells into 500m UTM sub-cells (16 sub-cells per 2km anchor: offsets [-750, -250, 250, 750]m)
            n_anchors = (max_cells // 16 + 1) if max_cells else len(master_df)
            base_subset = master_df.iloc[:n_anchors]
            coords_1d = [-750.0, -250.0, 250.0, 750.0]
            offsets = [(dx, dy) for dx in coords_1d for dy in coords_1d]
            records = []
            for _, row in base_subset.iterrows():
                ux, uy = row["utm_x"], row["utm_y"]
                gid = row["grid_id"]
                for si, (dx, dy) in enumerate(offsets):
                    r_dict = dict(row)
                    r_dict["grid_id"] = f"{gid}_500m_{si}"
                    r_dict["utm_x"] = ux + dx
                    r_dict["utm_y"] = uy + dy
                    records.append(r_dict)
            grid_df = pd.DataFrame(records)
            lons, lats = self.to_wgs.transform(grid_df["utm_x"].values, grid_df["utm_y"].values)
            grid_df["centroid_lon"] = lons.astype(np.float32)
            grid_df["centroid_lat"] = lats.astype(np.float32)
            grid_df["coriolis_param"] = (2.0 * 7.2921159e-5 * np.sin(np.radians(lats))).astype(np.float32)

        if subsample_step and subsample_step > 1:
            grid_df = grid_df.iloc[::subsample_step].copy().reset_index(drop=True)

        if max_cells and len(grid_df) > max_cells:
            grid_df = grid_df.iloc[:max_cells].copy().reset_index(drop=True)

        # Ensure dist_coast_km and coriolis_param are computed
        if "dist_coast_km" not in grid_df.columns:
            d_coast, coriolis = compute_geographic_proxies(grid_df["centroid_lat"].values, grid_df["centroid_lon"].values)
            grid_df["dist_coast_km"] = d_coast
            grid_df["coriolis_param"] = coriolis

        # Supplement with TPI index and local relief if missing
        if "tpi_index" not in grid_df.columns:
            grid_df["tpi_index"] = self.compute_tpi_index(grid_df)

        if "relief_m" not in grid_df.columns:
            grid_df["relief_m"] = self.compute_local_relief(grid_df)

        grid_df["resolution_km"] = np.float32(resolution_km)
        grid_df["exclusion_radius_km"] = np.float32(EXCLUSION_BUFFER_MAP[resolution_km])

        return grid_df

    def attach_solar_and_time_features(
        self,
        grid_df: pd.DataFrame,
        observation_dt: pd.Timestamp,
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon",
    ) -> pd.DataFrame:
        """
        Attaches Group 7 Solar Angle & Cyclical Time features to grid_df:
        - solar_zenith_angle
        - is_daytime
        - hour_sin, hour_cos
        - month_sin, month_cos
        """
        out_df = grid_df.copy()
        lats = out_df[lat_col].to_numpy(dtype=float)
        lons = out_df[lon_col].to_numpy(dtype=float)

        sza, is_day = compute_solar_angles_vectorized(observation_dt, lats, lons)
        out_df["solar_zenith_angle"] = sza
        out_df["is_daytime"] = is_day

        time_feats = compute_cyclical_time_features(pd.Series([observation_dt] * len(out_df)))
        for c in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            out_df[c] = time_feats[c].to_numpy()

        return out_df

    def extract_dem_zonal_stats(
        self,
        grid_df: pd.DataFrame,
        cell_size_m: float = 2000.0,
        dem_tile_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Extracts genuine 30m Copernicus DEM topography statistics for grid cells:
        - elevation_m: Mean elevation of 30m pixels in cell
        - elevation_std: Standard deviation of 30m pixels (Terrain Roughness / Drag)
        - relief_m: Max - Min elevation within cell
        - slope_deg: Average terrain slope from 30m gradient
        - aspect_sin, aspect_cos: Vector aspect of slope
        """
        out_df = grid_df.copy()
        if not HAS_RASTERIO:
            logger.warning("rasterio not available; skipping DEM GeoTIFF extraction.")
            return out_df

        tile_paths = []
        if dem_tile_path and dem_tile_path.exists():
            tile_paths.append(dem_tile_path)
        else:
            for d in self.dem_dirs:
                if d.exists():
                    tile_paths.extend(list(d.glob("*.tif")) + list(d.glob("*.tiff")))

        if not tile_paths:
            logger.debug("No DEM GeoTIFF files found in %s; keeping existing attributes.", self.dem_dirs)
            return out_df

        lat_col = "centroid_lat" if "centroid_lat" in out_df.columns else "lat"
        lon_col = "centroid_lon" if "centroid_lon" in out_df.columns else "lon"
        if lat_col not in out_df.columns or lon_col not in out_df.columns:
            return out_df

        half_px = max(1, int(round((cell_size_m / 2.0) / 30.0)))
        win_size = 2 * half_px + 1

        for t_path in tile_paths:
            try:
                with rasterio.open(t_path) as src:
                    bounds = src.bounds
                    mask = (
                        (out_df[lon_col] >= bounds.left) & (out_df[lon_col] <= bounds.right) &
                        (out_df[lat_col] >= bounds.bottom) & (out_df[lat_col] <= bounds.top)
                    )
                    matching_indices = out_df.index[mask].tolist()
                    if not matching_indices:
                        continue

                    logger.info("Extracting Copernicus DEM 30m stats for %d cells from %s", len(matching_indices), t_path.name)
                    for idx in matching_indices:
                        clat = float(out_df.loc[idx, lat_col])
                        clon = float(out_df.loc[idx, lon_col])
                        r, c = src.index(clon, clat)
                        if (r < half_px or r >= src.height - half_px or
                            c < half_px or c >= src.width - half_px):
                            continue

                        win = rasterio.windows.Window(c - half_px, r - half_px, win_size, win_size)
                        data = src.read(1, window=win).astype(np.float64)
                        valid = data[data > -9000]
                        if len(valid) < 4:
                            continue

                        out_df.loc[idx, "elevation_m"] = np.float32(np.mean(valid))
                        out_df.loc[idx, "elevation_std"] = np.float32(np.std(valid))
                        out_df.loc[idx, "relief_m"] = np.float32(np.max(valid) - np.min(valid))

                        # Slope & Aspect from 30m window
                        if data.shape[0] >= 3 and data.shape[1] >= 3:
                            dz_dx = (data[:, 2:] - data[:, :-2]) / 60.0
                            dz_dy = (data[:-2, :] - data[2:, :]) / 60.0
                            slp = np.degrees(np.arctan(np.hypot(dz_dx[1:-1, :], dz_dy[:, 1:-1])))
                            asp = np.radians(np.degrees(np.arctan2(dz_dy[:, 1:-1], -dz_dx[1:-1, :])))
                            out_df.loc[idx, "slope_deg"] = np.float32(np.nanmean(slp))
                            out_df.loc[idx, "aspect_sin"] = np.float32(np.nanmean(np.sin(asp)))
                            out_df.loc[idx, "aspect_cos"] = np.float32(np.nanmean(np.cos(asp)))
            except Exception as e:
                logger.warning("Error processing DEM tile %s: %s", t_path, e)

        return out_df
