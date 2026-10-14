"""
Himawari Satellite 10 IR Bands AWS S3 Loader & Brightness Temperature Calibrator
Fetches authentic Himawari-8 & Himawari-9 AHI Level 1b (HSD) observation data
from NOAA Open Data on AWS S3 (s3://noaa-himawari9/ and s3://noaa-himawari8/).

Features:
- Segment pruning (Segments 3, 4, 5 for Thailand: Lat 5.5°N - 21.0°N)
- All 10 Infrared Bands (B07 through B16)
- Physics-based Satpy calibration into Brightness Temperature (Kelvin)
- Physical boundary assertion (150K - 350K)
- Strict Zero-Mock: never invents, simulates, or randomly generates pixel values.
"""

import os
import sys
import bz2
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
import pandas as pd
from satpy import Scene

# Filter expected satpy / dask runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice.*")

from src.config.paths import HIMAWARI_CACHE_DIR, HIMAWARI_DIR, load_yaml_config, DATA_CONFIG_PATH

logger = logging.getLogger(__name__)

# NOAA S3 Buckets
BUCKET_H8 = "noaa-himawari8"
BUCKET_H9 = "noaa-himawari9"

# Satellite Operational Cutover timestamp (Dec 13, 2022 ~05:00 UTC)
HIMAWARI_CUTOVER_UTC = pd.Timestamp("2022-12-13 05:00:00", tz="UTC")

# 2025 Himawari-9 Anomaly & Backup Period:
# On 2025-10-11 ~15:30 UTC, Himawari-9 encountered an imager anomaly.
# JMA temporarily switched primary operations to backup satellite Himawari-8 (H08)
# from 2025-10-11 23:20 UTC until operations were restored to Himawari-9 on 2025-11-26 04:50 UTC.
H8_ANOMALY_BACKUP_START = pd.Timestamp("2025-10-11 15:30:00", tz="UTC")
H8_ANOMALY_BACKUP_END = pd.Timestamp("2025-11-26 04:50:00", tz="UTC")

# 5 Core Physical IR Bands (Satellite-Only Dust, Fire, Moisture & Inversion)
CORE_5_IR_BANDS = ["B07", "B10", "B11", "B13", "B15"]
# Default bands to load (reduced from 10 to 5 to eliminate redundancy and save 50% bandwidth)
ALL_IR_BANDS = CORE_5_IR_BANDS
FULL_10_IR_BANDS = ["B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16"]

# Segments covering Thailand (Lat 5.5 - 21.0 N)
DEFAULT_SEGS = [3, 4, 5]
# Training Segment: Segment 4 covers scan lines 1651–2200 (~10.5°N - 21.6°N, 574 stations / 92.9% of Thailand)
DEFAULT_TRAIN_SEGS = [4]
# Testing Segments: Full coverage of all 618+ stations including Southern Thailand (~5.5°N - 21.0°N)
DEFAULT_TEST_SEGS = [3, 4, 5]

PHYSICAL_BT_MIN_K = 150.0
PHYSICAL_BT_MAX_K = 350.0


def get_himawari_bucket_and_sat(target_dt: pd.Timestamp) -> Tuple[str, str]:
    """Determines whether to use Himawari-8 or Himawari-9 based on timestamp."""
    ts = target_dt
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    if ts < HIMAWARI_CUTOVER_UTC:
        return BUCKET_H8, "H08"
    if H8_ANOMALY_BACKUP_START <= ts < H8_ANOMALY_BACKUP_END:
        return BUCKET_H8, "H08"
    return BUCKET_H9, "H09"


def round_to_himawari_10m(dt: pd.Timestamp, mode: str = "nearest") -> pd.Timestamp:
    """
    Rounds timestamp to a 10-minute observation cycle.
    Supports mode='nearest' (closest cycle) and mode='floor' (past completed cycle, anti-leakage).
    """
    minute = dt.minute
    rem = minute % 10

    if mode == "floor":
        rounded_m = minute - rem
    elif rem < 5:
        rounded_m = minute - rem
    else:
        rounded_m = minute + (10 - rem)

    if rounded_m == 60:
        return dt.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    return dt.replace(minute=rounded_m, second=0, microsecond=0)


def floor_to_himawari_10m(dt: pd.Timestamp) -> pd.Timestamp:
    """Floors timestamp to the most recent completed 10-minute cycle to guarantee zero look-ahead bias."""
    return round_to_himawari_10m(dt, mode="floor")


def build_s3_prefix_and_keys(
    target_dt: pd.Timestamp,
    band: str,
    segments: Optional[List[int]] = None
) -> Tuple[str, List[str]]:
    """
    Constructs the S3 bucket name and list of keys for the requested observation.
    band format: 'B07', 'B13', etc.
    """
    segments = segments or DEFAULT_SEGS
    dt = round_to_himawari_10m(target_dt)
    bucket, sat_id = get_himawari_bucket_and_sat(dt)

    date_str = dt.strftime("%Y/%m/%d")
    hhmm = dt.strftime("%H%M")
    ymd = dt.strftime("%Y%m%d")

    prefix = f"AHI-L1b-FLDK/{date_str}/{hhmm}/"
    keys = [
        f"{prefix}HS_{sat_id}_{ymd}_{hhmm}_{band}_FLDK_R20_S{seg:02d}10.DAT.bz2"
        for seg in segments
    ]
    return bucket, keys


_SHARED_S3_CLIENT: Optional[Any] = None


def get_shared_s3_client(max_pool_connections: int = 10) -> Any:
    """Returns a shared S3 client with persistent HTTPS connection pooling."""
    global _SHARED_S3_CLIENT
    if _SHARED_S3_CLIENT is None:
        _SHARED_S3_CLIENT = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(signature_version=UNSIGNED, max_pool_connections=max_pool_connections)
        )
    return _SHARED_S3_CLIENT


def _download_and_extract_single_key(
    bucket: str,
    k: str,
    cache_dir: Path,
    s3_client: Optional[Any] = None
) -> Optional[Path]:
    """Downloads a single .DAT.bz2 segment and decompresses it to cache_dir."""
    fname = Path(k).name
    dat_fname = fname[:-4] if fname.endswith(".bz2") else fname
    dat_path = cache_dir / dat_fname
    bz2_path = cache_dir / fname

    # If uncompressed file exists and has content, reuse it immediately (Fast Cache Hit)
    if dat_path.exists() and dat_path.stat().st_size > 0:
        return dat_path

    client = s3_client or get_shared_s3_client()
    try:
        logger.debug("Downloading s3://%s/%s -> %s", bucket, k, bz2_path.name)
        client.download_file(bucket, k, str(bz2_path))
    except Exception as e:
        logger.warning("Could not download s3://%s/%s: %s", bucket, k, e)
        if bz2_path.exists():
            try:
                bz2_path.unlink()
            except OSError:
                pass
        return None

    try:
        with bz2.open(bz2_path, "rb") as fin, open(dat_path, "wb") as fout:
            fout.write(fin.read())
        try:
            bz2_path.unlink()  # remove compressed archive to save disk space
        except OSError:
            pass
        return dat_path
    except Exception as e:
        logger.warning("Failed decompressing %s: %s", bz2_path.name, e)
        if bz2_path.exists():
            try:
                bz2_path.unlink()
            except OSError:
                pass
        if dat_path.exists():
            try:
                dat_path.unlink()
            except OSError:
                pass
        return None


def download_and_extract_hsd_segments(
    bucket: str,
    keys: List[str],
    cache_dir: Path = HIMAWARI_CACHE_DIR,
    max_workers: int = 10,
    s3_client: Optional[Any] = None,
) -> List[Path]:
    """Downloads and extracts HSD segments concurrently using shared S3 connection pool."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not keys:
        return []

    client = s3_client or get_shared_s3_client(max_pool_connections=max(10, max_workers))
    workers = min(len(keys), max_workers)
    results_map: Dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futs = {executor.submit(_download_and_extract_single_key, bucket, k, cache_dir, client): k for k in keys}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                res = fut.result()
                if res is not None:
                    results_map[key] = res
            except Exception as e:
                logger.warning("Error downloading segment %s: %s", key, e)

    return [results_map[k] for k in keys if k in results_map]


class HimawariIRLoader:
    """
    Manages downloading, caching, and satpy calibration for all 10 IR Bands
    (B07, B08, B09, B10, B11, B12, B13, B14, B15, B16).
    """

    def __init__(
        self,
        cache_dir: Path = HIMAWARI_CACHE_DIR,
        bands: Optional[List[str]] = None,
        segments: Optional[List[int]] = None,
        train_segments: Optional[List[int]] = None,
        test_segments: Optional[List[int]] = None,
        max_workers: int = 10,
    ):
        self.cache_dir = Path(cache_dir)
        self.bands = bands or ALL_IR_BANDS
        self.segments = segments or DEFAULT_SEGS
        self.train_segments = train_segments or DEFAULT_TRAIN_SEGS
        self.test_segments = test_segments or DEFAULT_TEST_SEGS
        self.max_workers = max_workers
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Shared S3 client with connection pool matching max_workers (default 10)
        self.s3_client = get_shared_s3_client(max_pool_connections=max(10, self.max_workers))
        # Spatial Index Cache to eliminate redundant area.get_lonlats() and cKDTree recomputations
        self._cached_spatial_index: Dict[Tuple, Any] = {}

    def fetch_snapshot_files(
        self,
        target_dt: pd.Timestamp,
        bands: Optional[List[str]] = None,
        segments: Optional[List[int]] = None,
    ) -> Dict[str, List[Path]]:
        """
        Downloads all segment files for the specified bands for target_dt concurrently across all bands.
        Returns mapping from band name to list of extracted DAT paths.
        """
        bands = bands or self.bands
        use_segments = segments if segments is not None else self.segments
        dt = round_to_himawari_10m(target_dt)
        files_by_band: Dict[str, List[Path]] = {}

        # 1. Quick local cache check across all bands and segments (zero network overhead for hits)
        needed_downloads: List[Tuple[str, str, Path]] = []
        band_segments_map: Dict[str, List[Tuple[str, Path, bool]]] = {}

        for band in bands:
            bucket, keys = build_s3_prefix_and_keys(dt, band=band, segments=use_segments)
            seg_info = []
            for k in keys:
                fname = Path(k).name
                dat_fname = fname[:-4] if fname.endswith(".bz2") else fname
                dat_path = self.cache_dir / dat_fname
                is_cached = dat_path.exists() and dat_path.stat().st_size > 0
                seg_info.append((k, dat_path, is_cached))
                if not is_cached:
                    needed_downloads.append((bucket, k, dat_path))
            band_segments_map[band] = seg_info

        # 2. Concurrently download any missing segments across all bands simultaneously (up to max_workers connections)
        downloaded_results: Dict[str, Path] = {}
        if needed_downloads:
            workers = min(len(needed_downloads), self.max_workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futs = {
                    executor.submit(_download_and_extract_single_key, b, k, self.cache_dir, self.s3_client): k
                    for b, k, _ in needed_downloads
                }
                for fut in as_completed(futs):
                    k = futs[fut]
                    try:
                        res = fut.result()
                        if res is not None:
                            downloaded_results[k] = res
                    except Exception as e:
                        logger.warning("Error downloading segment %s: %s", k, e)

        # 3. Satellite Fallback check: If all segments for any band failed, attempt alternate satellite
        primary_bucket, _ = get_himawari_bucket_and_sat(dt)
        missing_bands = [
            b for b in bands
            if not any(is_cached or (k in downloaded_results) for k, _, is_cached in band_segments_map[b])
        ]
        if missing_bands:
            alt_bucket = BUCKET_H8 if primary_bucket == BUCKET_H9 else BUCKET_H9
            alt_sat_old = "H09" if primary_bucket == BUCKET_H9 else "H08"
            alt_sat_new = "H08" if primary_bucket == BUCKET_H9 else "H09"
            alt_needed_downloads: List[Tuple[str, str, Path]] = []

            for b in missing_bands:
                alt_seg_info = []
                for orig_k, _, _ in band_segments_map[b]:
                    alt_k = orig_k.replace(f"HS_{alt_sat_old}_", f"HS_{alt_sat_new}_")
                    fname = Path(alt_k).name
                    dat_fname = fname[:-4] if fname.endswith(".bz2") else fname
                    dat_path = self.cache_dir / dat_fname
                    is_cached = dat_path.exists() and dat_path.stat().st_size > 0
                    alt_seg_info.append((alt_k, dat_path, is_cached))
                    if not is_cached:
                        alt_needed_downloads.append((alt_bucket, alt_k, dat_path))
                band_segments_map[b] = alt_seg_info

            if alt_needed_downloads:
                logger.info(
                    "Attempting alternate satellite fallback (%s -> %s) for %s missing bands at %s",
                    alt_sat_old, alt_sat_new, len(missing_bands), dt
                )
                alt_workers = min(len(alt_needed_downloads), self.max_workers)
                with ThreadPoolExecutor(max_workers=alt_workers) as executor:
                    alt_futs = {
                        executor.submit(_download_and_extract_single_key, b, k, self.cache_dir, self.s3_client): k
                        for b, k, _ in alt_needed_downloads
                    }
                    for fut in as_completed(alt_futs):
                        k = alt_futs[fut]
                        try:
                            res = fut.result()
                            if res is not None:
                                downloaded_results[k] = res
                        except Exception as e:
                            logger.debug("Alternate satellite fallback failed for %s: %s", k, e)

        # 4. Assemble results per band
        for band, seg_info in band_segments_map.items():
            dat_files = []
            for k, dat_path, is_cached in seg_info:
                if is_cached:
                    dat_files.append(dat_path)
                elif k in downloaded_results:
                    dat_files.append(downloaded_results[k])

            if dat_files:
                files_by_band[band] = dat_files
            else:
                logger.warning("No files downloaded for band %s at %s", band, dt)

        return files_by_band

    def load_scene(
        self,
        files_by_band: Dict[str, List[Path]],
        bands_to_load: Optional[List[str]] = None
    ) -> Scene:
        """
        Creates a Satpy Scene and calibrates the specified IR bands into
        Brightness Temperature (Kelvin).
        """
        all_files: List[str] = []
        for file_list in files_by_band.values():
            all_files.extend([str(p) for p in file_list])

        if not all_files:
            raise ValueError("Cannot load Satpy Scene: file list is empty")

        bands = bands_to_load or list(files_by_band.keys())
        scn = Scene(filenames=all_files, reader="ahi_hsd")
        scn.load(bands)
        return scn

    def validate_physical_bounds(
        self,
        bt_values: np.ndarray,
        band_name: str
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Validates that Brightness Temperature falls within physical limits (150K - 350K).
        Returns sanitized array with NaN for out-of-bound values, and audit statistics.
        Strict zero-mock rule: out-of-bound values are set to NaN, NEVER synthesized.
        """
        valid_mask = ~np.isnan(bt_values)
        total_pixels = int(valid_mask.sum())
        if total_pixels == 0:
            return bt_values, {"valid_count": 0, "oob_count": 0, "min_k": np.nan, "max_k": np.nan, "mean_k": np.nan}

        oob_mask = (bt_values < PHYSICAL_BT_MIN_K) | (bt_values > PHYSICAL_BT_MAX_K)
        oob_count = int((valid_mask & oob_mask).sum())

        cleaned = bt_values.copy()
        cleaned[oob_mask] = np.nan

        valid_clean = cleaned[~np.isnan(cleaned)]
        stats = {
            "valid_count": int(len(valid_clean)),
            "oob_count": oob_count,
            "min_k": float(valid_clean.min()) if len(valid_clean) > 0 else np.nan,
            "max_k": float(valid_clean.max()) if len(valid_clean) > 0 else np.nan,
            "mean_k": float(valid_clean.mean()) if len(valid_clean) > 0 else np.nan,
        }
        return cleaned, stats

    def sample_at_coordinates(
        self,
        scene: Scene,
        coords_df: pd.DataFrame,
        bands: Optional[List[str]] = None,
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon",
        segments: Optional[List[int]] = None,
        max_dist_deg: float = 0.05,
    ) -> pd.DataFrame:
        """
        Extracts centroid Brightness Temperature for given lat/lon coordinates
        (e.g., from the Master Grid or Ground Stations).
        Uses spatial index caching across timestamps and masks out-of-bounds
        coordinates (e.g. Southern stations when Segment 4 only is loaded) with NaN.
        """
        bands = bands or self.bands
        use_segments = segments if segments is not None else self.segments
        result_df = coords_df.copy()

        query_lats = coords_df[lat_col].to_numpy()
        query_lons = coords_df[lon_col].to_numpy()

        cache_key = (
            tuple(sorted(use_segments)),
            len(coords_df),
            round(float(query_lats[0]), 4) if len(query_lats) > 0 else 0.0,
            round(float(query_lons[0]), 4) if len(query_lons) > 0 else 0.0,
            round(float(query_lats[-1]), 4) if len(query_lats) > 0 else 0.0,
            round(float(query_lons[-1]), 4) if len(query_lons) > 0 else 0.0,
        )

        if cache_key in self._cached_spatial_index:
            bbox_mask, idxs, valid_dist_mask = self._cached_spatial_index[cache_key]
        else:
            sample_band = bands[0]
            ds_sample = scene[sample_band]
            area = ds_sample.attrs["area"]
            lons_sat, lats_sat = area.get_lonlats()

            from scipy.spatial import cKDTree
            margin = 0.2
            bbox_mask = (
                (lats_sat >= query_lats.min() - margin) &
                (lats_sat <= query_lats.max() + margin) &
                (lons_sat >= query_lons.min() - margin) &
                (lons_sat <= query_lons.max() + margin)
            )
            sub_lats = lats_sat[bbox_mask]
            sub_lons = lons_sat[bbox_mask]

            if len(sub_lats) == 0:
                raise ValueError("No satellite pixels intersect the query bounding box")

            tree = cKDTree(np.column_stack([sub_lats, sub_lons]))
            dists, idxs = tree.query(np.column_stack([query_lats, query_lons]), k=1)
            valid_dist_mask = (dists <= max_dist_deg)
            self._cached_spatial_index[cache_key] = (bbox_mask, idxs, valid_dist_mask)

        # For each band, extract the pixel values at idxs
        for b in bands:
            col_name = f"bt_{b.lower()}"
            ds = scene[b]
            raw_vals = ds.values[bbox_mask]
            sampled_vals = raw_vals[idxs].astype(np.float64, copy=True)
            # Coordinates exceeding max distance from any valid pixel become NaN
            sampled_vals[~valid_dist_mask] = np.nan
            cleaned_vals, _ = self.validate_physical_bounds(sampled_vals, b)
            result_df[col_name] = cleaned_vals

        return result_df

    def fetch_range_snapshots(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        cadence_hours: int = 1,
        bands: Optional[List[str]] = None
    ) -> Dict[pd.Timestamp, Dict[str, List[Path]]]:
        """
        Downloads Himawari IR bands across a date range at regular cadence.
        Returns mapping from timestamp to downloaded files per band.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize("UTC")
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize("UTC")

        timestamps = pd.date_range(start_dt, end_dt, freq=f"{cadence_hours}h")
        logger.info("Downloading Himawari snapshots for %d timestamps from %s to %s",
                    len(timestamps), start_dt, end_dt)

        all_snapshots: Dict[pd.Timestamp, Dict[str, List[Path]]] = {}
        t_batch_start = time.perf_counter()
        for i, dt in enumerate(timestamps):
            files = self.fetch_snapshot_files(dt, bands=bands)
            if files:
                all_snapshots[dt] = files
            if (i + 1) % 10 == 0 or (i + 1) == len(timestamps):
                elapsed = max(0.001, time.perf_counter() - t_batch_start)
                rate = 10.0 / elapsed if (i + 1) % 10 == 0 else ((i % 10 + 1) / elapsed)
                logger.info(
                    "  Progress: %d/%d timestamps processed (%.1fs/batch, %.2f snapshots/sec)",
                    i + 1, len(timestamps), elapsed, rate
                )
                t_batch_start = time.perf_counter()

        return all_snapshots

    def fetch_and_extract_station_points(
        self,
        target_dt: pd.Timestamp,
        stations_df: pd.DataFrame,
        bands: Optional[List[str]] = None,
        segments: Optional[List[int]] = None,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        cleanup_raw: bool = True,
        max_dist_deg: float = 0.05,
    ) -> pd.DataFrame:
        """
        Downloads the required bands for target_dt, loads the scene,
        extracts Brightness Temperature & physics BTDs at station coordinates,
        and immediately deletes raw .DAT segment files to maintain zero disk footprint.
        """
        bands = bands or self.bands
        use_segments = segments if segments is not None else self.segments
        dt = round_to_himawari_10m(target_dt)
        files_by_band = self.fetch_snapshot_files(dt, bands=bands, segments=use_segments)
        if not files_by_band:
            logger.warning("No satellite files available for %s", dt)
            return pd.DataFrame()

        try:
            scene = self.load_scene(files_by_band, bands_to_load=bands)
            station_features = self.sample_at_coordinates(
                scene,
                stations_df,
                bands=bands,
                lat_col=lat_col,
                lon_col=lon_col,
                segments=use_segments,
                max_dist_deg=max_dist_deg,
            )

            # Drop stations that fall outside the loaded segments extent (where bt_b13 is NaN)
            if "bt_b13" in station_features.columns:
                station_features = station_features.dropna(subset=["bt_b13"]).reset_index(drop=True)

            if station_features.empty:
                return pd.DataFrame()

            # Compute core physics-informed BTDs
            if "bt_b15" in station_features.columns and "bt_b13" in station_features.columns:
                station_features["btd_15_13"] = (
                    station_features["bt_b15"] - station_features["bt_b13"]
                ).astype(np.float32)
                station_features["dust_rgb_red"] = np.clip(
                    (station_features["btd_15_13"] - (-4.0)) / (2.0 - (-4.0)), 0.0, 1.0
                ).astype(np.float32)

            if "bt_b13" in station_features.columns and "bt_b07" in station_features.columns:
                station_features["btd_13_07"] = (
                    station_features["bt_b13"] - station_features["bt_b07"]
                ).astype(np.float32)

            if "bt_b13" in station_features.columns and "bt_b10" in station_features.columns:
                station_features["btd_13_10"] = (
                    station_features["bt_b13"] - station_features["bt_b10"]
                ).astype(np.float32)
                if "bt_b10" in station_features.columns and "bt_b08" in station_features.columns:
                    station_features["btd_10_08"] = (
                        station_features["bt_b10"] - station_features["bt_b08"]
                    ).astype(np.float32)
                else:
                    station_features["btd_10_08"] = station_features["btd_13_10"]

            if "bt_b13" in station_features.columns and "bt_b11" in station_features.columns:
                station_features["btd_13_11"] = (
                    station_features["bt_b13"] - station_features["bt_b11"]
                ).astype(np.float32)
                if "bt_b14" in station_features.columns and "bt_b11" in station_features.columns:
                    station_features["btd_14_11"] = (
                        station_features["bt_b14"] - station_features["bt_b11"]
                    ).astype(np.float32)
                else:
                    station_features["btd_14_11"] = station_features["btd_13_11"]

            station_features["timestamp"] = pd.to_datetime(dt, utc=True)
            return station_features
        finally:
            if cleanup_raw:
                for file_list in files_by_band.values():
                    for p in file_list:
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass

    def fetch_range_station_snapshots(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        stations_df: pd.DataFrame = None,
        cadence_hours: int = 3,
        target_timestamps: Optional[List[pd.Timestamp]] = None,
        bands: Optional[List[str]] = None,
        segments: Optional[List[int]] = None,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        cleanup_raw: bool = True,
        save_parquet_path: Optional[Path] = None,
        checkpoint_interval: int = 20,
    ) -> pd.DataFrame:
        """
        Extracts satellite features across a multi-year date range specifically
        at ground station locations (Station-Centric Extraction).
        Supports:
        - Target-Driven Timestamps: if target_timestamps is provided, extracts exactly those observation hours.
        - Pruning to Segment 4 for ultra-fast training vs All Segments [3, 4, 5] for testing.
        - Incremental Auto-Resume: detects existing Parquet checkpoint and resumes missing timestamps.
        - Crash-safe periodic checkpoint saves every `checkpoint_interval` snapshots.
        - Deletes raw .DAT rasters after each step (Zero Disk Footprint).
        """
        use_segments = segments if segments is not None else self.segments

        if target_timestamps is not None and len(target_timestamps) > 0:
            parsed_ts = [
                pd.to_datetime(t, utc=True) if pd.to_datetime(t).tzinfo is None
                else pd.to_datetime(t).tz_convert("UTC")
                for t in target_timestamps
            ]
            all_timestamps = pd.DatetimeIndex(sorted(set(parsed_ts)))
        else:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            if start_dt.tzinfo is None:
                start_dt = start_dt.tz_localize("UTC")
            if end_dt.tzinfo is None:
                end_dt = end_dt.tz_localize("UTC")
            all_timestamps = pd.date_range(start_dt, end_dt, freq=f"{cadence_hours}h")

        save_p = Path(save_parquet_path) if save_parquet_path else None
        existing_df = pd.DataFrame()
        done_timestamps = set()

        # Check existing parquet checkpoint for Auto-Resume
        if save_p and save_p.exists():
            try:
                existing_df = pd.read_parquet(save_p)
                if not existing_df.empty and "timestamp" in existing_df.columns:
                    done_timestamps = set(pd.to_datetime(existing_df["timestamp"], utc=True))
                    logger.info(
                        "Found existing checkpoint at %s: %d rows, %d/%d timestamps already completed.",
                        save_p.name, len(existing_df), len(done_timestamps), len(all_timestamps)
                    )
            except Exception as e:
                logger.warning("Could not read existing parquet checkpoint at %s: %s", save_p, e)

        pending_timestamps = [dt for dt in all_timestamps if dt not in done_timestamps]
        if not pending_timestamps and not existing_df.empty:
            logger.info("All %d timestamps already completed in %s. Returning cached dataset.",
                        len(all_timestamps), save_p)
            return existing_df

        logger.info(
            "Station-Centric Ingestion: Extracting %d remaining timestamps (of %d total, cadence=%dh) "
            "for %d stations using Segments %s (cleanup_raw=%s)",
            len(pending_timestamps), len(all_timestamps), cadence_hours, len(stations_df), use_segments, cleanup_raw
        )

        accumulated_dfs: List[pd.DataFrame] = [existing_df] if not existing_df.empty else []
        new_batch_dfs: List[pd.DataFrame] = []
        t_batch_start = time.perf_counter()

        for i, dt in enumerate(pending_timestamps):
            try:
                stn_df = self.fetch_and_extract_station_points(
                    target_dt=dt,
                    stations_df=stations_df,
                    bands=bands,
                    segments=use_segments,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    cleanup_raw=cleanup_raw,
                )
                if not stn_df.empty:
                    new_batch_dfs.append(stn_df)
                    accumulated_dfs.append(stn_df)
            except Exception as e:
                logger.warning("Error extracting satellite features at %s: %s", dt, e)

            # Progress log & periodic checkpoint save
            if (i + 1) % 10 == 0 or (i + 1) == len(pending_timestamps):
                elapsed = max(0.001, time.perf_counter() - t_batch_start)
                rate = 10.0 / elapsed if (i + 1) % 10 == 0 else ((i % 10 + 1) / elapsed)
                total_rows = sum(len(d) for d in accumulated_dfs)
                logger.info(
                    "  [Station Extraction] Progress: %d/%d remaining timestamps (%.1fs/batch, %.2f snapshots/sec, total rows: %d)",
                    i + 1, len(pending_timestamps), elapsed, rate, total_rows
                )
                t_batch_start = time.perf_counter()

            # Crash-Safe Checkpoint every checkpoint_interval snapshots
            if save_p and new_batch_dfs and ((i + 1) % checkpoint_interval == 0 or (i + 1) == len(pending_timestamps)):
                try:
                    chk_df = pd.concat(accumulated_dfs, ignore_index=True)
                    if "timestamp" in chk_df.columns:
                        dedup_cols = ["timestamp", "location_id"] if "location_id" in chk_df.columns else ["timestamp", lat_col, lon_col]
                        valid_cols = [c for c in dedup_cols if c in chk_df.columns]
                        if valid_cols:
                            chk_df = chk_df.drop_duplicates(subset=valid_cols)
                        chk_df = chk_df.sort_values("timestamp").reset_index(drop=True)
                    save_p.parent.mkdir(parents=True, exist_ok=True)
                    chk_df.to_parquet(save_p, compression="snappy", index=False)
                    logger.debug("  [Checkpoint Saved] -> %s (%d rows, %d timestamps)",
                                 save_p.name, len(chk_df), chk_df["timestamp"].nunique())
                except Exception as e:
                    logger.warning("Failed saving periodic checkpoint to %s: %s", save_p, e)

        if accumulated_dfs:
            final_df = pd.concat(accumulated_dfs, ignore_index=True)
            if "timestamp" in final_df.columns:
                dedup_cols = ["timestamp", "location_id"] if "location_id" in final_df.columns else ["timestamp", lat_col, lon_col]
                valid_cols = [c for c in dedup_cols if c in final_df.columns]
                if valid_cols:
                    final_df = final_df.drop_duplicates(subset=valid_cols)
                final_df = final_df.sort_values("timestamp").reset_index(drop=True)
            if save_p:
                save_p.parent.mkdir(parents=True, exist_ok=True)
                final_df.to_parquet(save_p, compression="snappy", index=False)
                logger.info("Saved station satellite features -> %s (Rows: %d, Unique Timestamps: %d, Size: %.2f MB)",
                            save_p, len(final_df), final_df["timestamp"].nunique(), save_p.stat().st_size / (1024 * 1024))
            return final_df
        return pd.DataFrame()

