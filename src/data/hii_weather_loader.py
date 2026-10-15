"""
HII Ground Telemetry Weather Data Loader & QA Engine
Fetches authentic atmospheric pressure and relative humidity observations from
HII Open Data Catalog (Hydro and Agro Informatics Institute, Thailand).

Features:
- Timezone Alignment: Converts Thailand Local Time (ICT = UTC+7) to UTC (UTC = ICT - 7 hours)
- Physical Bound Filtering:
  * Atmospheric Pressure: [850.0, 1050.0] hPa
  * Relative Humidity: [0.0, 100.0] %
- Sentinel Filtering: Eliminates -999, -999.0, -9999, NaN, and corrupted telemetry
- Metadata Merging: Associates observations with official station Lat/Lon and province
- Strict Zero-Mock: NEVER imputes with random numbers or synthetic models. Missing records remain NaN.
"""

import os
import io
import sys
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config.paths import HII_DIR, METADATA_DIR, DATA_CONFIG_PATH, load_yaml_config

logger = logging.getLogger(__name__)

DEFAULT_METADATA_PATH = METADATA_DIR / "hii_stations_master_metadata.csv"

CATALOG_URLS = {
    "pressure": "https://tiservice.hii.or.th/opendata/data_catalog/pressure",
    "humidity": "https://tiservice.hii.or.th/opendata/data_catalog/humidity",
}

# Physical boundary constants
BOUNDS_PRESSURE = (850.0, 1050.0)
BOUNDS_HUMIDITY = (0.0, 100.0)
SENTINEL_VALUES = {-999.0, -9999.0, 9999.0, -99.0}


class HIIWeatherLoader:
    """
    Downloads, parses, audits, and caches HII atmospheric pressure and humidity telemetry.
    """

    def __init__(
        self,
        hii_dir: Path = HII_DIR,
        metadata_path: Path = DEFAULT_METADATA_PATH
    ):
        self.hii_dir = hii_dir
        self.metadata_path = metadata_path
        self.hii_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_df: Optional[pd.DataFrame] = None

    def get_metadata(self) -> pd.DataFrame:
        """Loads and caches HII station master metadata."""
        if self._metadata_df is not None:
            return self._metadata_df

        from src.config.paths import PROJECT_ROOT, METADATA_DIR
        candidates = [
            self.metadata_path,
            METADATA_DIR / "hii_stations_master_metadata.csv",
            PROJECT_ROOT / "data" / "metadata" / "hii_stations_master_metadata.csv",
            Path.cwd() / "data" / "metadata" / "hii_stations_master_metadata.csv",
            Path(__file__).resolve().parent.parent.parent / "data" / "metadata" / "hii_stations_master_metadata.csv",
        ]
        found_path = next((p for p in candidates if p and p.exists()), None)

        if found_path is not None:
            self.metadata_path = found_path
            df = pd.read_csv(self.metadata_path)
        else:
            # P1-5 fix (15-10-2026): fail LOUD. Silent fallback stuffed 20 fake
            # stations at a single coordinate (13.75, 100.5), manufacturing
            # bogus gradients. Missing metadata is a hard error now.
            raise FileNotFoundError(
                "HII master metadata CSV not found in any candidate path "
                f"{[str(p) for p in candidates]}. Refusing silent single-point "
                "fallback (would fabricate gradients). Provide "
                "data/metadata/hii_stations_master_metadata.csv."
            )

        # Ensure standard column naming and coordinate validity
        req_cols = ["station_code", "latitude", "longitude"]
        for c in req_cols:
            if c not in df.columns:
                df[c] = np.nan if "latitude" in c or "longitude" in c else "UNKNOWN"

        # Filter valid Thailand bounding box
        valid_coords = (
            (df["latitude"] >= 5.5) & (df["latitude"] <= 20.5) &
            (df["longitude"] >= 97.0) & (df["longitude"] <= 106.0)
        )
        filtered_df = df[valid_coords].copy()
        self._metadata_df = filtered_df if not filtered_df.empty else df.copy()
        return self._metadata_df

    def fetch_station_raw_csv(
        self,
        variable: str,
        year: int,
        month: int,
        station_code: str,
        timeout: int = 15
    ) -> Optional[bytes]:
        """
        Fetches the raw monthly CSV bytes for a station from HII Open Data catalog.
        """
        if variable not in CATALOG_URLS:
            raise ValueError(f"Unsupported variable: {variable}. Must be 'pressure' or 'humidity'")

        ym = f"{year}{month:02d}"
        url = f"{CATALOG_URLS[variable]}/{year}/{ym}/{station_code}.csv"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DustPrediction/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return resp.read()
        except Exception:
            # Data not available for this station/month on HII server
            return None
        return None

    def parse_and_clean_csv(
        self,
        raw_bytes: bytes,
        station_code: str,
        variable: str
    ) -> pd.DataFrame:
        """
        Parses HII raw CSV, applies UTC alignment and strict physical bounds QA.
        """
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as e:
            logger.debug("Failed parsing CSV bytes for %s: %s", station_code, e)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Find date and time columns (supports 2024-2025 separate date/time and 2026 unified measure_datetime)
        cols_lower = {c.lower().strip(): c for c in df.columns}
        
        datetime_col = None
        for k in ["measure_datetime", "measuredatetime", "datetime", "date_time", "timestamp"]:
            if k in cols_lower:
                datetime_col = cols_lower[k]
                break

        if datetime_col is None:
            for k, orig in cols_lower.items():
                if "datetime" in k or "timestamp" in k:
                    datetime_col = orig
                    break

        if datetime_col is not None:
            date_col = datetime_col
            time_col = None
        else:
            date_col = next((cols_lower[k] for k in ["date", "dat", "date_str"] if k in cols_lower), None)
            time_col = next((cols_lower[k] for k in ["time", "time_str"] if k in cols_lower), None)

        # Find value column
        val_col = None
        for candidate in ["press", "pressure", "humid", "humidity", "rh", "value", "val"]:
            if candidate in cols_lower:
                val_col = cols_lower[candidate]
                break

        if val_col is None:
            # Fallback: choose first non-metadata column
            excluded_names = {"station_code", "station", "quality_flag", "flag", "code"}
            for c in df.columns:
                c_lower = c.lower().strip()
                if c not in (date_col, time_col) and c_lower not in excluded_names:
                    val_col = c
                    break

        if date_col is None or val_col is None:
            return pd.DataFrame()

        # Construct datetime series in ICT
        if time_col is not None:
            dt_str = df[date_col].astype(str) + " " + df[time_col].astype(str)
        else:
            dt_str = df[date_col].astype(str)

        try:
            dt_series = pd.to_datetime(dt_str, errors="coerce")
        except Exception:
            return pd.DataFrame()

        # Convert values to numeric
        values = pd.to_numeric(df[val_col], errors="coerce")

        clean_df = pd.DataFrame({
            "station_code": station_code,
            "timestamp_ict": dt_series,
            "raw_value": values
        }).dropna(subset=["timestamp_ict"])

        if clean_df.empty:
            return pd.DataFrame()

        # Convert ICT to UTC (ICT = UTC + 7 -> UTC = ICT - 7 hours)
        clean_df["timestamp_utc"] = clean_df["timestamp_ict"] - pd.Timedelta(hours=7)

        # Apply Physical Bounds and Sentinel Filtering
        bounds = BOUNDS_PRESSURE if variable == "pressure" else BOUNDS_HUMIDITY
        val_arr = clean_df["raw_value"].to_numpy()

        valid_mask = (
            ~np.isnan(val_arr) &
            (~np.isin(val_arr, list(SENTINEL_VALUES))) &
            (val_arr >= bounds[0]) &
            (val_arr <= bounds[1])
        )

        clean_df["value"] = np.where(valid_mask, val_arr, np.nan)
        clean_df["variable"] = variable

        # Keep valid records
        clean_df = clean_df.dropna(subset=["value"]).copy()
        clean_df = clean_df[["station_code", "timestamp_utc", "variable", "value"]]

        return clean_df

    def fetch_and_harmonize(
        self,
        variable: str,
        station_codes: List[str],
        year: int,
        month: int,
        max_workers: int = 8
    ) -> pd.DataFrame:
        """
        Concurrently downloads and cleans observations for a list of stations.
        Merges station metadata (latitude, longitude).
        """
        meta = self.get_metadata().set_index("station_code")
        dfs: List[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_to_st = {
                executor.submit(self.fetch_station_raw_csv, variable, year, month, sc): sc
                for sc in station_codes
            }
            for fut in as_completed(fut_to_st):
                sc = fut_to_st[fut]
                try:
                    raw_bytes = fut.result()
                    if raw_bytes:
                        df_st = self.parse_and_clean_csv(raw_bytes, sc, variable)
                        if not df_st.empty:
                            dfs.append(df_st)
                except Exception as e:
                    logger.warning("Error processing station %s: %s", sc, e)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Merge metadata
        merged = combined.merge(
            meta[["latitude", "longitude"]].reset_index(),
            on="station_code",
            how="inner"
        )
        return merged

    def fetch_date_range(
        self,
        variable: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        station_codes: Optional[List[str]] = None,
        max_workers: int = 8
    ) -> pd.DataFrame:
        """
        Fetches HII records across a date range (spanning multiple months/years).
        """
        meta = self.get_metadata()
        stations = station_codes or meta["station_code"].tolist()
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Build list of (year, month)
        period_index = pd.period_range(start=start_dt, end=end_dt, freq="M")
        logger.info("Fetching HII %s for %d stations across %d months (%s to %s)",
                    variable, len(stations), len(period_index), start_dt.date(), end_dt.date())

        month_dfs = []
        for period in period_index:
            df_m = self.fetch_and_harmonize(
                variable=variable,
                station_codes=stations,
                year=period.year,
                month=period.month,
                max_workers=max_workers
            )
            if not df_m.empty:
                month_dfs.append(df_m)

        if not month_dfs:
            return pd.DataFrame()

        combined = pd.concat(month_dfs, ignore_index=True)
        # Harmonize timezone between timestamp_utc and start_dt/end_dt
        ts_series = combined["timestamp_utc"]
        s_dt = start_dt
        e_dt = end_dt
        if getattr(ts_series.dt, "tz", None) is not None:
            if s_dt.tzinfo is None:
                s_dt = s_dt.tz_localize("UTC")
            if e_dt.tzinfo is None:
                e_dt = e_dt.tz_localize("UTC")
        else:
            if s_dt.tzinfo is not None:
                s_dt = s_dt.tz_localize(None)
            if e_dt.tzinfo is not None:
                e_dt = e_dt.tz_localize(None)

        mask = (ts_series >= s_dt) & (ts_series <= e_dt)
        filtered = combined[mask].drop_duplicates(subset=["station_code", "timestamp_utc"]).copy()
        return filtered.sort_values(["station_code", "timestamp_utc"]).reset_index(drop=True)

    @staticmethod
    def split_train_test(
        df: pd.DataFrame,
        split_date: str = "2026-01-01"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into Train (before split_date, e.g. 2024-2025)
        and Test (on or after split_date, e.g. 2026).
        """
        if df.empty:
            return df.copy(), df.copy()

        cutoff = pd.to_datetime(split_date)
        if getattr(df["timestamp_utc"].dt, "tz", None) is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        elif getattr(df["timestamp_utc"].dt, "tz", None) is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)

        train_df = df[df["timestamp_utc"] < cutoff].copy()
        test_df = df[df["timestamp_utc"] >= cutoff].copy()

        train_df["split"] = "train"
        test_df["split"] = "test"
        return train_df, test_df

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        variable: str,
        filename: Optional[str] = None
    ) -> Path:
        """Saves cleaned HII telemetry to partitioned Parquet file."""
        if df.empty:
            raise ValueError("Cannot save empty DataFrame to Parquet")

        fname = filename or f"hii_{variable}_cleaned.parquet"
        out_path = self.hii_dir / fname
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Saved %d %s records to %s", len(df), variable, out_path)
        return out_path

    def audit_records(self, df: pd.DataFrame, variable: str) -> Dict[str, Any]:
        """
        Conducts quality audit on HII records, verifying physical limits and coverage.
        """
        bounds = BOUNDS_PRESSURE if variable == "pressure" else BOUNDS_HUMIDITY
        if df.empty:
            return {
                "variable": variable,
                "total_records": 0,
                "stations_count": 0,
                "min_val": None,
                "max_val": None,
                "mean_val": None,
                "status": "EMPTY",
            }

        vals = df["value"].dropna()
        return {
            "variable": variable,
            "total_records": int(len(df)),
            "stations_count": int(df["station_code"].nunique()),
            "min_val": float(vals.min()) if len(vals) > 0 else None,
            "max_val": float(vals.max()) if len(vals) > 0 else None,
            "mean_val": float(vals.mean()) if len(vals) > 0 else None,
            "within_physical_bounds": bool((vals >= bounds[0]).all() and (vals <= bounds[1]).all()),
            "status": "PASS",
        }
