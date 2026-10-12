"""
Ground Truth PM2.5 Target Data Acquisition & Sensor QA Engine
Ingests real physical particulate matter observations from official reference sources:
1. PCD Air4Thai (Pollution Control Department of Thailand) - Regulatory Monitors
2. OpenAQ Historical Archive & API v3 (618 Government Reference Stations nationwide)
3. Historical Monitoring Station Parquet/CSV Archives

Note: CMU DustBoy low-cost optical sensor network is excluded per project design
to preserve 100% regulatory-grade reference data integrity (Zero-Mock / Zero-Sensor-Noise).

Quality Assurance Rules:
- Physical Bounds QA: PM2.5 in [0.0, 500.0] µg/m³
- Stuck Sensor Filter: Rejects sensor streams with 0 variance over > 6 consecutive hours
- Timezone Alignment: Converts Thailand Local Time (ICT = UTC+7) to UTC
- Strict Zero-Mock: NEVER generates synthetic plumes or random target values.
"""

import os
import ssl
import json
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config.paths import TARGET_DIR, DATA_CONFIG_PATH, load_yaml_config
from src.data.openaq_loader import OpenAQLoader, DEFAULT_THAILAND_OPENAQ_LOCATIONS

logger = logging.getLogger(__name__)

AIR4THAI_API_URL = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"

PHYSICAL_PM25_MIN = 0.0
PHYSICAL_PM25_MAX = 500.0
PHYSICAL_PM10_MIN = 0.0
PHYSICAL_PM10_MAX = 1000.0
MAX_STUCK_HOURS = 6


class GroundTruthLoader:
    """
    Manages loading, cleaning, and validating real physical PM2.5 and PM10 ground truth observations.
    """

    def __init__(self, target_dir: Path = TARGET_DIR, openaq_api_key: Optional[str] = None):
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.openaq_loader = OpenAQLoader(target_dir=self.target_dir, api_key=openaq_api_key)

    def fetch_openaq_historical(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        location_ids: Optional[List[int]] = None,
        region: Optional[str] = None,
        prefer_api: bool = True,
        pollutant: str = "pm25",
    ) -> pd.DataFrame:
        """
        Fetches historical PM2.5 or PM10 observations from OpenAQ (mirroring PCD Air4Thai stations)
        via OpenAQ v3 API or AWS S3 archive.
        Supports regional filtering: 'north', 'northeast', 'central', 'east', 'south', 'all'.
        """
        return self.openaq_loader.fetch_benchmark_stations(
            start_date=start_date,
            end_date=end_date,
            location_ids=location_ids,
            region=region,
            prefer_api=prefer_api,
            pollutant=pollutant,
        )

    def _get_ssl_context(self) -> ssl.SSLContext:
        """Creates an SSL context that handles self-signed or internal government certs."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def fetch_live_air4thai(self, timeout: int = 15, pollutant: str = "pm25") -> pd.DataFrame:
        """
        Fetches live PM2.5 or PM10 observations from all active PCD Air4Thai monitoring stations.
        If pollutant='pm25', retains stations with valid PM2.5 readings.
        If pollutant='pm10', retains stations with valid PM10 readings.
        If pollutant='all', retains stations with at least one valid reading.
        Both observed_pm25 and observed_pm10 columns are preserved when available.
        """
        req = urllib.request.Request(
            AIR4THAI_API_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DustPrediction/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self._get_ssl_context()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error("Failed fetching live Air4Thai data: %s", e)
            return pd.DataFrame()

        stations = data.get("stations", [])
        records = []
        for s in stations:
            st_id = s.get("stationID")
            st_name = s.get("nameEN") or s.get("nameTH", "")
            lat_str = s.get("lat")
            lon_str = s.get("long")
            aqi_last = s.get("AQILast", {})
            pm25_info = aqi_last.get("PM25", {})
            pm10_info = aqi_last.get("PM10", {})
            val_25_str = pm25_info.get("value")
            val_10_str = pm10_info.get("value")
            d_str = aqi_last.get("date")
            t_str = aqi_last.get("time")

            if not (st_id and lat_str and lon_str and d_str and t_str):
                continue

            try:
                lat = float(lat_str)
                lon = float(lon_str)
                dt_ict = pd.to_datetime(f"{d_str} {t_str}")
                dt_utc = dt_ict - pd.Timedelta(hours=7)
            except (ValueError, TypeError):
                continue

            val_pm25 = np.nan
            if val_25_str not in (None, "", "-1"):
                try:
                    v = float(val_25_str)
                    if PHYSICAL_PM25_MIN <= v <= PHYSICAL_PM25_MAX:
                        val_pm25 = v
                except (ValueError, TypeError):
                    pass

            val_pm10 = np.nan
            if val_10_str not in (None, "", "-1"):
                try:
                    v = float(val_10_str)
                    if PHYSICAL_PM10_MIN <= v <= PHYSICAL_PM10_MAX:
                        val_pm10 = v
                except (ValueError, TypeError):
                    pass

            # Filter according to requested pollutant
            p_lower = pollutant.lower().replace(".", "").replace("-", "")
            if p_lower == "pm25" and np.isnan(val_pm25):
                continue
            elif p_lower == "pm10" and np.isnan(val_pm10):
                continue
            elif p_lower in ("all", "both") and np.isnan(val_pm25) and np.isnan(val_pm10):
                continue

            records.append({
                "station_code": f"PCD_{st_id}",
                "station_name": st_name.strip(),
                "timestamp_utc": dt_utc,
                "observed_pm25": val_pm25,
                "observed_pm10": val_pm10,
                "latitude": lat,
                "longitude": lon,
                "source": "PCD_Air4Thai"
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Apply strict spatial bounds for Thailand
        df = self.filter_thailand_bounds(df)
        target_col = "observed_pm10" if pollutant.lower() == "pm10" else "observed_pm25"
        df = self.apply_physical_bounds(df, target_col=target_col)
        return df

    def filter_thailand_bounds(
        self,
        df: pd.DataFrame,
        lat_min: float = 5.5,
        lat_max: float = 20.5,
        lon_min: float = 97.0,
        lon_max: float = 106.0
    ) -> pd.DataFrame:
        """Filters observations within Thailand territorial boundaries."""
        if df.empty:
            return df
        mask = (
            (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max) &
            (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
        )
        return df[mask].copy()

    def apply_physical_bounds(
        self,
        df: pd.DataFrame,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        target_col: str = "observed_pm25",
    ) -> pd.DataFrame:
        """
        Enforces realistic atmospheric particulate physical bounds:
        PM2.5: [0.0, 500.0] µg/m³, PM10: [0.0, 1000.0] µg/m³.
        Negative values, NaNs, and extreme electrical spikes are dropped.
        """
        if df.empty:
            return df
        eff_col = target_col if target_col in df.columns else ("observed_pm10" if "observed_pm10" in df.columns else "observed_pm25")
        if eff_col not in df.columns:
            return df

        if min_val is None:
            min_val = PHYSICAL_PM10_MIN if "pm10" in eff_col.lower() else PHYSICAL_PM25_MIN
        if max_val is None:
            max_val = PHYSICAL_PM10_MAX if "pm10" in eff_col.lower() else PHYSICAL_PM25_MAX

        val_mask = (
            df[eff_col].notna() &
            (df[eff_col] >= min_val) &
            (df[eff_col] <= max_val)
        )
        return df[val_mask].copy()

    def filter_stuck_sensors(
        self,
        df: pd.DataFrame,
        max_consecutive_hours: int = MAX_STUCK_HOURS,
        target_col: str = "observed_pm25",
        consecutive_threshold: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Identifies and removes malfunctioning sensors where the reading remains
        completely identical (zero variance) for more than max_consecutive_hours.
        """
        threshold = consecutive_threshold if consecutive_threshold is not None else max_consecutive_hours
        if df.empty or len(df) <= threshold:
            return df

        eff_col = target_col if target_col in df.columns else ("observed_pm10" if "observed_pm10" in df.columns else "observed_pm25")
        if eff_col not in df.columns:
            return df

        # Sort by station and timestamp
        df_sorted = df.sort_values(["station_code", "timestamp_utc"]).copy()
        valid_indices = []

        for _, group in df_sorted.groupby("station_code"):
            if len(group) <= threshold:
                valid_indices.extend(group.index.tolist())
                continue

            vals = group[eff_col].to_numpy()
            times = group["timestamp_utc"].to_numpy()
            n = len(vals)
            is_stuck = np.zeros(n, dtype=bool)

            run_len = 1
            for i in range(1, n):
                # Check if consecutive readings are identical and consecutive in time (<= 2h step)
                dt_diff = pd.Timedelta(times[i] - times[i - 1]).total_seconds() / 3600.0
                if vals[i] == vals[i - 1] and dt_diff <= 1.5:
                    run_len += 1
                    if run_len > threshold:
                        is_stuck[i - run_len + 1 : i + 1] = True
                else:
                    run_len = 1

            valid_group_idx = group.index[~is_stuck].tolist()
            valid_indices.extend(valid_group_idx)

        return df.loc[valid_indices].copy()

    def load_ground_truth_archive(
        self,
        file_path: Path,
        target_col: str = "observed_pm25",
    ) -> pd.DataFrame:
        """Loads and standardizes an archived PM2.5 or PM10 CSV or Parquet file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Ground truth archive not found: {file_path}")

        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        # Standardize column naming
        cols = {c.lower(): c for c in df.columns}
        st_col = cols.get("station_code") or cols.get("stationid") or cols.get("station_id")
        pm25_col = cols.get("observed_pm25") or cols.get("pm25") or cols.get("pm2.5")
        pm10_col = cols.get("observed_pm10") or cols.get("pm10")
        pm_generic = cols.get("value")

        if "pm10" in target_col.lower() and pm10_col:
            pm_col = pm10_col
        else:
            pm_col = pm25_col or pm10_col or pm_generic

        ts_col = cols.get("timestamp_utc") or cols.get("timestamp") or cols.get("datetime")
        lat_col = cols.get("latitude") or cols.get("lat")
        lon_col = cols.get("longitude") or cols.get("lon") or cols.get("long")

        if not all([st_col, pm_col, ts_col, lat_col, lon_col]):
            raise ValueError(f"Ground truth file missing essential columns in {file_path}")

        target_name = "observed_pm10" if "pm10" in target_col.lower() else "observed_pm25"
        standard_df = pd.DataFrame({
            "station_code": df[st_col].astype(str),
            "timestamp_utc": pd.to_datetime(df[ts_col]),
            target_name: pd.to_numeric(df[pm_col], errors="coerce"),
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
            "source": "Historical_Archive"
        })

        standard_df = self.filter_thailand_bounds(standard_df)
        standard_df = self.apply_physical_bounds(standard_df, target_col=target_name)
        standard_df = self.filter_stuck_sensors(standard_df, target_col=target_name)
        return standard_df

    def save_ground_truth(
        self,
        df: pd.DataFrame,
        filename: str = "ground_truth_pm25.parquet"
    ) -> Path:
        """Saves clean ground truth observations to Parquet."""
        if df.empty:
            raise ValueError("Cannot save empty DataFrame to Parquet")

        out_path = self.target_dir / filename
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Saved %d particulate observations to %s", len(df), out_path)
        return out_path

    @staticmethod
    def split_train_test(
        df: pd.DataFrame,
        split_date: str = "2026-01-01"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits ground truth records into Train (before split_date, e.g. 2024-2025)
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

    def audit_target_records(
        self,
        df: pd.DataFrame,
        target_col: str = "observed_pm25",
        pollutant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Performs quality audit and returns statistical summary of ground target (PM2.5 or PM10).
        """
        if pollutant is not None and "pm10" in pollutant.lower():
            target_col = "observed_pm10"

        if df.empty or (target_col not in df.columns and "observed_pm25" not in df.columns and "observed_pm10" not in df.columns):
            return {
                "total_records": 0,
                "stations_count": 0,
                "pollutant": "pm10" if (pollutant and "pm10" in pollutant.lower()) or "pm10" in target_col.lower() else "pm25",
                "min_pm25": None,
                "max_pm25": None,
                "mean_pm25": None,
                "status": "EMPTY",
            }

        eff_col = target_col if target_col in df.columns else ("observed_pm10" if "observed_pm10" in df.columns else "observed_pm25")
        is_pm10 = "pm10" in eff_col.lower() or (pollutant is not None and "pm10" in pollutant.lower())
        max_bound = PHYSICAL_PM10_MAX if is_pm10 else PHYSICAL_PM25_MAX

        vals = df[eff_col].dropna()
        min_v = float(vals.min()) if len(vals) > 0 else None
        max_v = float(vals.max()) if len(vals) > 0 else None
        mean_v = float(vals.mean()) if len(vals) > 0 else None

        res = {
            "total_records": int(len(df)),
            "stations_count": int(df["station_code"].nunique()),
            "sources": df["source"].unique().tolist() if "source" in df.columns else [],
            "target_column": eff_col,
            "pollutant": "pm10" if is_pm10 else "pm25",
            "min_target": min_v,
            "max_target": max_v,
            "mean_target": mean_v,
            "min_pm25": min_v,
            "max_pm25": max_v,
            "mean_pm25": mean_v,
            "within_bounds": bool((vals >= 0.0).all() and (vals <= max_bound).all()) if len(vals) > 0 else False,
            "within_bounds_0_to_500": bool((vals >= 0.0).all() and (vals <= 500.0).all()) if len(vals) > 0 else False,
            "status": "PASS" if len(vals) > 0 else "EMPTY",
        }
        if is_pm10:
            res["min_pm10"] = min_v
            res["max_pm10"] = max_v
            res["mean_pm10"] = mean_v
            res["within_bounds_0_to_1000"] = bool((vals >= 0.0).all() and (vals <= 1000.0).all()) if len(vals) > 0 else False
        return res
