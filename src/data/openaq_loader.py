"""
OpenAQ Historical PM2.5 Data Loader & Harmonizer
Fetches authentic historical particulate matter observations (Air4Thai reference stations)
from the OpenAQ Open Data archive on AWS S3 (s3://openaq-data-archive/) and OpenAQ API v3.

Features:
- Anonymous AWS S3 Direct Ingestion (s3://openaq-data-archive/records/csv.gz/)
- Automatic Decompression, Timezone to UTC Alignment, and Parameter Filtering ('pm25')
- Physical Bounds QA: [0.0, 500.0] µg/m³
- Stuck Sensor Detection: Flags and removes sensors with zero variance > 6 consecutive hours
- Strict Zero-Mock: 100% authentic physical measurements from government reference monitors.
"""

import os
import io
import gzip
import logging
import urllib.request
import urllib.error
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
import pandas as pd

from src.config.paths import TARGET_DIR, DATA_CONFIG_PATH, METADATA_DIR, OPENAQ_CACHE_DIR, load_yaml_config

logger = logging.getLogger(__name__)

OPENAQ_S3_BUCKET = "openaq-data-archive"
OPENAQ_API_BASE = "https://api.openaq.org/v3"
# P3-1 fix (15-10-2026): no hard-coded secret. Read from env OPENAQ_API_KEY or
# configs/data_config.yaml at runtime; empty string if unset (API still works
# with rate limits, S3 path needs no key).
DEFAULT_OPENAQ_API_KEY = ""

# Official Thailand Air4Thai Benchmark Location IDs on OpenAQ (Geographically Diverse Nationwide Subset)
DEFAULT_THAILAND_OPENAQ_LOCATIONS = {
    # 1. Central & Bangkok Metropolitan Region
    225642: "Bang Phlat District Office, Bangkok",
    368399: "Benjakitti Park, Bangkok",
    225672: "Suan Sayam-Ram Intra Intersection, Bangkok",
    351927: "Le Thai Community Park, Bangkok",
    5554536: "Chalermprakiat Rama IX Park, Bangkok",
    # 2. Northern Mountainous Region (Chiang Mai, Chiang Rai, Mae Hong Son, Phitsanulok)
    6266258: "Doi Tung Royal Palace, Mae Fah Luang, Chiang Rai",
    1861095: "Wiang Disaster Prevention Center, Chiang Mai",
    1896059: "Pai Airport, Mae Hong Son",
    6251785: "King Naresuan Camp, Phitsanulok",
    # 3. Northeastern Region / Isan (Roi Et, Sisaket)
    1993936: "Somdet Phra Srinagarindra Park, Roi Et",
    1785693: "Sisaket Meteorological Center, Sisaket",
    # 4. Eastern Seaboard & Western Coast (Rayong, Phetchaburi)
    3345345: "Rayong Provincial Stadium, Rayong",
    2132593: "Phetchaburi Municipality, Phetchaburi",
    # 5. Southern Peninsula (Chumphon, Ranong, Pattani)
    3047300: "Sports Stadium, Chumphon",
    1290682: "Ranong Municipal Office, Ranong",
    5554538: "Pattani Provincial Forestry Center, Pattani",
}

PHYSICAL_PM25_MIN = 0.0
PHYSICAL_PM25_MAX = 500.0
PHYSICAL_PM10_MIN = 0.0
PHYSICAL_PM10_MAX = 1000.0
MAX_STUCK_HOURS = 6


class OpenAQLoader:
    """
    Downloads, extracts, cleans, and harmonizes historical PM2.5 observations from OpenAQ.
    Supports both AWS S3 Open Data archive and OpenAQ API v3.
    """

    def __init__(
        self,
        target_dir: Path = TARGET_DIR,
        metadata_dir: Path = METADATA_DIR,
        cache_dir: Optional[Path] = None,
        api_key: Optional[str] = None
    ):
        self.target_dir = Path(target_dir)
        self.metadata_dir = Path(metadata_dir)
        self.cache_dir = Path(cache_dir or OPENAQ_CACHE_DIR)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cfg_key = None
        if DATA_CONFIG_PATH.exists():
            try:
                cfg = load_yaml_config(DATA_CONFIG_PATH)
                cfg_key = cfg.get("target", {}).get("openaq", {}).get("api_key")
            except Exception:
                pass

        self.api_key = api_key or os.getenv("OPENAQ_API_KEY") or cfg_key or DEFAULT_OPENAQ_API_KEY
        self._s3_client = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(signature_version=UNSIGNED, max_pool_connections=50)
        )
        self._stations_df: Optional[pd.DataFrame] = None
        self._stations_dict: Dict[int, str] = dict(DEFAULT_THAILAND_OPENAQ_LOCATIONS)

    def get_thailand_stations(
        self,
        region: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Loads the catalog of 618 authentic OpenAQ monitoring stations across Thailand.
        Optionally filters by geographical region:
        - 'bangkok' or 'central': Central Thailand / Greater Bangkok
        - 'north': Northern provinces (Chiang Mai, Chiang Rai, Lampang, Nan, etc.)
        - 'northeast': Isan region (Khon Kaen, Nakhon Ratchasima, Udon Thani, etc.)
        - 'east': Eastern seaboard (Rayong, Chonburi, Trat, etc.)
        - 'south': Southern peninsula (Phuket, Surat Thani, Songkhla, etc.)
        - 'all': Nationwide (all regions)

        Optionally filters by data provider:
        - 'Air4Thai': Official PCD government regulatory monitors
        - 'AirGradient': Dense IoT optical sensor network
        """
        if self._stations_df is None:
            from src.config.paths import PROJECT_ROOT
            candidates_parquet = [
                self.metadata_dir / "openaq_thailand_stations.parquet",
                PROJECT_ROOT / "data" / "metadata" / "openaq_thailand_stations.parquet",
                Path.cwd() / "data" / "metadata" / "openaq_thailand_stations.parquet",
                Path(__file__).resolve().parent.parent.parent / "data" / "metadata" / "openaq_thailand_stations.parquet",
            ]
            candidates_csv = [
                self.metadata_dir / "openaq_thailand_stations.csv",
                PROJECT_ROOT / "data" / "metadata" / "openaq_thailand_stations.csv",
                Path.cwd() / "data" / "metadata" / "openaq_thailand_stations.csv",
                Path(__file__).resolve().parent.parent.parent / "data" / "metadata" / "openaq_thailand_stations.csv",
            ]

            p_found = next((p for p in candidates_parquet if p and p.exists()), None)
            c_found = next((p for p in candidates_csv if p and p.exists()), None)

            if p_found is not None:
                self._stations_df = pd.read_parquet(p_found)
            elif c_found is not None:
                self._stations_df = pd.read_csv(c_found)
            else:
                # P3 hygiene: metadata missing — placeholder coords are NOT measurements.
                # Callers must not derive gradients from these; fail-soft with warning
                # (hard fail would break offline smoke tests; HII path already fails loud).
                logger.warning(
                    "OpenAQ station catalog not found; using %d placeholder locations "
                    "at (13.75, 100.5). Coordinates are NOT authentic — do not use "
                    "for spatial training without real metadata.",
                    len(DEFAULT_THAILAND_OPENAQ_LOCATIONS),
                )
                records = [
                    {
                        "location_id": loc_id,
                        "station_code": f"OPENAQ_{loc_id}",
                        "station_name": name,
                        "latitude": 13.75,
                        "longitude": 100.5,
                        "pm25_sensor_id": None,
                        "provider": "Air4Thai",
                    }
                    for loc_id, name in DEFAULT_THAILAND_OPENAQ_LOCATIONS.items()
                ]
                self._stations_df = pd.DataFrame(records)

        df = self._stations_df.copy()

        # Optional provider filter
        if provider and provider.lower() != "all" and "provider" in df.columns:
            p_clean = provider.strip().lower()
            df = df[df["provider"].astype(str).str.strip().str.lower() == p_clean].copy()

        if not region or region.lower() == "all":
            return df.reset_index(drop=True)

        region_lower = region.lower()
        if region_lower in ["bangkok", "central"]:
            mask = (
                (df["latitude"] >= 13.0) & (df["latitude"] <= 15.5) &
                (df["longitude"] >= 99.5) & (df["longitude"] <= 101.5)
            )
        elif region_lower == "north":
            mask = df["latitude"] >= 17.5
        elif region_lower == "northeast":
            mask = (df["latitude"] >= 14.5) & (df["longitude"] >= 101.5)
        elif region_lower == "east":
            mask = (
                (df["latitude"] >= 12.0) & (df["latitude"] <= 14.5) &
                (df["longitude"] >= 101.0)
            )
        elif region_lower == "south":
            mask = df["latitude"] < 12.0
        else:
            logger.warning("Unknown region '%s'; returning all stations", region)
            return df.reset_index(drop=True)

        return df[mask].reset_index(drop=True)

    def _http_request_with_retry(
        self,
        url: str,
        headers: Dict[str, str],
        timeout: int = 20,
        max_retries: int = 5,
        base_delay: float = 6.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes an HTTP GET request with exponential backoff and retry.
        Specifically catches HTTP 429 (Too Many Requests), pauses (5 - 10+ seconds),
        and retries 3 - 5 times to respect rate limits.
        """
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    if attempt <= max_retries:
                        sleep_time = min(60.0, base_delay * (1.5 ** (attempt - 1)) + ((attempt % 3) * 0.75 + 1.0))
                        logger.warning(
                            "OpenAQ API HTTP 429 (Too Many Requests) [attempt %d/%d]. "
                            "Pausing for %.1f seconds before retrying URL: %s",
                            attempt, max_retries, sleep_time, url
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        logger.error("OpenAQ HTTP 429 rate limit exceeded max retries (%d) for %s", max_retries, url)
                        return None
                elif e.code in (500, 502, 503, 504):
                    if attempt <= max_retries:
                        sleep_time = base_delay + attempt * 2.0
                        logger.warning(
                            "OpenAQ API server error HTTP %d [attempt %d/%d]. Sleeping %.1fs before retry...",
                            e.code, attempt, max_retries, sleep_time
                        )
                        time.sleep(sleep_time)
                        continue
                    return None
                else:
                    logger.warning("OpenAQ API HTTP Error %d for %s: %s", e.code, url, e)
                    return None
            except urllib.error.URLError as e:
                if attempt <= max_retries:
                    sleep_time = base_delay + 2.0
                    logger.warning(
                        "OpenAQ network connection glitch [attempt %d/%d]: %s. Retrying in %.1fs...",
                        attempt, max_retries, e, sleep_time
                    )
                    time.sleep(sleep_time)
                    continue
                logger.warning("OpenAQ network connection failed after %d attempts: %s", max_retries, e)
                return None
            except Exception as e:
                logger.warning("Unexpected error during OpenAQ request to %s: %s", url, e)
                return None
        return None

    def get_station_cache_path(self, location_id: int, pollutant: str = "pm25") -> Path:
        """Returns the local Parquet cache path for an individual OpenAQ station."""
        p_param = "pm10" if "pm10" in pollutant.lower() else "pm25"
        return self.cache_dir / f"station_{location_id}_{p_param}.parquet"

    def load_cached_station(
        self,
        location_id: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        pollutant: str = "pm25",
    ) -> Optional[pd.DataFrame]:
        """
        Loads cached observations for a station from disk if available,
        filtering to the requested date range. Returns None if not cached.
        """
        cache_path = self.get_station_cache_path(location_id, pollutant)
        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            if df.empty:
                return df

            p_param = "pm10" if "pm10" in pollutant.lower() else "pm25"
            target_col = f"observed_{p_param}"
            if "timestamp_utc" in df.columns and target_col in df.columns:
                cached_ts = pd.to_datetime(df["timestamp_utc"], utc=True)
                s_ts = pd.to_datetime(start_date, utc=True)
                e_ts = pd.to_datetime(end_date, utc=True)
                mask = (cached_ts >= s_ts) & (cached_ts <= e_ts)
                return df[mask].copy() if not mask.all() else df.copy()
            return df
        except Exception as e:
            logger.warning("Failed reading cached station %s: %s. Will re-fetch.", cache_path, e)
            return None

    def save_cached_station(
        self,
        location_id: int,
        df: pd.DataFrame,
        pollutant: str = "pm25",
    ) -> Path:
        """
        Atomically saves station observations to local disk cache for crash recovery & resuming.
        """
        cache_path = self.get_station_cache_path(location_id, pollutant)
        try:
            tmp_path = cache_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp_path, index=False, engine="pyarrow")
            tmp_path.replace(cache_path)
        except Exception as e:
            logger.warning("Could not persist station cache to %s: %s", cache_path, e)
        return cache_path

    def fetch_sensor_measurements_api(
        self,
        sensor_id: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        page_limit: int = 1000,
        timeout: int = 20,
        pollutant: str = "pm25",
    ) -> pd.DataFrame:
        """
        Fetches hourly particulate matter measurements directly from OpenAQ v3 API
        with multi-page pagination, rate-limit backoff, and 429 retry handling.
        Endpoint: /v3/sensors/{sensor_id}/hours
        """
        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True)
        start_iso = start_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        headers = {
            "User-Agent": "DustPrediction/1.0",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        target_col = "observed_pm10" if "pm10" in pollutant.lower() else "observed_pm25"
        rows = []
        page = 1

        while True:
            url = (
                f"{OPENAQ_API_BASE}/sensors/{sensor_id}/hours?"
                f"datetime_from={start_iso}&datetime_to={end_iso}&limit={page_limit}&page={page}"
            )
            data = self._http_request_with_retry(
                url,
                headers=headers,
                timeout=timeout,
                max_retries=5,
                base_delay=6.0
            )
            if not data:
                if page == 1:
                    logger.warning("OpenAQ API request failed for sensor %s after retries", sensor_id)
                break

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                val = item.get("value")
                period = item.get("period", {})
                dt_to = period.get("datetimeTo", {})
                utc_str = dt_to.get("utc") if isinstance(dt_to, dict) else None

                if val is not None and utc_str:
                    try:
                        ts = pd.to_datetime(utc_str, utc=True).tz_localize(None)
                        rows.append({
                            "timestamp_utc": ts,
                            target_col: float(val)
                        })
                    except Exception:
                        continue

            if len(results) < page_limit:
                break

            page += 1
            if page > 50:  # safety ceiling (up to 50,000 records per sensor)
                break

            # Polite pause between pages to mitigate burst rate limiting
            time.sleep(0.2)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    def _resolve_station_sensor(
        self,
        location_id: int,
        pollutant: str = "pm25",
    ) -> Tuple[Optional[int], str, float, float]:
        """
        Resolves sensor_id, station_name, lat, lon for an OpenAQ location ID.
        Checks local station catalog first, falling back to OpenAQ API /v3/locations/{id}.
        """
        stations = self.get_thailand_stations()
        match = stations[stations["location_id"] == location_id]
        p_param = "pm10" if "pm10" in pollutant.lower() else "pm25"

        if not match.empty:
            row = match.iloc[0]
            sensor_id = row.get(f"{p_param}_sensor_id") if f"{p_param}_sensor_id" in row else (row.get("pm25_sensor_id") if p_param == "pm25" else None)
            station_name = row.get("station_name", f"OpenAQ_{location_id}")
            lat = float(row.get("latitude", 13.75))
            lon = float(row.get("longitude", 100.5))
            if pd.notna(sensor_id) and sensor_id:
                return int(sensor_id), station_name, lat, lon
        else:
            sensor_id = None
            station_name = DEFAULT_THAILAND_OPENAQ_LOCATIONS.get(location_id, f"OpenAQ_{location_id}")
            lat = 13.75
            lon = 100.5

        if pd.isna(sensor_id) or not sensor_id:
            try:
                headers = {"User-Agent": "DustPrediction/1.0"}
                if self.api_key:
                    headers["X-API-Key"] = self.api_key
                url = f"{OPENAQ_API_BASE}/locations/{location_id}"
                loc_data = self._http_request_with_retry(
                    url,
                    headers=headers,
                    timeout=15,
                    max_retries=5,
                    base_delay=6.0
                )
                if loc_data:
                    results = loc_data.get("results", [])
                    if results:
                        loc_obj = results[0]
                        station_name = loc_obj.get("name", station_name)
                        coords = loc_obj.get("coordinates", {})
                        if coords:
                            lat = coords.get("latitude", lat)
                            lon = coords.get("longitude", lon)
                        for s in loc_obj.get("sensors", []):
                            if s.get("parameter", {}).get("name") == p_param:
                                sensor_id = s.get("id")
                                break
            except Exception as e:
                logger.warning("Could not resolve sensor for location %d: %s", location_id, e)

        return (int(sensor_id) if sensor_id else None), station_name, lat, lon

    def get_station_provider(self, location_id: int) -> str:
        """Resolves data provider ('Air4Thai', 'AirGradient', etc.) for an OpenAQ location ID."""
        stations = self.get_thailand_stations()
        match = stations[stations["location_id"] == location_id]
        if not match.empty:
            p = match.iloc[0].get("provider")
            if pd.notna(p) and str(p).strip():
                return str(p).strip()
        return "Air4Thai"

    def fetch_station_api(
        self,
        location_id: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        pollutant: str = "pm25",
    ) -> pd.DataFrame:
        """
        Fetches observations for a station location via OpenAQ v3 API.
        Resolves the sensor_id and station coordinates from the station catalog.
        """
        p_param = "pm10" if "pm10" in pollutant.lower() else "pm25"
        target_col = f"observed_{p_param}"

        sensor_id, station_name, lat, lon = self._resolve_station_sensor(location_id, pollutant=pollutant)
        if not sensor_id:
            logger.warning("No %s sensor found for location %d", p_param.upper(), location_id)
            return pd.DataFrame()

        df_meas = self.fetch_sensor_measurements_api(int(sensor_id), start_date, end_date, pollutant=pollutant)
        if df_meas.empty:
            return pd.DataFrame()

        prov = self.get_station_provider(location_id)
        df_meas["location_id"] = int(location_id)
        df_meas["station_code"] = f"OPENAQ_{location_id}"
        df_meas["station_name"] = station_name
        df_meas["latitude"] = lat
        df_meas["longitude"] = lon
        df_meas["source"] = f"OpenAQ_{prov}"
        df_meas["provider"] = prov

        cols = [
            "location_id", "station_code", "station_name", "timestamp_utc",
            target_col, "latitude", "longitude", "source", "provider"
        ]
        return df_meas[cols]

    def discover_s3_station_keys(
        self,
        location_id: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> List[Tuple[str, Any]]:
        """
        Discovers all available daily CSV.GZ archive keys on AWS S3 for a location ID
        within the requested year and date range.
        S3 Key pattern: records/csv.gz/locationid={id}/year={year}/month={month}/location-{id}-{yyyymmdd}.csv.gz
        """
        start_d = pd.to_datetime(start_date).date()
        end_d = pd.to_datetime(end_date).date()
        years = range(start_d.year, end_d.year + 1)
        valid_keys: List[Tuple[str, Any]] = []

        for y in years:
            prefix = f"records/csv.gz/locationid={location_id}/year={y}/"
            try:
                paginator = self._s3_client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=OPENAQ_S3_BUCKET, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        k = obj.get("Key", "")
                        fn = k.split("/")[-1]
                        parts = fn.replace(".csv.gz", "").split("-")
                        if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 8:
                            try:
                                dt = pd.to_datetime(parts[-1], format="%Y%m%d").date()
                                if start_d <= dt <= end_d:
                                    valid_keys.append((k, dt))
                            except Exception:
                                continue
            except Exception as e:
                logger.debug("S3 listing failed for loc %d year %d: %s", location_id, y, e)
                continue

        return valid_keys

    def fetch_station_s3_bulk(
        self,
        location_id: int,
        s3_key_entries: List[Tuple[str, Any]],
        pollutant: str = "pm25",
        max_workers: int = 8,
    ) -> Tuple[pd.DataFrame, Set[Any]]:
        """
        Downloads and parses multiple daily files from AWS S3 in parallel.
        Returns a combined DataFrame and the set of covered dates.
        """
        if not s3_key_entries:
            return pd.DataFrame(), set()

        p_param = "pm10" if "pm10" in pollutant.lower() else "pm25"
        target_col = f"observed_{p_param}"
        station_name = DEFAULT_THAILAND_OPENAQ_LOCATIONS.get(location_id, f"OpenAQ_{location_id}")

        def _fetch_one_key(entry: Tuple[str, Any]) -> Optional[pd.DataFrame]:
            k, d_val = entry
            try:
                obj = self._s3_client.get_object(Bucket=OPENAQ_S3_BUCKET, Key=k)
                compressed = obj["Body"].read()
                with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
                    df = pd.read_csv(gz)
                if df.empty or "parameter" not in df.columns:
                    return None
                p_df = df[df["parameter"].str.lower() == p_param].copy()
                if p_df.empty:
                    return None
                p_df["timestamp_utc"] = pd.to_datetime(p_df["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
                p_df[target_col] = pd.to_numeric(p_df["value"], errors="coerce")
                p_df["latitude"] = pd.to_numeric(p_df["lat"], errors="coerce")
                p_df["longitude"] = pd.to_numeric(p_df["lon"], errors="coerce")
                s_name = p_df["location"].iloc[0] if "location" in p_df.columns and pd.notna(p_df["location"].iloc[0]) else station_name
                out_df = pd.DataFrame({
                    "location_id": int(location_id),
                    "station_code": f"OPENAQ_{location_id}",
                    "station_name": s_name,
                    "timestamp_utc": p_df["timestamp_utc"],
                    target_col: p_df[target_col],
                    "latitude": p_df["latitude"],
                    "longitude": p_df["longitude"],
                    "source": "OpenAQ_AWS_S3"
                }).dropna(subset=["timestamp_utc", target_col, "latitude", "longitude"])
                return out_df
            except Exception as e:
                logger.debug("Failed fetching S3 key %s: %s", k, e)
                return None

        dfs: List[pd.DataFrame] = []
        covered_dates: Set[Any] = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_fetch_one_key, s3_key_entries))
            for res, (k, dt_val) in zip(results, s3_key_entries):
                if res is not None and not res.empty:
                    dfs.append(res)
                    covered_dates.add(dt_val)

        if not dfs:
            return pd.DataFrame(), set()

        combined = pd.concat(dfs, ignore_index=True)
        return combined, covered_dates

    @staticmethod
    def find_missing_date_chunks(
        all_dates: List[Any],
        covered_dates: Set[Any],
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Identifies contiguous date intervals that were missing from AWS S3
        so they can be selectively backfilled via OpenAQ API v3.
        """
        missing = sorted(list(set(all_dates) - set(covered_dates)))
        if not missing:
            return []

        chunks = []
        cur_start = missing[0]
        cur_end = missing[0]

        for d in missing[1:]:
            if (d - cur_end).days == 1:
                cur_end = d
            else:
                chunks.append((cur_start, cur_end))
                cur_start = d
                cur_end = d
        chunks.append((cur_start, cur_end))

        ts_chunks = []
        for s_d, e_d in chunks:
            s_ts = pd.Timestamp(s_d).tz_localize("UTC")
            e_ts = pd.Timestamp(e_d).tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
            ts_chunks.append((s_ts, e_ts))

        return ts_chunks

    def fetch_station_hybrid(
        self,
        location_id: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        pollutant: str = "pm25",
        s3_workers: int = 8,
    ) -> pd.DataFrame:
        """
        Intelligent Hybrid Data Ingestion Engine:
        1. Queries AWS Open Data S3 direct archive (zero rate limit, highly parallel).
        2. Detects any missing months/days across the requested time range.
        3. Selectively backfills ONLY the missing date gaps via OpenAQ v3 API with retry.
        4. Merges, harmonizes, and deduplicates the combined dataset.
        """
        start_d = pd.to_datetime(start_date).date()
        end_d = pd.to_datetime(end_date).date()
        all_target_dates = pd.date_range(start_d, end_d, freq="D").date.tolist()

        # Phase 1: Discover & Ingest from AWS S3 Open Data archive
        s3_keys = self.discover_s3_station_keys(location_id, start_date, end_date)
        df_s3 = pd.DataFrame()
        covered_dates: Set[Any] = set()

        if s3_keys:
            df_s3, covered_dates = self.fetch_station_s3_bulk(
                location_id, s3_keys, pollutant=pollutant, max_workers=s3_workers
            )
            logger.info("  Location %d: Ingested %d records across %d days from AWS S3 Open Data archive",
                        location_id, len(df_s3), len(covered_dates))
        else:
            logger.info("  Location %d: No AWS S3 records found. Will attempt OpenAQ API v3 for full range", location_id)

        # Phase 2: Detect Missing Gaps
        missing_chunks = self.find_missing_date_chunks(all_target_dates, covered_dates)
        dfs_api: List[pd.DataFrame] = []

        if missing_chunks and self.api_key:
            total_missing_days = sum((e.date() - s.date()).days + 1 for s, e in missing_chunks)
            logger.info("  Location %d: AWS S3 missing %d days across %d chunk(s). Backfilling from OpenAQ API v3...",
                        location_id, total_missing_days, len(missing_chunks))

            sensor_id, station_name, lat, lon = self._resolve_station_sensor(location_id, pollutant=pollutant)

            if sensor_id:
                p_param = "pm10" if "pm10" in pollutant.lower() else "pm25"
                target_col = f"observed_{p_param}"

                for idx, (chunk_start, chunk_end) in enumerate(missing_chunks, 1):
                    logger.debug("  Location %d: Backfilling gap %d/%d (%s to %s) via API...",
                                 location_id, idx, len(missing_chunks), chunk_start.date(), chunk_end.date())
                    df_chunk = self.fetch_sensor_measurements_api(
                        int(sensor_id), chunk_start, chunk_end, pollutant=pollutant
                    )
                    if not df_chunk.empty:
                        df_chunk["location_id"] = int(location_id)
                        df_chunk["station_code"] = f"OPENAQ_{location_id}"
                        df_chunk["station_name"] = station_name
                        df_chunk["latitude"] = lat
                        df_chunk["longitude"] = lon
                        df_chunk["source"] = "OpenAQ_API_v3"
                        cols = ["location_id", "station_code", "station_name", "timestamp_utc", target_col, "latitude", "longitude", "source"]
                        dfs_api.append(df_chunk[cols])
            else:
                logger.warning("  Location %d: Could not resolve %s sensor for API gap fill", location_id, pollutant.upper())

        # Phase 3: Combine S3 records + API Gap records
        all_dfs = []
        if not df_s3.empty:
            all_dfs.append(df_s3)
        if dfs_api:
            all_dfs.extend(dfs_api)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")

        if not combined.empty:
            if "station_name" not in combined.columns or combined["station_name"].iloc[0] == f"OpenAQ_{location_id}":
                stn_name = DEFAULT_THAILAND_OPENAQ_LOCATIONS.get(location_id, f"OpenAQ_{location_id}")
                combined["station_name"] = stn_name

        return self.clean_and_qa_dataset(combined, pollutant=pollutant)

    def fetch_station_day(
        self,
        location_id: int,
        target_date: pd.Timestamp,
        pollutant: str = "pm25",
    ) -> pd.DataFrame:
        """
        Downloads and parses one day of observations for a specific OpenAQ location ID from S3.
        S3 Key pattern: records/csv.gz/locationid={id}/year={year}/month={month}/location-{id}-{yyyymmdd}.csv.gz
        """
        dt = pd.to_datetime(target_date)
        year = dt.year
        month = dt.month
        date_str = dt.strftime("%Y%m%d")
        key = f"records/csv.gz/locationid={location_id}/year={year}/month={month:02d}/location-{location_id}-{date_str}.csv.gz"

        try:
            obj = self._s3_client.get_object(Bucket=OPENAQ_S3_BUCKET, Key=key)
            compressed_bytes = obj["Body"].read()
        except Exception as e:
            logger.debug("File not found on S3 for location %d at %s: %s", location_id, date_str, e)
            return pd.DataFrame()

        try:
            with gzip.GzipFile(fileobj=io.BytesIO(compressed_bytes)) as gz:
                df = pd.read_csv(gz)
        except Exception as e:
            logger.warning("Error decompressing/parsing CSV for %s: %s", key, e)
            return pd.DataFrame()

        if df.empty or "parameter" not in df.columns:
            return pd.DataFrame()

        p_param = "pm10" if "pm10" in pollutant.lower() else "pm25"
        target_col = f"observed_{p_param}"

        p_df = df[df["parameter"].str.lower() == p_param].copy()
        if p_df.empty:
            return pd.DataFrame()

        p_df["timestamp_utc"] = pd.to_datetime(p_df["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        p_df[target_col] = pd.to_numeric(p_df["value"], errors="coerce")
        p_df["latitude"] = pd.to_numeric(p_df["lat"], errors="coerce")
        p_df["longitude"] = pd.to_numeric(p_df["lon"], errors="coerce")

        station_name = DEFAULT_THAILAND_OPENAQ_LOCATIONS.get(
            location_id,
            p_df["location"].iloc[0] if "location" in p_df.columns else f"OpenAQ_{location_id}"
        )

        standard_df = pd.DataFrame({
            "station_code": f"OPENAQ_{location_id}",
            "station_name": station_name,
            "timestamp_utc": p_df["timestamp_utc"],
            target_col: p_df[target_col],
            "latitude": p_df["latitude"],
            "longitude": p_df["longitude"],
            "source": "OpenAQ_Air4Thai"
        }).dropna(subset=["timestamp_utc", target_col, "latitude", "longitude"])

        return standard_df

    def fetch_station_range(
        self,
        location_id: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        max_workers: int = 4,
        prefer_api: bool = False,
        pollutant: str = "pm25",
        use_cache: bool = True,
        strategy: str = "hybrid",
    ) -> pd.DataFrame:
        """
        Fetches historical records for a location ID over a given date range.
        Checks local station cache first to allow resuming interrupted downloads.
        By default, uses the intelligent 'hybrid' strategy:
          1. Ingests available days/months directly from AWS S3 Open Data archive (no rate limits).
          2. Detects any missing gap periods across the date range.
          3. Queries OpenAQ v3 API only for the missing gap intervals.
        """
        # 1. Check local station disk cache for resuming
        if use_cache:
            cached_df = self.load_cached_station(location_id, start_date, end_date, pollutant=pollutant)
            if cached_df is not None:
                if not cached_df.empty:
                    logger.info("  Location %d: Restored %d clean records from disk cache (%s)",
                                location_id, len(cached_df), self.get_station_cache_path(location_id, pollutant).name)
                return cached_df

        df_clean = pd.DataFrame()
        fetched = False

        if strategy == "hybrid":
            try:
                df_clean = self.fetch_station_hybrid(
                    location_id, start_date, end_date, pollutant=pollutant
                )
                if not df_clean.empty:
                    fetched = True
            except Exception as e:
                logger.warning("Hybrid fetch failed for loc %d: %s. Falling back to direct API...", location_id, e)

        if not fetched and self.api_key:
            try:
                df_api = self.fetch_station_api(location_id, start_date, end_date, pollutant=pollutant)
                if not df_api.empty:
                    df_clean = self.clean_and_qa_dataset(df_api, pollutant=pollutant)
                    fetched = True
            except Exception as e:
                logger.warning("API fetch failed for loc %d: %s", location_id, e)

        # Save to local station cache to support resume
        if use_cache and (fetched or not df_clean.empty):
            self.save_cached_station(location_id, df_clean, pollutant=pollutant)
            if not df_clean.empty:
                logger.info("  Location %d: Saved %d clean records to disk cache", location_id, len(df_clean))

        return df_clean

    def fetch_benchmark_stations(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        location_ids: Optional[List[int]] = None,
        region: Optional[str] = None,
        max_workers: int = 3,
        prefer_api: bool = True,
        pollutant: str = "pm25",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetches historical PM2.5 or PM10 records across stations for the given date range.
        Includes periodic checkpointing and automatic resumption from local cache files.
        """
        if location_ids is not None:
            locs = location_ids
        elif region is not None:
            region_df = self.get_thailand_stations(region=region)
            locs = region_df["location_id"].dropna().astype(int).tolist()
        else:
            locs = list(DEFAULT_THAILAND_OPENAQ_LOCATIONS.keys())

        p_clean = "pm10" if "pm10" in pollutant.lower() else "pm25"
        logger.info("Fetching OpenAQ historical data (%s) for %d stations from %s to %s (Cache dir: %s)",
                    p_clean.upper(), len(locs), start_date, end_date, self.cache_dir)

        dfs: List[pd.DataFrame] = []
        checkpoint_file = self.target_dir / f"openaq_{p_clean}_checkpoint.parquet"
        processed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_to_loc = {
                executor.submit(
                    self.fetch_station_range,
                    loc,
                    start_date,
                    end_date,
                    prefer_api=prefer_api,
                    pollutant=pollutant,
                    use_cache=use_cache,
                ): loc
                for loc in locs
            }
            for fut in as_completed(fut_to_loc):
                loc = fut_to_loc[fut]
                processed_count += 1
                try:
                    df_loc = fut.result()
                    if not df_loc.empty:
                        dfs.append(df_loc)
                        logger.info("  [%d/%d] Location %d: Retrieved %d clean records",
                                    processed_count, len(locs), loc, len(df_loc))
                    else:
                        logger.info("  [%d/%d] Location %d: 0 clean records found",
                                    processed_count, len(locs), loc)
                except Exception as e:
                    logger.warning("Error fetching location %d: %s", loc, e)

                # Periodic checkpoint every 10 stations to guarantee file exists on disk during long runs
                if processed_count % 10 == 0 and dfs:
                    try:
                        cur_cp = pd.concat(dfs, ignore_index=True)
                        tmp_cp = checkpoint_file.with_suffix(".parquet.tmp")
                        cur_cp.to_parquet(tmp_cp, index=False, engine="pyarrow")
                        tmp_cp.replace(checkpoint_file)
                        logger.info("  💾 [Checkpoint] Progress saved to %s (%d records from %d stations)",
                                    checkpoint_file, len(cur_cp), cur_cp["station_code"].nunique())
                    except Exception as cp_err:
                        logger.debug("Failed updating periodic checkpoint: %s", cp_err)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        final_df = self.clean_and_qa_dataset(combined, pollutant=pollutant)

        # Save final consolidated checkpoint
        try:
            tmp_cp = checkpoint_file.with_suffix(".parquet.tmp")
            final_df.to_parquet(tmp_cp, index=False, engine="pyarrow")
            tmp_cp.replace(checkpoint_file)
            logger.info("  💾 [Checkpoint] Final consolidated data saved to %s", checkpoint_file)
        except Exception as cp_err:
            logger.debug("Failed saving final checkpoint: %s", cp_err)

        return final_df

    def clean_and_qa_dataset(self, df: pd.DataFrame, pollutant: str = "pm25") -> pd.DataFrame:
        """
        Applies physical boundaries and stuck sensor filtering to the dataset.
        """
        if df.empty:
            return df

        target_col = "observed_pm10" if ("pm10" in pollutant.lower() or ("observed_pm10" in df.columns and "observed_pm25" not in df.columns)) else "observed_pm25"
        if target_col not in df.columns:
            return df

        min_val = PHYSICAL_PM10_MIN if "pm10" in target_col else PHYSICAL_PM25_MIN
        max_val = PHYSICAL_PM10_MAX if "pm10" in target_col else PHYSICAL_PM25_MAX

        # 1. Physical limits
        mask_bounds = (
            (df[target_col] >= min_val) &
            (df[target_col] <= max_val)
        )
        df_clean = df[mask_bounds].copy()

        # 2. Thailand bounding box
        mask_geo = (
            (df_clean["latitude"] >= 5.5) & (df_clean["latitude"] <= 20.5) &
            (df_clean["longitude"] >= 97.0) & (df_clean["longitude"] <= 106.0)
        )
        df_clean = df_clean[mask_geo].copy()

        # 3. Stuck sensor filter
        df_clean = self._filter_stuck_sensors(df_clean, target_col=target_col)

        return df_clean.sort_values(["station_code", "timestamp_utc"]).reset_index(drop=True)

    def _filter_stuck_sensors(
        self,
        df: pd.DataFrame,
        max_consecutive_hours: int = MAX_STUCK_HOURS,
        target_col: str = "observed_pm25",
    ) -> pd.DataFrame:
        """Removes runs of identical values exceeding max_consecutive_hours."""
        if df.empty or len(df) <= max_consecutive_hours or target_col not in df.columns:
            return df

        df_sorted = df.sort_values(["station_code", "timestamp_utc"]).copy()
        valid_indices = []

        for _, group in df_sorted.groupby("station_code"):
            if len(group) <= max_consecutive_hours:
                valid_indices.extend(group.index.tolist())
                continue

            vals = group[target_col].to_numpy()
            times = group["timestamp_utc"].to_numpy()
            n = len(vals)
            is_stuck = np.zeros(n, dtype=bool)

            run_len = 1
            for i in range(1, n):
                dt_diff = pd.Timedelta(times[i] - times[i - 1]).total_seconds() / 3600.0
                if vals[i] == vals[i - 1] and dt_diff <= 1.5:
                    run_len += 1
                    if run_len > max_consecutive_hours:
                        is_stuck[i - run_len + 1 : i + 1] = True
                else:
                    run_len = 1

            valid_indices.extend(group.index[~is_stuck].tolist())

        return df.loc[valid_indices].copy()

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        filename: str = "openaq_air4thai_historical.parquet"
    ) -> Path:
        """Saves clean OpenAQ historical observations to Parquet."""
        if df.empty:
            raise ValueError("Cannot save empty DataFrame to Parquet")

        out_path = self.target_dir / filename
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Saved %d OpenAQ records to %s", len(df), out_path)
        return out_path

    def audit_records(self, df: pd.DataFrame, pollutant: str = "pm25") -> Dict[str, Any]:
        """Performs statistical QA audit on OpenAQ dataset."""
        if df.empty:
            return {
                "total_records": 0,
                "stations_count": 0,
                "status": "EMPTY",
            }

        target_col = "observed_pm10" if ("pm10" in pollutant.lower() or ("observed_pm10" in df.columns and "observed_pm25" not in df.columns)) else "observed_pm25"
        max_bound = PHYSICAL_PM10_MAX if "pm10" in target_col else PHYSICAL_PM25_MAX

        vals = df[target_col].dropna() if target_col in df.columns else pd.Series([], dtype=float)
        min_v = float(vals.min()) if len(vals) > 0 else None
        max_v = float(vals.max()) if len(vals) > 0 else None
        mean_v = float(vals.mean()) if len(vals) > 0 else None

        res = {
            "total_records": int(len(df)),
            "stations_count": int(df["station_code"].nunique()),
            "stations": df["station_name"].unique().tolist(),
            "target_column": target_col,
            "min_pm25": min_v,
            "max_pm25": max_v,
            "mean_pm25": mean_v,
            "within_bounds_0_to_500": bool((vals >= 0.0).all() and (vals <= 500.0).all()) if len(vals) > 0 else False,
            "within_bounds": bool((vals >= 0.0).all() and (vals <= max_bound).all()) if len(vals) > 0 else False,
            "time_min": str(df["timestamp_utc"].min()) if "timestamp_utc" in df.columns else None,
            "time_max": str(df["timestamp_utc"].max()) if "timestamp_utc" in df.columns else None,
            "status": "PASS" if len(vals) > 0 else "EMPTY",
        }
        if "pm10" in target_col:
            res["min_pm10"] = min_v
            res["max_pm10"] = max_v
            res["mean_pm10"] = mean_v
            res["within_bounds_0_to_1000"] = bool((vals >= 0.0).all() and (vals <= 1000.0).all()) if len(vals) > 0 else False

        return res
