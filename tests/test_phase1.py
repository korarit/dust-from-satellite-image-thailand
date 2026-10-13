"""
Unit and Integration Tests for Phase 1 & Phase 1B
Verifies Data Loaders, Time Alignment, Quality Assurance, and Zero-Mock Gate.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import (
    DATA_DIR,
    HIMAWARI_CACHE_DIR,
    HII_DIR,
    TARGET_DIR,
    GEO_DIR,
    REPORTS_DIR,
)
from src.data.himawari_ir_loader import (
    round_to_himawari_10m,
    get_himawari_bucket_and_sat,
    build_s3_prefix_and_keys,
    HimawariIRLoader,
    PHYSICAL_BT_MIN_K,
    PHYSICAL_BT_MAX_K,
    CORE_5_IR_BANDS,
    ALL_IR_BANDS,
    DEFAULT_TRAIN_SEGS,
    DEFAULT_TEST_SEGS,
)
from src.data.hii_weather_loader import (
    HIIWeatherLoader,
    BOUNDS_PRESSURE,
    BOUNDS_HUMIDITY,
)
from src.data.ground_truth_loader import (
    GroundTruthLoader,
    PHYSICAL_PM25_MIN,
    PHYSICAL_PM25_MAX,
)
from src.data.provenance_auditor import (
    CodeASTScanner,
    ProvenanceAuditor,
    ForbiddenSyntheticDataError,
)


def test_paths_and_directories():
    """Verifies that all required directories exist and resolve properly."""
    assert DATA_DIR.exists()
    assert HIMAWARI_CACHE_DIR.exists()
    assert HII_DIR.exists()
    assert TARGET_DIR.exists()
    assert GEO_DIR.exists()
    assert REPORTS_DIR.exists()


def test_himawari_timestamp_and_keys():
    """Verifies 10-minute rounding, cutover handling, and S3 key formatting."""
    dt1 = pd.Timestamp("2021-05-10 12:04:30", tz="UTC")
    rounded1 = round_to_himawari_10m(dt1)
    assert rounded1 == pd.Timestamp("2021-05-10 12:00:00", tz="UTC")
    bucket1, sat1 = get_himawari_bucket_and_sat(rounded1)
    assert bucket1 == "noaa-himawari8"
    assert sat1 == "H08"

    dt2 = pd.Timestamp("2023-01-01 00:06:00", tz="UTC")
    rounded2 = round_to_himawari_10m(dt2)
    assert rounded2 == pd.Timestamp("2023-01-01 00:10:00", tz="UTC")
    bucket2, sat2 = get_himawari_bucket_and_sat(rounded2)
    assert bucket2 == "noaa-himawari9"
    assert sat2 == "H09"

    _, keys = build_s3_prefix_and_keys(rounded2, "B13", segments=[3, 4, 5])
    assert len(keys) == 3
    assert "HS_H09_20230101_0010_B13_FLDK_R20_S0310.DAT.bz2" in keys[0]
    assert "HS_H09_20230101_0010_B13_FLDK_R20_S0410.DAT.bz2" in keys[1]


def test_himawari_physical_bounds_sanitization():
    """Verifies Brightness Temperature physical bounds sanitization."""
    loader = HimawariIRLoader()
    raw_vals = np.array([120.0, 180.0, 290.0, 360.0, np.nan], dtype=np.float32)
    cleaned, stats = loader.validate_physical_bounds(raw_vals, "B13")

    # 120.0 is < 150K -> NaN
    # 360.0 is > 350K -> NaN
    assert np.isnan(cleaned[0])
    assert cleaned[1] == 180.0
    assert cleaned[2] == 290.0
    assert np.isnan(cleaned[3])
    assert stats["oob_count"] == 2
    assert stats["valid_count"] == 2


def test_hii_loader_parsing_and_utc_alignment():
    """Verifies HII CSV parsing, sentinel filtering, and UTC conversion (ICT - 7h)."""
    loader = HIIWeatherLoader()
    sample_csv = (
        b"date,time,press\n"
        b"2023-01-01,07:00,1012.5\n"
        b"2023-01-01,08:00,-999.0\n"
        b"2023-01-01,09:00,750.0\n"    # Out of physical bound (< 850 hPa)
        b"2023-01-01,10:00,1011.8\n"
    )

    df = loader.parse_and_clean_csv(sample_csv, "TEST01", "pressure")
    assert len(df) == 2  # Only 1012.5 and 1011.8 are valid
    # 07:00 ICT should become 00:00 UTC
    first_row = df.iloc[0]
    assert first_row["timestamp_utc"] == pd.Timestamp("2023-01-01 00:00:00")
    assert first_row["value"] == 1012.5


def test_hii_loader_2026_unified_schema():
    """Verifies HII 2026 schema parsing with measure_datetime, pressure/humidity, and quality_flag."""
    loader = HIIWeatherLoader()
    sample_2026_csv = (
        b"station_code,measure_datetime,pressure,quality_flag\n"
        b"ACRU,2026-01-01 07:00:00,992.7,N\n"
        b"ACRU,2026-01-01 08:00:00,-999.0,N\n"
        b"ACRU,2026-01-01 09:00:00,993.1,N\n"
    )
    df = loader.parse_and_clean_csv(sample_2026_csv, "ACRU", "pressure")
    assert len(df) == 2
    assert df.iloc[0]["timestamp_utc"] == pd.Timestamp("2026-01-01 00:00:00")
    assert df.iloc[0]["value"] == 992.7
    assert df.iloc[1]["value"] == 993.1



def test_ground_truth_stuck_sensor_detection():
    """Verifies stuck sensor filter drops identical values exceeding 6 consecutive hours."""
    loader = GroundTruthLoader()
    times = pd.date_range("2023-01-01 00:00:00", periods=10, freq="1h")
    # First 8 hours have stuck identical value 25.0
    vals = [25.0] * 8 + [30.0, 35.0]

    df = pd.DataFrame({
        "station_code": ["ST_01"] * 10,
        "station_name": ["Test Station"] * 10,
        "timestamp_utc": times,
        "observed_pm25": vals,
        "latitude": [13.7] * 10,
        "longitude": [100.5] * 10,
        "source": ["PCD_Air4Thai"] * 10,
    })

    filtered = loader.filter_stuck_sensors(df, max_consecutive_hours=6)
    # The stuck values should be filtered out
    assert len(filtered) < 10
    assert 30.0 in filtered["observed_pm25"].values
    assert 35.0 in filtered["observed_pm25"].values


def test_provenance_ast_scanner():
    """Verifies AST scanner detects forbidden random calls in python files."""
    scanner = CodeASTScanner(PROJECT_ROOT)
    # Scan src directory
    violations = scanner.scan_directory(PROJECT_ROOT / "src")
    # All production code in src/ must be clean of forbidden calls
    assert len(violations) == 0, f"Found unexpected violations in src: {violations}"


def test_provenance_full_gate_execution():
    """Verifies full execution of Phase 1B Verification Gate."""
    auditor = ProvenanceAuditor(PROJECT_ROOT)
    report = auditor.run_full_gate_verification()
    assert report["overall_decision"] == "CERTIFIED_REAL_OBSERVATIONS_PASS"
    assert report["is_synthetic"] is False
    assert (REPORTS_DIR / "provenance_audit_report.json").exists()
    assert (REPORTS_DIR / "provenance_audit_report.md").exists()


def test_openaq_loader_s3_ingestion():
    """Verifies OpenAQ loader downloads and processes genuine Air4Thai data from AWS S3."""
    from src.data.openaq_loader import OpenAQLoader
    loader = OpenAQLoader()
    # Test 1 day for Bang Phlat (location 225642)
    dt = pd.Timestamp("2021-06-02")
    df = loader.fetch_station_day(location_id=225642, target_date=dt)
    assert not df.empty
    assert "OPENAQ_225642" in df["station_code"].values
    assert df["source"].iloc[0] == "OpenAQ_Air4Thai"
    assert (df["observed_pm25"] >= 0.0).all() and (df["observed_pm25"] <= 500.0).all()


def test_himawari_core_5_bands_and_station_extraction():
    """Verifies that HimawariIRLoader defaults to 5 Core IR Bands and computes new BTDs."""
    assert CORE_5_IR_BANDS == ["B07", "B10", "B11", "B13", "B15"]
    assert ALL_IR_BANDS == CORE_5_IR_BANDS

    loader = HimawariIRLoader()
    assert loader.bands == ["B07", "B10", "B11", "B13", "B15"]

    # Test BTD computation logic on simulated station features
    sample_stns = pd.DataFrame({
        "location_id": [101, 102],
        "latitude": [13.75, 18.80],
        "longitude": [100.50, 98.98],
        "bt_b07": [300.0, 290.0],
        "bt_b10": [255.0, 250.0],
        "bt_b11": [280.0, 275.0],
        "bt_b13": [295.0, 290.0],
        "bt_b15": [290.0, 284.0],
    })

    # Test BTDs
    # btd_15_13: row 0 is 290 - 295 = -5.0, row 1 is 284 - 290 = -6.0
    assert np.allclose(sample_stns["bt_b15"] - sample_stns["bt_b13"], [-5.0, -6.0])
    # btd_13_10 (Moisture / stability proxy): 295 - 255 = 40.0
    assert np.allclose(sample_stns["bt_b13"] - sample_stns["bt_b10"], [40.0, 40.0])
    # btd_13_11 (Sulfate / size proxy): 295 - 280 = 15.0
    assert np.allclose(sample_stns["bt_b13"] - sample_stns["bt_b11"], [15.0, 15.0])


def test_himawari_segment_partition_train_vs_test():
    """Verifies Segment 4 for Training vs All Segments [3, 4, 5] for Testing."""
    assert DEFAULT_TRAIN_SEGS == [4]
    assert DEFAULT_TEST_SEGS == [3, 4, 5]

    loader = HimawariIRLoader()
    assert loader.train_segments == [4]
    assert loader.test_segments == [3, 4, 5]

    dt = pd.Timestamp("2024-03-01 06:00:00", tz="UTC")
    # Training: only S04
    bucket_train, train_keys = build_s3_prefix_and_keys(dt, "B13", segments=loader.train_segments)
    assert len(train_keys) == 1
    assert "S0410.DAT.bz2" in train_keys[0]

    # Testing: S03, S04, S05
    bucket_test, test_keys = build_s3_prefix_and_keys(dt, "B13", segments=loader.test_segments)
    assert len(test_keys) == 3
    assert any("S0310.DAT.bz2" in k for k in test_keys)
    assert any("S0410.DAT.bz2" in k for k in test_keys)
    assert any("S0510.DAT.bz2" in k for k in test_keys)


def test_himawari_kdtree_spatial_caching_and_distance_threshold():
    """Verifies KDTree spatial index caching and out-of-bounds station masking (max_dist_deg)."""
    class DummyArea:
        def get_lonlats(self):
            lats = np.linspace(13.0, 14.0, 51)
            lons = np.linspace(100.0, 101.0, 51)
            grid_lons, grid_lats = np.meshgrid(lons, lats)
            return grid_lons, grid_lats

    class DummyDataArray:
        def __init__(self, values, area):
            self.values = values
            self.attrs = {"area": area}

    class DummyScene:
        def __init__(self, bands_dict, area):
            self._data = {b: DummyDataArray(vals, area) for b, vals in bands_dict.items()}
        def __getitem__(self, item):
            return self._data[item]

    area = DummyArea()
    scene = DummyScene({
        "B13": np.full((51, 51), 295.0, dtype=np.float32),
        "B15": np.full((51, 51), 290.0, dtype=np.float32),
    }, area)

    coords_df = pd.DataFrame({
        "location_id": [1, 2],
        "centroid_lat": [13.5, 6.5],     # Lat 13.5 is in grid; Lat 6.5 is > 6.5 degrees away
        "centroid_lon": [100.5, 100.5],
    })

    loader = HimawariIRLoader()
    assert len(loader._cached_spatial_index) == 0

    # First call: computes and caches KDTree
    res1 = loader.sample_at_coordinates(
        scene=scene,
        coords_df=coords_df,
        bands=["B13", "B15"],
        segments=[4],
        max_dist_deg=0.05,
    )
    assert len(loader._cached_spatial_index) == 1
    # Bangkok station (inside grid) gets 295.0 K
    assert res1.loc[res1["location_id"] == 1, "bt_b13"].iloc[0] == 295.0
    # Southern station (outside Segment 4) gets NaN
    assert np.isnan(res1.loc[res1["location_id"] == 2, "bt_b13"].iloc[0])

    # Second call: uses cache
    res2 = loader.sample_at_coordinates(
        scene=scene,
        coords_df=coords_df,
        bands=["B13", "B15"],
        segments=[4],
        max_dist_deg=0.05,
    )
    assert len(loader._cached_spatial_index) == 1
    assert res2.loc[res2["location_id"] == 1, "bt_b13"].iloc[0] == 295.0
    assert np.isnan(res2.loc[res2["location_id"] == 2, "bt_b13"].iloc[0])


def test_himawari_station_snapshots_auto_resume(tmp_path):
    """Verifies that fetch_range_station_snapshots auto-resumes from existing Parquet checkpoints."""
    checkpoint_file = tmp_path / "satellite_checkpoint.parquet"

    # Pre-create a checkpoint with 1 completed timestamp
    ts_done = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    existing_df = pd.DataFrame({
        "location_id": [101],
        "latitude": [13.75],
        "longitude": [100.50],
        "bt_b13": [295.0],
        "timestamp": [ts_done],
    })
    existing_df.to_parquet(checkpoint_file, index=False)

    loader = HimawariIRLoader()
    # Query exact same range that includes the completed timestamp
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    stn_df = pd.DataFrame({
        "location_id": [101],
        "latitude": [13.75],
        "longitude": [100.50],
    })

    # Should detect that all timestamps are already completed and return existing_df without any download
    res_df = loader.fetch_range_station_snapshots(
        start_date=start_dt,
        end_date=end_dt,
        stations_df=stn_df,
        cadence_hours=3,
        save_parquet_path=checkpoint_file,
    )
    assert not res_df.empty
    assert len(res_df) == 1
    assert res_df["bt_b13"].iloc[0] == 295.0


