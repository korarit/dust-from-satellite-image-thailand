"""
Unit and Integration Tests for Phase 3: Unified Feature Pipeline & Anti-Leakage Protocol
Verifies:
- Temporal Anti-Leakage Validator (Strict Zero Look-Ahead Bias)
- Spatial Anti-Leakage Validator (Strict Proximity Exclusion & Target Segregation)
- Spatial-Temporal Target PM2.5 Matcher
- Deterministic Physics Fallback Imputer (Zero Mock, Zero Random)
- Unified Feature Matrix Assembly across 52+ features + keys + target
- Apache Parquet storage, Snappy compression, and Columnar Projection
- Multi-Year Train/Test Splitting (2024-2025 Train vs 2026 Out-of-Time Test)
- CodeASTScanner Compliance (0 violations across entire src/ tree)
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

from src.config.paths import FEATURES_DIR, REPORTS_DIR
from src.features.feature_matrix_builder import (
    FeatureMatrixBuilder,
    TemporalAntiLeakageValidator,
    SpatialAntiLeakageValidator,
    TemporalAntiLeakageError,
    SpatialAntiLeakageError,
    TargetFeatureIsolationError,
    PhysicsFallbackImputer,
    TargetMatcher,
    KEY_COLUMNS,
    SAT_BT_COLUMNS,
    SAT_BTD_COLUMNS,
    SAT_TEXTURE_COLUMNS,
    SAT_TEMPORAL_COLUMNS,
    WEATHER_PRESSURE_COLUMNS,
    WEATHER_HUMIDITY_COLUMNS,
    TOPO_SOLAR_COLUMNS,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.features.grid_engine import MultiResolutionGridEngine
from src.data.provenance_auditor import CodeASTScanner


def test_temporal_anti_leakage_validator():
    """
    Verifies that observations strictly in the future relative to the run timestamp
    raise TemporalAntiLeakageError, while valid past/present timestamps pass.
    """
    run_dt = pd.Timestamp("2024-03-15 12:00:00", tz="UTC")

    # 1. Past observation passes
    past_dt = pd.Timestamp("2024-03-15 11:50:00", tz="UTC")
    assert TemporalAntiLeakageValidator.validate_timestamp(past_dt, run_dt, "Past Obs")

    # 2. Present observation passes
    assert TemporalAntiLeakageValidator.validate_timestamp(run_dt, run_dt, "Current Obs")

    # 3. Future observation raises TemporalAntiLeakageError
    future_dt = pd.Timestamp("2024-03-15 12:10:00", tz="UTC")
    with pytest.raises(TemporalAntiLeakageError):
        TemporalAntiLeakageValidator.validate_timestamp(future_dt, run_dt, "Future Obs")

    # 4. DataFrame batch check
    df_valid = pd.DataFrame({"timestamp": [past_dt, run_dt]})
    assert TemporalAntiLeakageValidator.validate_dataframe(df_valid, run_dt)

    df_invalid = pd.DataFrame({"timestamp": [past_dt, future_dt]})
    with pytest.raises(TemporalAntiLeakageError):
        TemporalAntiLeakageValidator.validate_dataframe(df_invalid, run_dt)


def test_spatial_anti_leakage_validator():
    """
    Verifies that weather stations within the exclusion radius (< 2.0 km)
    trigger SpatialAntiLeakageError, and target feature isolation is strictly enforced.
    """
    cell_utm_km = np.array([650.0, 1500.0])
    exclusion_radius_km = 2.0

    # Stations at various distances
    station_coords_km = np.array([
        [650.0, 1501.2],  # Distance = 1.2 km (< 2.0 km -> BREACH)
        [650.0, 1504.0],  # Distance = 4.0 km (Valid)
        [655.0, 1500.0],  # Distance = 5.0 km (Valid)
    ])

    # Case 1: Valid mask only using stations >= 2.0 km
    valid_mask = np.array([False, True, True])
    assert SpatialAntiLeakageValidator.verify_exclusion(
        cell_utm_km, station_coords_km, valid_mask, exclusion_radius_km
    )

    # Case 2: Breach when using the station at 1.2 km
    invalid_mask = np.array([True, True, True])
    with pytest.raises(SpatialAntiLeakageError):
        SpatialAntiLeakageValidator.verify_exclusion(
            cell_utm_km, station_coords_km, invalid_mask, exclusion_radius_km
        )

    # Case 3: Target feature isolation
    safe_features = ["bt_b13", "p_north", "elevation_m"]
    assert SpatialAntiLeakageValidator.verify_target_feature_isolation(safe_features)

    leaked_features = ["bt_b13", "observed_pm25", "elevation_m"]
    with pytest.raises(TargetFeatureIsolationError):
        SpatialAntiLeakageValidator.verify_target_feature_isolation(leaked_features)


def test_target_matcher_spatial_proximity():
    """
    Verifies KDTree matching of ground truth PM2.5 stations to grid cells
    and filtering of training cells.
    """
    # 3 Grid cells: Bangkok area
    grid_df = pd.DataFrame({
        "grid_id": ["G1", "G2", "G3"],
        "centroid_lat": [13.75, 13.80, 14.20],
        "centroid_lon": [100.50, 100.55, 101.00],
    })

    # Ground truth station near G1 (within 0.5 km)
    gt_df = pd.DataFrame({
        "station_code": ["PCD_01", "PCD_FAR"],
        "latitude": [13.752, 18.00],  # PCD_01 is ~0.2 km from G1; PCD_FAR is far away in Chiang Mai
        "longitude": [100.501, 99.00],
        "observed_pm25": [45.5, 90.0],
    })

    matcher = TargetMatcher()
    matched = matcher.attach_ground_truth(grid_df, gt_df, max_distance_km=2.0)

    # G1 must receive 45.5
    assert not np.isnan(matched.loc[matched["grid_id"] == "G1", "observed_pm25"].iloc[0])
    assert np.isclose(matched.loc[matched["grid_id"] == "G1", "observed_pm25"].iloc[0], 45.5)

    # G2 and G3 must remain NaN (no station within 2km)
    assert np.isnan(matched.loc[matched["grid_id"] == "G2", "observed_pm25"].iloc[0])
    assert np.isnan(matched.loc[matched["grid_id"] == "G3", "observed_pm25"].iloc[0])

    # Filter training cells: should only have 1 row (G1)
    train_cells = matcher.filter_training_cells(matched)
    assert len(train_cells) == 1
    assert train_cells["grid_id"].iloc[0] == "G1"


def test_physics_fallback_imputer():
    """
    Verifies deterministic physics fallback imputation:
    - Fills NaNs without random numbers
    - Detects deep convective clouds and sets cloud_occluded_flag
    - Recomputes BTDs
    - Sets imputed_flag correctly
    """
    df = pd.DataFrame({
        "grid_id": ["G_CLEAR", "G_CLOUD", "G_MISSING"],
        "bt_b13": [295.0, 210.0, np.nan],       # G_CLOUD has BT13 < 220 K (deep cloud)
        "bt_b15": [293.0, 212.0, np.nan],
        "p_north": [1014.0, np.nan, 1012.0],
        "pressure_mean_surround": [1013.0, 1013.0, 1013.0],
        "humidity_mean_surround": [75.0, 90.0, np.nan],
    })

    imputer = PhysicsFallbackImputer()
    imputed = imputer.impute_features(df)

    # 1. Zero NaNs in imputed columns
    for col in ALL_FEATURE_COLUMNS:
        if col in imputed.columns:
            assert not imputed[col].isna().any(), f"Column {col} still has NaNs after imputation!"

    # 2. Cloud Occluded Flag
    # G_CLEAR: 0, G_CLOUD: 1 (BT13 = 210 < 220), G_MISSING: 1 (NaN BT13)
    assert imputed.loc[imputed["grid_id"] == "G_CLEAR", "cloud_occluded_flag"].iloc[0] == 0
    assert imputed.loc[imputed["grid_id"] == "G_CLOUD", "cloud_occluded_flag"].iloc[0] == 1
    assert imputed.loc[imputed["grid_id"] == "G_MISSING", "cloud_occluded_flag"].iloc[0] == 1

    # 3. Imputed flag
    assert imputed.loc[imputed["grid_id"] == "G_CLOUD", "imputed_flag"].iloc[0] == 1
    assert imputed.loc[imputed["grid_id"] == "G_MISSING", "imputed_flag"].iloc[0] == 1

    # 4. BTD recomputed: G_CLEAR btd_15_13 = 293 - 295 = -2.0
    assert np.isclose(imputed.loc[imputed["grid_id"] == "G_CLEAR", "btd_15_13"].iloc[0], -2.0)


def test_build_snapshot_matrix_end_to_end():
    """
    Verifies end-to-end matrix assembly with standard keys, features, and float32 precision.
    """
    target_dt = pd.Timestamp("2024-03-21 05:00:00", tz="UTC")

    # Sample Grid
    grid_df = pd.DataFrame({
        "grid_id": ["G_TEST_01", "G_TEST_02"],
        "centroid_lat": [13.75, 14.00],
        "centroid_lon": [100.50, 100.60],
        "utm_x": [662000.0, 673000.0],
        "utm_y": [1520000.0, 1548000.0],
        "elevation_m": [15.0, 25.0],
        "elevation_std": [2.0, 3.0],
        "slope_deg": [0.5, 1.2],
        "aspect_sin": [0.0, 0.5],
        "aspect_cos": [1.0, 0.866],
        "dist_coast_km": [25.0, 50.0],
        "coriolis_param": [3.4e-5, 3.5e-5],
    })

    # Sample Satellite Features
    sat_df = pd.DataFrame({
        "grid_id": ["G_TEST_01", "G_TEST_02"],
        "bt_b13": [298.0, 296.0],
        "bt_b15": [295.0, 292.0],
        "btd_15_13": [-3.0, -4.0],
        "dust_rgb_red": [0.166, 0.0],
        "b13_mean_3x3": [298.0, 296.0],
        "b13_std_3x3": [0.4, 0.5],
    })

    # Sample Weather Features
    weather_df = pd.DataFrame({
        "grid_id": ["G_TEST_01", "G_TEST_02"],
        "p_north": [1014.0, 1015.0],
        "p_south": [1011.0, 1012.0],
        "p_east": [1013.0, 1013.5],
        "p_west": [1012.5, 1013.0],
        "delta_p_ns": [3.0, 3.0],
        "delta_p_ew": [0.5, 0.5],
        "pressure_grad_mag": [1.2, 1.2],
        "pressure_mean_surround": [1012.6, 1013.4],
        "humidity_mean_surround": [72.0, 70.0],
        "humidity_std_surround": [4.0, 3.5],
        "humidity_grad_mag": [0.8, 0.7],
    })

    # Sample PM2.5 Target
    gt_df = pd.DataFrame({
        "latitude": [13.751],
        "longitude": [100.502],
        "observed_pm25": [38.5],
    })

    builder = FeatureMatrixBuilder()
    matrix = builder.build_snapshot_matrix(
        target_dt=target_dt,
        grid_df=grid_df,
        satellite_df=sat_df,
        weather_df=weather_df,
        ground_truth_df=gt_df,
        resolution_km=2.0,
        impute_missing=True,
    )

    # 1. Standard Keys present
    for k in KEY_COLUMNS:
        assert k in matrix.columns, f"Key column {k} missing!"

    # 2. Solar and temporal features attached
    for st in ["solar_zenith_angle", "is_daytime", "hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert st in matrix.columns, f"Solar/Temporal feature {st} missing!"

    # 3. Target present and matched for G_TEST_01
    assert TARGET_COLUMN in matrix.columns
    assert np.isclose(matrix.loc[matrix["grid_id"] == "G_TEST_01", TARGET_COLUMN].iloc[0], 38.5)

    # 4. Check feature column data types are float32
    for fc in ALL_FEATURE_COLUMNS:
        if fc in matrix.columns:
            assert matrix[fc].dtype == np.float32, f"Feature {fc} is not float32!"


def test_parquet_storage_and_columnar_projection(tmp_path: Path):
    """
    Verifies Parquet serialization with Snappy compression and fast columnar loading.
    """
    df = pd.DataFrame({
        "grid_id": ["G1", "G2", "G3"],
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"),
        "lat": [13.7, 13.8, 13.9],
        "lon": [100.5, 100.6, 100.7],
        "utm_x": [660000.0, 670000.0, 680000.0],
        "utm_y": [1520000.0, 1530000.0, 1540000.0],
        "bt_b13": [295.0, 296.0, 297.0],
        "p_north": [1014.0, 1013.0, 1012.0],
        "observed_pm25": [32.0, 45.0, 28.0],
    })

    builder = FeatureMatrixBuilder(output_dir=tmp_path)
    file_path = tmp_path / "test_matrix.parquet"
    builder.save_matrix(df, file_path)

    assert file_path.exists()
    assert file_path.stat().st_size > 0

    # Columnar load of specific columns
    loaded = builder.load_matrix(file_path, columns=["grid_id", "bt_b13", "observed_pm25"])
    assert list(loaded.columns) == ["grid_id", "bt_b13", "observed_pm25"]
    assert len(loaded) == 3


def test_temporal_train_test_split():
    """
    Verifies System Invariant #7:
    Train Split: 2024-2025
    Out-of-Time Test Split: 2026
    """
    df = pd.DataFrame({
        "grid_id": [f"G_{i}" for i in range(5)],
        "timestamp": pd.to_datetime([
            "2024-01-15 00:00:00",
            "2024-08-20 12:00:00",
            "2025-12-31 23:00:00",
            "2026-01-01 00:00:00",
            "2026-06-15 06:00:00",
        ], utc=True),
        "observed_pm25": [25.0, 30.0, 35.0, 40.0, 45.0],
    })

    train_df, test_df = FeatureMatrixBuilder.split_train_test(df, split_date="2026-01-01")

    assert len(train_df) == 3
    assert len(test_df) == 2
    assert (pd.to_datetime(train_df["timestamp"]).dt.year < 2026).all()
    assert (pd.to_datetime(test_df["timestamp"]).dt.year >= 2026).all()


def test_ast_scanner_zero_violations_phase3():
    """
    Verifies that the entire src/ tree, including new Phase 3 features,
    has 0 violations in the AST Code Scanner.
    """
    scanner = CodeASTScanner(PROJECT_ROOT)
    violations = scanner.scan_directory(PROJECT_ROOT / "src")
    assert len(violations) == 0, f"Found forbidden calls or mock keywords in src: {violations}"
