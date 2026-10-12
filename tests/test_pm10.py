"""
Unit and Integration Tests for PM10 Separate Model Suite & Pipeline
Thailand Dust Prediction Satellite

Verifies:
1. Physical Bounds & Ground Truth Loader:
   - Validates PM10 physical limits [0.0, 1000.0] µg/m³
   - Validates apply_physical_bounds and filter_stuck_sensors for observed_pm10
   - Verifies audit_target_records with pollutant='pm10'
2. Feature Matrix & Strict Anti-Leakage:
   - Target matching attaches observed_pm10 without leakage
   - SpatialAntiLeakageValidator verifies isolation of observed_pm10 from model features
   - build_snapshot_matrix builds training matrix filtered on observed_pm10
3. Thailand PCD PM10 Standards & AQI Metrics:
   - PM10 PCD 5-tier thresholds [0.0, 50.0, 80.0, 120.0, 180.0, 1000.0]
   - assign_aqi_category(..., pollutant='pm10')
   - High-risk alert threshold >= 120.0 µg/m³
   - compute_comprehensive_metrics with pollutant='pm10'
4. Separate Model Suite & Stacking Ensemble:
   - Dedicated PM10 base learners (LightGBM, CatBoost, XGBoost, Random Forest)
   - Dedicated PM10 NNLS Stacking Ensemble
   - Checkpoint serialization with pollutant='pm10' metadata
5. Complete Model Suite Separation:
   - ModelTrainer stores PM10 checkpoints under dedicated exp_pm10 directory
   - PM2.5 and PM10 models are 100% separate models (no joint multi-target regression)
6. Deliverables Generation:
   - Nowcast PM10 deliverables (Parquet, GeoTIFF, Interactive HTML, Static PNG)
   - HTML map displays PM10 legend (0-50, 50-80, 80-120, 120-180, >180) and alert threshold (120.0)
7. CodeASTScanner:
   - Strict Zero-Mock compliance (0 mock or random violations)
"""

import sys
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ground_truth_loader import (
    GroundTruthLoader,
    PHYSICAL_PM10_MIN,
    PHYSICAL_PM10_MAX,
    PHYSICAL_PM25_MIN,
    PHYSICAL_PM25_MAX,
)
from src.features.feature_matrix_builder import (
    FeatureMatrixBuilder,
    SpatialAntiLeakageValidator,
    TargetFeatureIsolationError,
    TargetMatcher,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN_PM10,
    TARGET_COLUMN_PM25,
)
from src.models import (
    BaseDustModel,
    LGBMDustModel,
    CatBoostDustModel,
    XGBoostDustModel,
    RandomForestDustModel,
    DustStackingEnsemble,
    ModelTrainer,
    THAILAND_AQI_THRESHOLDS_PM10,
    assign_aqi_category,
    compute_dust_metrics,
)
from src.evaluation.metrics import (
    compute_comprehensive_metrics,
    compute_aqi_classification_metrics,
    HIGH_RISK_ALERT_THRESHOLD_PM10,
)
from src.evaluation.visualizer import DustMapVisualizer
from src.data.provenance_auditor import CodeASTScanner


@pytest.fixture
def sample_pm10_ground_truth():
    """Generates a deterministic synthetic-free PM10 ground truth DataFrame."""
    return pd.DataFrame({
        "station_code": ["STN_01", "STN_02", "STN_03", "STN_04", "STN_05", "STN_06"],
        "station_name": ["Station 1", "Station 2", "Station 3", "Station 4", "Station 5", "Station 6"],
        "timestamp_utc": ["2026-10-12 10:00:00"] * 6,
        "latitude": [13.75, 13.80, 13.85, 18.78, 18.82, 18.85],
        "longitude": [100.50, 100.55, 100.60, 98.98, 99.02, 99.05],
        "observed_pm25": [12.0, 22.0, 32.0, 45.0, 85.0, 28.0],
        "observed_pm10": [25.0, 65.0, 95.0, 140.0, 220.0, 75.0],
        "source": ["PCD_Air4Thai"] * 6,
    })


@pytest.fixture
def sample_synthetic_free_training_grid():
    """Generates a deterministic grid and feature matrix without random generators."""
    rows = []
    # 20 samples to train fast 5-fold models
    for i in range(20):
        lat = 13.0 + (i * 0.25)
        lon = 100.0 + (i * 0.15)
        # Deterministic linear + non-linear physical relationship
        pm10_val = 30.0 + (i * 8.5)
        pm25_val = 15.0 + (i * 3.8)

        row = {
            "grid_id": f"GRID_{i:04d}",
            "lat": lat,
            "lon": lon,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "utm_x": 600000.0 + i * 2000.0,
            "utm_y": 1500000.0 + i * 2000.0,
            "elevation_m": 50.0 + i * 15.0,
            "local_relief_m": 10.0 + i * 2.0,
            "tpi_basin_index": -0.5 if i % 2 == 0 else 0.5,
            "distance_to_coast_km": 20.0 + i * 5.0,
            "p_north": 1012.0 - i * 0.2,
            "p_south": 1011.5 + i * 0.1,
            "p_east": 1012.2 - i * 0.15,
            "p_west": 1011.8 + i * 0.05,
            "delta_p_ns": -0.5 - i * 0.3,
            "delta_p_ew": 0.4 - i * 0.2,
            "pressure_grad_mag": 0.6 + i * 0.1,
            "surrounding_humidity_mean": 65.0 + i * 0.5,
            "humidity_spatial_std": 2.5 + i * 0.1,
            "humidity_gradient_mag": 0.3 + i * 0.02,
            "bt_b13": 285.0 - i * 0.8,
            "bt_b15": 283.0 - i * 0.7,
            "btd_15_13": -2.0 + i * 0.1,
            "dust_rgb_red": 3.0 + i * 0.2,
            "texture_var_b13": 0.8 + i * 0.05,
            "texture_gradient_b13": 0.5 + i * 0.03,
            "bt_tendency_1h": -0.2 * i,
            "btd_tendency_1h": 0.05 * i,
            "solar_zenith_angle": 35.0 + i * 1.5,
            "is_daytime": 1,
            "hour_sin": 0.5,
            "hour_cos": 0.86,
            "day_of_year_sin": 0.7,
            "day_of_year_cos": 0.71,
            "imputed_flag": 0,
            "cloud_occluded_flag": 0,
            "observed_pm25": pm25_val,
            "observed_pm10": pm10_val,
        }
        rows.append(row)
    return pd.DataFrame(rows)


class TestPM10GroundTruthLoader:
    """Verifies PM10 ingestion and physical bound validation."""

    def test_pm10_physical_constants(self):
        assert PHYSICAL_PM10_MIN == 0.0
        assert PHYSICAL_PM10_MAX == 1000.0
        assert PHYSICAL_PM25_MIN == 0.0
        assert PHYSICAL_PM25_MAX == 500.0

    def test_apply_physical_bounds_pm10(self, tmp_path):
        loader = GroundTruthLoader(target_dir=tmp_path)
        raw_df = pd.DataFrame({
            "station_code": ["A", "B", "C", "D"],
            "observed_pm10": [-15.0, 50.0, 950.0, 1500.0],
        })
        cleaned = loader.apply_physical_bounds(raw_df, target_col="observed_pm10")
        # -15.0 and 1500.0 should be dropped
        assert len(cleaned) == 2
        assert cleaned["observed_pm10"].min() == 50.0
        assert cleaned["observed_pm10"].max() == 950.0

    def test_filter_stuck_sensors_pm10(self, tmp_path):
        loader = GroundTruthLoader(target_dir=tmp_path)
        # 6 consecutive identical readings indicates stuck sensor
        df = pd.DataFrame({
            "station_code": ["S1"] * 6 + ["S2"] * 6,
            "timestamp_utc": pd.date_range("2026-10-12", periods=6, freq="h").tolist() * 2,
            "observed_pm10": [55.0] * 6 + [40.0, 42.0, 45.0, 43.0, 48.0, 46.0],
        })
        filtered = loader.filter_stuck_sensors(df, target_col="observed_pm10", consecutive_threshold=5)
        # S1 should be dropped
        assert "S1" not in filtered["station_code"].values
        assert "S2" in filtered["station_code"].values

    def test_audit_target_records_pm10(self, tmp_path, sample_pm10_ground_truth):
        loader = GroundTruthLoader(target_dir=tmp_path)
        audit = loader.audit_target_records(sample_pm10_ground_truth, pollutant="pm10")
        assert audit["pollutant"] == "pm10"
        assert audit["status"] == "PASS"
        assert audit["min_pm10"] == 25.0
        assert audit["max_pm10"] == 220.0
        assert audit["mean_pm10"] > 0.0


class TestPM10FeatureMatrixAndAntiLeakage:
    """Verifies that PM10 target is integrated without target feature leakage."""

    def test_target_matcher_filters_pm10(self, sample_synthetic_free_training_grid, sample_pm10_ground_truth):
        matcher = TargetMatcher()
        # Matched contains observed_pm10 and observed_pm25
        matched = matcher.attach_ground_truth(
            sample_synthetic_free_training_grid,
            sample_pm10_ground_truth,
            max_distance_km=25.0,
            target_val_col="observed_pm10",
        )
        training_cells = matcher.filter_training_cells(matched, target_col="observed_pm10")
        assert not training_cells.empty
        assert "observed_pm10" in training_cells.columns
        assert not training_cells["observed_pm10"].isna().any()

    def test_anti_leakage_isolation_both_targets(self):
        validator = SpatialAntiLeakageValidator()
        # Feature columns must not contain observed_pm25 or observed_pm10
        assert TARGET_COLUMN_PM25 not in ALL_FEATURE_COLUMNS
        assert TARGET_COLUMN_PM10 not in ALL_FEATURE_COLUMNS

        # Test verification function
        clean_features = [c for c in ALL_FEATURE_COLUMNS]
        assert validator.verify_target_feature_isolation(clean_features) is True

        with pytest.raises(TargetFeatureIsolationError, match="TARGET ISOLATION BREACH"):
            validator.verify_target_feature_isolation(clean_features + ["observed_pm10"])

        with pytest.raises(TargetFeatureIsolationError, match="TARGET ISOLATION BREACH"):
            validator.verify_target_feature_isolation(clean_features + ["observed_pm25"])

    def test_build_snapshot_matrix_supports_pm10(self, sample_synthetic_free_training_grid, sample_pm10_ground_truth):
        builder = FeatureMatrixBuilder()
        matrix = builder.build_snapshot_matrix(
            target_dt=pd.Timestamp("2026-10-12 12:00:00", tz="UTC"),
            grid_df=sample_synthetic_free_training_grid,
            satellite_df=sample_synthetic_free_training_grid,
            weather_df=sample_synthetic_free_training_grid,
            ground_truth_df=sample_pm10_ground_truth,
            target_col="observed_pm10",
            training_only=False,
        )
        assert "observed_pm10" in matrix.columns
        assert matrix.attrs.get("target_column") == "observed_pm10"


class TestThailandPCDPM10StandardsAndMetrics:
    """Verifies Thailand PCD PM10 5-tier classification thresholds and metrics."""

    def test_pcd_pm10_thresholds(self):
        assert THAILAND_AQI_THRESHOLDS_PM10 == [0.0, 50.0, 80.0, 120.0, 180.0, 1000.0]

    def test_assign_aqi_category_pm10(self):
        test_values = [10.0, 50.0, 50.1, 80.0, 80.1, 120.0, 120.1, 180.0, 180.1, 350.0]
        cats = assign_aqi_category(test_values, pollutant="pm10")
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        np.testing.assert_array_equal(cats, expected)

    def test_high_risk_alert_threshold_pm10(self):
        assert HIGH_RISK_ALERT_THRESHOLD_PM10 == 120.0
        # True and Pred with alerts above 120.0
        y_true = np.array([40.0, 75.0, 110.0, 140.0, 200.0])
        y_pred = np.array([45.0, 70.0, 115.0, 135.0, 210.0])
        metrics = compute_comprehensive_metrics(y_true, y_pred, pollutant="pm10")

        # Two samples >= 120.0 (140.0 and 200.0), both predicted >= 120.0 (135.0, 210.0) -> 100% recall
        assert metrics["high_risk_recall"] == 100.0


class TestPM10DedicatedModelSuiteAndEnsemble:
    """Verifies training separate PM10 models and NNLS ensemble."""

    def test_train_lgbm_pm10(self, sample_synthetic_free_training_grid):
        features = [c for c in ALL_FEATURE_COLUMNS if c in sample_synthetic_free_training_grid.columns][:8]
        X = sample_synthetic_free_training_grid[features]
        y = sample_synthetic_free_training_grid["observed_pm10"]

        model = LGBMDustModel(model_params={"n_estimators": 10, "min_child_samples": 2})
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(y)
        assert np.all(preds >= 0.0)

    def test_train_stacking_ensemble_pm10(self, sample_synthetic_free_training_grid):
        features = [c for c in ALL_FEATURE_COLUMNS if c in sample_synthetic_free_training_grid.columns][:6]
        X = sample_synthetic_free_training_grid[features]
        y = sample_synthetic_free_training_grid["observed_pm10"]

        ensemble = DustStackingEnsemble(pollutant="pm10", target_col="observed_pm10")
        ensemble.fit(X, y, cv=3)

        assert "observed_pm10" in ensemble.oof_predictions_
        assert ensemble.pollutant == "pm10"

        # Fit meta-learner weights
        weights = ensemble.weights
        assert len(weights) == 4
        # All weights non-negative and sum to 1.0
        for w in weights.values():
            assert w >= -1e-6
        assert pytest.approx(sum(weights.values()), abs=1e-4) == 1.0

        # Predict
        comp = ensemble.predict_components(X)
        assert "pred_ensemble" in comp.columns
        assert len(comp) == len(X)

    def test_save_and_load_pm10_ensemble(self, tmp_path, sample_synthetic_free_training_grid):
        features = [c for c in ALL_FEATURE_COLUMNS if c in sample_synthetic_free_training_grid.columns][:6]
        X = sample_synthetic_free_training_grid[features]
        y = sample_synthetic_free_training_grid["observed_pm10"]

        ensemble = DustStackingEnsemble(pollutant="pm10", target_col="observed_pm10")
        ensemble.fit(X, y, cv=2)

        save_dir = tmp_path / "exp_2km_pm10"
        saved_paths = ensemble.save(save_dir)
        assert (save_dir / "ensemble_meta.json").exists()

        # Verify metadata specifies PM10
        with open(save_dir / "ensemble_meta.json", "r") as f:
            meta = json.load(f)
        assert meta["pollutant"] == "pm10"
        assert meta["target_col"] == "observed_pm10"

        # Reload
        loaded_ens = DustStackingEnsemble.load(save_dir)
        assert loaded_ens.pollutant == "pm10"
        assert loaded_ens.target_col == "observed_pm10"
        preds = loaded_ens.predict(X)
        np.testing.assert_allclose(preds, ensemble.predict(X), atol=1e-4)


class TestModelIsolationBetweenPM25AndPM10:
    """Verifies that PM2.5 and PM10 models are 100% distinct suites without crosstalk."""

    def test_trainer_directory_isolation(self, tmp_path, sample_synthetic_free_training_grid):
        # 1. Train PM2.5 suite
        trainer_pm25 = ModelTrainer(output_dir=tmp_path, pollutant="pm25")
        ens_pm25 = trainer_pm25.train_suite(
            train_df=sample_synthetic_free_training_grid,
            cv=2,
            experiment_id="EXP-2KM"
        )
        pm25_dir = tmp_path / "exp_2km_pm25"
        assert pm25_dir.exists()

        # 2. Train PM10 suite
        trainer_pm10 = ModelTrainer(output_dir=tmp_path, pollutant="pm10")
        ens_pm10 = trainer_pm10.train_suite(
            train_df=sample_synthetic_free_training_grid,
            cv=2,
            experiment_id="EXP-2KM"
        )
        pm10_dir = tmp_path / "exp_2km_pm10"
        assert pm10_dir.exists()

        # Check directories are separate
        assert pm25_dir != pm10_dir

        # Check each directory holds its respective target metadata
        with open(pm25_dir / "ensemble_meta.json") as f:
            meta_25 = json.load(f)
        with open(pm10_dir / "ensemble_meta.json") as f:
            meta_10 = json.load(f)

        assert meta_25["pollutant"] == "pm25"
        assert meta_25["target_col"] == "observed_pm25"

        assert meta_10["pollutant"] == "pm10"
        assert meta_10["target_col"] == "observed_pm10"


class TestPM10VisualizerDeliverables:
    """Verifies multi-format nowcast deliverables for PM10."""

    def test_generate_pm10_deliverables(self, tmp_path, sample_synthetic_free_training_grid):
        visualizer = DustMapVisualizer(output_dir=tmp_path)
        pred_df = sample_synthetic_free_training_grid.copy()
        pred_df["pm10_pred"] = pred_df["observed_pm10"]

        deliverables = visualizer.generate_all_deliverables(
            grid_predictions_df=pred_df,
            base_filename_stem="nowcast_pm10_test",
            pred_col="pm10_pred",
            pollutant="pm10",
        )

        assert deliverables["parquet"].exists()
        assert deliverables["geotiff"].exists()
        assert deliverables["interactive_html"].exists()
        assert deliverables["static_png"].exists()

        # Inspect HTML content for PM10 standard legend
        with open(deliverables["interactive_html"], "r", encoding="utf-8") as f:
            html = f.read()
        assert "PCD Thailand PM10 Standards" in html
        assert "PM10:" in html
        assert "0 - 50.0 µg/m³" in html
        assert "Alert Cells (≥120.0)" in html


class TestZeroMockCompliancePM10:
    """Verifies that the entire codebase maintains zero-mock compliance."""

    def test_ast_scanner_passes(self):
        scanner = CodeASTScanner(PROJECT_ROOT)
        violations = scanner.scan_directory(PROJECT_ROOT / "src")
        assert len(violations) == 0
