"""
Unit and Integration Tests for Phase 5: Evaluation, Benchmarking & Nationwide Inference
Verifies:
- Continuous Metrics Engine (MAE, RMSE, MAPE, R², Mean Bias, Pearson r, Spearman rho)
- Thailand PCD AQI Discretization and Classification Metrics (5-tier Color Accuracy, Confusion Matrix, Macro F1, High-Risk Recall)
- Spatial Geographic Region Partitioning (North, Northeast, Central, East, South)
- Regional Hold-Out Cross-Validation (Spatial generalization on unseen geographic domains)
- Nationwide Nowcast Map Deliverables Suite:
  * Apache Parquet Table with enriched AQI fields
  * Georeferenced GeoTIFF Raster via rasterio
  * Standalone Interactive Leaflet Web Map HTML
  * Publication-Quality Static PNG Map
- Multi-Resolution Benchmark Suite (Leaderboard, latency/throughput profiling, and Research Question synthesis)
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

from src.evaluation import (
    compute_continuous_metrics,
    compute_aqi_classification_metrics,
    compute_comprehensive_metrics,
    assign_aqi_category,
    get_aqi_metadata,
    format_evaluation_summary_table,
    RegionalHoldoutCV,
    SpatialTemporalEvaluator,
    assign_thailand_region,
    attach_region_column,
    DustMapVisualizer,
    MultiResolutionBenchmarkSuite,
    EvaluationPlotter,
    THAILAND_AQI_THRESHOLDS,
)
from src.models import (
    LGBMDustModel,
    DustStackingEnsemble,
    MODEL_KEYS,
)
from src.data.provenance_auditor import CodeASTScanner


@pytest.fixture
def mock_free_ground_truth_and_predictions():
    """Generates deterministic evaluation arrays without synthetic/mock calls."""
    # 5 samples across all 5 PCD color tiers:
    # Tier 0 (0-15): 10.0
    # Tier 1 (15-25): 20.0
    # Tier 2 (25-37.5): 30.0
    # Tier 3 (37.5-75): 50.0
    # Tier 4 (>75): 80.0
    y_true = np.array([10.0, 20.0, 30.0, 50.0, 80.0], dtype=np.float32)
    y_pred = np.array([12.0, 19.0, 32.0, 55.0, 78.0], dtype=np.float32)
    return y_true, y_pred


@pytest.fixture
def regional_grid_dataset():
    """Generates deterministic spatial grid dataset covering Thailand regions."""
    # North (Chiang Mai ~18.8N, 98.9E), Northeast (Khon Kaen ~16.4N, 102.8E),
    # Central (Bangkok ~13.7N, 100.5E), East (Chonburi ~13.3N, 101.0E), South (Phuket ~7.9N, 98.4E)
    coords = [
        {"lat": 18.8, "lon": 98.9, "expected_reg": "north"},
        {"lat": 16.4, "lon": 102.8, "expected_reg": "northeast"},
        {"lat": 13.7, "lon": 100.5, "expected_reg": "central"},
        {"lat": 13.3, "lon": 101.2, "expected_reg": "east"},
        {"lat": 7.9, "lon": 98.4, "expected_reg": "south"},
    ]
    rows = []
    for i, c in enumerate(coords):
        for rep in range(4):
            lat_off = c["lat"] + rep * 0.05
            lon_off = c["lon"] + rep * 0.05
            rows.append({
                "grid_id": f"cell_{i}_{rep}",
                "lat": lat_off,
                "lon": lon_off,
                "utm_x": 600000.0 + i * 20000.0 + rep * 2000.0,
                "utm_y": 1500000.0 + i * 20000.0 + rep * 2000.0,
                "elevation_m": 50.0 + i * 100.0,
                "feature_a": 10.0 + i * 2.0 + rep * 0.5,
                "feature_b": 20.0 - i * 1.5,
                "observed_pm25": 15.0 + i * 12.0 + rep * 2.0,
                "pm25_pred": 16.0 + i * 11.5 + rep * 1.8,
            })
    return pd.DataFrame(rows)


def test_continuous_metrics_engine(mock_free_ground_truth_and_predictions):
    """Verifies continuous regression metrics calculation."""
    y_true, y_pred = mock_free_ground_truth_and_predictions
    metrics = compute_continuous_metrics(y_true, y_pred)

    assert metrics["mae"] > 0.0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["r2"] > 0.8
    assert metrics["pearson_r"] > 0.95
    assert metrics["spearman_rho"] > 0.95
    assert metrics["n_samples"] == len(y_true)
    # y_pred [12, 19, 32, 55, 78] vs y_true [10, 20, 30, 50, 80] -> sum diff = 2 - 1 + 2 + 5 - 2 = 6 / 5 = 1.2
    assert np.isclose(metrics["mean_bias"], 1.2, atol=0.01)


def test_aqi_classification_engine(mock_free_ground_truth_and_predictions):
    """Verifies Thailand PCD AQI categorization and classification metrics."""
    y_true, y_pred = mock_free_ground_truth_and_predictions

    # Test categorization
    cats = assign_aqi_category(y_true)
    assert np.array_equal(cats, [0, 1, 2, 3, 4])

    pred_cats = assign_aqi_category(y_pred)
    assert np.array_equal(pred_cats, [0, 1, 2, 3, 4])

    # Test classification metrics
    aqi_res = compute_aqi_classification_metrics(y_true, y_pred)
    assert aqi_res["accuracy"] == 100.0
    assert aqi_res["macro_f1"] == 100.0
    assert aqi_res["high_risk_recall"] == 100.0
    assert len(aqi_res["per_tier"]) == 5

    cm = np.array(aqi_res["confusion_matrix"])
    assert cm.shape == (5, 5)
    assert np.trace(cm) == len(y_true)


def test_aqi_metadata_retrieval():
    """Verifies metadata resolution for PCD color tiers."""
    meta_blue = get_aqi_metadata(0)
    assert meta_blue["color_name"] == "Blue"
    assert meta_blue["color_hex"] == "#00b0f0"
    assert meta_blue["label_th"] == "ดีมาก"

    meta_red = get_aqi_metadata(4)
    assert meta_red["color_name"] == "Red"
    assert meta_red["color_hex"] == "#ff0000"
    assert meta_red["label_th"] == "มีผลกระทบต่อสุขภาพ"


def test_spatial_region_assignment():
    """Verifies physical bounding box assignment for Thailand regions."""
    assert assign_thailand_region(19.0, 99.0) == "north"
    assert assign_thailand_region(16.5, 103.0) == "northeast"
    assert assign_thailand_region(13.8, 100.5) == "central"
    assert assign_thailand_region(13.0, 101.5) == "east"
    assert assign_thailand_region(8.0, 99.0) == "south"


def test_regional_holdout_cv(regional_grid_dataset):
    """Verifies Spatial K-Fold partitioning across unseen geographic regions."""
    df = regional_grid_dataset
    cv = RegionalHoldoutCV()

    splits = list(cv.split(df))
    assert len(splits) == 5

    observed_regions = set()
    for reg, train_idx, val_idx in splits:
        observed_regions.add(reg)
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        # No overlap between train and validation indices
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(train_idx) + len(val_idx) == len(df)

    assert observed_regions == {"north", "northeast", "central", "east", "south"}


def test_spatial_temporal_evaluator(regional_grid_dataset):
    """Verifies regional hold-out evaluation of model generalization."""
    df = regional_grid_dataset
    evaluator = SpatialTemporalEvaluator(target_col="observed_pm25")

    def model_factory():
        return LGBMDustModel(model_params={"n_estimators": 5, "min_child_samples": 2})

    results = evaluator.evaluate_regional_generalization(
        df=df,
        feature_cols=["feature_a", "feature_b", "elevation_m"],
        model_factory_fn=model_factory,
    )

    assert "by_region" in results
    assert "overall_unseen_spatial_metrics" in results
    assert len(results["by_region"]) == 5
    assert results["total_evaluated_samples"] == len(df)

    report = evaluator.format_regional_report(results)
    assert "Spatial Generalization" in report
    assert "MACRO OVERALL" in report


def test_dust_map_visualizer_deliverables(regional_grid_dataset, tmp_path):
    """
    Verifies generation of all 4 nationwide nowcast map deliverables:
    1. Parquet Table with enriched fields
    2. GeoTIFF Raster via rasterio
    3. Interactive Leaflet HTML
    4. Static PNG Map
    """
    df = regional_grid_dataset
    visualizer = DustMapVisualizer(output_dir=tmp_path / "maps")

    deliverables = visualizer.generate_all_deliverables(
        grid_predictions_df=df,
        base_filename_stem="test_nowcast",
        pred_col="pm25_pred",
        resolution_m=2000.0,
    )

    # 1. Parquet Check
    p_parquet = deliverables["parquet"]
    assert p_parquet.exists()
    df_read = pd.read_parquet(p_parquet)
    assert "aqi_category" in df_read.columns
    assert "aqi_color" in df_read.columns
    assert "aqi_label_th" in df_read.columns

    # 2. GeoTIFF Check
    p_tif = deliverables["geotiff"]
    assert p_tif.exists()
    assert p_tif.stat().st_size > 0

    # 3. Interactive Leaflet HTML Check
    p_html = deliverables["interactive_html"]
    assert p_html.exists()
    with open(p_html, "r", encoding="utf-8") as f:
        html_content = f.read()
    assert "leaflet" in html_content
    assert "PCD Thailand PM2.5 Standards" in html_content
    assert "Thailand PM2.5 Satellite Nowcast" in html_content

    # 4. Static PNG Check
    p_png = deliverables["static_png"]
    assert p_png.exists()
    assert p_png.stat().st_size > 0


def test_multi_resolution_benchmark_suite(regional_grid_dataset, tmp_path):
    """
    Verifies multi-resolution benchmark runner, inference latency profiling,
    leaderboard table generation, and Research Questions synthesis.
    """
    df = regional_grid_dataset
    f_cols = ["feature_a", "feature_b", "elevation_m"]

    # Train simple ensemble
    cfg = {
        "models": {
            "lightgbm": {"n_estimators": 5, "min_child_samples": 2},
            "catboost": {"iterations": 5, "depth": 2},
            "xgboost": {"n_estimators": 5, "max_depth": 2},
            "random_forest": {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2},
        }
    }
    ensemble = DustStackingEnsemble(config=cfg, name="BenchEnsemble")
    ensemble.fit(df[f_cols], df["observed_pm25"], feature_names=f_cols, cv=2)

    trained_ensembles = {
        3.0: ensemble,
        2.0: ensemble,
        1.0: ensemble,
        0.5: ensemble,
    }
    test_matrices = {
        3.0: df.copy(),
        2.0: df.copy(),
        1.0: df.copy(),
        0.5: df.copy(),
    }

    suite = MultiResolutionBenchmarkSuite(output_dir=tmp_path / "reports")
    benchmark_res = suite.run_full_benchmark(
        trained_ensembles_by_res=trained_ensembles,
        test_matrices_by_res=test_matrices,
        feature_cols=f_cols,
        target_col="observed_pm25",
    )

    assert "leaderboard" in benchmark_res
    assert len(benchmark_res["leaderboard"]) == 4 * (len(MODEL_KEYS) + 1)  # 4 resolutions x (5 base models + ensemble)
    assert "research_questions" in benchmark_res
    rqs = benchmark_res["research_questions"]
    assert "RQ1_Optimal_Resolution" in rqs
    assert "RQ2_Topography_Gain_vs_Sensor_Blur" in rqs
    assert "RQ3_Best_Model_Architecture" in rqs
    assert "RQ4_Compute_Tradeoff_Pareto" in rqs

    # Reports generated (both pollutant-suffixed and legacy alias)
    assert (tmp_path / "reports" / "phase5_multi_resolution_benchmark_pm25.json").exists()
    assert (tmp_path / "reports" / "phase5_multi_resolution_benchmark_pm25.md").exists()
    assert (tmp_path / "reports" / "phase5_multi_resolution_benchmark.json").exists()
    assert (tmp_path / "reports" / "phase5_multi_resolution_benchmark.md").exists()


def test_evaluation_plotter_suite(regional_grid_dataset, tmp_path):
    """
    Verifies that EvaluationPlotter successfully creates all 7 publication-quality
    scientific diagnostic plots in outputs/evaluation/.
    """
    df = regional_grid_dataset
    y_true = df["observed_pm25"].values
    y_pred = df["pm25_pred"].values

    eval_dir = tmp_path / "evaluation"
    plotter = EvaluationPlotter(output_dir=eval_dir)

    leaderboard_df = pd.DataFrame([
        {"resolution": "2 km × 2 km (Baseline)", "model": "CATBOOST", "mae": 0.24, "latency_ms": 1.3},
        {"resolution": "2 km × 2 km (Baseline)", "model": "Stacking Ensemble (NNLS)", "mae": 0.24, "latency_ms": 40.3},
    ])

    regional_results = {
        "by_region": {
            "north": {"region_th": "ภาคเหนือ", "metrics": {"mae": 5.25, "rmse": 5.84, "aqi_accuracy": 13.3}},
            "central": {"region_th": "ภาคกลาง", "metrics": {"mae": 4.32, "rmse": 5.55, "aqi_accuracy": 34.4}},
        }
    }

    attribution_df = pd.DataFrame([
        {"feature": "pressure_grad_mag", "ensemble_importance": 0.35},
        {"feature": "btd_15_13", "ensemble_importance": 0.25},
        {"feature": "elevation_m", "ensemble_importance": 0.20},
    ])

    weights = {"lightgbm": 0.1, "catboost": 0.6, "xgboost": 0.2, "random_forest": 0.1}

    plots = plotter.generate_all_evaluation_plots(
        y_true=y_true,
        y_pred=y_pred,
        leaderboard_df=leaderboard_df,
        regional_results=regional_results,
        attribution_df=attribution_df,
        ensemble_weights=weights,
    )

    assert len(plots) == 7
    for plot_key, p_path in plots.items():
        assert p_path.exists(), f"Plot {plot_key} was not generated!"
        assert p_path.stat().st_size > 0, f"Plot {plot_key} is empty!"

    # Verify evaluation csv folder and csv tables
    csv_dir = eval_dir / "csv"
    assert csv_dir.exists()
    assert (csv_dir / "parity_predictions_pm25.csv").exists()
    assert (csv_dir / "aqi_confusion_matrix_pm25.csv").exists()
    assert (csv_dir / "multi_resolution_benchmark_pm25.csv").exists()
    assert (csv_dir / "regional_generalization_pm25.csv").exists()
    assert (csv_dir / "feature_importance_pm25.csv").exists()
    assert (csv_dir / "ensemble_weights_pm25.csv").exists()


def test_codebase_ast_scanner_compliance():
    """
    Verifies that the entire src/ tree complies 100% with Phase 1B Zero-Mock Gate.
    0 forbidden random calls, 0 synthetic keywords across all Phase 1-5 modules.
    """
    scanner = CodeASTScanner(PROJECT_ROOT)
    violations = []

    src_dir = PROJECT_ROOT / "src"
    for py_file in src_dir.rglob("*.py"):
        file_violations = scanner.scan_file(py_file)
        violations.extend(file_violations)

    assert len(violations) == 0, f"Found {len(violations)} AST violations in src/: {violations}"
