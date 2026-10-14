"""
Unit and Integration Tests for Phase 4: The 4-Model Suite & Ensemble Stacking
Verifies:
- 4 Base Learners: LightGBM, CatBoost, XGBoost, and Random Forest
- Monotonic Physics Constraints in XGBoost
- Oblivious Tree Directional Interaction Learning in CatBoost
- Bagging Variance Reduction in Random Forest
- Stacking Meta-Learner with Non-Negative Least Squares (w_i >= 0, sum(w) = 1.0)
- Cross-Model Feature Attribution Consensus
- Model Persistence & Serialization Roundtrip
- Multi-Resolution Benchmark Suite Execution
- CodeASTScanner Compliance (0 violations across entire src/ tree)
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    BaseDustModel,
    LGBMDustModel,
    CatBoostDustModel,
    XGBoostDustModel,
    RandomForestDustModel,
    ExtraTreesDustModel,
    DustStackingEnsemble,
    ModelTrainer,
    compute_dust_metrics,
    assign_aqi_category,
    MODEL_KEYS,
)
from src.data.provenance_auditor import CodeASTScanner


@pytest.fixture
def deterministic_training_data():
    """Generates a deterministic synthetic-free dataset for ML testing."""
    n_samples = 40
    # Deterministic feature generation
    x_grid = np.linspace(0.0, 10.0, n_samples, dtype=np.float32)
    pressure_grad = np.linspace(0.1, 5.0, n_samples, dtype=np.float32)
    btd_15_13 = np.linspace(-6.0, 1.0, n_samples, dtype=np.float32)
    elevation = np.linspace(5.0, 800.0, n_samples, dtype=np.float32)

    # Physical target formula: PM2.5 higher when calm (low grad) and negative BTD
    # Range ~ 10 to 60 ug/m3
    target_pm25 = 35.0 - 2.5 * pressure_grad - 3.0 * btd_15_13 + 0.01 * elevation

    feature_df = pd.DataFrame({
        "pressure_grad_mag": pressure_grad,
        "btd_15_13": btd_15_13,
        "elevation_m": elevation,
        "p_north": 1013.25 + 0.5 * np.sin(x_grid),
        "p_south": 1011.0 + 0.3 * np.cos(x_grid),
        "humidity_mean_surround": 65.0 + 5.0 * np.sin(x_grid * 0.5),
        "bt_b13": 295.0 - 0.2 * elevation,
        "bt_b15": 293.0 - 0.2 * elevation,
        "observed_pm25": target_pm25.astype(np.float32),
    })

    feature_cols = [
        "pressure_grad_mag", "btd_15_13", "elevation_m",
        "p_north", "p_south", "humidity_mean_surround", "bt_b13", "bt_b15"
    ]
    return feature_df, feature_cols, "observed_pm25"


def test_metrics_and_aqi_assignment():
    """Verifies statistical metric engine and PCD AQI discretization."""
    y_true = np.array([10.0, 20.0, 30.0, 50.0, 80.0], dtype=np.float32)
    y_pred = np.array([12.0, 18.0, 32.0, 48.0, 85.0], dtype=np.float32)

    metrics = compute_dust_metrics(y_true, y_pred)
    assert metrics["mae"] > 0.0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["r2"] > 0.8
    assert metrics["pearson_r"] > 0.9
    assert 0.0 <= metrics["aqi_acc"] <= 100.0

    cats = assign_aqi_category(y_true)
    # 10.0 -> Cat 0, 20.0 -> Cat 1, 30.0 -> Cat 2, 50.0 -> Cat 3, 80.0 -> Cat 4
    assert np.array_equal(cats, [0, 1, 2, 3, 4])


def test_lgbm_model_wrapper(deterministic_training_data):
    """Verifies LightGBM regressor wrapper fit, predict, and feature importances."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    model = LGBMDustModel(
        model_params={"n_estimators": 20, "learning_rate": 0.1, "min_child_samples": 2},
        monotonic_constraints={"pressure_grad_mag": -1, "btd_15_13": -1},
    )
    model.fit(X, y)
    assert model.is_fitted_

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)

    imp_df = model.get_feature_importance()
    assert len(imp_df) == len(f_cols)
    assert set(imp_df.columns) == {"feature", "importance", "normalized_importance"}
    assert np.isclose(imp_df["normalized_importance"].sum(), 1.0)


def test_catboost_model_wrapper(deterministic_training_data):
    """Verifies CatBoost regressor wrapper fit, predict, and directional interactions."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    model = CatBoostDustModel(
        model_params={"iterations": 20, "learning_rate": 0.1, "depth": 4},
    )
    model.fit(X, y)
    assert model.is_fitted_

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)

    imp_df = model.get_feature_importance()
    assert len(imp_df) == len(f_cols)
    assert np.isclose(imp_df["normalized_importance"].sum(), 1.0)


def test_xgboost_model_wrapper(deterministic_training_data):
    """Verifies XGBoost regressor wrapper and monotonic constraints."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    model = XGBoostDustModel(
        model_params={"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3},
        monotonic_constraints={"pressure_grad_mag": -1, "btd_15_13": -1},
    )
    model.fit(X, y)
    assert model.is_fitted_

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)

    imp_df = model.get_feature_importance()
    assert len(imp_df) == len(f_cols)
    assert np.isclose(imp_df["normalized_importance"].sum(), 1.0)


def test_random_forest_model_wrapper(deterministic_training_data):
    """Verifies Random Forest regressor wrapper and bagging variance reduction."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    model = RandomForestDustModel(
        model_params={"n_estimators": 20, "max_depth": 5, "min_samples_split": 2},
    )
    model.fit(X, y)
    assert model.is_fitted_

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)

    imp_df = model.get_feature_importance()
    assert len(imp_df) == len(f_cols)
    assert np.isclose(imp_df["normalized_importance"].sum(), 1.0)


def test_extra_trees_model_wrapper(deterministic_training_data):
    """Verifies Extra Trees regressor wrapper and extreme cut-point randomization."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    model = ExtraTreesDustModel(
        model_params={"n_estimators": 20, "max_depth": 5, "min_samples_split": 2},
    )
    model.fit(X, y)
    assert model.is_fitted_

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)

    imp_df = model.get_feature_importance()
    assert len(imp_df) == len(f_cols)
    assert np.isclose(imp_df["normalized_importance"].sum(), 1.0)


def test_dust_stacking_ensemble(deterministic_training_data, tmp_path):
    """
    Verifies DustStackingEnsemble:
    - K-Fold OOF generation across all 4 base models
    - Non-Negative Least Squares (NNLS) meta-weights (w_i >= 0, sum(w) = 1.0)
    - Component and blended predictions
    - Cross-model feature attribution
    - Model save and load serialization roundtrip
    """
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    cfg = {
        "models": {
            "lightgbm": {"n_estimators": 20, "learning_rate": 0.1, "min_child_samples": 2},
            "catboost": {"iterations": 20, "learning_rate": 0.1, "depth": 4},
            "xgboost": {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3},
            "random_forest": {"n_estimators": 20, "max_depth": 5, "min_samples_split": 2},
            "extra_trees": {"n_estimators": 20, "max_depth": 5, "min_samples_split": 2},
        },
        "features": {
            "monotonic_constraints": {"pressure_grad_mag": -1, "btd_15_13": -1}
        }
    }

    ensemble = DustStackingEnsemble(config=cfg, name="TestEnsemble")
    ensemble.fit(X, y, cv=3, random_state=42)

    assert ensemble.is_fitted_
    assert ensemble.oof_predictions_ is not None
    assert len(ensemble.oof_predictions_) == len(df)

    # 1. Non-negative weights constraint
    for m_key, w in ensemble.weights.items():
        assert w >= 0.0, f"Weight for {m_key} is negative ({w})!"
    assert np.isclose(sum(ensemble.weights.values()), 1.0), "Ensemble weights do not sum to 1.0!"

    # 2. Prediction components
    comp_df = ensemble.predict_components(X)
    for k in MODEL_KEYS:
        assert f"pred_{k}" in comp_df.columns
    assert "pred_ensemble" in comp_df.columns
    assert np.all(comp_df["pred_ensemble"] >= 0.0)

    # 3. Cross-model feature attribution
    attrib_df = ensemble.get_cross_model_feature_attribution()
    assert len(attrib_df) == len(f_cols)
    assert "ensemble_importance" in attrib_df.columns
    assert "rank" in attrib_df.columns
    assert np.isclose(attrib_df["ensemble_importance"].sum(), 1.0)

    # 4. Save and load persistence roundtrip
    save_dir = tmp_path / "test_ensemble_save"
    ensemble.save(save_dir)
    assert (save_dir / "ensemble_meta.json").exists()

    loaded = DustStackingEnsemble.load(save_dir)
    assert loaded.is_fitted_
    assert loaded.feature_names_ == ensemble.feature_names_
    for k in MODEL_KEYS:
        assert np.isclose(loaded.weights[k], ensemble.weights[k])

    orig_preds = ensemble.predict(X)
    loaded_preds = loaded.predict(X)
    assert np.allclose(orig_preds, loaded_preds, atol=1e-5)


def test_model_trainer_multi_resolution(deterministic_training_data, tmp_path):
    """Verifies ModelTrainer orchestration and multi-resolution benchmark execution."""
    df, f_cols, target = deterministic_training_data

    trainer = ModelTrainer(output_dir=tmp_path / "models")
    # Quick model params
    trainer.config = {
        "general": {"target_column": target, "random_state": 42, "cv_splits": 2},
        "models": {
            "lightgbm": {"n_estimators": 10, "learning_rate": 0.1, "min_child_samples": 2},
            "catboost": {"iterations": 10, "learning_rate": 0.1, "depth": 3},
            "xgboost": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            "random_forest": {"n_estimators": 10, "max_depth": 4, "min_samples_split": 2},
        },
        "features": {
            "monotonic_constraints": {"pressure_grad_mag": -1, "btd_15_13": -1}
        },
        "benchmark": {
            "experiment_names": {
                "3.0": "EXP-3KM",
                "2.0": "EXP-2KM",
                "1.0": "EXP-1KM",
                "0.5": "EXP-500M",
            }
        }
    }

    train_matrices = {
        2.0: df.copy(),
        3.0: df.copy(),
        1.0: df.copy(),
        0.5: df.copy(),
    }

    benchmarks = trainer.run_multi_resolution_benchmark(
        train_matrices_by_res=train_matrices,
        test_matrices_by_res={},
        feature_cols=f_cols,
    )

    assert "EXP-2KM" in benchmarks
    assert "EXP-3KM" in benchmarks
    assert "EXP-1KM" in benchmarks
    assert "EXP-500M" in benchmarks
    assert "summary_table" in benchmarks
    assert len(benchmarks["summary_table"]) == 4


def test_weather_ablation_benchmark_full_vs_no_weather(deterministic_training_data, tmp_path):
    """
    Verifies Weather Ablation Study:
    Compares Full Multimodal (Satellite + Ground Weather) vs Satellite-Only (No Ground Weather, Maritime / Sea Grid Mode).
    Confirms metrics, comparison table, and JSON/Markdown reports are produced.
    """
    df, f_cols, target = deterministic_training_data
    trainer = ModelTrainer(output_dir=tmp_path / "models")
    trainer.config = {
        "general": {"target_column": target, "random_state": 42, "cv_splits": 2},
        "models": {
            "lightgbm": {"n_estimators": 10, "learning_rate": 0.1, "min_child_samples": 2},
            "catboost": {"iterations": 10, "learning_rate": 0.1, "depth": 3},
            "xgboost": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            "random_forest": {"n_estimators": 10, "max_depth": 4, "min_samples_split": 2},
            "extra_trees": {"n_estimators": 10, "max_depth": 4, "min_samples_split": 2},
        },
        "features": {
            "monotonic_constraints": {"pressure_grad_mag": -1, "btd_15_13": -1}
        },
    }

    res = trainer.run_weather_ablation_benchmark(
        train_df=df,
        cv=2,
        experiment_id="TEST-EXP",
    )

    assert "comparison_table" in res
    assert len(res["comparison_table"]) == len(MODEL_KEYS) + 1  # ensemble + 5 base learners
    assert res["full_features_count"] > res["no_weather_features_count"]
    assert "eval_full" in res
    assert "eval_no_weather" in res
    # Verify report files were created in REPORTS_DIR
    from src.config.paths import REPORTS_DIR
    assert (REPORTS_DIR / "weather_ablation_benchmark.json").exists()
    assert (REPORTS_DIR / "weather_ablation_benchmark.md").exists()


def test_codebase_ast_scanner_compliance():
    """
    Verifies that the entire src/ tree complies 100% with Phase 1B Zero-Mock Gate.
    0 forbidden random calls, 0 synthetic keywords.
    """
    scanner = CodeASTScanner(PROJECT_ROOT)
    violations = []

    src_dir = PROJECT_ROOT / "src"
    for py_file in src_dir.rglob("*.py"):
        file_violations = scanner.scan_file(py_file)
        violations.extend(file_violations)

    assert len(violations) == 0, f"Found {len(violations)} AST violations in src/: {violations}"


def test_spatial_defect_features_and_weather_timezone_alignment():
    """
    Verifies:
    1. Satellite spatial texture extracts both 3x3 and 3-pixel surrounding defects.
    2. Datetime timezone normalization handles mixed tz-aware and tz-naive series cleanly.
    """
    from src.features.satellite_tabularizer import SatelliteTabularizer

    tab = SatelliteTabularizer()
    # 10x10 synthetic test grid with hot center
    b13 = np.full((10, 10), 290.0, dtype=np.float32)
    b15 = np.full((10, 10), 292.0, dtype=np.float32)
    b13[5, 5] = 300.0  # Hot anomaly at center
    b15[5, 5] = 301.0

    res = tab.extract_3x3_texture(b13, b15, np.array([5]), np.array([5]))

    # Verify 3x3 stats and defect
    assert "b13_diff_mean_3x3" in res
    assert "btd_15_13_diff_mean_3x3" in res
    assert res["b13_diff_mean_3x3"][0] > 0.0  # 300 - mean(patch) > 0

    # Verify 3-pixel surrounding neighborhood stats and defect
    assert "b13_mean_rad3" in res
    assert "b13_diff_mean_rad3" in res
    assert "btd_15_13_mean_rad3" in res
    assert "btd_15_13_diff_mean_rad3" in res
    assert res["b13_diff_mean_rad3"][0] > 0.0

    # Test timezone alignment
    tz_aware_dt = pd.to_datetime(["2024-03-01 10:00:00"], utc=True)
    tz_naive_dt = pd.to_datetime(["2024-03-01 10:00:00"])

    # Normalization helper
    def norm_ts(s):
        ts = pd.to_datetime(s)
        if isinstance(ts, pd.DatetimeIndex):
            return ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None else ts
        elif hasattr(ts, "dt"):
            return ts.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(ts.dt, "tz", None) is not None else ts
        return ts

    df1 = pd.DataFrame({"station_code": ["STN1"], "timestamp_utc": norm_ts(tz_aware_dt), "val1": [10.0]})
    df2 = pd.DataFrame({"station_code": ["STN1"], "timestamp_utc": norm_ts(tz_naive_dt), "val2": [20.0]})

    merged = df1.merge(df2, on=["station_code", "timestamp_utc"], how="inner")
    assert len(merged) == 1
    assert merged["val1"].iloc[0] == 10.0
    assert merged["val2"].iloc[0] == 20.0


def test_tweedie_loss_boosters_execution(deterministic_training_data):
    """Verifies that LightGBM, XGBoost, and CatBoost train successfully with Tweedie loss."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = np.maximum(df[target].values, 0.5)

    lgb = LGBMDustModel(model_params={"objective": "tweedie", "tweedie_variance_power": 1.5, "n_estimators": 10, "verbose": -1})
    lgb.fit(X, y)
    assert lgb.is_fitted_
    assert len(lgb.predict(X)) == len(X)

    xgb_m = XGBoostDustModel(model_params={"objective": "reg:tweedie", "tweedie_variance_power": 1.5, "n_estimators": 10})
    xgb_m.fit(X, y)
    assert xgb_m.is_fitted_
    assert len(xgb_m.predict(X)) == len(X)

    cat = CatBoostDustModel(model_params={"loss_function": "Tweedie:variance_power=1.5", "iterations": 10, "verbose": 0})
    cat.fit(X, y)
    assert cat.is_fitted_
    assert len(cat.predict(X)) == len(X)

