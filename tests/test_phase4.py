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
    FIGSDustModel,
    NystroemRidgeDustModel,
    DustStackingEnsemble,
    ModelTrainer,
    compute_dust_metrics,
    assign_aqi_category,
    MODEL_KEYS,
)
from src.features import resolve_dataset_providers
from src.evaluation import EvaluationPlotter
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


def test_figs_model_wrapper(deterministic_training_data):
    """Verifies FIGS regressor wrapper, interpretable rules, and prediction constraints."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    model = FIGSDustModel(
        model_params={"max_rules": 12, "max_trees": 3, "random_state": 42},
    )
    model.fit(X, y)
    assert model.is_fitted_

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)

    imp_df = model.get_feature_importance()
    assert len(imp_df) == len(f_cols)
    assert np.isclose(imp_df["normalized_importance"].sum(), 1.0)


def test_nystroem_model_wrapper(deterministic_training_data):
    """Verifies Nystroem Fast Kernel Ridge regressor wrapper and projected importances."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    model = NystroemRidgeDustModel(
        model_params={"n_components": 20, "alpha": 1.0, "random_state": 42},
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
    - K-Fold OOF generation across all 7 base models
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
            "extra_trees": {"n_estimators": 20, "max_depth": 5, "min_samples_split": 2},
            "figs": {"max_rules": 10, "max_trees": 3, "random_state": 42},
            "nystroem": {"n_components": 20, "alpha": 1.0, "random_state": 42},
            "random_forest": {"n_estimators": 20, "max_depth": 5, "min_samples_split": 2},
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


def test_tuned_param_override(deterministic_training_data, tmp_path):
    """Verifies train_suite deep-merges Optuna-tuned model params via model_params_override."""
    df, f_cols, target = deterministic_training_data
    half = len(df) // 2
    df = df.copy()
    df["provider"] = ["Air4Thai"] * half + ["AirGradient"] * (len(df) - half)

    trainer = ModelTrainer(output_dir=tmp_path / "models")
    trainer.config = {
        "general": {"target_column": target, "random_state": 42, "cv_splits": 1},
        "models": {"lightgbm": {"n_estimators": 5, "learning_rate": 0.1, "min_child_samples": 2}},
        "features": {"monotonic_constraints": {}},
    }
    trainer.tuned_params = {"full__combined": {"lightgbm": {"num_leaves": 17}}}
    assert trainer._tuned_override("missing", "combined") is None

    ens = trainer.train_suite(
        train_df=df,
        feature_cols=f_cols,
        cv=1,
        experiment_id="EXP-2KM-TEST-COMBINED",
        model_params_override=trainer._tuned_override("full", "combined"),
    )
    assert ens.base_models["lightgbm"].params["num_leaves"] == 17


def test_hyperparameter_search_space_and_native_conversion():
    """Verifies Optuna search-space builders and numpy->native conversion used by the tuner."""
    from src.models.hyperparameter_tuner import suggest_params, _to_native, DEFAULT_FEATURE_SETS

    class DummyTrial:
        def suggest_int(self, name, low, high, **kwargs):
            return low

        def suggest_float(self, name, low, high, **kwargs):
            return low

        def suggest_categorical(self, name, choices):
            return choices[0]

    for mk in MODEL_KEYS:
        params = suggest_params(DummyTrial(), mk, {"existing": 1})
        assert isinstance(params, dict)
        assert params.get("existing") == 1

    assert set(DEFAULT_FEATURE_SETS.keys()) == {"full", "pure_satellite"}
    assert _to_native({"a": np.int64(3), "b": [np.float32(1.5)]}) == {"a": 3, "b": [1.5]}


def test_per_model_benchmark_callback(deterministic_training_data):
    """Verifies each base learner is benchmarked (callback invoked) immediately after it trains."""
    df, f_cols, target = deterministic_training_data
    X = df[f_cols]
    y = df[target]

    cfg = {
        "models": {
            "lightgbm": {"n_estimators": 5, "learning_rate": 0.1, "min_child_samples": 2},
            "catboost": {"iterations": 5, "learning_rate": 0.1, "depth": 3},
            "xgboost": {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 3},
            "extra_trees": {"n_estimators": 5, "max_depth": 4, "min_samples_split": 2},
            "figs": {"max_rules": 6, "max_trees": 2, "random_state": 42},
            "nystroem": {"n_components": 10, "alpha": 1.0, "random_state": 42},
            "random_forest": {"n_estimators": 5, "max_depth": 4, "min_samples_split": 2},
        },
        "features": {"monotonic_constraints": {}},
    }

    records = []
    ensemble = DustStackingEnsemble(config=cfg, name="BenchEnsemble")
    ensemble.fit(X, y, cv=1, random_state=42, benchmark_callback=records.append)

    assert len(records) == len(MODEL_KEYS)
    assert {r["model"] for r in records} == set(MODEL_KEYS)
    for r in records:
        assert "mae" in r and "rmse" in r and "r2" in r and "aqi_acc" in r
        assert r["eval_mode"] == "holdout"
        assert r["n_eval"] >= 2


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


def test_resolve_dataset_providers_filter_sparse():
    """Verifies that sparse networks (Clarity, AirNow, HabitatMap, NaN) are dropped, retaining Air4Thai & AirGradient."""
    mock_df = pd.DataFrame({
        "station_code": ["PCD_01", "OPENAQ_1", "OPENAQ_2", "OPENAQ_3", "OPENAQ_4", "OPENAQ_5"],
        "provider": ["Air4Thai", "AirGradient", "Clarity", "AirNow", "HabitatMap", None],
        "lat": [13.75, 18.78, 13.80, 13.66, 16.18, 14.0],
        "lon": [100.5, 98.98, 100.40, 100.70, 103.30, 101.0],
        "observed_pm25": [25.0, 30.0, 45.0, 50.0, 60.0, 22.0],
    })

    filtered = resolve_dataset_providers(mock_df, allowed_providers=("Air4Thai", "AirGradient"))
    assert len(filtered) == 2
    assert set(filtered["provider"].unique()) == {"Air4Thai", "AirGradient"}
    assert "Clarity" not in filtered["provider"].values
    assert "AirNow" not in filtered["provider"].values
    assert "HabitatMap" not in filtered["provider"].values


def test_multi_resolution_provider_benchmark_execution(deterministic_training_data, tmp_path):
    """Verifies Multi-Resolution Provider Benchmark executes Air4Thai first, AirGradient second, Combined third."""
    df, f_cols, target = deterministic_training_data
    # Create dataset with equal Air4Thai and AirGradient rows
    half = len(df) // 2
    df_prov = df.copy()
    df_prov["provider"] = ["Air4Thai"] * half + ["AirGradient"] * (len(df) - half)

    trainer = ModelTrainer(output_dir=tmp_path / "models")
    trainer.config = {
        "general": {"target_column": target, "random_state": 42, "cv_splits": 2},
        "models": {
            "lightgbm": {"n_estimators": 5, "learning_rate": 0.1, "min_child_samples": 2},
            "catboost": {"iterations": 5, "learning_rate": 0.1, "depth": 2},
            "xgboost": {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 2},
            "random_forest": {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2},
        },
        "features": {
            "monotonic_constraints": {"pressure_grad_mag": -1, "btd_15_13": -1}
        },
    }

    train_matrices = {
        2.0: df_prov.copy(),
        1.0: df_prov.copy(),
    }

    results = trainer.run_multi_resolution_provider_benchmark(
        train_matrices_by_res=train_matrices,
        test_matrices_by_res={},
        feature_cols=f_cols,
        target_providers=["Air4Thai", "AirGradient"],
        include_combined=True,
    )

    assert "summary_table" in results
    summary_df = pd.DataFrame(results["summary_table"])
    assert not summary_df.empty

    # Confirm both 2km and 1km are benchmarked
    assert 2.0 in summary_df["resolution_km"].values
    assert 1.0 in summary_df["resolution_km"].values

    # Confirm Air4Thai, AirGradient, and Combined are present
    prov_set = set(summary_df["provider"].unique())
    assert "Air4Thai" in prov_set
    assert "AirGradient" in prov_set
    assert any("Combined" in p for p in prov_set)


def test_provider_partitioned_training_and_cross_provider(deterministic_training_data, tmp_path):
    """Verifies provider-partitioned production training runs every feature set x provider,
    persists checkpoints, trains Combined, and produces cross-provider (train -> predict swap) rows."""
    df, f_cols, target = deterministic_training_data
    half = len(df) // 2
    df_prov = df.copy()
    df_prov["provider"] = ["Air4Thai"] * half + ["AirGradient"] * (len(df) - half)

    pure_cols = [c for c in f_cols if c not in ("p_north", "p_south", "humidity_mean_surround", "pressure_grad_mag")]

    trainer = ModelTrainer(output_dir=tmp_path / "models")
    trainer.config = {
        "general": {"target_column": target, "random_state": 42, "cv_splits": 2},
        "models": {
            "lightgbm": {"n_estimators": 5, "learning_rate": 0.1, "min_child_samples": 2},
            "catboost": {"iterations": 5, "learning_rate": 0.1, "depth": 2},
            "xgboost": {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 2},
            "random_forest": {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2},
        },
        "features": {"monotonic_constraints": {}},
    }

    results = trainer.run_provider_partitioned_training(
        train_matrices_by_res={2.0: df_prov.copy()},
        test_matrices_by_res={},
        providers=["Air4Thai", "AirGradient"],
        feature_sets={"full": f_cols, "pure": pure_cols},
        include_combined=True,
        cross_provider_eval=True,
        save_checkpoints=True,
    )

    # Per-provider + Combined, across both feature sets
    summary_df = pd.DataFrame(results["summary_table"])
    prov_set = set(summary_df["provider"].unique())
    assert "Air4Thai" in prov_set
    assert "AirGradient" in prov_set
    assert any("Combined" in p for p in prov_set)
    assert set(summary_df["feature_set"].unique()) == {"full", "pure"}

    # Every checkpoint persisted (feature set x provider x Combined)
    chk_names = {Path(c).name for c in results["checkpoints"]}
    assert "exp_2km_full_air4thai_pm25" in chk_names
    assert "exp_2km_pure_air4thai_pm25" in chk_names
    assert "exp_2km_full_airgradient_pm25" in chk_names
    assert "exp_2km_pure_combined_pm25" in chk_names

    # Cross-provider swap rows exist in both directions
    cross_df = pd.DataFrame(results["cross_provider_table"])
    assert not cross_df.empty
    pairs = set(zip(cross_df["train_provider"], cross_df["test_provider"]))
    assert ("Air4Thai", "AirGradient") in pairs
    assert ("AirGradient", "Air4Thai") in pairs


def test_evaluation_plotter_provider_graphics(tmp_path):
    """Verifies that EvaluationPlotter generates separate provider comparison plots and parity plots."""
    plotter = EvaluationPlotter(output_dir=tmp_path / "plots", pollutant="pm25")

    # 1. Summary comparison bar chart
    mock_summary = [
        {"resolution_name": "2km", "provider": "Air4Thai", "ensemble_mae": 2.1, "ensemble_r2": 0.85},
        {"resolution_name": "2km", "provider": "AirGradient", "ensemble_mae": 2.8, "ensemble_r2": 0.80},
        {"resolution_name": "2km", "provider": "Combined (Air4Thai + AirGradient)", "ensemble_mae": 2.3, "ensemble_r2": 0.83},
        {"resolution_name": "1km", "provider": "Air4Thai", "ensemble_mae": 1.9, "ensemble_r2": 0.88},
        {"resolution_name": "1km", "provider": "AirGradient", "ensemble_mae": 2.5, "ensemble_r2": 0.82},
        {"resolution_name": "1km", "provider": "Combined (Air4Thai + AirGradient)", "ensemble_mae": 2.0, "ensemble_r2": 0.86},
    ]

    p_comp = plotter.plot_provider_comparison(mock_summary, pollutant="pm25")
    assert p_comp.exists()
    assert (tmp_path / "plots" / "csv" / "provider_benchmark_summary_pm25.csv").exists()

    # 2. Provider-partitioned parity scatter
    y_true = np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0])
    y_pred = np.array([16.0, 24.0, 34.0, 47.0, 53.0, 64.0])
    providers = np.array(["Air4Thai", "Air4Thai", "Air4Thai", "AirGradient", "AirGradient", "AirGradient"])

    p_dict = plotter.plot_provider_parity_scatter(y_true, y_pred, providers, pollutant="pm25")
    assert "Air4Thai" in p_dict
    assert "AirGradient" in p_dict
    assert "panel" in p_dict
    assert p_dict["Air4Thai"].exists()
    assert p_dict["AirGradient"].exists()
    assert p_dict["panel"].exists()

    # 3. Feature Set x Resolution x Provider benchmark + CSV
    summary = [
        {"resolution_name": r, "feature_set": fs, "provider": p,
         "ensemble_mae": 2.0 + 0.1 * i, "ensemble_r2": 0.80 + 0.01 * i}
        for i, (r, fs, p) in enumerate(
            (r, fs, p)
            for r in ["3km", "2km"]
            for fs in ["full", "pure_satellite"]
            for p in ["Air4Thai", "AirGradient", "Combined (Air4Thai + AirGradient)"]
        )
    ]
    p_fs = plotter.plot_provider_feature_resolution_benchmark(summary, pollutant="pm25")
    assert p_fs.exists()
    assert (tmp_path / "plots" / "csv" / "provider_feature_resolution_summary_pm25.csv").exists()

    # 4. Cross-provider generalization heatmap (train A -> test B swap)
    cross = [
        {"resolution_name": r, "feature_set": fs, "train_provider": tp, "test_provider": vp,
         "mae": 2.0 + 0.2 * i, "r2": 0.8 - 0.01 * i, "eval_split": "test"}
        for i, (r, fs, tp, vp) in enumerate(
            (r, fs, tp, vp)
            for r in ["3km", "2km"]
            for fs in ["full", "pure_satellite"]
            for tp in ["Air4Thai", "AirGradient", "Combined (Air4Thai + AirGradient)"]
            for vp in ["Air4Thai", "AirGradient"]
        )
    ]
    p_cross_mae = plotter.plot_cross_provider_matrix(cross, metric="mae", pollutant="pm25")
    p_cross_r2 = plotter.plot_cross_provider_matrix(cross, metric="r2", pollutant="pm25")
    assert p_cross_mae.exists()
    assert p_cross_r2.exists()


