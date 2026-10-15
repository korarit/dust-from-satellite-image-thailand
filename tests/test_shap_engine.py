"""
Unit Tests for the SHAP Attribution Engine (Phase 5 Extension)
Verifies:
- Ensemble-consensus SHAP computation on fitted tree models
- Deterministic stride subsampling (no randomness)
- Attribute table ordering, shares, and directional sign logic
- Artifact export: CSV + summary bar plot + beeswarm plot
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

from src.models import DustStackingEnsemble
from src.evaluation.shap_engine import ShapAttributionEngine


@pytest.fixture
def deterministic_training_data():
    """Deterministic physically-motivated dataset (no random calls)."""
    n_samples = 120
    x_grid = np.linspace(0.0, 10.0, n_samples, dtype=np.float32)
    pressure_grad = np.linspace(0.1, 5.0, n_samples, dtype=np.float32)
    btd_15_13 = np.linspace(-6.0, 1.0, n_samples, dtype=np.float32)
    elevation = np.linspace(5.0, 800.0, n_samples, dtype=np.float32)

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


@pytest.fixture
def small_ensemble(deterministic_training_data):
    df, _, target = deterministic_training_data
    config = {
        "models": {
            "lightgbm": {"n_estimators": 30, "learning_rate": 0.1, "min_child_samples": 5, "verbose": -1},
            "catboost": {"iterations": 30, "learning_rate": 0.1, "depth": 3, "verbose": 0},
            "xgboost": {"n_estimators": 30, "learning_rate": 0.1, "max_depth": 3},
            "extra_trees": {"n_estimators": 30, "max_depth": 6},
            "figs": {"max_rules": 12},
            "nystroem": {"n_components": 24},
            "random_forest": {"n_estimators": 30, "max_depth": 6},
            "hist_gbr": {"max_iter": 30, "max_depth": 3},
        },
        "general": {"eval_metric": "mae"},
        "features": {"monotonic_constraints": {}},
    }
    ensemble = DustStackingEnsemble(config=config, pollutant="pm25", target_col=target)
    X = df.drop(columns=[target])
    y = df[target]
    ensemble.fit(X, y, cv=1)
    return ensemble, X, y, ensemble.feature_names_


def test_shap_consensus_compute(small_ensemble):
    ensemble, X, _, _ = small_ensemble
    engine = ShapAttributionEngine(ensemble=ensemble, pollutant="pm25")
    attrib = engine.compute(X)

    assert attrib is not None
    assert list(attrib["rank"]) == list(range(1, len(attrib) + 1))
    assert set(["feature", "mean_abs_shap", "mean_signed_shap", "shap_share"]).issubset(attrib.columns)

    # Shares normalized to 1.0 and sorted descending
    assert abs(attrib["shap_share"].sum() - 1.0) < 1e-6
    assert attrib["mean_abs_shap"].is_monotonic_decreasing

    # At least one tree model contributed
    assert len(engine.contributed_models_) >= 1
    assert engine.consensus_shap_.shape[1] == len(engine.X_sample_.columns)


def test_shap_deterministic_subsampling(small_ensemble):
    ensemble, X, _, _ = small_ensemble
    engine = ShapAttributionEngine(ensemble=ensemble, pollutant="pm25", max_samples=50)
    engine.compute(X)

    # Stride sampling must be deterministic across repeated calls
    idx_a = np.linspace(0, len(X) - 1, 50).astype(np.int64)
    sampled = X.iloc[idx_a].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        engine.X_sample_.reset_index(drop=True),
        sampled[list(engine.X_sample_.columns)],
    )


def test_shap_consensus_matrix_matches_weights(small_ensemble):
    """Consensus SHAP = sum_i w_i * SHAP_i (weighted linear blend, verified additivity)."""
    ensemble, X, _, _ = small_ensemble

    # Force weights to only LightGBM so consensus equals raw LGBM SHAP
    for k in ensemble.weights:
        ensemble.weights[k] = 1.0 if k == "lightgbm" else 0.0

    import shap as shap_lib

    engine = ShapAttributionEngine(ensemble=ensemble, pollutant="pm25")
    engine.compute(X)
    f_names = list(engine.X_sample_.columns)
    explainer = shap_lib.TreeExplainer(ensemble.base_models["lightgbm"].estimator, model_output="raw")
    sv = explainer.shap_values(engine.X_sample_, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[0]
    assert np.allclose(engine.consensus_shap_, np.asarray(sv, dtype=np.float64), atol=1e-6)
    assert len(f_names) == len(ensemble.feature_names_)


def test_shap_export_artifacts(small_ensemble, tmp_path):
    ensemble, X, _, _ = small_ensemble
    engine = ShapAttributionEngine(
        ensemble=ensemble,
        output_dir=tmp_path,
        pollutant="pm25",
        max_samples=100,
        forest_max_samples=60,
    )
    result = engine.compute_and_export(X)

    csv_path = result["csv"]
    bar_path = result["summary_bar"]
    assert csv_path.exists() and csv_path.name == "shap_importance_pm25.csv"
    assert bar_path.exists() and bar_path.name == "shap_summary_bar_pm25.png"
    if result["beeswarm"] is not None:
        assert result["beeswarm"].exists() and result["beeswarm"].name == "shap_beeswarm_pm25.png"

    df = pd.read_csv(csv_path)
    assert len(df) == len(ensemble.feature_names_)


def test_shap_requires_fitted(tmp_path):
    from src.models import DustStackingEnsemble as Ens
    bare = Ens(config={}, pollutant="pm25")
    engine = ShapAttributionEngine(ensemble=bare, output_dir=tmp_path)
    with pytest.raises(RuntimeError):
        engine.compute(pd.DataFrame({"a": [1.0]}))
