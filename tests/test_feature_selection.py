"""Tests for scripts/correlation_clustering.py (Step A-lite)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_script(monkeypatch, tmp_path, df: pd.DataFrame, extra_args=None):
    import scripts.correlation_clustering as cc

    mat = tmp_path / "matrix.parquet"
    df.to_parquet(mat)
    argv = [
        "correlation_clustering.py",
        "--matrix", str(mat),
        "--features", "all_physical",
        "--threshold", "0.9",
        "--out-dir", str(tmp_path),
        "--tag", "t",
    ] + (extra_args or [])
    monkeypatch.setattr(sys, "argv", argv)
    cc.main()
    feats = pd.read_csv(tmp_path / "correlation_features_t.csv")
    clus = pd.read_csv(tmp_path / "correlation_clusters_t.csv")
    return feats, clus


def test_correlated_pair_clustered_and_medoid_kept(monkeypatch, tmp_path):
    rng = np.random.RandomState(0)
    n = 200
    a = rng.normal(size=n)
    # b is ~identical to a -> same cluster; c is independent -> singleton.
    b = a + rng.normal(scale=1e-6, size=n)
    c = rng.normal(size=n)
    df = pd.DataFrame({"bt_b13": a, "b13_mean_3x3": b, "elevation_m": c})
    feats, clus = _run_script(monkeypatch, tmp_path, df)

    assert len(clus) == 2
    multi = clus[clus["cluster_size"] == 2].iloc[0]
    assert set(str(multi["members"]).split(";")) == {"bt_b13", "b13_mean_3x3"}
    keepers = feats[feats["is_keeper"]]
    assert len(keepers) == 2  # one keeper per cluster
    assert str(multi["keeper"]) in ("bt_b13", "b13_mean_3x3")


def test_constant_column_reported_not_clustered(monkeypatch, tmp_path):
    rng = np.random.RandomState(1)
    n = 50
    df = pd.DataFrame({
        "bt_b13": rng.normal(size=n),
        "elevation_m": np.full(n, 123.0),
    })
    feats, clus = _run_script(monkeypatch, tmp_path, df)
    # Constant col excluded from corr clustering entirely.
    assert "elevation_m" not in set(feats["feature"])
    assert set(feats["feature"]) == {"bt_b13"}


def test_importance_csv_selects_keeper(monkeypatch, tmp_path):
    rng = np.random.RandomState(2)
    n = 200
    a = rng.normal(size=n)
    df = pd.DataFrame({
        "bt_b13": a,
        "b13_mean_3x3": a + rng.normal(scale=1e-6, size=n),
    })
    imp = tmp_path / "imp.csv"
    pd.DataFrame({
        "feature": ["bt_b13", "b13_mean_3x3"],
        "ensemble_importance": [0.01, 0.99],
    }).to_csv(imp, index=False)
    feats, clus = _run_script(
        monkeypatch, tmp_path, df, extra_args=["--importance", str(imp)]
    )
    assert clus.iloc[0]["keeper"] == "b13_mean_3x3"
    assert clus.iloc[0]["drop_candidates"] == "bt_b13"
