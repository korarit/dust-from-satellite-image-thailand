"""Tests for scripts/repair_multiyear_matrix.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _fixture(tmp_path: Path):
    hii_dir = tmp_path / "hii"
    hours = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC").tz_convert(None)
    p_rows, h_rows = [], []
    for sc, la, lo in [("H1", 15.0, 100.0), ("H2", 16.0, 101.0)]:
        for i, ts in enumerate(hours[:5]):  # 5 of 6 hours -> one matchable-but-missing hour
            p_rows.append({"station_code": sc, "timestamp_utc": ts,
                          "variable": "pressure", "value": 1000.0 + i,
                          "latitude": la, "longitude": lo})
            h_rows.append({"station_code": sc, "timestamp_utc": ts,
                          "variable": "humidity", "value": 60.0 + i,
                          "latitude": la, "longitude": lo})
    _write(hii_dir / "hii_pressure_train.parquet", pd.DataFrame(p_rows))
    _write(hii_dir / "hii_humidity_train.parquet", pd.DataFrame(h_rows))

    # Matrix: S1 sits ~0.15 km from H1 (maps); S2 is far (no map).
    # S1X is outside HII time coverage (matchable-station, out-of-range ts).
    rng = np.random.RandomState(0)
    rows = []
    for sc, la, lo, ts_list in [
        ("S1", 15.001, 100.001, list(hours) + [pd.Timestamp("2025-06-01")]),
        ("S2", 50.0, 50.0, list(hours)),
    ]:
        for ts in ts_list:
            rows.append({"station_code": sc, "timestamp_utc": ts,
                         "lat": la, "lon": lo,
                         "elevation_m": 10.0,
                         "pressure_mean_surround": 999.0,  # bogus, must be overwritten
                         "humidity_mean_surround": 70.0,
                         "bt_b13": 280.0 + rng.normal(),
                         "observed_pm25": 20.0 + rng.normal()})
    mat_path = tmp_path / "matrix.parquet"
    _write(mat_path, pd.DataFrame(rows))

    # Master grid with asymmetric, dense elevation around S1 (TPI/relief nonzero).
    glat, glon, gelev = [], [], []
    for i in range(5):
        for j in range(5):
            glat.append(15.0 + (i - 2) * 0.01)
            glon.append(100.0 + (j - 2) * 0.01)
            gelev.append(20.0 + i * 3.0 + j * 1.0)
    grid_path = tmp_path / "grid.parquet"
    _write(grid_path, pd.DataFrame({
        "grid_id": [f"g{k}" for k in range(25)],
        "centroid_lat": glat, "centroid_lon": glon, "elevation_m": gelev,
    }))
    return mat_path, hii_dir, grid_path


def _run(monkeypatch, *argv):
    import scripts.repair_multiyear_matrix as rep
    monkeypatch.setattr(sys, "argv", ["repair_multiyear_matrix.py"] + list(argv))
    rep.main()


def test_repair_merges_overwrites_and_topo(monkeypatch, tmp_path):
    mat_path, hii_dir, grid_path = _fixture(tmp_path)
    out = tmp_path / "repaired.parquet"
    audit_dir = tmp_path / "audit"
    _run(monkeypatch,
          "--matrix", str(mat_path), "--hii-dir", str(hii_dir),
          "--master-grid", str(grid_path), "--out", str(out),
          "--out-dir", str(audit_dir), "--split-date", "2026-01-01",
          "--max-match-km", "50", "--min-match-rate", "0.0")

    assert out.exists()
    rep = pd.read_parquet(out)
    # S1 rows inside HII coverage get REAL HII values (bogus 999 overwritten).
    s1_in = rep[(rep["station_code"] == "S1") &
                (rep["timestamp_utc"] < pd.Timestamp("2024-01-01 05:00"))]
    assert len(s1_in) == 5
    assert (s1_in["pressure_mean_surround"] < 1010).all()
    assert (s1_in["pressure_mean_surround"] >= 1000).all()
    assert (s1_in["humidity_mean_surround"] >= 60).all()
    # S2 (far) keeps no HII values but is fully imputed (no NaNs).
    assert float(rep["pressure_mean_surround"].isna().mean()) == 0.0
    # TPI/relief attached from terrain (vary around S1 grid).
    assert "tpi_index" in rep.columns and "relief_m" in rep.columns
    assert float(rep["tpi_index"].std()) > 0 or float(rep["relief_m"].std()) > 0
    # Unmatched cells were NaNed then re-imputed + flagged ...
    s2 = rep[rep["station_code"] == "S2"]
    assert (s2["imputed_flag"] == 1).all()
    # pressure: train median scaled by barometric elevation factor (elev=10m);
    # humidity: plain train median.
    import numpy as _np
    f_baro = (1.0 - (0.0065 * 10.0) / 288.15) ** 5.25588
    _np.testing.assert_allclose(
        s2["pressure_mean_surround"].to_numpy(), 1002.0 * f_baro, rtol=1e-5)
    assert (s2["humidity_mean_surround"] == 62.0).all()  # median of 60..64
    # ... while matched cells keep real values (bogus 999 gone).
    # NOTE: imputed_flag is 1 for every fixture row because the tiny fixture
    # lacks most schema columns (imputer flags those globally); the NaN->impute
    # behavior itself is proven by the S2 value assertions above.
    assert (s1_in["pressure_mean_surround"] != 999.0).all()
    assert (audit_dir / "repair_audit.json").exists()
    import json as _ja
    assert _ja.loads((audit_dir / "repair_audit.json").read_text())["naned_for_reimpute"][
        "pressure_mean_surround"] == 8  # 6x S2 + S1 05:00 + S1 out-of-range
    # Audit artifacts exist with match info.
    import json
    audit = json.loads((audit_dir / "repair_audit.json").read_text(encoding="utf-8"))
    assert audit["mapping"]["mapped_pressure_within_km"] == 1
    assert audit["match"]["pressure"]["matched_rows"] > 0
    assert (audit_dir / "repair_audit.md").exists()


def test_dead_station_excluded_from_mapping(monkeypatch, tmp_path):
    """A station with coords but zero valid hourly rows must not receive mapping."""
    import json as _json

    hii_dir = tmp_path / "hii_dead"
    hii_dir.mkdir(parents=True, exist_ok=True)
    good_ts = pd.date_range("2024-01-01", periods=3, freq="h")
    good_str = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in good_ts]
    _write(hii_dir / "hii_pressure_t.parquet", pd.DataFrame(
        [{"station_code": "H-DEAD", "timestamp_utc": "not-a-time",
          "variable": "pressure", "value": 1000.0,
          "latitude": 15.0, "longitude": 100.0}] +
        [{"station_code": "H-FAR", "timestamp_utc": ts,
          "variable": "pressure", "value": 1000.0,
          "latitude": 60.0, "longitude": 60.0} for ts in good_str]))
    _write(hii_dir / "hii_humidity_t.parquet", pd.DataFrame([
        {"station_code": "H-FAR", "timestamp_utc": ts, "variable": "humidity",
         "value": 60.0, "latitude": 60.0, "longitude": 60.0} for ts in good_str]))
    mat_path = tmp_path / "m3.parquet"
    _write(mat_path, pd.DataFrame([
        {"station_code": "SX", "timestamp_utc": ts, "lat": 15.0, "lon": 100.0,
         "elevation_m": 10.0, "pressure_mean_surround": 999.0,
         "humidity_mean_surround": 70.0, "bt_b13": 280.0, "observed_pm25": 20.0}
        for ts in good_ts]))
    _run(monkeypatch,
          "--matrix", str(mat_path), "--hii-dir", str(hii_dir),
          "--out", str(tmp_path / "repaired3.parquet"),
          "--out-dir", str(tmp_path / "audit3"),
          "--min-match-rate", "0.0")
    audit = _json.loads((tmp_path / "audit3" / "repair_audit.json").read_text())
    # H-DEAD has no valid rows -> excluded; H-FAR is >50km away -> no mapping.
    assert audit["mapping"]["mapped_pressure_within_km"] == 0
    rep = pd.read_parquet(tmp_path / "repaired3.parquet")
    assert float(rep["pressure_mean_surround"].isna().mean()) == 0.0


def test_fail_loud_on_low_match_rate(monkeypatch, tmp_path):
    # HII missing a MIDDLE hour -> matchable-but-unmatched row -> gate must fire.
    hii_dir = tmp_path / "hii2"
    hours = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC").tz_convert(None)
    keep = [h for i, h in enumerate(hours) if i != 2]
    _write(hii_dir / "hii_pressure_t.parquet", pd.DataFrame([
        {"station_code": "H1", "timestamp_utc": ts, "variable": "pressure",
         "value": 1000.0, "latitude": 15.0, "longitude": 100.0} for ts in keep]))
    _write(hii_dir / "hii_humidity_t.parquet", pd.DataFrame([
        {"station_code": "H1", "timestamp_utc": ts, "variable": "humidity",
         "value": 60.0, "latitude": 15.0, "longitude": 100.0} for ts in keep]))
    mat_path = tmp_path / "m2.parquet"
    _write(mat_path, pd.DataFrame([
        {"station_code": "S1", "timestamp_utc": ts, "lat": 15.001, "lon": 100.001,
         "elevation_m": 10.0, "pressure_mean_surround": 999.0,
         "humidity_mean_surround": 70.0, "bt_b13": 280.0, "observed_pm25": 20.0}
        for ts in hours]))
    with pytest.raises(SystemExit):
        _run(monkeypatch,
              "--matrix", str(mat_path), "--hii-dir", str(hii_dir),
              "--out", str(tmp_path / "repaired.parquet"),
              "--out-dir", str(tmp_path / "audit"),
              "--min-match-rate", "0.9")  # 3/4 matchable matched -> must fail loud
    # Gate failure still leaves a forensic partial audit behind.
    import json as _json
    fail_audit = _json.loads((tmp_path / "audit" / "repair_audit_failed.json").read_text())
    assert fail_audit["status"] == "failed_gate"
    assert fail_audit["gate_failure"]["variable"] in ("pressure", "humidity")
    assert "worst_stations" in fail_audit["gate_failure"]
    assert "by_month" in fail_audit["gate_failure"]


def test_no_hii_files_skips_merge_gracefully(monkeypatch, tmp_path):
    mat_path, _, grid_path = _fixture(tmp_path)
    empty_hii = tmp_path / "empty_hii"
    empty_hii.mkdir()
    out = tmp_path / "repaired.parquet"
    _run(monkeypatch,
          "--matrix", str(mat_path), "--hii-dir", str(empty_hii),
          "--master-grid", str(grid_path), "--out", str(out),
          "--out-dir", str(tmp_path / "audit2"),
          "--min-match-rate", "0.0")
    rep = pd.read_parquet(out)
    assert float(rep["pressure_mean_surround"].isna().mean()) == 0.0
