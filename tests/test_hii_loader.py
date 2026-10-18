"""Tests for HII full-coverage ingestion: station select, resume, retry."""

import io
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hii_weather_loader import HIIWeatherLoader


def _loader(tmp_path: Path, meta: pd.DataFrame) -> HIIWeatherLoader:
    loader = HIIWeatherLoader(hii_dir=tmp_path / "hii")
    loader._metadata_df = meta
    return loader


def _meta() -> pd.DataFrame:
    return pd.DataFrame({
        "station_code": ["S3", "S1", "S2"],
        "latitude": [15.0, 16.0, 17.0],
        "longitude": [100.0, 101.0, 102.0],
        "has_pressure": [True, True, False],
        "has_humidity": [False, True, True],
    })


def test_select_stations_capability_filter_and_cap(tmp_path):
    loader = _loader(tmp_path, _meta())
    assert loader.select_stations("pressure") == ["S1", "S3"]  # sorted, S2 lacks pressure
    assert loader.select_stations("humidity") == ["S1", "S2"]
    assert loader.select_stations("pressure", max_stations=1) == ["S1"]


def _month_df(station: str, hours) -> pd.DataFrame:
    return pd.DataFrame({
        "station_code": [station] * len(hours),
        "timestamp_utc": list(hours),
        "value": [1000.0] * len(hours),
        "variable": ["pressure"] * len(hours),
        "latitude": [15.0] * len(hours),
        "longitude": [100.0] * len(hours),
    })


def test_fetch_date_range_resumes_month_checkpoints(tmp_path, monkeypatch):
    loader = _loader(tmp_path, _meta())
    jan = pd.date_range("2024-01-01", periods=3, freq="h")
    feb = pd.date_range("2024-02-01", periods=2, freq="h")

    # Pre-existing January checkpoint (must not be refetched).
    ckpt = loader._month_checkpoint_path("pressure", 2024, 1)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    _month_df("S1", jan).to_parquet(ckpt, index=False)

    calls = []

    def fake_harmonize(variable, station_codes, year, month, max_workers=8):
        calls.append((year, month))
        if (year, month) == (2024, 2):
            return _month_df("S1", feb)
        return pd.DataFrame()

    monkeypatch.setattr(loader, "fetch_and_harmonize", fake_harmonize)
    out = loader.fetch_date_range(
        variable="pressure",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-02-29"),
        station_codes=["S1"],
    )
    assert calls == [(2024, 2)]  # January skipped, February fetched
    assert len(out) == 5  # 3 cached + 2 fetched
    assert loader._month_checkpoint_path("pressure", 2024, 2).exists()
    # Empty months leave no checkpoint behind.
    monkeypatch.setattr(loader, "fetch_and_harmonize", lambda **kw: pd.DataFrame())
    loader.fetch_date_range(
        variable="pressure",
        start_date=pd.Timestamp("2024-03-01"),
        end_date=pd.Timestamp("2024-03-31"),
        station_codes=["S1"],
    )
    assert not loader._month_checkpoint_path("pressure", 2024, 3).exists()


def test_ingest_hii_full_saves_splits(tmp_path, monkeypatch):
    import run_pipeline as rp

    monkeypatch.setattr(rp, "HII_DIR", tmp_path / "hii")
    ts_train = pd.date_range("2025-12-31 22:00", periods=2, freq="h", tz="UTC")
    ts_test = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({
        "station_code": ["S1"] * 5,
        "timestamp_utc": list(ts_train) + list(ts_test),
        "value": [1000.0] * 5,
        "variable": ["pressure"] * 5,
        "latitude": [15.0] * 5,
        "longitude": [100.0] * 5,
    })
    monkeypatch.setattr(
        HIIWeatherLoader, "fetch_date_range",
        lambda self, *a, **kw: df.copy(),
    )
    monkeypatch.setattr(
        HIIWeatherLoader, "select_stations",
        lambda self, variable, max_stations=None: ["S1"],
    )
    saved = rp.ingest_hii_full(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2026-02-01", tz="UTC"),
        split_date="2026-01-01",
    )
    assert set(saved) == {"pressure_train", "pressure_test",
                          "humidity_train", "humidity_test"}
    train = pd.read_parquet(saved["pressure_train"])
    test = pd.read_parquet(saved["pressure_test"])
    assert len(train) == 2 and len(test) == 3
    assert (train["split"] == "train").all() and (test["split"] == "test").all()


def test_fetch_retries_transient_failure_but_not_404(tmp_path, monkeypatch):
    loader = _loader(tmp_path, _meta())

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"a,b\n1,2\n"

    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection reset")
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    assert loader.fetch_station_raw_csv("pressure", 2024, 1, "S1") == b"a,b\n1,2\n"
    assert calls["n"] == 2

    def not_found(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO())

    monkeypatch.setattr("urllib.request.urlopen", not_found)
    calls["n"] = 0
    assert loader.fetch_station_raw_csv("pressure", 2024, 1, "NOPE") is None
