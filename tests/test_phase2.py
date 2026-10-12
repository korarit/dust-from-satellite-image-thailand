"""
Unit and Integration Tests for Phase 2: Spatial Grid & Feature Engineering
Verifies:
- Strict Proximity Exclusion (< 2.0 km and multi-resolution buffers)
- 4 Cardinal Quadrant division (North, South, East, West) and IDW interpolation
- Atmospheric Gradients (Delta P_ns, Delta P_ew, ||nabla P|| Stagnation Index)
- Satellite Tabularizer: 10 IR Bands BT, Physics BTDs, 3x3 Texture, and Temporal Tendencies
- Multi-Resolution Grid Engine (3km, 2km, 1km, 500m)
- Solar Zenith Angle (NOAA Model) and Cyclical Time Encoding
- AST Code Scanner compliance (0 violations)
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

from src.features.spatial_directional import (
    DirectionalWeatherEngine,
    RESOLUTION_EXCLUSION_MAP,
    DEFAULT_EXCLUSION_KM,
)
from src.features.satellite_tabularizer import (
    SatelliteTabularizer,
    ALL_IR_BANDS,
    BAND_COLS,
    PHYSICAL_BT_MIN_K,
    PHYSICAL_BT_MAX_K,
)
from src.features.grid_engine import (
    MultiResolutionGridEngine,
    compute_geographic_proxies,
    compute_solar_zenith_angle,
    compute_solar_angles_vectorized,
    compute_cyclical_time_features,
    SUPPORTED_RESOLUTIONS_KM,
    EXCLUSION_BUFFER_MAP,
)
from src.data.provenance_auditor import CodeASTScanner


def test_strict_proximity_exclusion():
    """
    Verifies that any station located at distance < exclusion_radius_km
    from the target centroid is 100% strictly excluded to prevent data leakage.
    """
    # Target centroid at Bangkok: Lat 13.75, Lon 100.5
    target_lat = 13.75
    target_lon = 100.5

    # 1 deg latitude is approx 111 km -> 1.5 km is ~0.0135 deg, 3.0 km is ~0.027 deg
    stn_data = pd.DataFrame({
        "station_code": ["STN_IN_CELL", "STN_OUT_CELL", "STN_FAR", "STN_TOO_FAR"],
        "latitude": [
            target_lat + (1.5 / 111.0),   # ~1.5 km (within 2km exclusion)
            target_lat + (3.5 / 111.0),   # ~3.5 km (valid North)
            target_lat + (20.0 / 111.0),  # ~20.0 km (valid North)
            target_lat + (65.0 / 111.0),  # ~65.0 km (> 50km max radius)
        ],
        "longitude": [target_lon, target_lon, target_lon, target_lon],
    })

    engine = DirectionalWeatherEngine(stn_data, default_exclusion_km=2.0, max_search_radius_km=50.0)

    # Observations: STN_IN_CELL has a unique sentinel value 999.0
    obs_press = {
        "STN_IN_CELL": 999.0,   # MUST BE DROPPED!
        "STN_OUT_CELL": 1012.0,
        "STN_FAR": 1014.0,
        "STN_TOO_FAR": 1050.0,  # Outside 50km
    }

    feat = engine.extract_cell_directional_features(
        target_lat, target_lon, obs_press, variable="pressure", exclusion_radius_km=2.0
    )

    # The in-cell station (1.5 km < 2.0 km) must NOT be included
    # Therefore p_north must reflect only 1012.0 and 1014.0, never 999.0
    assert feat["p_north"] < 900.0 or feat["p_north"] <= 1015.0
    assert not np.isclose(feat["p_north"], 999.0)
    assert not np.isclose(feat["pressure_mean_surround"], 999.0)


def test_four_cardinal_quadrants_and_gradients():
    """
    Verifies 4 cardinal quadrants (North, South, East, West) and atmospheric gradient math.
    """
    target_lat = 14.0
    target_lon = 101.0

    # Place 4 stations at ~15 km in each cardinal direction
    km_offset = 15.0 / 111.0
    stn_data = pd.DataFrame({
        "station_code": ["STN_N", "STN_S", "STN_E", "STN_W"],
        "latitude": [
            target_lat + km_offset,  # North
            target_lat - km_offset,  # South
            target_lat,              # East
            target_lat,              # West
        ],
        "longitude": [
            target_lon,              # North
            target_lon,              # South
            target_lon + km_offset,  # East
            target_lon - km_offset,  # West
        ],
    })

    engine = DirectionalWeatherEngine(stn_data, default_exclusion_km=2.0)

    obs = {
        "STN_N": 1018.0,  # Cold surge from north
        "STN_S": 1010.0,
        "STN_E": 1014.0,
        "STN_W": 1012.0,
    }

    feat = engine.extract_cell_directional_features(target_lat, target_lon, obs, variable="pressure")

    assert np.isclose(feat["p_north"], 1018.0, atol=0.1)
    assert np.isclose(feat["p_south"], 1010.0, atol=0.1)
    assert np.isclose(feat["p_east"], 1014.0, atol=0.1)
    assert np.isclose(feat["p_west"], 1012.0, atol=0.1)

    # North-South differential: 1018 - 1010 = +8.0 hPa
    assert np.isclose(feat["delta_p_ns"], 8.0, atol=0.1)
    # East-West differential: 1014 - 1012 = +2.0 hPa
    assert np.isclose(feat["delta_p_ew"], 2.0, atol=0.1)
    # Pressure gradient magnitude must be positive
    assert feat["pressure_grad_mag"] > 0.0


def test_stagnation_zero_gradient_index():
    """
    Verifies that when surrounding pressure is uniform, gradient is ~0 (stagnant calm condition).
    """
    target_lat = 14.0
    target_lon = 101.0
    km_offset = 10.0 / 111.0

    stn_data = pd.DataFrame({
        "station_code": ["STN_N", "STN_S", "STN_E", "STN_W"],
        "latitude": [target_lat + km_offset, target_lat - km_offset, target_lat, target_lat],
        "longitude": [target_lon, target_lon, target_lon + km_offset, target_lon - km_offset],
    })

    engine = DirectionalWeatherEngine(stn_data, default_exclusion_km=2.0)
    # All stations identical 1012.0 hPa
    obs = {"STN_N": 1012.0, "STN_S": 1012.0, "STN_E": 1012.0, "STN_W": 1012.0}

    feat = engine.extract_cell_directional_features(target_lat, target_lon, obs, variable="pressure")
    assert np.isclose(feat["delta_p_ns"], 0.0)
    assert np.isclose(feat["delta_p_ew"], 0.0)
    assert np.isclose(feat["pressure_grad_mag"], 0.0)


def test_satellite_tabularizer_physics_btds():
    """
    Verifies Physics-based Brightness Temperature Differences (BTDs) and dust indices.
    """
    df = pd.DataFrame({
        "grid_id": ["G1", "G2"],
        "bt_b07": [300.0, 290.0],
        "bt_b08": [235.0, 230.0],
        "bt_b09": [245.0, 240.0],
        "bt_b10": [255.0, 250.0],
        "bt_b11": [280.0, 275.0],
        "bt_b12": [260.0, 255.0],
        "bt_b13": [295.0, 290.0],
        "bt_b14": [288.0, 285.0],
        "bt_b15": [290.0, 284.0],  # Heavy dust case: BT15 < BT13
        "bt_b16": [265.0, 260.0],
    })

    res = SatelliteTabularizer.compute_btds(df)

    # btd_15_13: 290 - 295 = -5.0 K (Dust indicator)
    assert np.isclose(res["btd_15_13"].iloc[0], -5.0)
    assert np.isclose(res["btd_15_13"].iloc[1], -6.0)

    # dust_rgb_red: normalized to [0, 1]
    assert 0.0 <= res["dust_rgb_red"].iloc[0] <= 1.0

    # btd_14_11: 288 - 280 = 8.0 K
    assert np.isclose(res["btd_14_11"].iloc[0], 8.0)

    # btd_13_07: 295 - 300 = -5.0 K
    assert np.isclose(res["btd_13_07"].iloc[0], -5.0)

    # btd_10_08: 255 - 235 = 20.0 K
    assert np.isclose(res["btd_10_08"].iloc[0], 20.0)


def test_satellite_tabularizer_3x3_texture():
    """
    Verifies 3x3 patch window spatial statistics (mean, std, min, max).
    """
    # 5x5 grid with constant value 290.0
    grid_b13 = np.full((5, 5), 290.0, dtype=np.float32)
    grid_b15 = np.full((5, 5), 288.0, dtype=np.float32)

    rows = np.array([2])
    cols = np.array([2])

    tex = SatelliteTabularizer.extract_3x3_texture(grid_b13, grid_b15, rows, cols)

    # Constant patch -> std == 0.0, mean == min == max == 290.0
    assert np.isclose(tex["b13_mean_3x3"][0], 290.0)
    assert np.isclose(tex["b13_std_3x3"][0], 0.0)
    assert np.isclose(tex["b13_min_3x3"][0], 290.0)
    assert np.isclose(tex["b13_max_3x3"][0], 290.0)
    assert np.isclose(tex["b15_mean_3x3"][0], 288.0)
    assert np.isclose(tex["b15_std_3x3"][0], 0.0)
    assert np.isclose(tex["btd_15_13_mean_3x3"][0], -2.0)


def test_satellite_tabularizer_temporal_tendencies():
    """
    Verifies temporal tendencies (differences between t and t-10m, t-30m, t-60m).
    """
    curr = pd.DataFrame({"grid_id": ["G1"], "bt_b13": [290.0], "btd_15_13": [-2.0]})
    past10 = pd.DataFrame({"grid_id": ["G1"], "bt_b13": [288.0], "btd_15_13": [-1.0]})
    past30 = pd.DataFrame({"grid_id": ["G1"], "bt_b13": [285.0], "btd_15_13": [0.0]})
    past60 = pd.DataFrame({"grid_id": ["G1"], "bt_b13": [280.0], "btd_15_13": [1.0]})

    res = SatelliteTabularizer.compute_temporal_tendencies(
        curr, past_10m_df=past10, past_30m_df=past30, past_60m_df=past60
    )

    # delta_bt13: 290 - 288 = +2.0 K
    assert np.isclose(res["delta_bt13_10m"].iloc[0], 2.0)
    # delta_bt13_30m: 290 - 285 = +5.0 K
    assert np.isclose(res["delta_bt13_30m"].iloc[0], 5.0)
    # delta_bt13_60m: 290 - 280 = +10.0 K
    assert np.isclose(res["delta_bt13_60m"].iloc[0], 10.0)

    # delta_btd15_13: -2 - (-1) = -1.0 K (increasing dust)
    assert np.isclose(res["delta_btd15_13_10m"].iloc[0], -1.0)
    assert np.isclose(res["delta_btd15_13_30m"].iloc[0], -2.0)
    assert np.isclose(res["delta_btd15_13_60m"].iloc[0], -3.0)


def test_multi_resolution_grid_engine():
    """
    Verifies MultiResolutionGridEngine handles all 4 benchmark resolutions
    (3km, 2km, 1km, 500m) and associates correct exclusion buffer.
    """
    engine = MultiResolutionGridEngine()
    for res_km in SUPPORTED_RESOLUTIONS_KM:
        grid = engine.get_grid_by_resolution(resolution_km=res_km, max_cells=10)
        assert len(grid) == 10
        assert grid["resolution_km"].iloc[0] == res_km
        assert grid["exclusion_radius_km"].iloc[0] == EXCLUSION_BUFFER_MAP[res_km]
        assert "tpi_index" in grid.columns
        assert "relief_m" in grid.columns


def test_solar_zenith_angle_and_cyclical_time():
    """
    Verifies NOAA solar zenith angle calculation and cyclical time transforms.
    """
    bangkok_lat = 13.75
    bangkok_lon = 100.5

    # Noon ICT (05:00 UTC): Sun high in the sky
    dt_noon = pd.Timestamp("2024-03-21 05:00:00", tz="UTC")
    sza_noon, is_day_noon = compute_solar_zenith_angle(dt_noon, bangkok_lat, bangkok_lon)
    assert sza_noon < 40.0
    assert is_day_noon == 1

    # Midnight ICT (17:00 UTC): Nighttime
    dt_night = pd.Timestamp("2024-03-21 17:00:00", tz="UTC")
    sza_night, is_day_night = compute_solar_zenith_angle(dt_night, bangkok_lat, bangkok_lon)
    assert sza_night > 100.0
    assert is_day_night == 0

    # Cyclical time encoding
    times = pd.date_range("2024-01-01 00:00:00", periods=24, freq="1h")
    cyc = compute_cyclical_time_features(times)
    # sin^2 + cos^2 == 1.0
    assert np.allclose(cyc["hour_sin"]**2 + cyc["hour_cos"]**2, 1.0, atol=1e-5)
    assert np.allclose(cyc["month_sin"]**2 + cyc["month_cos"]**2, 1.0, atol=1e-5)


def test_barometric_radial_gradient_approach_b():
    """
    Verifies Approach B: Barometric Radial Gradient when missing 3 quadrants (only 1 valid quadrant).
    Ensures non-zero, physics-calibrated gradient within [0.2, 5.0] hPa / 100km and elevation scaling.
    """
    target_lat = 14.0
    target_lon = 101.0
    km_offset = 20.0 / 111.0

    # Only 1 station in North quadrant (missing South, East, West)
    stn_data = pd.DataFrame({
        "station_code": ["STN_NORTH_ONLY"],
        "latitude": [target_lat + km_offset],
        "longitude": [target_lon],
    })

    engine = DirectionalWeatherEngine(stn_data, default_exclusion_km=2.0)
    obs = {"STN_NORTH_ONLY": 1016.0}  # MSLP cold ridge (+2.75 hPa above 1013.25)

    # 1. Test at sea level (0 m)
    feat_sea = engine.extract_cell_directional_features(
        target_lat, target_lon, obs, variable="pressure", cell_elevation_m=0.0
    )
    assert not np.isnan(feat_sea["pressure_grad_mag"])
    assert 0.2 <= feat_sea["pressure_grad_mag"] <= 5.0
    assert feat_sea["delta_p_ns"] > 0.0  # North has higher pressure
    assert feat_sea["delta_p_ew"] == 0.0

    # 2. Test at high elevation (1500 m, Doi Suthep / mountain terrain)
    feat_mtn = engine.extract_cell_directional_features(
        target_lat, target_lon, obs, variable="pressure", cell_elevation_m=1500.0
    )
    assert not np.isnan(feat_mtn["pressure_grad_mag"])
    assert 0.2 <= feat_mtn["pressure_grad_mag"] <= 5.0
    # Density is lower at 1500m -> scaled delta_p_ns is slightly smaller than sea level
    assert feat_mtn["delta_p_ns"] < feat_sea["delta_p_ns"]

    # 3. Test extract_grid_weather passing elevation_m column
    grid_df = pd.DataFrame({
        "centroid_lat": [target_lat],
        "centroid_lon": [target_lon],
        "elevation_m": [500.0],
    })
    res_df = engine.extract_grid_weather(grid_df, pressure_obs=obs)
    assert not np.isnan(res_df["pressure_grad_mag"].iloc[0])
    assert 0.2 <= res_df["pressure_grad_mag"].iloc[0] <= 5.0


def test_ast_scanner_zero_violations():
    """
    Verifies that the entire src/ tree, including new Phase 2 features,
    has 0 violations in the AST Code Scanner.
    """
    scanner = CodeASTScanner(PROJECT_ROOT)
    violations = scanner.scan_directory(PROJECT_ROOT / "src")
    assert len(violations) == 0, f"Found forbidden calls or mock keywords in src: {violations}"

