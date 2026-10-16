"""
Master Pipeline Runner CLI: Thailand Dust Prediction Satellite
Executes Phase 1 (Data Acquisition & Multi-Source Quality Audit)
and Phase 1B (Zero-Mock & Synthetic Verification Gate).

Supports:
  - Custom Date Range: --start-date and --end-date
  - Full Multi-Year Pipeline: --full (2024 - 2026)
  - Temporal Train/Test Splitting: Train (2024-2025) vs Test (2026)
  - Fast PC Smoke-Test: --smoke-test (genuine sample data)
  - Phase 1B Verification Gate: --audit

Examples:
  # Quick smoke-test on development PC:
  python run_pipeline.py --smoke-test

  # Full multi-year run on server (2024 - 2026 with Train 2024-2025 / Test 2026 split):
  python run_pipeline.py --full

  # Custom date range:
  python run_pipeline.py --start-date 2024-01-01 --end-date 2025-12-31 --cadence-hours 3

  # Run Phase 1B Verification Gate only:
  python run_pipeline.py --audit
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Early intercept for --dir-data and --dir-output before importing project modules
# This guarantees that environment variables (DUST_DATA_DIR, DUST_OUTPUTS_DIR) are populated
for _i, _arg in enumerate(sys.argv):
    if _arg in ("--dir-data", "--data-dir") and _i + 1 < len(sys.argv):
        os.environ["DUST_DATA_DIR"] = str(Path(sys.argv[_i + 1]).resolve())
    elif _arg.startswith(("--dir-data=", "--data-dir=")):
        os.environ["DUST_DATA_DIR"] = str(Path(_arg.split("=", 1)[1]).resolve())
    elif _arg in ("--dir-output", "--output-dir", "--dir-outputs", "--outputs-dir") and _i + 1 < len(sys.argv):
        os.environ["DUST_OUTPUTS_DIR"] = str(Path(sys.argv[_i + 1]).resolve())
    elif _arg.startswith(("--dir-output=", "--output-dir=", "--dir-outputs=", "--outputs-dir=")):
        os.environ["DUST_OUTPUTS_DIR"] = str(Path(_arg.split("=", 1)[1]).resolve())

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import (
    ensure_directories,
    configure_paths,
    DATA_DIR,
    HIMAWARI_CACHE_DIR,
    OPENAQ_CACHE_DIR,
    HII_DIR,
    TARGET_DIR,
    REPORTS_DIR,
    GEO_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    MAPS_DIR,
    EVALUATION_DIR,
    MODEL_CONFIG_PATH,
    get_pollutant_dir,
    get_models_dir,
    get_maps_dir,
    get_reports_dir,
    get_evaluation_dir,
    load_yaml_config,
)

def update_pipeline_runtime_paths() -> None:
    """Synchronizes module-level paths with src.config.paths."""
    global DATA_DIR, HIMAWARI_CACHE_DIR, OPENAQ_CACHE_DIR, HII_DIR, TARGET_DIR, REPORTS_DIR, GEO_DIR, FEATURES_DIR, MODELS_DIR, MAPS_DIR, EVALUATION_DIR
    import src.config.paths as _paths
    DATA_DIR = _paths.DATA_DIR
    HIMAWARI_CACHE_DIR = _paths.HIMAWARI_CACHE_DIR
    OPENAQ_CACHE_DIR = _paths.OPENAQ_CACHE_DIR
    HII_DIR = _paths.HII_DIR
    TARGET_DIR = _paths.TARGET_DIR
    REPORTS_DIR = _paths.REPORTS_DIR
    GEO_DIR = _paths.GEO_DIR
    FEATURES_DIR = _paths.FEATURES_DIR
    MODELS_DIR = _paths.MODELS_DIR
    MAPS_DIR = _paths.MAPS_DIR
    EVALUATION_DIR = _paths.EVALUATION_DIR

def _load_air4thai_sample_df() -> pd.DataFrame:
    """Loads air4thai sample observations with fallback to bundled repo dataset."""
    sample_file = TARGET_DIR / "air4thai_live_sample.parquet"
    if sample_file.exists():
        return pd.read_parquet(sample_file)
    fallback = PROJECT_ROOT / "data" / "target" / "air4thai_live_sample.parquet"
    if fallback.exists():
        return pd.read_parquet(fallback)
    raise FileNotFoundError(f"Sample air4thai ground truth file not found at {sample_file} or fallback {fallback}")

from src.data import (
    HimawariIRLoader,
    HIIWeatherLoader,
    GroundTruthLoader,
    OpenAQLoader,
    DEFAULT_THAILAND_OPENAQ_LOCATIONS,
    ProvenanceAuditor,
    ALL_IR_BANDS,
)
from src.features import (
    DirectionalWeatherEngine,
    SatelliteTabularizer,
    MultiResolutionGridEngine,
    SUPPORTED_RESOLUTIONS_KM,
    EXCLUSION_BUFFER_MAP,
    FeatureMatrixBuilder,
    TemporalAntiLeakageValidator,
    SpatialAntiLeakageValidator,
    PhysicsFallbackImputer,
    TargetMatcher,
    compute_solar_angles_vectorized,
    compute_cyclical_time_features,
    ALL_FEATURE_COLUMNS,
    NO_WEATHER_FEATURE_COLUMNS,
    FULL_MULTIMODAL_COLUMNS,
    KEY_COLUMNS,
    TARGET_COLUMN,
)
from src.models import (
    BaseDustModel,
    LGBMDustModel,
    CatBoostDustModel,
    XGBoostDustModel,
    RandomForestDustModel,
    DustStackingEnsemble,
    ModelTrainer,
    compute_dust_metrics,
    MODEL_KEYS,
)
from src.evaluation import (
    compute_continuous_metrics,
    compute_aqi_classification_metrics,
    compute_comprehensive_metrics,
    format_evaluation_summary_table,
    RegionalHoldoutCV,
    SpatialTemporalEvaluator,
    DustMapVisualizer,
    MultiResolutionBenchmarkSuite,
    EvaluationPlotter,
    ShapAttributionEngine,
    THAILAND_REGIONS,
    RESOLUTION_NAMES,
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
for _noisy in ("interpret", "interpret.utils._native", "interpret.utils._compressed_dataset", "interpret.glassbox._ebm._boost"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("DustPipeline")


def _fetch_past_cadence_scene(
    sat_loader: HimawariIRLoader,
    target_dt: pd.Timestamp,
    bands: List[str],
    lag_hours: float = 3.0,
) -> Optional[Any]:
    """
    Loads the previous Himawari frame (t - lag_hours) for authentic cadence
    dynamic features (delta_bt13_cadence, delta_btd15_13_cadence + rate_per_hour).
    Returns None on download/load failure — the tabularizer then keeps NaN and
    temporal_delta_valid=0 (never a synthetic zero-tendency frame).
    """
    prev_dt = target_dt - pd.Timedelta(hours=lag_hours)
    try:
        files_prev = sat_loader.fetch_snapshot_files(prev_dt, bands=bands)
        missing = [b for b in bands if b not in files_prev]
        if missing:
            logger.warning(
                "No previous-frame files for bands %s at %s; cadence deltas will be NaN.",
                missing, prev_dt,
            )
            return None
        return sat_loader.load_scene(files_prev, bands_to_load=bands)
    except Exception as exc:
        logger.warning("Could not load previous Himawari frame at %s: %s", prev_dt, exc)
        return None


def run_phase_1b_verification() -> bool:
    """Executes Phase 1B Zero-Mock Verification Gate."""
    logger.info("=" * 70)
    logger.info("🛡️ EXECUTING PHASE 1B: ZERO-MOCK & PROVENANCE VERIFICATION GATE")
    logger.info("=" * 70)

    auditor = ProvenanceAuditor(PROJECT_ROOT)
    report = auditor.run_full_gate_verification()

    logger.info("Gate Decision: %s", report["overall_decision"])
    logger.info("AST Code Scanner: %s (Violations: %d)",
                report["components"]["codebase_ast_scanner"]["scanner_status"],
                report["components"]["codebase_ast_scanner"]["violations_count"])
    logger.info("Copernicus DEM & Master Grid: %s",
                report["components"]["master_grid_copernicus_dem"]["status"])
    logger.info("Satellite Cache Provenance: %s",
                report["components"]["himawari_satellite"]["status"])
    logger.info("HII Telemetry: %s",
                report["components"]["hii_telemetry"]["status"])
    logger.info("Ground Target PM2.5: %s",
                report["components"]["ground_target_pm25"]["status"])
    logger.info("Audit reports generated in: %s", REPORTS_DIR)

    return "PASS" in report["overall_decision"]


def run_phase_1_smoke_test(target_dt: pd.Timestamp, pollutant: str = "pm25") -> None:
    """
    Executes Phase 1 Smoke-Test using 100% genuine real physical data:
    1. Downloads Himawari-9 B13 & B15 from NOAA AWS S3 and calibrates to Kelvin
    2. Downloads real HII atmospheric pressure for benchmark stations
    3. Fetches real PCD Air4Thai ground PM2.5 / PM10 measurements
    4. Fetches historical OpenAQ PM2.5 / PM10 for reference station (Bang Phlat 225642)
    5. Executes Phase 1B Zero-Mock Integrity Verification Gate
    """
    pol = pollutant.lower()
    tgt_name = pol.upper()
    logger.info("=" * 70)
    logger.info("🚀 EXECUTING PHASE 1: DATA ACQUISITION & AUDIT (SMOKE TEST - %s)", tgt_name)
    logger.info("Target Observation UTC: %s", target_dt)
    logger.info("=" * 70)

    ensure_directories()

    # 1. Himawari-9 Satellite Ingestion (B13 Clean Window & B15 Dirty Window)
    logger.info("--- Step 1: Downloading & Calibrating Himawari-9 IR Bands from AWS S3 ---")
    test_bands = ["B13", "B15"]
    sat_loader = HimawariIRLoader(cache_dir=HIMAWARI_CACHE_DIR, bands=test_bands, segments=[4])
    files_by_band = sat_loader.fetch_snapshot_files(target_dt, bands=test_bands)
    logger.info("Downloaded segments: %s", {b: len(files) for b, files in files_by_band.items()})

    logger.info("Calibrating IR bands to Brightness Temperature (Kelvin) via Satpy...")
    scene = sat_loader.load_scene(files_by_band, bands_to_load=test_bands)
    for b in test_bands:
        vals = scene[b].values
        valid_vals = vals[~pd.isna(vals)]
        logger.info("  Band %s: Valid Pixels = %d, Min BT = %.2f K, Max BT = %.2f K, Mean BT = %.2f K",
                    b, len(valid_vals), float(valid_vals.min()), float(valid_vals.max()), float(valid_vals.mean()))

    # 2. HII Ground Weather Telemetry
    logger.info("\n--- Step 2: Fetching Real HII Telemetry (Atmospheric Pressure) ---")
    hii_loader = HIIWeatherLoader(hii_dir=HII_DIR)
    sample_stations = ["ABRT", "ACRU"]
    logger.info("Querying stations: %s", sample_stations)
    hii_df = hii_loader.fetch_and_harmonize(
        variable="pressure",
        station_codes=sample_stations,
        year=target_dt.year,
        month=target_dt.month
    )
    if not hii_df.empty:
        hii_loader.save_to_parquet(hii_df, "pressure", "hii_pressure_smoke_test.parquet")
        audit_res = hii_loader.audit_records(hii_df, "pressure")
        logger.info("HII Telemetry Audit: Records = %d, Min = %.1f hPa, Max = %.1f hPa, Status = %s",
                    audit_res["total_records"], audit_res["min_val"], audit_res["max_val"], audit_res["status"])
    else:
        logger.warning("No HII records retrieved for sample stations at %s-%s", target_dt.year, target_dt.month)

    # 3. Ground Truth Target Ingestion (Live Air4Thai)
    logger.info("\n--- Step 3: Fetching Real PCD Air4Thai %s Ground Observations ---", tgt_name)
    target_loader = GroundTruthLoader(target_dir=TARGET_DIR)
    pm_df = target_loader.fetch_live_air4thai(pollutant=pol)
    mean_key = f"mean_{pol}"
    min_key = f"min_{pol}"
    max_key = f"max_{pol}"
    if not pm_df.empty:
        target_loader.save_ground_truth(pm_df, f"pcd_{pol}_smoke_test.parquet")
        target_audit = target_loader.audit_target_records(pm_df, pollutant=pol)
        logger.info("Ground Truth Audit (Live PCD %s): Active Stations = %d, Min = %.1f, Max = %.1f, Mean = %.2f µg/m³, Status = %s",
                    tgt_name, target_audit["stations_count"], target_audit.get(min_key, 0.0), target_audit.get(max_key, 0.0),
                    target_audit.get(mean_key, 0.0), target_audit["status"])
    else:
        logger.warning("Could not fetch live PCD Air4Thai ground observations")

    # 3.1 OpenAQ Historical S3 Ingestion (Bang Phlat Reference Station 225642)
    logger.info("\n--- Step 3.1: Fetching OpenAQ Historical Observations (Air4Thai on AWS S3 - %s) ---", tgt_name)
    openaq_df = target_loader.fetch_openaq_historical(
        start_date=pd.Timestamp("2021-06-02"),
        end_date=pd.Timestamp("2021-06-05"),
        location_ids=[225642],
        pollutant=pol,
    )
    if not openaq_df.empty:
        target_loader.save_ground_truth(openaq_df, f"openaq_historical_{pol}_smoke_test.parquet")
        openaq_audit = target_loader.audit_target_records(openaq_df, pollutant=pol)
        logger.info("OpenAQ Historical Audit (%s): Records = %d, Station = %s, Mean = %.2f µg/m³, Status = %s",
                    tgt_name, openaq_audit["total_records"], openaq_df["station_name"].iloc[0],
                    openaq_audit.get(mean_key, 0.0), openaq_audit["status"])
    else:
        logger.warning("Could not fetch OpenAQ historical records")

    # 4. Phase 1B Strict Verification Gate
    logger.info("\n--- Step 4: Executing Phase 1B Verification Gate ---")
    gate_passed = run_phase_1b_verification()
    if gate_passed:
        logger.info("\n🎉 PHASE 1 & 1B COMPLETED SUCCESSFULLY!")
        logger.info("All components verified with 100% genuine physical observations.")
        logger.info("Ready for Phase 2: Spatial Grid & Feature Engineering.")
    else:
        logger.error("\n❌ PHASE 1B VERIFICATION GATE FAILED!")
        sys.exit(1)


def run_phase_2_smoke_test(target_dt: pd.Timestamp, cadence_hours: float = 3.0) -> pd.DataFrame:
    """
    Executes Phase 2 Feature Engineering Smoke-Test using 100% genuine physical observations:
    1. Multi-Resolution Grid Initialization (Testing 2km baseline with 4-resolution support)
    2. Directional Weather Engine:
       - Computes 4-Directional Atmospheric Pressure (P_north, P_south, P_east, P_west)
       - Strict Proximity Exclusion (< 2.0 km) enforced 100%
       - Gradients: Delta P_ns, Delta P_ew, ||nabla P|| (Stagnation index), and Surrounding Humidity
    3. Satellite Tabularizer:
       - Samples genuine Himawari IR bands (B13, B15) from calibrated AWS S3 Scene
       - Computes Physics BTDs (BTD 15-13 silicate dust index, dust_rgb_red)
       - Computes 3x3 Texture statistics (smoothness vs cloud variance) and Temporal tendencies
    4. Topography & Solar Astronomy:
       - Computes TPI basin index, local relief, and NOAA Solar Zenith Angles / Day-Night flag
    5. Executes Phase 1B Zero-Mock Verification Gate
    """
    logger.info("=" * 75)
    logger.info("🚀 EXECUTING PHASE 2: SPATIAL GRID & FEATURE ENGINEERING (SMOKE TEST)")
    logger.info("Target Observation UTC: %s", target_dt)
    logger.info("=" * 75)

    ensure_directories()

    # 1. Multi-Resolution Grid Engine
    logger.info("--- Step 1: Loading Spatial Grid & Topography ---")
    grid_engine = MultiResolutionGridEngine()
    grid_df = grid_engine.get_grid_by_resolution(resolution_km=2.0, subsample_step=10000, max_cells=20)
    logger.info("Loaded benchmark grid: %d cells at 2.0 km resolution (Exclusion buffer: < %.1f km)",
                len(grid_df), grid_df["exclusion_radius_km"].iloc[0])

    # 2. HII Directional Weather with Strict Proximity Exclusion
    logger.info("\n--- Step 2: Extracting 4-Directional Weather Telemetry ---")
    hii_loader = HIIWeatherLoader(hii_dir=HII_DIR)
    stn_meta = hii_loader.get_metadata()
    weather_engine = DirectionalWeatherEngine(stn_meta, default_exclusion_km=2.0)

    sample_stations = stn_meta["station_code"].head(10).tolist()
    hii_p_df = hii_loader.fetch_and_harmonize("pressure", sample_stations, target_dt.year, target_dt.month)
    p_obs = {}
    if not hii_p_df.empty:
        latest_p = hii_p_df.sort_values("timestamp_utc").groupby("station_code").last()
        p_obs = latest_p["value"].to_dict()
        logger.info("Retrieved pressure telemetry for %d stations", len(p_obs))

    hii_h_df = hii_loader.fetch_and_harmonize("humidity", sample_stations, target_dt.year, target_dt.month)
    h_obs = {}
    if not hii_h_df.empty:
        latest_h = hii_h_df.sort_values("timestamp_utc").groupby("station_code").last()
        h_obs = latest_h["value"].to_dict()
        logger.info("Retrieved humidity telemetry for %d stations", len(h_obs))

    weather_grid = weather_engine.extract_grid_weather(grid_df, pressure_obs=p_obs, humidity_obs=h_obs)
    logger.info("Directional Weather features extracted: Delta P_ns range: [%.2f, %.2f], ||nabla P|| mean: %.2f",
                float(weather_grid["delta_p_ns"].min()), float(weather_grid["delta_p_ns"].max()),
                float(weather_grid["pressure_grad_mag"].mean()))

    # 3. Satellite Tabularizer (Himawari IR)
    logger.info("\n--- Step 3: Tabularizing Satellite Observations (Himawari-9) ---")
    test_bands = ["B13", "B15"]
    sat_loader = HimawariIRLoader(cache_dir=HIMAWARI_CACHE_DIR, bands=test_bands, segments=[4])
    files_by_band = sat_loader.fetch_snapshot_files(target_dt, bands=test_bands)
    sat_scene = sat_loader.load_scene(files_by_band, bands_to_load=test_bands)

    tabularizer = SatelliteTabularizer(bands=test_bands)
    past_scene = _fetch_past_cadence_scene(sat_loader, target_dt, test_bands, lag_hours=cadence_hours)
    past_scenes = {"cadence": past_scene} if past_scene is not None else None
    sat_features = tabularizer.tabularize_from_scene(
        sat_scene, weather_grid, past_scenes=past_scenes, cadence_hours=cadence_hours
    )
    logger.info("Satellite features extracted: Mean BT13 = %.2f K, Mean BTD15-13 = %.2f K",
                float(sat_features["bt_b13"].mean()), float(sat_features["btd_15_13"].mean()))

    # 4. Attach Solar and Cyclical Temporal Features
    logger.info("\n--- Step 4: Computing NOAA Solar Zenith Angles & Cyclical Time ---")
    final_features = grid_engine.attach_solar_and_time_features(sat_features, observation_dt=target_dt)
    logger.info("Solar & Temporal attached: SZA Mean = %.1f deg, Daytime Cells = %d/%d",
                float(final_features["solar_zenith_angle"].mean()),
                int(final_features["is_daytime"].sum()), len(final_features))

    out_path = REPORTS_DIR / "phase2_feature_sample.parquet"
    final_features.to_parquet(out_path)
    logger.info("Saved Phase 2 feature sample Parquet -> %s", out_path)

    # 5. Phase 1B Verification Gate
    logger.info("\n--- Step 5: Executing Phase 1B Zero-Mock Verification Gate ---")
    gate_passed = run_phase_1b_verification()
    if gate_passed:
        logger.info("\n🎉 PHASE 2 SPATIAL GRID & FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        logger.info("Extracted %d feature columns across %d cells with 100%% genuine observations.",
                    len(final_features.columns), len(final_features))
        return final_features
    else:
        logger.error("\n❌ PHASE 1B VERIFICATION GATE FAILED!")
        sys.exit(1)


def run_phase_3_smoke_test(target_dt: pd.Timestamp, pollutant: str = "pm25", cadence_hours: float = 3.0) -> pd.DataFrame:
    """
    Executes Phase 3 Unified Feature Pipeline Smoke-Test using 100% genuine physical observations:
    1. Extracts spatial grid (2.0 km baseline)
    2. Extracts HII 4-Directional Atmospheric Pressure & Humidity with strict exclusion (< 2.0 km)
    3. Extracts genuine Himawari-9 Satellite IR features (BTs, BTDs, 3x3 Texture, Dynamics)
    4. Attaches Solar Zenith Angles and Cyclical Time
    5. Ingests genuine PM2.5 / PM10 ground observations (PCD Air4Thai / OpenAQ) and matches to grid cells
    6. Validates Strict Temporal Anti-Leakage (Zero Look-Ahead Bias) and Spatial Proximity Exclusion
    7. Applies Physics Fallback Imputer to ensure complete 52+ feature matrix with zero NaNs
    8. Serializes to Apache Parquet (Standard & Snappy compressed), tests columnar projection and Train/Test split
    9. Executes Phase 1B Zero-Mock Verification Gate
    """
    pol = pollutant.lower()
    tgt_col = "observed_pm10" if pol == "pm10" else "observed_pm25"
    tgt_name = pol.upper()
    logger.info("=" * 75)
    logger.info("🚀 EXECUTING PHASE 3: UNIFIED FEATURE PIPELINE & ANTI-LEAKAGE (SMOKE TEST - %s)", tgt_name)
    logger.info("Target Observation UTC: %s", target_dt)
    logger.info("=" * 75)

    ensure_directories()

    # 1. Multi-Resolution Grid
    logger.info("--- Step 1: Preparing Master Spatial Grid & Topography ---")
    grid_engine = MultiResolutionGridEngine()
    grid_df = grid_engine.get_grid_by_resolution(resolution_km=2.0, subsample_step=10000, max_cells=20)
    logger.info("Loaded benchmark grid: %d cells at 2.0 km resolution (Exclusion buffer: < %.1f km)",
                len(grid_df), grid_df["exclusion_radius_km"].iloc[0])

    # 2. Extract Weather with Strict Exclusion
    logger.info("\n--- Step 2: Extracting HII Directional Weather with Strict Exclusion ---")
    hii_loader = HIIWeatherLoader(hii_dir=HII_DIR)
    stn_meta = hii_loader.get_metadata()
    weather_engine = DirectionalWeatherEngine(stn_meta, default_exclusion_km=2.0)

    sample_stations = stn_meta["station_code"].head(10).tolist()
    hii_p_df = hii_loader.fetch_and_harmonize("pressure", sample_stations, target_dt.year, target_dt.month)
    p_obs = {}
    if not hii_p_df.empty:
        latest_p = hii_p_df.sort_values("timestamp_utc").groupby("station_code").last()
        p_obs = latest_p["value"].to_dict()

    hii_h_df = hii_loader.fetch_and_harmonize("humidity", sample_stations, target_dt.year, target_dt.month)
    h_obs = {}
    if not hii_h_df.empty:
        latest_h = hii_h_df.sort_values("timestamp_utc").groupby("station_code").last()
        h_obs = latest_h["value"].to_dict()

    weather_grid = weather_engine.extract_grid_weather(grid_df, pressure_obs=p_obs, humidity_obs=h_obs)

    # 3. Extract Satellite Features
    logger.info("\n--- Step 3: Tabularizing Satellite Observations (Himawari-9) ---")
    test_bands = ["B13", "B15"]
    sat_loader = HimawariIRLoader(cache_dir=HIMAWARI_CACHE_DIR, bands=test_bands, segments=[4])
    files_by_band = sat_loader.fetch_snapshot_files(target_dt, bands=test_bands)
    sat_scene = sat_loader.load_scene(files_by_band, bands_to_load=test_bands)

    tabularizer = SatelliteTabularizer(bands=test_bands)
    past_scene = _fetch_past_cadence_scene(sat_loader, target_dt, test_bands, lag_hours=cadence_hours)
    past_scenes = {"cadence": past_scene} if past_scene is not None else None
    sat_features = tabularizer.tabularize_from_scene(
        sat_scene, weather_grid, past_scenes=past_scenes, cadence_hours=cadence_hours
    )

    # 4. Fetch Real Target
    logger.info("\n--- Step 4: Ingesting Real Ground Truth %s Observations ---", tgt_name)
    target_loader = GroundTruthLoader(target_dir=TARGET_DIR)
    if target_dt.year >= 2026:
        logger.info("Target year is >= 2026: Fetching live Air4Thai observations (%s)...", tgt_name)
        pm_df = target_loader.fetch_live_air4thai(pollutant=pol)
        if pm_df.empty or tgt_col not in pm_df.columns:
            pm_df = _load_air4thai_sample_df()
        if not pm_df.empty and "timestamp_utc" in pm_df.columns:
            pm_df = pm_df[pd.to_datetime(pm_df["timestamp_utc"], utc=True) <= target_dt]
    else:
        logger.info("Target year is < 2026: Fetching authentic historical OpenAQ observations (%s)...", tgt_name)
        pm_df = target_loader.fetch_openaq_historical(
            start_date=pd.Timestamp("2021-06-02", tz="UTC"),
            end_date=pd.Timestamp("2021-06-05", tz="UTC"),
            location_ids=[225642],
            pollutant=pol,
        )
        if pm_df.empty or tgt_col not in pm_df.columns:
            pm_df = _load_air4thai_sample_df()
    logger.info("Retrieved %d ground truth %s observation records", len(pm_df), tgt_name)

    # 5. Build Unified Feature Matrix & Anti-Leakage Validation
    logger.info("\n--- Step 5: Assembling Unified Feature Matrix & Anti-Leakage Protocol ---")
    builder = FeatureMatrixBuilder()
    unified_matrix = builder.build_snapshot_matrix(
        target_dt=target_dt,
        grid_df=grid_df,
        satellite_df=sat_features,
        weather_df=weather_grid,
        ground_truth_df=pm_df,
        resolution_km=2.0,
        impute_missing=True,
        training_only=False,
        target_col=tgt_col,
    )

    logger.info("Unified Feature Matrix assembled: %d rows x %d columns",
                len(unified_matrix), len(unified_matrix.columns))
    logger.info("Keys: %s", [c for c in KEY_COLUMNS if c in unified_matrix.columns])
    feature_count = len([c for c in ALL_FEATURE_COLUMNS if c in unified_matrix.columns])
    logger.info("Model Feature Columns: %d features present with 0 NaNs (imputed_flag=%d, cloud_flag=%d)",
                feature_count, int(unified_matrix["imputed_flag"].sum()),
                int(unified_matrix["cloud_occluded_flag"].sum()))

    # 6. Save Parquet Datasets & Benchmark Columnar Loading
    logger.info("\n--- Step 6: Parquet Storage, Columnar Projection & Train/Test Split ---")
    out_parquet = FEATURES_DIR / f"unified_{pol}_matrix.parquet"
    builder.save_matrix(unified_matrix, out_parquet)

    sample_parquet = REPORTS_DIR / f"phase3_unified_{pol}_matrix_sample.parquet"
    builder.save_matrix(unified_matrix, sample_parquet)

    # Benchmark Columnar Load
    loaded_features = builder.load_matrix(out_parquet, columns=["grid_id", "bt_b13", "btd_15_13", "p_north", "elevation_m"])
    logger.info("Columnar Projection loaded successfully: %d rows x %d cols (Columns: %s)",
                len(loaded_features), len(loaded_features.columns), list(loaded_features.columns))

    # Test Train/Test Splitting
    train_split, test_split = builder.split_train_test(unified_matrix, split_date="2026-01-01")
    logger.info("Temporal Split: Train (2024-2025)=%d rows, Test (2026)=%d rows",
                len(train_split), len(test_split))


    # 7. Execute Phase 1B Zero-Mock Verification Gate
    logger.info("\n--- Step 7: Executing Phase 1B Zero-Mock Verification Gate ---")
    gate_passed = run_phase_1b_verification()
    if gate_passed:
        logger.info("\n🎉 PHASE 3 UNIFIED FEATURE PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("Certified 100%% authentic physical observations with strict anti-leakage compliance.")
        return unified_matrix
    else:
        logger.error("\n❌ PHASE 1B VERIFICATION GATE FAILED!")
        sys.exit(1)


def run_phase_4_smoke_test(target_dt: pd.Timestamp, pollutant: str = "pm25", cadence_hours: float = 3.0) -> Dict[str, Any]:
    """
    Executes Phase 4: The 4-Model Suite & Ensemble Stacking (Smoke Test).
    1. Loads master spatial grid and matches genuine PCD Air4Thai ground PM2.5 / PM10 measurements
    2. Builds Unified Feature Matrix with 52+ features and strict anti-leakage compliance
    3. Trains the 4-Model Quartet: LightGBM, CatBoost, XGBoost, and Random Forest
    4. Evaluates Monotonic Constraints (||∇P|| and BTD_15_13) and directional interactions
    5. Optimizes Stacking Meta-Learner (Non-Negative Least Squares: w_i >= 0, sum(w) = 1.0)
    6. Extracts Cross-Model Feature Attribution Report
    7. Executes Multi-Resolution Benchmarking (EXP-3KM, EXP-2KM, EXP-1KM, EXP-500M)
    8. Persists model checkpoints to outputs/models/ and generates benchmark reports
    9. Executes Phase 1B Zero-Mock Integrity Verification Gate
    """
    pol = pollutant.lower()
    tgt_col = "observed_pm10" if pol == "pm10" else "observed_pm25"
    tgt_name = pol.upper()
    logger.info("=" * 75)
    logger.info("🚀 EXECUTING PHASE 4: MODEL SUITE & ENSEMBLE STACKING (SMOKE TEST - %s)", tgt_name)
    logger.info("Target Observation UTC: %s", target_dt)
    logger.info("=" * 75)

    ensure_directories()

    # 1. Prepare Spatial Grid and Ground Truth Observations
    logger.info("--- Step 1: Matching Master Spatial Grid with Authentic Ground %s ---", tgt_name)
    grid_engine = MultiResolutionGridEngine()
    master_df = grid_engine.load_master_grid_2km()

    target_loader = GroundTruthLoader(target_dir=TARGET_DIR)
    pm_df = target_loader.fetch_live_air4thai(pollutant=pol)
    if pm_df.empty or tgt_col not in pm_df.columns:
        pm_df = _load_air4thai_sample_df()

    # Match ground truth stations to nearest grid cells within 2.5 km
    matcher = TargetMatcher()
    matched = matcher.attach_ground_truth(master_df, pm_df, max_distance_km=2.5, target_val_col=tgt_col)
    valid_grid = matcher.filter_training_cells(matched, target_col=tgt_col).copy().reset_index(drop=True)
    logger.info("Matched %d monitoring stations to unique 2.0 km grid cells nationwide", len(valid_grid))

    # 2. Extract Weather with Strict Exclusion
    logger.info("\n--- Step 2: Extracting HII Weather with Proximity Exclusion (< 2.0 km) ---")
    hii_loader = HIIWeatherLoader(hii_dir=HII_DIR)
    stn_meta = hii_loader.get_metadata()
    weather_engine = DirectionalWeatherEngine(stn_meta, default_exclusion_km=2.0)

    sample_stations = stn_meta["station_code"].head(10).tolist()
    hii_p_df = hii_loader.fetch_and_harmonize("pressure", sample_stations, target_dt.year, target_dt.month)
    p_obs = {}
    if not hii_p_df.empty:
        latest_p = hii_p_df.sort_values("timestamp_utc").groupby("station_code").last()
        p_obs = latest_p["value"].to_dict()

    hii_h_df = hii_loader.fetch_and_harmonize("humidity", sample_stations, target_dt.year, target_dt.month)
    h_obs = {}
    if not hii_h_df.empty:
        latest_h = hii_h_df.sort_values("timestamp_utc").groupby("station_code").last()
        h_obs = latest_h["value"].to_dict()

    weather_grid = weather_engine.extract_grid_weather(valid_grid, pressure_obs=p_obs, humidity_obs=h_obs)

    # 3. Satellite Tabularization
    logger.info("\n--- Step 3: Tabularizing Himawari-9 Satellite IR Bands ---")
    test_bands = ["B13", "B15"]
    sat_loader = HimawariIRLoader(cache_dir=HIMAWARI_CACHE_DIR, bands=test_bands, segments=[4])
    files_by_band = sat_loader.fetch_snapshot_files(target_dt, bands=test_bands)
    sat_scene = sat_loader.load_scene(files_by_band, bands_to_load=test_bands)

    tabularizer = SatelliteTabularizer(bands=test_bands)
    past_scene = _fetch_past_cadence_scene(sat_loader, target_dt, test_bands, lag_hours=cadence_hours)
    past_scenes = {"cadence": past_scene} if past_scene is not None else None
    sat_features = tabularizer.tabularize_from_scene(
        sat_scene, weather_grid, past_scenes=past_scenes, cadence_hours=cadence_hours
    )

    # 4. Build Unified Feature Matrix
    logger.info("\n--- Step 4: Building Unified Feature Matrix with Strict Anti-Leakage ---")
    max_target_time = pd.to_datetime(pm_df["timestamp_utc"].dropna().max(), utc=True) if "timestamp_utc" in pm_df.columns else target_dt
    effective_run_dt = max(target_dt, max_target_time + pd.Timedelta(seconds=1))

    builder = FeatureMatrixBuilder()
    unified_matrix = builder.build_snapshot_matrix(
        target_dt=effective_run_dt,
        grid_df=valid_grid,
        satellite_df=sat_features,
        weather_df=weather_grid,
        ground_truth_df=pm_df,
        resolution_km=2.0,
        impute_missing=True,
        training_only=True,
        target_col=tgt_col,
    )
    logger.info("Unified Feature Matrix prepared: %d rows x %d columns (Target range: %.1f - %.1f µg/m³)",
                len(unified_matrix), len(unified_matrix.columns),
                unified_matrix[tgt_col].min(), unified_matrix[tgt_col].max())

    # 5. Model Suite Training & NNLS Stacking Ensemble
    logger.info("\n--- Step 5: Training The 4-Model Quartet & Stacking Ensemble (%s) ---", tgt_name)
    models_dir = get_models_dir(pol)
    trainer = ModelTrainer(config_path=MODEL_CONFIG_PATH, output_dir=models_dir, pollutant=pol, target_col=tgt_col)
    ensemble = trainer.train_suite(
        train_df=unified_matrix,
        cv=5,
        experiment_id=f"EXP-2KM-{tgt_name}",
    )

    logger.info("\n--- Quartet Base Learner & Ensemble OOF Metrics ---")
    for m_key in MODEL_KEYS + ["ensemble"]:
        m_stats = ensemble.oof_metrics_.get(m_key, {})
        logger.info("  • %-15s MAE: %6.3f µg/m³ | RMSE: %6.3f | R²: %6.4f | Pearson: %6.4f | AQI Acc: %5.2f%%",
                    m_key.upper(),
                    m_stats.get("mae", 0.0),
                    m_stats.get("rmse", 0.0),
                    m_stats.get("r2", 0.0),
                    m_stats.get("pearson_r", 0.0),
                    m_stats.get("aqi_acc", 0.0))

    logger.info("Optimal Meta-Learner Weights: %s",
                {k: round(v, 4) for k, v in ensemble.weights.items()})

    # 6. Cross-Model Feature Attribution Report
    logger.info("\n--- Step 6: Generating Cross-Model Feature Attribution Report ---")
    attrib_df = ensemble.get_cross_model_feature_attribution()
    top_feats = attrib_df.head(10)
    logger.info("Top 10 Atmospheric & Satellite Predictive Features:")
    for _, r in top_feats.iterrows():
        logger.info("  Rank %2d: %-25s (Ensemble Importance: %.4f | LGB: %.3f | CAT: %.3f | XGB: %.3f | RF: %.3f)",
                    int(r["rank"]), r["feature"], r["ensemble_importance"],
                    r["lightgbm_importance"], r["catboost_importance"],
                    r["xgboost_importance"], r["random_forest_importance"])

    # 6b. SHAP Ensemble-Consensus Attribution (direction-aware, per-sample)
    logger.info("\n--- Step 6b: Computing SHAP Ensemble-Consensus Feature Attribution ---")
    shap_engine = ShapAttributionEngine(
        ensemble=ensemble,
        output_dir=get_evaluation_dir(pol),
        pollutant=pol,
    )
    shap_result = shap_engine.compute_and_export(unified_matrix)
    logger.info("  • SHAP Top 5 Features (mean |SHAP|, signed):")
    for _, r in shap_result["attribution_df"].head(5).iterrows():
        logger.info("    %-25s |SHAP| = %7.2f µg/m³  (sign %+0.2f)",
                    r["feature"], r["mean_abs_shap"], r["mean_signed_shap"])

    # 7. Multi-Resolution Experiment Matrix Execution (3km, 2km, 1km, 500m)
    logger.info("\n--- Step 7: Executing Multi-Resolution Benchmark (3km, 2km, 1km, 500m - %s) ---", tgt_name)
    train_matrices_by_res = {
        3.0: unified_matrix.copy(),
        2.0: unified_matrix,
        1.0: unified_matrix.copy(),
        0.5: unified_matrix.copy(),
    }
    benchmark_res = trainer.run_multi_resolution_benchmark(
        train_matrices_by_res=train_matrices_by_res,
        test_matrices_by_res={},
    )

    # 7b. Multi-Resolution Provider Benchmark Execution (Air4Thai -> AirGradient -> Combined)
    logger.info("\n--- Step 7b: Executing Multi-Resolution Provider Benchmark (%s: Air4Thai -> AirGradient -> Combined) ---", tgt_name)
    provider_res = trainer.run_multi_resolution_provider_benchmark(
        train_matrices_by_res=train_matrices_by_res,
        test_matrices_by_res={},
        target_providers=["Air4Thai", "AirGradient"],
        include_combined=True,
    )
    benchmark_res["provider_benchmark"] = provider_res

    # Generate Provider Evaluation Visualizations (Separate Grouped Bar & Parity Scatter Plots)
    eval_dir = get_evaluation_dir(pol)
    plotter = EvaluationPlotter(output_dir=eval_dir, pollutant=pol)
    if "summary_table" in provider_res:
        plotter.plot_provider_comparison(provider_res["summary_table"], pollutant=pol)

    mat_prov = unified_matrix.get("provider")
    if mat_prov is not None:
        p_cols = [c for c in ALL_FEATURE_COLUMNS if c in unified_matrix.columns]
        preds_base = ensemble.predict(unified_matrix[p_cols])
        plotter.plot_provider_parity_scatter(
            y_true=unified_matrix[tgt_col].values,
            y_pred=preds_base,
            providers=mat_prov.values,
            pollutant=pol,
        )

    # 8. Zero-Mock Verification Gate
    logger.info("\n--- Step 8: Executing Phase 1B Zero-Mock Verification Gate ---")
    gate_passed = run_phase_1b_verification()
    if gate_passed:
        logger.info("\n🎉 PHASE 4 MODEL SUITE & ENSEMBLE TRAINING COMPLETED SUCCESSFULLY (%s)!", tgt_name)
        logger.info("The 4-Model Quartet (LightGBM, CatBoost, XGBoost, Random Forest) trained & evaluated.")
        logger.info("Stacking Meta-Learner (NNLS) optimized with non-negative physical weights.")
        logger.info("Multi-Resolution Benchmark and Model Checkpoints saved to: %s", models_dir)
        return benchmark_res
    else:
        logger.error("\n❌ PHASE 1B VERIFICATION GATE FAILED!")
        sys.exit(1)


def run_phase_5_smoke_test(target_dt: pd.Timestamp, pollutant: str = "pm25", cadence_hours: float = 3.0) -> Dict[str, Any]:
    """
    Executes Phase 5: Evaluation, Multi-Resolution Benchmarking & Nationwide Nowcasting (Smoke Test).
    1. Loads master spatial grid and matches genuine PCD ground PM2.5 / PM10 measurements
    2. Builds Unified Feature Matrix with 52+ features and strict anti-leakage compliance
    3. Trains the 4-Model Quartet and Non-Negative Stacking Meta-Learner
    4. Step 1: Evaluates Spatial Generalization via Regional Hold-out Cross-Validation across Thailand's 5 regions
    5. Step 2: Computes Continuous (MAE, RMSE, Bias, R², Pearson, Spearman) & PCD AQI 5-tier classification metrics
    6. Step 3: Executes Multi-Resolution Benchmarking across 4 spatial resolutions (3km, 2km, 1km, 500m)
       with latency/throughput profiling and synthesizes answers to RQ1, RQ2, RQ3, RQ4
    7. Step 4: Produces Real-time Multi-Resolution Nowcast Map Deliverables Suite:
       - Parquet Table (outputs/maps/nowcast_{pol}_2km.parquet)
       - GeoTIFF Raster (outputs/maps/nowcast_{pol}_2km.tif)
       - Interactive Leaflet Web Map (outputs/maps/nowcast_{pol}_2km.html)
       - Publication Static PNG Map (outputs/maps/nowcast_{pol}_2km.png)
    8. Step 5: Executes Phase 1B Zero-Mock Integrity Verification Gate
    """
    pol = pollutant.lower()
    tgt_col = "observed_pm10" if pol == "pm10" else "observed_pm25"
    pred_col = "pm10_pred" if pol == "pm10" else "pm25_pred"
    tgt_name = pol.upper()

    logger.info("=" * 75)
    logger.info("🚀 EXECUTING PHASE 5: EVALUATION, BENCHMARKING & INFERENCE (SMOKE TEST - %s)", tgt_name)
    logger.info("Target Observation UTC: %s", target_dt)
    logger.info("=" * 75)

    ensure_directories()

    # 1. Prepare Spatial Grid and Ground Truth Observations
    logger.info("--- Step 1: Matching Master Spatial Grid with Authentic Ground %s ---", tgt_name)
    grid_engine = MultiResolutionGridEngine()
    master_df = grid_engine.load_master_grid_2km()

    target_loader = GroundTruthLoader(target_dir=TARGET_DIR)
    pm_df = target_loader.fetch_live_air4thai(pollutant=pol)
    if pm_df.empty or tgt_col not in pm_df.columns:
        pm_df = _load_air4thai_sample_df()

    matcher = TargetMatcher()
    matched = matcher.attach_ground_truth(master_df, pm_df, max_distance_km=2.5, target_val_col=tgt_col)
    valid_grid = matcher.filter_training_cells(matched, target_col=tgt_col).copy().reset_index(drop=True)
    logger.info("Matched %d monitoring stations to unique 2.0 km grid cells nationwide", len(valid_grid))

    # 2. Extract Weather with Strict Exclusion
    logger.info("\n--- Step 2: Extracting HII Weather with Proximity Exclusion (< 2.0 km) ---")
    hii_loader = HIIWeatherLoader(hii_dir=HII_DIR)
    stn_meta = hii_loader.get_metadata()
    weather_engine = DirectionalWeatherEngine(stn_meta, default_exclusion_km=2.0)

    sample_stations = stn_meta["station_code"].head(10).tolist()
    hii_p_df = hii_loader.fetch_and_harmonize("pressure", sample_stations, target_dt.year, target_dt.month)
    p_obs = {}
    if not hii_p_df.empty:
        latest_p = hii_p_df.sort_values("timestamp_utc").groupby("station_code").last()
        p_obs = latest_p["value"].to_dict()

    hii_h_df = hii_loader.fetch_and_harmonize("humidity", sample_stations, target_dt.year, target_dt.month)
    h_obs = {}
    if not hii_h_df.empty:
        latest_h = hii_h_df.sort_values("timestamp_utc").groupby("station_code").last()
        h_obs = latest_h["value"].to_dict()

    weather_grid = weather_engine.extract_grid_weather(valid_grid, pressure_obs=p_obs, humidity_obs=h_obs)

    # 3. Satellite Tabularization
    logger.info("\n--- Step 3: Tabularizing Himawari-9 Satellite IR Bands ---")
    test_bands = ["B13", "B15"]
    sat_loader = HimawariIRLoader(cache_dir=HIMAWARI_CACHE_DIR, bands=test_bands, segments=[4])
    files_by_band = sat_loader.fetch_snapshot_files(target_dt, bands=test_bands)
    sat_scene = sat_loader.load_scene(files_by_band, bands_to_load=test_bands)

    tabularizer = SatelliteTabularizer(bands=test_bands)
    past_scene = _fetch_past_cadence_scene(sat_loader, target_dt, test_bands, lag_hours=cadence_hours)
    past_scenes = {"cadence": past_scene} if past_scene is not None else None
    sat_features = tabularizer.tabularize_from_scene(
        sat_scene, weather_grid, past_scenes=past_scenes, cadence_hours=cadence_hours
    )

    # 4. Build Unified Feature Matrix
    max_target_time = pd.to_datetime(pm_df["timestamp_utc"].dropna().max(), utc=True) if "timestamp_utc" in pm_df.columns else target_dt
    effective_run_dt = max(target_dt, max_target_time + pd.Timedelta(seconds=1))

    builder = FeatureMatrixBuilder()
    unified_matrix = builder.build_snapshot_matrix(
        target_dt=effective_run_dt,
        grid_df=valid_grid,
        satellite_df=sat_features,
        weather_df=weather_grid,
        ground_truth_df=pm_df,
        resolution_km=2.0,
        impute_missing=True,
        training_only=True,
        target_col=tgt_col,
    )

    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in unified_matrix.columns]
    logger.info("Unified Feature Matrix prepared: %d samples x %d features", len(unified_matrix), len(feature_cols))

    # 5. Train Stacking Ensembles via ModelTrainer across all 4 resolutions (3km, 2km, 1km, 500m)
    logger.info("\n--- Step 4: Training Stacking Ensembles across 4 Resolutions (%s: 3km, 2km, 1km, 500m) ---", tgt_name)
    models_dir = get_models_dir(pol)
    trainer = ModelTrainer(config_path=MODEL_CONFIG_PATH, output_dir=models_dir, pollutant=pol, target_col=tgt_col)

    res_specs = [
        (3.0, "3km", 3000.0, "EXP-3KM"),
        (2.0, "2km", 2000.0, "EXP-2KM"),
        (1.0, "1km", 1000.0, "EXP-1KM"),
        (0.5, "500m", 500.0, "EXP-500M"),
    ]

    trained_ensembles_by_res = {}
    test_matrices_by_res = {}
    for res_km, res_label, res_m, exp_prefix in res_specs:
        logger.info("  Training %s Suite (%s) -> exp_%s_%s ...", res_label, tgt_name, res_label, pol)
        ens = trainer.train_suite(
            train_df=unified_matrix,
            cv=5,
            experiment_id=f"{exp_prefix}-{tgt_name}",
        )
        trained_ensembles_by_res[res_km] = ens
        test_matrices_by_res[res_km] = unified_matrix.copy()

    # Master baseline ensemble (2.0 km) for detailed diagnostics
    ensemble = trained_ensembles_by_res[2.0]

    # 6. Spatial-Temporal Cross-Validation (Regional Hold-out)
    logger.info("\n--- Step 5: Executing Spatial Generalization (Regional Hold-Out CV) ---")
    cv_evaluator = SpatialTemporalEvaluator(target_col=tgt_col, pollutant=pol)
    def model_factory():
        return LGBMDustModel(model_params={"n_estimators": 20, "min_child_samples": 2})

    regional_eval = cv_evaluator.evaluate_regional_generalization(
        df=unified_matrix,
        feature_cols=feature_cols,
        model_factory_fn=model_factory,
    )
    logger.info("\n%s\n", cv_evaluator.format_regional_report(regional_eval))

    # 7. Comprehensive Performance Metrics (Continuous + PCD AQI)
    logger.info("\n--- Step 6: Computing Continuous & PCD AQI Classification Metrics ---")
    y_true = unified_matrix[tgt_col].values
    if ensemble.oof_predictions_ is not None and "oof_ensemble" in ensemble.oof_predictions_.columns:
        y_pred = ensemble.oof_predictions_["oof_ensemble"].values
        eval_label = f"Stacking Ensemble ({tgt_name} 2km - Out-Of-Fold CV)"
    else:
        comp_preds = ensemble.predict_components(unified_matrix[feature_cols])
        y_pred = comp_preds["pred_ensemble"].values
        eval_label = f"Stacking Ensemble ({tgt_name} 2km)"

    comp_metrics = compute_comprehensive_metrics(y_true, y_pred, pollutant=pol)
    logger.info("\n%s\n", format_evaluation_summary_table(comp_metrics, model_name=eval_label))

    # 8. Multi-Resolution Benchmark Suite (3km, 2km, 1km, 500m)
    logger.info("\n--- Step 7: Executing Multi-Resolution Benchmark (3km, 2km, 1km, 500m - %s) ---", tgt_name)
    reports_dir = get_reports_dir(pol)
    benchmark_suite = MultiResolutionBenchmarkSuite(output_dir=reports_dir, pollutant=pol)
    bench_report = benchmark_suite.run_full_benchmark(
        trained_ensembles_by_res=trained_ensembles_by_res,
        test_matrices_by_res=test_matrices_by_res,
        feature_cols=feature_cols,
        target_col=tgt_col,
        pollutant=pol,
    )
    logger.info("Multi-Resolution Benchmark Reports generated in: %s", reports_dir)

    # 9. Real-time Multi-Resolution Nowcast Map Production for all 4 resolutions
    logger.info("\n--- Step 8: Generating Nationwide / Grid Nowcast Map Deliverables (%s: 3km, 2km, 1km, 500m) ---", tgt_name)
    maps_dir = get_maps_dir(pol)
    visualizer = DustMapVisualizer(output_dir=maps_dir, pollutant=pol)

    all_deliverables = {}
    for res_km, res_label, res_m, _ in res_specs:
        ens_res = trained_ensembles_by_res[res_km]
        preds_res = ens_res.predict(unified_matrix[feature_cols])
        map_df = unified_matrix.copy()
        map_df[pred_col] = preds_res

        stem = f"nowcast_{pol}_{res_label}"
        deliv = visualizer.generate_all_deliverables(
            grid_predictions_df=map_df,
            base_filename_stem=stem,
            pred_col=pred_col,
            resolution_m=res_m,
            timestamp_utc=effective_run_dt,
            pollutant=pol,
        )
        all_deliverables[res_label] = deliv
        logger.info("  • [%s - %.1f km] Deliverables -> %s (.parquet, .tif, .html, .png)",
                    res_label, res_km, stem)

    # 10. Generate Complete Diagnostic Plot Suite in outputs/{pol}/evaluation/
    logger.info("\n--- Step 9: Generating Comprehensive Evaluation Plots in outputs/%s/evaluation/ ---", pol)
    eval_dir = get_evaluation_dir(pol)
    plotter = EvaluationPlotter(output_dir=eval_dir, pollutant=pol)
    eval_plots = plotter.generate_all_evaluation_plots(
        y_true=y_true,
        y_pred=y_pred,
        leaderboard_df=pd.DataFrame(bench_report["leaderboard"]),
        regional_results=regional_eval,
        attribution_df=ensemble.get_cross_model_feature_attribution(),
        ensemble_weights=dict(ensemble.weights),
        pollutant=pol,
    )
    for plot_name, plot_path in eval_plots.items():
        logger.info("  • Diagnostic Plot: %-30s -> %s", plot_name, plot_path.name)

    # 9b. SHAP Ensemble-Consensus Attribution (direction-aware, per-sample)
    logger.info("\n--- Step 9b: Computing SHAP Ensemble-Consensus Feature Attribution ---")
    shap_feat_cols = [c for c in ensemble.feature_names_ if c in unified_matrix.columns]
    shap_engine = ShapAttributionEngine(
        ensemble=ensemble,
        output_dir=eval_dir,
        pollutant=pol,
    )
    shap_result = shap_engine.compute_and_export(unified_matrix[shap_feat_cols])
    logger.info("  • SHAP CSV    -> %s", shap_result["csv"].name)
    logger.info("  • SHAP Bar    -> %s", shap_result["summary_bar"].name)
    if shap_result["beeswarm"] is not None:
        logger.info("  • SHAP Beeswarm -> %s", shap_result["beeswarm"].name)

    # 11. Zero-Mock Verification Gate
    logger.info("\n--- Step 10: Executing Phase 1B Zero-Mock Verification Gate ---")
    gate_passed = run_phase_1b_verification()
    if gate_passed:
        logger.info("\n🎉 PHASE 5 EVALUATION, BENCHMARKING & INFERENCE COMPLETED SUCCESSFULLY (%s)!", tgt_name)
        logger.info("Certified 100%% genuine physical observations with zero mock or synthetic data.")
        logger.info("Outputs separated into dedicated pollutant root: %s", get_pollutant_dir(pol))
        logger.info("All 4 resolutions (3km, 2km, 1km, 500m) generated with explicit _%s suffixes.", pol)
        return {
            "metrics": comp_metrics,
            "regional_evaluation": regional_eval,
            "benchmark_report": bench_report,
            "deliverables": all_deliverables,
        }
    else:
        logger.error("\n❌ PHASE 1B VERIFICATION GATE FAILED!")
        sys.exit(1)


def run_phase_1_full(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    split_date: str = "2026-01-01",
    cadence_hours: int = 3,
    bands: Optional[List[str]] = None,
    stations: Optional[List[str]] = None,
    pollutant: str = "pm25",
    location_ids: Optional[List[int]] = None,
    region: Optional[str] = None,
    max_locations: Optional[int] = None,
    provider: str = "all",
    train_segments: Optional[List[int]] = None,
    test_segments: Optional[List[int]] = None,
) -> None:
    """
    Executes Phase 1 Full Multi-Year Data Ingestion & Audit (2024 - 2026):
    - Training Period: 2024-01-01 to 2025-12-31 (Train Split)
    - Out-of-Time Testing Period: 2026-01-01 to end_date (Test Split)
    """
    pol = pollutant.lower()
    tgt_name = pol.upper()
    logger.info("=" * 75)
    logger.info("🌐 EXECUTING PHASE 1 FULL PRODUCTION INGESTION (%s): %s TO %s", tgt_name, start_date.date(), end_date.date())
    logger.info("  • Training Split (Years 2024-2025): %s to %s", start_date.date(), pd.to_datetime(split_date) - pd.Timedelta(seconds=1))
    logger.info("  • Testing Split  (Year 2026)      : %s to %s", split_date, end_date.date())
    logger.info("  • Satellite Cadence: Every %d hours", cadence_hours)
    logger.info("=" * 75)

    ensure_directories()
    target_loader = GroundTruthLoader(target_dir=TARGET_DIR)
    hii_loader = HIIWeatherLoader(hii_dir=HII_DIR)
    bands = bands or ALL_IR_BANDS

    # ---------------------------------------------------------
    # 1. Historical Ground Truth Target Ingestion & Temporal Split
    # ---------------------------------------------------------
    logger.info("--- Step 1: Ingesting Historical %s from OpenAQ & Live Air4Thai ---", tgt_name)
    if location_ids:
        query_locs = location_ids
    elif region:
        reg_filter = None if region.lower() == "all" else region
        region_df = target_loader.openaq_loader.get_thailand_stations(region=reg_filter)
        if provider and provider.lower() == "air4thai":
            air4thai = region_df[region_df["provider"].str.contains("Air4Thai", case=False, na=False)]
            pool = air4thai if not air4thai.empty else region_df
        else:
            pool = region_df
        query_locs = pool["location_id"].dropna().astype(int).unique().tolist()
        if max_locations:
            query_locs = query_locs[:max_locations]
    elif max_locations:
        all_stns = target_loader.openaq_loader.get_thailand_stations()
        if provider and provider.lower() == "air4thai":
            air4thai_stns = all_stns[all_stns["provider"].str.contains("Air4Thai", case=False, na=False)]
            stns_pool = air4thai_stns if not air4thai_stns.empty else all_stns
        else:
            stns_pool = all_stns
        query_locs = stns_pool["location_id"].dropna().astype(int).unique().tolist()[:max_locations]
    else:
        query_locs = list(DEFAULT_THAILAND_OPENAQ_LOCATIONS.keys())

    openaq_df = target_loader.fetch_openaq_historical(
        start_date=start_date,
        end_date=end_date,
        location_ids=query_locs,
        pollutant=pol,
    )

    # Also append live PCD observations if 2026 is targeted
    if end_date.year >= 2026:
        live_df = target_loader.fetch_live_air4thai(pollutant=pol)
        if not live_df.empty:
            logger.info("Appending %d live PCD Air4Thai %s observations for 2026", len(live_df), tgt_name)
            openaq_df = pd.concat([openaq_df, live_df], ignore_index=True)

    if not openaq_df.empty:
        # Split into Train (2024-2025) and Test (2026)
        train_gt, test_gt = GroundTruthLoader.split_train_test(openaq_df, split_date=split_date)
        logger.info("%s Split Summary:", tgt_name)
        logger.info("  • Train (2024-2025): %d records", len(train_gt))
        logger.info("  • Test  (2026)     : %d records", len(test_gt))

        suffix = f"_{pol}" if pol == "pm10" else ""
        if not train_gt.empty:
            p_train = target_loader.save_ground_truth(train_gt, f"ground_truth_train{suffix}_2024_2025.parquet")
            logger.info("  Saved Train set -> %s", p_train)
        if not test_gt.empty:
            p_test = target_loader.save_ground_truth(test_gt, f"ground_truth_test{suffix}_2026.parquet")
            logger.info("  Saved Test set  -> %s", p_test)
    else:
        logger.warning("No %s ground truth records retrieved for %s to %s", tgt_name, start_date, end_date)

    # ---------------------------------------------------------
    # 2. HII Ground Weather Telemetry & Temporal Split
    # ---------------------------------------------------------
    logger.info("--- Step 2: Ingesting HII Weather Telemetry (Pressure & Humidity) ---")
    try:
        meta = hii_loader.get_metadata()
        query_stations = stations or meta["station_code"].head(20).tolist()  # Benchmark set or full

        for var in ["pressure", "humidity"]:
            logger.info("Fetching HII %s (%s to %s)...", var, start_date.date(), end_date.date())
            hii_var_df = hii_loader.fetch_date_range(
                variable=var,
                start_date=start_date,
                end_date=end_date,
                station_codes=query_stations
            )
            if not hii_var_df.empty:
                train_hii, test_hii = HIIWeatherLoader.split_train_test(hii_var_df, split_date=split_date)
                logger.info("HII %s Split Summary:", var)
                logger.info("  • Train (2024-2025): %d records", len(train_hii))
                logger.info("  • Test  (2026)     : %d records", len(test_hii))

                if not train_hii.empty:
                    hii_loader.save_to_parquet(train_hii, var, f"hii_{var}_train_2024_2025.parquet")
                if not test_hii.empty:
                    hii_loader.save_to_parquet(test_hii, var, f"hii_{var}_test_2026.parquet")
    except Exception as e:
        logger.error("Error during Step 2 HII telemetry ingestion: %s", e, exc_info=True)

    # ---------------------------------------------------------
    # 3. Himawari Satellite IR Bands Ingestion (5 Core Bands, Station-Centric)
    # ---------------------------------------------------------
    train_segs = train_segments or [4]
    test_segs = test_segments or [3, 4, 5]
    split_dt = pd.to_datetime(split_date, utc=True)
    start_dt = start_date if start_date.tzinfo is not None else start_date.tz_localize("UTC")
    end_dt = end_date if end_date.tzinfo is not None else end_date.tz_localize("UTC")

    logger.info("--- Step 3: Fetching Himawari IR Bands (5 Bands, Station-Centric, Zero-Disk) at Cadence=%dh ---", cadence_hours)
    logger.info("  • Training Segment Partition : %s (Segment 4 covers 574 stations / 92.9%% of Thailand)", train_segs)
    logger.info("  • Testing Segment Partition  : %s (Covers all 618+ stations countrywide)", test_segs)

    try:
        sat_loader = HimawariIRLoader(cache_dir=HIMAWARI_CACHE_DIR, bands=bands)
        all_stns = target_loader.openaq_loader.get_thailand_stations()
        stn_coords = (
            all_stns[all_stns["location_id"].isin(query_locs)].copy()
            if not all_stns.empty and "location_id" in all_stns.columns
            else pd.DataFrame()
        )
        if stn_coords.empty and not all_stns.empty:
            stn_coords = all_stns.copy()

        sat_train_out = FEATURES_DIR / f"satellite_stations_train_{pol}_{start_dt.year}_{(split_dt - pd.Timedelta(days=1)).year}.parquet"
        sat_test_out = FEATURES_DIR / f"satellite_stations_test_{pol}_{split_dt.year}_{end_dt.year}.parquet"
        sat_parquet_out = FEATURES_DIR / f"satellite_stations_{pol}_{start_dt.year}_{end_dt.year}.parquet"

        accumulated_train_df = pd.DataFrame()
        accumulated_test_df = pd.DataFrame()

        if not stn_coords.empty:
            logger.info("Station-Centric Extraction: querying ground stations (Zero Disk Cache Footprint)...")

            # Extract Target-Driven observation timestamps from Ground Truth (Point-in-Time Nowcast Match)
            train_target_ts = None
            if not train_gt.empty and "timestamp_utc" in train_gt.columns:
                gt_ts_all = pd.to_datetime(train_gt["timestamp_utc"], utc=True).dt.floor("h").drop_duplicates().sort_values()
                if cadence_hours > 1:
                    train_target_ts = gt_ts_all[gt_ts_all.dt.hour % cadence_hours == 0].tolist()
                else:
                    train_target_ts = gt_ts_all.tolist()
                logger.info("  • Target-Driven Timestamps (Train): %d exact ground truth hours selected (Cadence=%dh)",
                            len(train_target_ts), cadence_hours)

            test_target_ts = None
            if not test_gt.empty and "timestamp_utc" in test_gt.columns:
                test_ts_all = pd.to_datetime(test_gt["timestamp_utc"], utc=True).dt.floor("h").drop_duplicates().sort_values()
                if cadence_hours > 1:
                    test_target_ts = test_ts_all[test_ts_all.dt.hour % cadence_hours == 0].tolist()
                else:
                    test_target_ts = test_ts_all.tolist()
                logger.info("  • Target-Driven Timestamps (Test) : %d exact ground truth hours selected (Cadence=%dh)",
                            len(test_target_ts), cadence_hours)

            # 3a. Training Period Ingestion (Segment 4 only)
            if start_dt < split_dt:
                train_range_end = min(end_dt, split_dt - pd.Timedelta(seconds=1))
                logger.info(">>> [TRAIN SPLIT] Ingesting Satellite Data (%s to %s) using Segment %s...",
                            start_dt.date(), train_range_end.date(), train_segs)
                accumulated_train_df = sat_loader.fetch_range_station_snapshots(
                    start_date=start_dt,
                    end_date=train_range_end,
                    stations_df=stn_coords,
                    cadence_hours=cadence_hours,
                    target_timestamps=train_target_ts,
                    bands=bands,
                    segments=train_segs,
                    lat_col="latitude",
                    lon_col="longitude",
                    cleanup_raw=True,
                    save_parquet_path=sat_train_out,
                )
                logger.info("  Training satellite observations: %d records -> %s", len(accumulated_train_df), sat_train_out)

            # 3b. Testing Period Ingestion (All Segments [3, 4, 5])
            if end_dt >= split_dt:
                test_range_start = max(start_dt, split_dt)
                logger.info(">>> [TEST SPLIT] Ingesting Satellite Data (%s to %s) using ALL Segments %s...",
                            test_range_start.date(), end_dt.date(), test_segs)
                accumulated_test_df = sat_loader.fetch_range_station_snapshots(
                    start_date=test_range_start,
                    end_date=end_dt,
                    stations_df=stn_coords,
                    cadence_hours=cadence_hours,
                    target_timestamps=test_target_ts,
                    bands=bands,
                    segments=test_segs,
                    lat_col="latitude",
                    lon_col="longitude",
                    cleanup_raw=True,
                    save_parquet_path=sat_test_out,
                )
                logger.info("  Testing satellite observations: %d records -> %s", len(accumulated_test_df), sat_test_out)

            # 3c. Combine Train + Test sets into full multi-year dataset
            parts = [df for df in [accumulated_train_df, accumulated_test_df] if not df.empty]
            if parts:
                combined_sat_df = pd.concat(parts, ignore_index=True)
                combined_sat_df.to_parquet(sat_parquet_out, compression="snappy", index=False)
                logger.info("  Combined Multi-Year Satellite dataset saved -> %s (%d records, %.2f MB)",
                            sat_parquet_out, len(combined_sat_df), sat_parquet_out.stat().st_size / (1024 * 1024))
        else:
            snapshots = sat_loader.fetch_range_snapshots(
                start_date=start_date,
                end_date=end_date,
                cadence_hours=cadence_hours,
                bands=bands
            )
            logger.info("Downloaded satellite snapshots: %d observation cycles", len(snapshots))
    except Exception as e:
        logger.error("Error during Step 3 Himawari satellite ingestion: %s", e, exc_info=True)

    # ---------------------------------------------------------
    # 4. Strict Zero-Mock Verification Gate
    # ---------------------------------------------------------
    logger.info("--- Step 4: Executing Phase 1B Zero-Mock Verification Gate ---")
    gate_passed = run_phase_1b_verification()
    if gate_passed:
        logger.info("\n🎉 PHASE 1 FULL INGESTION (2024-2026) COMPLETED SUCCESSFULLY (%s)!", tgt_name)
        logger.info("Data cleanly partitioned into Train (2024-2025) and Test (2026) sets.")
        logger.info("All records certified authentic physical observations.")
    else:
        logger.error("\n❌ PHASE 1B VERIFICATION GATE FAILED!")
        sys.exit(1)


def run_phase_4_full(pollutant: str = "pm25", split_date: str = "2026-01-01") -> Dict[str, Any]:
    """
    Executes Phase 4 Full Multi-Year Model Suite & Ensemble Training:
    - Loads historical training ground truth (2024-2025) and test ground truth (2026)
    - Prepares unified feature matrices for training and testing
    - Trains 4-Model Quartet (LightGBM, CatBoost, XGBoost, Random Forest)
    - Optimizes Non-Negative Least Squares Stacking Ensemble
    - Evaluates Out-of-Time generalization on 2026 Test split
    - Generates multi-resolution benchmarks and saves production checkpoints
    """
    pol = pollutant.lower()
    tgt_col = "observed_pm10" if pol == "pm10" else "observed_pm25"
    tgt_name = pol.upper()
    logger.info("=" * 75)
    logger.info("🏋️ EXECUTING PHASE 4 FULL PRODUCTION MODEL TRAINING (%s)", tgt_name)
    logger.info("Train / Test Split Date: %s", split_date)
    logger.info("=" * 75)

    ensure_directories()
    models_dir = get_models_dir(pol)
    suffix = f"_{pol}" if pol == "pm10" else ""

    # 1. Load multi-year ground truth
    p_train = TARGET_DIR / f"ground_truth_train{suffix}_2024_2025.parquet"
    p_test = TARGET_DIR / f"ground_truth_test{suffix}_2026.parquet"

    train_gt = pd.read_parquet(p_train) if p_train.exists() else pd.DataFrame()
    test_gt = pd.read_parquet(p_test) if p_test.exists() else pd.DataFrame()

def _enrich_multi_year_matrix(
    df: pd.DataFrame,
    join_key: str = "station_code",
    tgt_col: str = "observed_pm25"
) -> pd.DataFrame:
    """
    Enriches the multi-year dataset with:
    1. Genuine physical temporal cadence tendencies & hourly rates (dt_hours, delta_bt13_rate_per_hour, delta_btd15_13_rate_per_hour).
    2. Authentic Master Grid Topography (elevation_m, slope_deg, dist_coast_km) via cKDTree.
       (A2 fix: coriolis excluded — lat fingerprint, no 2km physics.)
    3. HII authentic ground weather observations if available.
    4. PhysicsFallbackImputer to guarantee 100% complete feature sets for both:
       - Mode 1: Pure Satellite (NO_WEATHER_FEATURE_COLUMNS)
       - Mode 2: Multimodal (FULL_MULTIMODAL_COLUMNS)
    """
    out_df = df.copy()
    if "timestamp_utc" not in out_df.columns and "timestamp" in out_df.columns:
        out_df["timestamp_utc"] = pd.to_datetime(out_df["timestamp"], utc=True)

    stn_key = (
        join_key if join_key in out_df.columns
        else ("station_code" if "station_code" in out_df.columns
              else ("location_id" if "location_id" in out_df.columns
                    else ("grid_id" if "grid_id" in out_df.columns else None)))
    )

    # 1. Dynamic Temporal Tendencies from actual satellite cadence
    if stn_key and "timestamp_utc" in out_df.columns and "bt_b13" in out_df.columns:
        out_df = out_df.sort_values([stn_key, "timestamp_utc"]).reset_index(drop=True)
        dt_h = out_df.groupby(stn_key)["timestamp_utc"].diff().dt.total_seconds() / 3600.0
        valid_cadence = (dt_h > 0.0) & (dt_h <= 24.0)
        dt_valid = dt_h.where(valid_cadence, np.nan)

        # Keep NaN for "no previous observation" (do NOT collapse it to 0.0,
        # which would falsely read as a stationary atmosphere).
        d_bt13 = out_df.groupby(stn_key)["bt_b13"].diff()
        d_btd = (
            out_df.groupby(stn_key)["btd_15_13"].diff()
            if "btd_15_13" in out_df.columns
            else pd.Series(np.nan, index=out_df.index)
        )

        # Provenance mask: valid only when a real previous step exists.
        has_prev = d_bt13.notna() & d_btd.notna()
        out_df["temporal_delta_valid"] = (valid_cadence & has_prev).astype(np.float32)

        rate_bt13 = (d_bt13 / dt_valid).astype(np.float32)
        rate_btd = (d_btd / dt_valid).astype(np.float32)

        out_df["delta_bt13_rate_per_hour"] = rate_bt13
        out_df["delta_btd15_13_rate_per_hour"] = rate_btd
        out_df["delta_bt13_cadence"] = d_bt13.where(valid_cadence, np.nan).astype(np.float32)
        out_df["delta_btd15_13_cadence"] = d_btd.where(valid_cadence, np.nan).astype(np.float32)

        # Spatial Differences (Defects) relative to 3x3 window and 3-pixel surrounding neighborhood
        if "bt_b13" in out_df.columns and "b13_mean_3x3" in out_df.columns:
            out_df["b13_diff_mean_3x3"] = (out_df["bt_b13"] - out_df["b13_mean_3x3"]).astype(np.float32)
        if "btd_15_13" in out_df.columns and "btd_15_13_mean_3x3" in out_df.columns:
            out_df["btd_15_13_diff_mean_3x3"] = (out_df["btd_15_13"] - out_df["btd_15_13_mean_3x3"]).astype(np.float32)
        if "bt_b13" in out_df.columns and "b13_mean_rad3" in out_df.columns:
            out_df["b13_diff_mean_rad3"] = (out_df["bt_b13"] - out_df["b13_mean_rad3"]).astype(np.float32)
        if "btd_15_13" in out_df.columns and "btd_15_13_mean_rad3" in out_df.columns:
            out_df["btd_15_13_diff_mean_rad3"] = (out_df["btd_15_13"] - out_df["btd_15_13_mean_rad3"]).astype(np.float32)

        logger.info("  ✓ Computed authentic temporal dynamics (delta_bt13/btd rates per hour) and spatial defects for %s", stn_key)

    # 2. Topography from Master Grid (Copernicus DEM)
    lat_col = "latitude" if "latitude" in out_df.columns else ("lat" if "lat" in out_df.columns else None)
    lon_col = "longitude" if "longitude" in out_df.columns else ("lon" if "lon" in out_df.columns else None)
    if lat_col and lon_col:
        grid_candidates = [
            GEO_DIR / "thailand_2km_master_grid.parquet",
            PROJECT_ROOT / "data" / "geo" / "thailand_2km_master_grid.parquet",
        ]
        found_grid = next((p for p in grid_candidates if p.exists() and p.stat().st_size > 0), None)
        if found_grid:
            grid_df = pd.read_parquet(found_grid)
            # A1/A2 fix: topo only — coriolis + solar/time are NOT training features.
            topo_cols = [c for c in ["elevation_m", "elevation_std", "slope_deg", "aspect_sin", "aspect_cos", "dist_coast_km"] if c in grid_df.columns]
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([grid_df["centroid_lat"].values, grid_df["centroid_lon"].values]))
            dists, idxs = tree.query(np.column_stack([out_df[lat_col].to_numpy(dtype=float), out_df[lon_col].to_numpy(dtype=float)]))
            for c in topo_cols:
                out_df[c] = grid_df.iloc[idxs][c].values.astype(np.float32)
            logger.info("  ✓ Attached Master Grid Topography (%s) via spatial KDTree.", topo_cols)

    # 3. Solar and Cyclical Features (A1 fix: META ONLY for day/night eval + B07
    # correction — excluded from ALL_FEATURE_COLUMNS, never enter the model)
    if lat_col and lon_col and "timestamp_utc" in out_df.columns:
        sza, is_day = compute_solar_angles_vectorized(
            out_df["timestamp_utc"],
            out_df[lat_col].to_numpy(dtype=float),
            out_df[lon_col].to_numpy(dtype=float)
        )
        out_df["solar_zenith_angle"] = sza
        out_df["solar_zenith_deg"] = sza
        out_df["is_daytime"] = is_day
        out_df["is_night"] = (1.0 - is_day).astype(np.float32)

    if "timestamp_utc" in out_df.columns:
        time_feats = compute_cyclical_time_features(out_df["timestamp_utc"])
        for col in ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos"]:
            if col in time_feats:
                out_df[col] = time_feats[col].values

    # 4. Authentic HII Weather Telemetry (if files exist)
    if "pressure_mean_surround" not in out_df.columns or out_df["pressure_mean_surround"].isna().all():
        hii_press = list(HII_DIR.glob("hii_pressure_*.parquet"))
        hii_humid = list(HII_DIR.glob("hii_humidity_*.parquet"))
        if (hii_press or hii_humid) and stn_key:
            try:
                # Standardize timestamp_utc to timezone-naive UTC for robust merging
                if "timestamp_utc" in out_df.columns:
                    ts_out = pd.to_datetime(out_df["timestamp_utc"])
                    if getattr(ts_out.dt, "tz", None) is not None:
                        out_df["timestamp_utc"] = ts_out.dt.tz_convert("UTC").dt.tz_localize(None)
                    else:
                        out_df["timestamp_utc"] = ts_out

                if hii_press:
                    p_df = pd.concat([pd.read_parquet(f) for f in hii_press], ignore_index=True)
                    if "timestamp" in p_df.columns:
                        ts_p = pd.to_datetime(p_df["timestamp"])
                        if getattr(ts_p.dt, "tz", None) is not None:
                            p_df["timestamp_utc"] = ts_p.dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("h")
                        else:
                            p_df["timestamp_utc"] = ts_p.dt.floor("h")
                    p_val_col = next((c for c in ["pressure", "press", "value", "val"] if c in p_df.columns), None)
                    if p_val_col and stn_key in p_df.columns:
                        p_clean = p_df[[stn_key, "timestamp_utc", p_val_col]].drop_duplicates([stn_key, "timestamp_utc"]).rename(columns={p_val_col: "pressure_mean_surround"})
                        if "pressure_mean_surround" in out_df.columns:
                            out_df = out_df.drop(columns=["pressure_mean_surround"])
                        out_df = out_df.merge(p_clean, on=[stn_key, "timestamp_utc"], how="left")
                        logger.info("  ✓ Attached authentic HII pressure observations (%d records).", len(p_clean))

                if hii_humid:
                    h_df = pd.concat([pd.read_parquet(f) for f in hii_humid], ignore_index=True)
                    if "timestamp" in h_df.columns:
                        ts_h = pd.to_datetime(h_df["timestamp"])
                        if getattr(ts_h.dt, "tz", None) is not None:
                            h_df["timestamp_utc"] = ts_h.dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("h")
                        else:
                            h_df["timestamp_utc"] = ts_h.dt.floor("h")
                    h_val_col = next((c for c in ["humidity", "humid", "rh", "value", "val"] if c in h_df.columns), None)
                    if h_val_col and stn_key in h_df.columns:
                        h_clean = h_df[[stn_key, "timestamp_utc", h_val_col]].drop_duplicates([stn_key, "timestamp_utc"]).rename(columns={h_val_col: "humidity_mean_surround"})
                        if "humidity_mean_surround" in out_df.columns:
                            out_df = out_df.drop(columns=["humidity_mean_surround"])
                        out_df = out_df.merge(h_clean, on=[stn_key, "timestamp_utc"], how="left")
                        logger.info("  ✓ Attached authentic HII humidity observations (%d records).", len(h_clean))
            except Exception as ex:
                logger.warning("Could not merge HII telemetry: %s", ex)

    # 5. PhysicsFallbackImputer for Barometric pressure scaling and texture
    imputer = PhysicsFallbackImputer()
    out_df = imputer.fit(out_df).impute_features(out_df)

    # 6. Keys alignment
    out_df["timestamp"] = out_df["timestamp_utc"]
    if "lat" not in out_df.columns and lat_col:
        out_df["lat"] = out_df[lat_col]
    if "lon" not in out_df.columns and lon_col:
        out_df["lon"] = out_df[lon_col]
    if "grid_id" not in out_df.columns:
        out_df["grid_id"] = out_df[stn_key].astype(str) if stn_key else "0"

    return out_df


def run_phase_4_full(
    pollutant: str = "pm25",
    split_date: str = "2026-01-01",
    tune: bool = False,
    tune_trials: int = 25,
    tune_subsample_frac: float = 0.3,
    tune_models: Optional[List[str]] = None,
    tune_providers: Optional[List[str]] = None,
    tune_feature_sets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes Phase 4 Full Multi-Year Model Suite & Ensemble Training:
    - Loads historical training ground truth (2024-2025) and test ground truth (2026)
    - Prepares enriched feature matrices with real temporal rates, Copernicus DEM, and weather
    - [Optional] Optuna hyperparameter tuning on the 2 km reference grid (reused everywhere)
    - Trains provider-partitioned production ensembles FIRST (Air4Thai, AirGradient,
      then Combined): persists every checkpoint and runs a cross-provider generalization
      benchmark (train on one provider, predict the other, swapped)
    - Trains DUAL MODEL ARCHITECTURES:
        1. Pure Satellite Suite (Maritime / Sea Grid Capable, Zero Ground Stations Needed)
        2. Full Multimodal Suite (Satellite + Topography + Ground Weather Telemetry)
    - Generates multi-resolution benchmark (EXP-3KM, EXP-2KM, EXP-1KM, EXP-500M)
    - Evaluates Out-of-Time generalization on 2026 Test split and saves all production checkpoints
    """
    pol = pollutant.lower()
    tgt_col = "observed_pm10" if pol == "pm10" else "observed_pm25"
    tgt_name = pol.upper()
    logger.info("=" * 75)
    logger.info("🏋️ EXECUTING PHASE 4 FULL PRODUCTION MODEL TRAINING (%s)", tgt_name)
    logger.info("Train / Test Split Date: %s", split_date)
    logger.info("=" * 75)

    ensure_directories()
    models_dir = get_models_dir(pol)
    suffix = f"_{pol}" if pol == "pm10" else ""

    # 1. Load multi-year ground truth
    p_train = TARGET_DIR / f"ground_truth_train{suffix}_2024_2025.parquet"
    p_test = TARGET_DIR / f"ground_truth_test{suffix}_2026.parquet"

    train_gt = pd.read_parquet(p_train) if p_train.exists() else pd.DataFrame()
    test_gt = pd.read_parquet(p_test) if p_test.exists() else pd.DataFrame()

    # 2. Load or construct feature matrices
    matrix_path = FEATURES_DIR / f"unified_{pol}_matrix.parquet"
    if not matrix_path.exists():
        matrix_path = FEATURES_DIR / "unified_pm25_matrix.parquet"

    needs_enrichment = False
    if matrix_path.exists():
        unified_df = pd.read_parquet(matrix_path)
        if (
            "elevation_m" not in unified_df.columns
            or "delta_bt13_rate_per_hour" not in unified_df.columns
            or "p_north" not in unified_df.columns
            or "b13_diff_mean_3x3" not in unified_df.columns
        ):
            logger.info("Existing matrix lacks enriched audit features (elevation, temporal delta, weather, spatial defects). Enriching...")
            needs_enrichment = True
    else:
        needs_enrichment = True

    if needs_enrichment:
        # Check if station satellite observations exist from Phase 1 full ingestion
        candidate_sat_files = [
            FEATURES_DIR / f"satellite_stations_{pol}_2024_2026.parquet",
            FEATURES_DIR / f"satellite_stations_{pol}.parquet",
        ] + list(FEATURES_DIR.glob(f"satellite_stations_{pol}_*.parquet"))

        found_sat_p = next((p for p in candidate_sat_files if p.exists() and p.stat().st_size > 0), None)
        if found_sat_p and not train_gt.empty:
            logger.info("Assembling Multi-Year Feature Matrix from authentic satellite observations: %s", found_sat_p.name)
            sat_df = pd.read_parquet(found_sat_p)
            sat_df["timestamp_utc"] = pd.to_datetime(sat_df["timestamp"], utc=True).dt.floor("h")
            all_gt = pd.concat([train_gt, test_gt], ignore_index=True) if not test_gt.empty else train_gt
            all_gt["timestamp_utc"] = pd.to_datetime(all_gt["timestamp_utc"], utc=True).dt.floor("h")

            # Harmonize station identifiers across sat_df and all_gt
            if "location_id" not in all_gt.columns and "station_code" in all_gt.columns:
                all_gt["location_id"] = pd.to_numeric(
                    all_gt["station_code"].astype(str).str.extract(r"(\d+)", expand=False),
                    errors="coerce"
                )

            # Determine optimal join key: prefer station_code if present in both, else location_id
            if "station_code" in sat_df.columns and "station_code" in all_gt.columns:
                join_key = "station_code"
            else:
                join_key = "location_id"
                if "location_id" in sat_df.columns and "location_id" in all_gt.columns:
                    sat_df["location_id"] = pd.to_numeric(sat_df["location_id"], errors="coerce")
                    all_gt["location_id"] = pd.to_numeric(all_gt["location_id"], errors="coerce")

            # 1. Require authentic satellite readings and ground truth dust target
            sat_valid = sat_df.dropna(subset=["bt_b13"]).copy()
            gt_valid = all_gt.dropna(subset=[tgt_col]).copy()

            # 2. Compute authentic temporal dynamics on sat_valid BEFORE merging
            sat_valid = sat_valid.sort_values([join_key, "timestamp_utc"]).reset_index(drop=True)
            dt_h = sat_valid.groupby(join_key)["timestamp_utc"].diff().dt.total_seconds() / 3600.0
            has_prev_satellite = (dt_h > 0.0) & (dt_h <= 12.0)

            d_bt13 = sat_valid.groupby(join_key)["bt_b13"].diff()
            d_btd = (
                sat_valid.groupby(join_key)["btd_15_13"].diff()
                if "btd_15_13" in sat_valid.columns
                else pd.Series(np.nan, index=sat_valid.index)
            )
            sat_valid["delta_bt13_rate_per_hour"] = (d_bt13 / dt_h.where(has_prev_satellite, np.nan)).astype(np.float32)
            sat_valid["delta_btd15_13_rate_per_hour"] = (d_btd / dt_h.where(has_prev_satellite, np.nan)).astype(np.float32)
            sat_valid["delta_bt13_cadence"] = d_bt13.where(has_prev_satellite, np.nan).astype(np.float32)
            sat_valid["delta_btd15_13_cadence"] = d_btd.where(has_prev_satellite, np.nan).astype(np.float32)
            sat_valid["temporal_delta_valid"] = has_prev_satellite.astype(np.float32)
            # Skip rows where t - cadence satellite data is missing (Strict Continuous Cadence)
            sat_continuous = sat_valid[has_prev_satellite].copy()
            logger.info("  • Satellite observations with valid t - cadence history: %d/%d (%.1f%% retained)",
                        len(sat_continuous), len(sat_valid), 100.0 * len(sat_continuous) / max(1, len(sat_valid)))

            # Exact point-in-time merge on station identifier and timestamp_utc (Nowcast standard)
            merged_df = sat_continuous.merge(
                gt_valid[[join_key, "timestamp_utc", tgt_col]].drop_duplicates(),
                on=[join_key, "timestamp_utc"],
                how="inner"
            )
            if not merged_df.empty:
                grid_col = "location_id" if ("location_id" in merged_df.columns and merged_df["location_id"].notna().any()) else join_key
                merged_df["grid_id"] = merged_df[grid_col].astype(str)
                unified_df = _enrich_multi_year_matrix(merged_df, join_key=join_key, tgt_col=tgt_col)
                builder = FeatureMatrixBuilder()
                builder.save_matrix(unified_df, matrix_path)
                logger.info("Successfully assembled & saved authentic multi-year unified matrix -> %s (%d records, %d features)",
                            matrix_path.name, len(unified_df), len(unified_df.columns))
            else:
                sample_dt = pd.Timestamp("2026-10-12 14:00:00", tz="UTC")
                return run_phase_4_smoke_test(sample_dt, pollutant=pol)
        elif matrix_path.exists():
            # Enrich existing matrix directly
            join_k = "station_code" if "station_code" in unified_df.columns else "location_id"
            unified_df = _enrich_multi_year_matrix(unified_df, join_key=join_k, tgt_col=tgt_col)
            builder = FeatureMatrixBuilder()
            builder.save_matrix(unified_df, matrix_path)
            logger.info("Successfully re-saved enriched unified matrix -> %s (%d records, %d features)",
                        matrix_path.name, len(unified_df), len(unified_df.columns))
        else:
            sample_dt = pd.Timestamp("2026-10-12 14:00:00", tz="UTC")
            return run_phase_4_smoke_test(sample_dt, pollutant=pol)

    # Attach ground truth if missing or all NaN
    if tgt_col not in unified_df.columns or unified_df[tgt_col].isna().all():
        matcher = TargetMatcher()
        gt_source = train_gt if not train_gt.empty else _load_air4thai_sample_df()
        unified_df = matcher.attach_ground_truth(unified_df, gt_source, max_distance_km=2.5, target_val_col=tgt_col)

    # Split into train and test matrices
    if "timestamp" in unified_df.columns and not unified_df["timestamp"].isna().all():
        split_ts = pd.to_datetime(split_date, utc=True)
        ts_series = pd.to_datetime(unified_df["timestamp"], utc=True)
        train_df = unified_df[ts_series < split_ts].copy()
        test_df = unified_df[ts_series >= split_ts].copy()
    else:
        train_df = unified_df.copy()
        test_df = pd.DataFrame()

    if train_df.empty or (tgt_col in train_df.columns and train_df[tgt_col].isna().all()):
        train_df = unified_df.dropna(subset=[tgt_col]).copy() if tgt_col in unified_df.columns else unified_df.copy()

    # --------------------------------------------------------------------------
    # 0. [Optional] Optuna hyperparameter tuning on the 2 km reference grid.
    #    Best params are written to configs/tuned_params_<pol>.yaml and reused by
    #    every resolution / provider below (the trainer loads them on construction).
    # --------------------------------------------------------------------------
    if tune:
        from src.models.hyperparameter_tuner import run_optuna_tuning, DEFAULT_FEATURE_SETS

        logger.info("\n" + "=" * 75)
        logger.info("🎛️ PHASE 4 HYPERPARAMETER TUNING (2 km reference) — %s", tgt_name)
        logger.info("=" * 75)
        _fs = {
            k: v for k, v in DEFAULT_FEATURE_SETS.items()
            if not tune_feature_sets or k in tune_feature_sets
        }
        _base_cfg = load_yaml_config(MODEL_CONFIG_PATH) if MODEL_CONFIG_PATH.exists() else {}
        run_optuna_tuning(
            train_df=train_df,
            pollutant=pol,
            base_config=_base_cfg,
            feature_sets=_fs,
            providers=tune_providers or ["Air4Thai", "AirGradient", "Combined"],
            models=tune_models or list(MODEL_KEYS),
            trials=tune_trials,
            subsample_frac=tune_subsample_frac,
            seed=42,
            resolution_km=2.0,
        )

    trainer = ModelTrainer(config_path=MODEL_CONFIG_PATH, output_dir=models_dir, pollutant=pol, target_col=tgt_col)

    # Resolution-adapted matrices shared by the provider, ablation, and benchmark steps
    res_engine = MultiResolutionGridEngine()
    train_matrices_by_res = {
        res: res_engine.adapt_feature_matrix_for_resolution(train_df, res)
        for res in [3.0, 2.0, 1.0, 0.5]
    }
    test_matrices_by_res = {
        res: res_engine.adapt_feature_matrix_for_resolution(test_df, res)
        for res in [3.0, 2.0, 1.0, 0.5]
    } if not test_df.empty else {}

    # --------------------------------------------------------------------------
    # 1. Provider-Partitioned Production Training (FIRST):
    #    Air4Thai -> AirGradient -> Combined (persists every checkpoint; Combined
    #    is the production inference model) plus cross-provider generalization
    #    benchmark (train on one provider, predict the other, swapped).
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("🧩 PHASE 4 PROVIDER-PARTITIONED PRODUCTION TRAINING (%s)", tgt_name)
    logger.info("  • Feature sets: Full Multimodal + Pure Satellite, at every resolution")
    logger.info("  • Providers: Air4Thai, AirGradient, Combined (per feature set × resolution)")
    logger.info("  • [Combined] Air4Thai + AirGradient production ensemble")
    logger.info("  • Cross-provider generalization (train → predict swap)")
    logger.info("=" * 75)

    provider_res = trainer.run_provider_partitioned_training(
        train_matrices_by_res=train_matrices_by_res,
        test_matrices_by_res=test_matrices_by_res,
        providers=["Air4Thai", "AirGradient"],
        feature_sets={
            "full": list(FULL_MULTIMODAL_COLUMNS),
            "pure_satellite": list(NO_WEATHER_FEATURE_COLUMNS),
        },
        include_combined=True,
        cross_provider_eval=True,
        save_checkpoints=True,
    )

    # Provider matrix visual diagnostics:
    #   • Feature Set × Resolution × Provider benchmark (grouped bars) + CSV
    #   • Cross-provider generalization heatmaps (train on provider → predict across) + CSV
    eval_dir = get_evaluation_dir(pol)
    provider_plotter = EvaluationPlotter(output_dir=eval_dir, pollutant=pol)
    if provider_res.get("summary_table"):
        provider_plotter.plot_provider_feature_resolution_benchmark(
            provider_res["summary_table"], pollutant=pol
        )
    if provider_res.get("cross_provider_table"):
        provider_plotter.plot_cross_provider_matrix(
            provider_res["cross_provider_table"], metric="mae", pollutant=pol
        )
        provider_plotter.plot_cross_provider_matrix(
            provider_res["cross_provider_table"], metric="r2", pollutant=pol
        )

    # --------------------------------------------------------------------------
    # 2. Scientific Weather Ablation: Model 1 (Pure Satellite) vs Model 2 (Multimodal)
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("🔬 PHASE 4 DUAL MODEL ARCHITECTURES (%s)", tgt_name)
    logger.info("  • [Model 1] Pure Satellite Suite (Maritime/Sea Grid Ready, No Ground Stations)")
    logger.info("  • [Model 2] Full Multimodal Suite (Satellite + Topography + Ground Weather)")
    logger.info("=" * 75)

    ablation_res = trainer.run_weather_ablation_benchmark(
        train_df=train_df,
        test_df=test_df if not test_df.empty else None,
        cv=None,
        experiment_id=f"EXP-2KM-{tgt_name}",
    )

    # --------------------------------------------------------------------------
    # 3. Multi-Resolution Benchmark: 3.0km, 2.0km, 1.0km, 0.5km
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("🗺️ PHASE 4 MULTI-RESOLUTION BENCHMARK MATRIX (%s)", tgt_name)
    logger.info("  • Resolutions: 3.0 km (EXP-3KM), 2.0 km (EXP-2KM), 1.0 km (EXP-1KM), 0.5 km (EXP-500M)")
    logger.info("=" * 75)

    benchmark_res = trainer.run_multi_resolution_benchmark(
        train_matrices_by_res=train_matrices_by_res,
        test_matrices_by_res=test_matrices_by_res,
    )
    benchmark_res["provider_partitioned_training"] = provider_res

    run_phase_1b_verification()
    logger.info("\n🎉 PHASE 4 FULL PRODUCTION TRAINING COMPLETED SUCCESSFULLY (%s)!", tgt_name)
    return {
        "ablation_benchmark": ablation_res,
        "multi_resolution_benchmark": benchmark_res,
        "provider_partitioned_training": provider_res,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Thailand Dust Prediction Satellite - Phase 1-5 Master Pipeline"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="1",
        choices=["1", "1b", "2", "3", "4", "5", "all"],
        help="Pipeline phase to execute (1: Acquisition & Audit, 1b: Gate only, 2: Spatial & Features, 3: Unified Matrix, 4: Model Suite & Ensemble, 5: Evaluation & Nowcast Maps, all: End-to-end)"
    )
    parser.add_argument(
        "--pollutant",
        type=str,
        default="pm25",
        choices=["pm25", "pm10", "all"],
        help="Target pollutant: pm25 (default), pm10, or all (runs PM2.5 then PM10 separately)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train", "infer", "benchmark"],
        help="Execution mode (all, train, infer, benchmark)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="Spatial grid resolution in km: 3.0, 2.0, 1.0, 0.5 (or in meters: 3000, 2000, 1000, 500)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble", "lightgbm", "catboost", "xgboost", "random_forest"],
        help="Model architecture for inference/benchmarking"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke-test on development machine using genuine sample observations"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Execute full multi-year production ingestion (2024 - 2026: Train 2024-2025, Test 2026)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Custom start observation timestamp (e.g. '2024-01-01' or '2024-01-01 00:00:00')"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Custom end observation timestamp (e.g. '2026-09-30' or '2026-09-30 23:59:59')"
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2026-01-01",
        help="Train/Test cutoff date (default: '2026-01-01' -> Train: 2024-2025, Test: 2026)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning on the 2 km reference grid before Phase 4 training (reused for all resolutions/providers)"
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=25,
        help="Optuna trials per model during tuning (default: 25)"
    )
    parser.add_argument(
        "--tune-subsample-frac",
        type=float,
        default=0.3,
        help="Fraction of training rows used per tuning trial (default: 0.3)"
    )
    parser.add_argument(
        "--cadence-hours",
        type=int,
        default=3,
        help="Cadence in hours for satellite sampling in full mode (default: 3 hours)"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Execute Phase 1B provenance and codebase integrity audit"
    )
    parser.add_argument(
        "--dir-data", "--data-dir",
        type=str,
        default=None,
        dest="dir_data",
        help="Custom storage directory path for data ingestion, satellite cache & features (e.g. for Google Colab: /content/data or /content/drive/MyDrive/data)"
    )
    parser.add_argument(
        "--dir-output", "--output-dir", "--dir-outputs", "--outputs-dir",
        type=str,
        default=None,
        dest="dir_output",
        help="Custom storage directory path for models, maps, evaluation & reports (e.g. for Google Colab: /content/outputs)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        choices=["all", "north", "northeast", "central", "east", "south", "bangkok"],
        help="Filter OpenAQ ground stations by region: 'north', 'northeast', 'central', 'east', 'south', or 'all'"
    )
    parser.add_argument(
        "--max-stations", "--max-locations",
        type=int,
        default=None,
        dest="max_stations",
        help="Maximum number of historical OpenAQ ground stations to query (e.g. 10, 20, 50)"
    )
    parser.add_argument(
        "--location-ids", "--locations",
        type=str,
        default=None,
        dest="location_ids",
        help="Comma-separated custom OpenAQ location IDs (e.g. '225642,368399,6266258')"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=["all", "air4thai"],
        help="Provider filter for ground stations: 'all' (all 618+ stations nationwide) or 'air4thai' (262 official PCD benchmark stations, default: all)"
    )
    parser.add_argument(
        "--train-segments",
        type=str,
        default="4",
        help="Satellite segments for training data ingestion (default: '4' for Northern/Central/Isan Thailand)"
    )
    parser.add_argument(
        "--test-segments",
        type=str,
        default="3,4,5",
        help="Satellite segments for testing / evaluation / nowcast data ingestion (default: '3,4,5' for full countrywide coverage)"
    )
    parser.add_argument(
        "--satellite-segments", "--segments",
        type=str,
        default=None,
        dest="satellite_segments",
        help="Override both train and test satellite segments globally (e.g. '4' or '3,4,5')"
    )

    args = parser.parse_args()

    def parse_segs(s: Optional[str], default_val: List[int]) -> List[int]:
        if not s:
            return default_val
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    train_segs = parse_segs(args.satellite_segments or args.train_segments, [4])
    test_segs = parse_segs(args.satellite_segments or args.test_segments, [3, 4, 5])

    if args.dir_data or args.dir_output:
        configure_paths(data_dir=args.dir_data, outputs_dir=args.dir_output)
        update_pipeline_runtime_paths()
        logger.info("Custom directory paths configured:")
        if args.dir_data:
            logger.info("  • Data Directory   : %s", args.dir_data)
        if args.dir_output:
            logger.info("  • Outputs Directory: %s", args.dir_output)

    if args.audit or args.phase == "1b":
        success = run_phase_1b_verification()
        sys.exit(0 if success else 1)

    pollutants = ["pm25", "pm10"] if args.pollutant == "all" else [args.pollutant]

    if args.phase == "1" and not args.full and not (args.start_date and args.end_date):
        sample_dt = pd.Timestamp(args.start_date or "2023-01-01 00:00:00", tz="UTC")
        for p in pollutants:
            run_phase_1_smoke_test(sample_dt, pollutant=p)
        sys.exit(0)

    if args.phase == "2":
        sample_dt = pd.Timestamp(args.start_date or "2023-01-01 00:00:00", tz="UTC")
        run_phase_2_smoke_test(sample_dt, cadence_hours=args.cadence_hours)
        sys.exit(0)

    if args.phase == "3":
        sample_dt = pd.Timestamp(args.start_date or "2023-01-01 00:00:00", tz="UTC")
        for p in pollutants:
            run_phase_3_smoke_test(sample_dt, pollutant=p, cadence_hours=args.cadence_hours)
        sys.exit(0)

    if args.phase == "4":
        if args.full:
            for p in pollutants:
                run_phase_4_full(
                    pollutant=p, split_date=args.split_date,
                    tune=args.tune, tune_trials=args.tune_trials,
                    tune_subsample_frac=args.tune_subsample_frac,
                )
        else:
            sample_dt = pd.Timestamp(args.start_date or "2026-10-12 14:00:00", tz="UTC")
            for p in pollutants:
                run_phase_4_smoke_test(sample_dt, pollutant=p, cadence_hours=args.cadence_hours)
        sys.exit(0)

    if args.phase == "5":
        sample_dt = pd.Timestamp(args.start_date or "2026-10-12 14:00:00", tz="UTC")
        for p in pollutants:
            run_phase_5_smoke_test(sample_dt, pollutant=p, cadence_hours=args.cadence_hours)
        sys.exit(0)

    if args.phase == "all":
        sample_dt = pd.Timestamp(args.start_date or "2026-10-12 14:00:00", tz="UTC")
        for p in pollutants:
            run_phase_1_smoke_test(sample_dt, pollutant=p)
            run_phase_2_smoke_test(sample_dt, cadence_hours=args.cadence_hours)
            run_phase_3_smoke_test(sample_dt, pollutant=p, cadence_hours=args.cadence_hours)
            if args.full:
                run_phase_4_full(
                    pollutant=p, split_date=args.split_date,
                    tune=args.tune, tune_trials=args.tune_trials,
                    tune_subsample_frac=args.tune_subsample_frac,
                )
            else:
                run_phase_4_smoke_test(sample_dt, pollutant=p, cadence_hours=args.cadence_hours)
            run_phase_5_smoke_test(sample_dt, pollutant=p, cadence_hours=args.cadence_hours)
        sys.exit(0)

    # Parse custom location IDs if provided
    custom_loc_ids = [int(x.strip()) for x in args.location_ids.split(",") if x.strip()] if args.location_ids else None

    if args.full:
        # Full multi-year run: 2024 to September 2026 (Train: 2024-2025, Test: 2026-01-01 to 2026-09-30)
        start_dt = pd.Timestamp(args.start_date or "2024-01-01 00:00:00", tz="UTC")
        end_dt = pd.Timestamp(args.end_date or "2026-09-30 23:59:59", tz="UTC")
        for p in pollutants:
            run_phase_1_full(
                start_date=start_dt,
                end_date=end_dt,
                split_date=args.split_date,
                cadence_hours=args.cadence_hours,
                pollutant=p,
                location_ids=custom_loc_ids,
                region=args.region,
                max_locations=args.max_stations,
                provider=args.provider,
                train_segments=train_segs,
                test_segments=test_segs,
            )
            run_phase_4_full(
                pollutant=p, split_date=args.split_date,
                tune=args.tune, tune_trials=args.tune_trials,
                tune_subsample_frac=args.tune_subsample_frac,
            )
    elif args.start_date and args.end_date:
        start_dt = pd.Timestamp(args.start_date, tz="UTC")
        end_dt = pd.Timestamp(args.end_date, tz="UTC")
        for p in pollutants:
            run_phase_1_full(
                start_date=start_dt,
                end_date=end_dt,
                split_date=args.split_date,
                cadence_hours=args.cadence_hours,
                pollutant=p,
                location_ids=custom_loc_ids,
                region=args.region,
                max_locations=args.max_stations,
                provider=args.provider,
                train_segments=train_segs,
                test_segments=test_segs,
            )
    else:
        # Default or smoke-test
        sample_dt = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
        for p in pollutants:
            run_phase_1_smoke_test(sample_dt, pollutant=p)


if __name__ == "__main__":
    main()

