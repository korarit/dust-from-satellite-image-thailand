"""
Centralized Storage Path Configuration for dust-prediction-satelite
Provides robust path resolution for satellite, HII ground, target PM2.5,
geographic grids, caches, and report outputs.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Base Directories
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = Path(os.getenv("DUST_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUTS_DIR = Path(os.getenv("DUST_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

# Data Subdirectories
CACHE_DIR = DATA_DIR / "cache"
HIMAWARI_CACHE_DIR = CACHE_DIR / "himawari_s3"
OPENAQ_CACHE_DIR = CACHE_DIR / "openaq"
HIMAWARI_DIR = DATA_DIR / "himawari9"
HII_DIR = DATA_DIR / "hii"
TARGET_DIR = DATA_DIR / "target"
GEO_DIR = DATA_DIR / "geo"
METADATA_DIR = DATA_DIR / "metadata"
FEATURES_DIR = DATA_DIR / "features"

# Outputs Subdirectories & Pollutant Folders
OUTPUTS_PM25_DIR = OUTPUTS_DIR / "pm25"
OUTPUTS_PM10_DIR = OUTPUTS_DIR / "pm10"

def get_pollutant_dir(pollutant: str = "pm25") -> Path:
    """Returns the dedicated root output directory for a specific pollutant."""
    return OUTPUTS_PM10_DIR if pollutant.lower() == "pm10" else OUTPUTS_PM25_DIR

def get_models_dir(pollutant: str = "pm25") -> Path:
    """Returns the dedicated models directory for a specific pollutant."""
    return get_pollutant_dir(pollutant) / "models"

def get_maps_dir(pollutant: str = "pm25") -> Path:
    """Returns the dedicated maps directory for a specific pollutant."""
    return get_pollutant_dir(pollutant) / "maps"

def get_reports_dir(pollutant: str = "pm25") -> Path:
    """Returns the dedicated reports directory for a specific pollutant."""
    return get_pollutant_dir(pollutant) / "reports"

def get_evaluation_dir(pollutant: str = "pm25") -> Path:
    """Returns the dedicated evaluation directory for a specific pollutant."""
    return get_pollutant_dir(pollutant) / "evaluation"

def get_evaluation_csv_dir(pollutant: str = "pm25") -> Path:
    """Returns the dedicated evaluation/csv directory for a specific pollutant."""
    return get_evaluation_dir(pollutant) / "csv"

# Default subdirectories (defaulting to pm25 for backward compatibility)
REPORTS_DIR = OUTPUTS_PM25_DIR / "reports"
MODELS_DIR = OUTPUTS_PM25_DIR / "models"
MAPS_DIR = OUTPUTS_PM25_DIR / "maps"
EVALUATION_DIR = OUTPUTS_PM25_DIR / "evaluation"

# Config file paths
GRID_CONFIG_PATH = CONFIGS_DIR / "grid_config.yaml"
DATA_CONFIG_PATH = CONFIGS_DIR / "data_config.yaml"
MODEL_CONFIG_PATH = CONFIGS_DIR / "model_config.yaml"


def get_tuned_params_path(pollutant: str = "pm25") -> Path:
    """Returns the Optuna-tuned hyper-parameter YAML path for a pollutant."""
    return CONFIGS_DIR / f"tuned_params_{pollutant.lower()}.yaml"


def load_yaml_config(config_path: Path) -> dict:
    """Safely loads a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_paths(
    data_dir: Optional[Union[Path, str]] = None,
    outputs_dir: Optional[Union[Path, str]] = None,
) -> None:
    """
    Dynamically reconfigures DATA_DIR and OUTPUTS_DIR and all dependent subdirectories.
    Designed for Google Colab and cloud instances with custom storage volumes.
    """
    global DATA_DIR, OUTPUTS_DIR
    global CACHE_DIR, HIMAWARI_CACHE_DIR, OPENAQ_CACHE_DIR, HIMAWARI_DIR, HII_DIR, TARGET_DIR, GEO_DIR, METADATA_DIR, FEATURES_DIR
    global OUTPUTS_PM25_DIR, OUTPUTS_PM10_DIR, REPORTS_DIR, MODELS_DIR, MAPS_DIR, EVALUATION_DIR

    if data_dir is not None:
        DATA_DIR = Path(data_dir).resolve()
        os.environ["DUST_DATA_DIR"] = str(DATA_DIR)
        CACHE_DIR = DATA_DIR / "cache"
        HIMAWARI_CACHE_DIR = CACHE_DIR / "himawari_s3"
        OPENAQ_CACHE_DIR = CACHE_DIR / "openaq"
        HIMAWARI_DIR = DATA_DIR / "himawari9"
        HII_DIR = DATA_DIR / "hii"
        TARGET_DIR = DATA_DIR / "target"
        GEO_DIR = DATA_DIR / "geo"
        METADATA_DIR = DATA_DIR / "metadata"
        FEATURES_DIR = DATA_DIR / "features"

    if outputs_dir is not None:
        OUTPUTS_DIR = Path(outputs_dir).resolve()
        os.environ["DUST_OUTPUTS_DIR"] = str(OUTPUTS_DIR)
        OUTPUTS_PM25_DIR = OUTPUTS_DIR / "pm25"
        OUTPUTS_PM10_DIR = OUTPUTS_DIR / "pm10"
        REPORTS_DIR = OUTPUTS_PM25_DIR / "reports"
        MODELS_DIR = OUTPUTS_PM25_DIR / "models"
        MAPS_DIR = OUTPUTS_PM25_DIR / "maps"
        EVALUATION_DIR = OUTPUTS_PM25_DIR / "evaluation"

    ensure_directories(seed_reference_data=True)


def seed_metadata_if_needed() -> None:
    """
    If METADATA_DIR points to custom external storage (e.g. Google Drive),
    automatically seed essential reference metadata files from the repository.
    """
    import shutil
    candidate_sources = [
        PROJECT_ROOT / "data" / "metadata",
        Path(__file__).resolve().parent.parent.parent / "data" / "metadata",
        Path.cwd() / "data" / "metadata",
    ]
    source_dir = next((p for p in candidate_sources if p.exists() and p.is_dir()), None)
    if source_dir and source_dir.resolve() != METADATA_DIR.resolve():
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        for f in source_dir.glob("*"):
            dest_file = METADATA_DIR / f.name
            if not dest_file.exists() and f.is_file():
                try:
                    shutil.copy2(f, dest_file)
                except Exception:
                    pass

def seed_geo_if_needed() -> None:
    """
    If GEO_DIR points to custom external storage (e.g. Google Drive),
    automatically seed essential master grid and DEM files from the repository.
    """
    import shutil
    candidate_sources = [
        PROJECT_ROOT / "data" / "geo",
        Path(__file__).resolve().parent.parent.parent / "data" / "geo",
        Path.cwd() / "data" / "geo",
    ]
    source_dir = next((p for p in candidate_sources if p.exists() and p.is_dir()), None)
    if source_dir and source_dir.resolve() != GEO_DIR.resolve():
        GEO_DIR.mkdir(parents=True, exist_ok=True)
        # 1. Master Grid
        src_grid = source_dir / "thailand_2km_master_grid.parquet"
        dst_grid = GEO_DIR / "thailand_2km_master_grid.parquet"
        if src_grid.exists() and not dst_grid.exists():
            try:
                shutil.copy2(src_grid, dst_grid)
            except Exception:
                pass
        # 2. DEM folder and tiles
        src_dem = source_dir / "dem"
        dst_dem = GEO_DIR / "dem"
        if src_dem.exists() and src_dem.is_dir():
            dst_dem.mkdir(parents=True, exist_ok=True)
            for f in src_dem.glob("*"):
                dst_f = dst_dem / f.name
                if not dst_f.exists() and f.is_file():
                    try:
                        shutil.copy2(f, dst_f)
                    except Exception:
                        pass


def ensure_directories(seed_reference_data: bool = False) -> None:
    """Ensures all required project directories exist.

    P3-2 fix: importing this module must NOT copy/seed reference files.
    Seeding (metadata/geo copies) only runs when seed_reference_data=True
    (explicit pipeline start via configure_paths() or run_pipeline).
    """
    dirs = [
        CONFIGS_DIR,
        DATA_DIR,
        CACHE_DIR,
        HIMAWARI_CACHE_DIR,
        OPENAQ_CACHE_DIR,
        HIMAWARI_DIR,
        HII_DIR,
        TARGET_DIR,
        GEO_DIR,
        METADATA_DIR,
        FEATURES_DIR,
        OUTPUTS_DIR,
        OUTPUTS_PM25_DIR,
        OUTPUTS_PM10_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        MAPS_DIR,
        EVALUATION_DIR,
        OUTPUTS_PM10_DIR / "models",
        OUTPUTS_PM10_DIR / "maps",
        OUTPUTS_PM10_DIR / "reports",
        OUTPUTS_PM10_DIR / "evaluation",
        OUTPUTS_PM10_DIR / "evaluation" / "csv",
        OUTPUTS_PM25_DIR / "evaluation" / "csv",
        OUTPUTS_DIR / "reports",
        OUTPUTS_DIR / "models",
        OUTPUTS_DIR / "maps",
        OUTPUTS_DIR / "evaluation",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    if seed_reference_data:
        seed_metadata_if_needed()
        seed_geo_if_needed()


# P3-2 fix: import-time side effect limited to mkdir (no file seeding/writes).
# Pass seed_reference_data=True explicitly at pipeline start if needed.
ensure_directories(seed_reference_data=False)

