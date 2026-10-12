"""
Centralized Storage Path Configuration for dust-prediction-satelite
Provides robust path resolution for satellite, HII ground, target PM2.5,
geographic grids, caches, and report outputs.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Base Directories
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = Path(os.getenv("DUST_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUTS_DIR = Path(os.getenv("DUST_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

# Data Subdirectories
CACHE_DIR = DATA_DIR / "cache"
HIMAWARI_CACHE_DIR = CACHE_DIR / "himawari_s3"
HIMAWARI_DIR = DATA_DIR / "himawari9"
HII_DIR = DATA_DIR / "hii"
TARGET_DIR = DATA_DIR / "target"
GEO_DIR = DATA_DIR / "geo"
METADATA_DIR = DATA_DIR / "metadata"
FEATURES_DIR = DATA_DIR / "features"

# Outputs Subdirectories
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"
MAPS_DIR = OUTPUTS_DIR / "maps"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

# Config file paths
GRID_CONFIG_PATH = CONFIGS_DIR / "grid_config.yaml"
DATA_CONFIG_PATH = CONFIGS_DIR / "data_config.yaml"
MODEL_CONFIG_PATH = CONFIGS_DIR / "model_config.yaml"



def load_yaml_config(config_path: Path) -> dict:
    """Safely loads a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories() -> None:
    """Ensures all required project directories exist."""
    dirs = [
        CONFIGS_DIR,
        DATA_DIR,
        CACHE_DIR,
        HIMAWARI_CACHE_DIR,
        HIMAWARI_DIR,
        HII_DIR,
        TARGET_DIR,
        GEO_DIR,
        METADATA_DIR,
        FEATURES_DIR,
        OUTPUTS_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        MAPS_DIR,
        EVALUATION_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()
