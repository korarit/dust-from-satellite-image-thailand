"""
Model Suite & Ensemble Stacking Module (Phase 4)
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Exports:
- BaseDustModel: Abstract base class
- compute_dust_metrics: Statistical and AQI evaluation
- assign_aqi_category: Discretization into PCD color categories
- LGBMDustModel: Fast Leaf-wise LightGBM
- CatBoostDustModel: Symmetric/Oblivious Trees
- XGBoostDustModel: Depth-wise trees with monotonic constraints
- RandomForestDustModel: Bagging variance reducer
- DustStackingEnsemble: NNLS Stacking Meta-Learner & Feature Attribution
- ModelTrainer: High-level trainer and multi-resolution benchmark orchestrator
"""

from src.models.base_model import (
    BaseDustModel,
    compute_dust_metrics,
    assign_aqi_category,
    THAILAND_AQI_THRESHOLDS,
    THAILAND_AQI_THRESHOLDS_PM10,
)
from src.models.lgbm_model import LGBMDustModel
from src.models.catboost_model import CatBoostDustModel
from src.models.xgboost_model import XGBoostDustModel
from src.models.random_forest_model import RandomForestDustModel
from src.models.ensemble_model import DustStackingEnsemble, MODEL_KEYS
from src.models.trainer import ModelTrainer

__all__ = [
    "BaseDustModel",
    "compute_dust_metrics",
    "assign_aqi_category",
    "THAILAND_AQI_THRESHOLDS",
    "THAILAND_AQI_THRESHOLDS_PM10",
    "LGBMDustModel",
    "CatBoostDustModel",
    "XGBoostDustModel",
    "RandomForestDustModel",
    "DustStackingEnsemble",
    "MODEL_KEYS",
    "ModelTrainer",
]
