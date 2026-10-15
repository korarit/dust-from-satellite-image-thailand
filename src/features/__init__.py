"""
Feature Engineering Modules for Thailand Dust Prediction Satellite
Exports:
- DirectionalWeatherEngine: 4-Directional Atmospheric Pressure & Surrounding Weather
- SatelliteTabularizer: 10 IR Bands BT, Physics BTDs, 3x3 Texture, and Temporal Tendencies
- MultiResolutionGridEngine: Multi-Resolution Spatial Grids, Copernicus DEM, TPI, Solar & Temporal Cycles
"""

from src.features.spatial_directional import (
    DirectionalWeatherEngine,
    RESOLUTION_EXCLUSION_MAP,
    DEFAULT_EXCLUSION_KM,
    MAX_SEARCH_RADIUS_KM,
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
from src.features.feature_matrix_builder import (
    FeatureMatrixBuilder,
    TemporalAntiLeakageValidator,
    SpatialAntiLeakageValidator,
    TemporalAntiLeakageError,
    SpatialAntiLeakageError,
    TargetFeatureIsolationError,
    PhysicsFallbackImputer,
    TargetMatcher,
    resolve_dataset_providers,
    KEY_COLUMNS,
    SAT_BT_COLUMNS,
    SAT_BTD_COLUMNS,
    SAT_TEXTURE_COLUMNS,
    SAT_TEMPORAL_COLUMNS,
    WEATHER_PRESSURE_COLUMNS,
    WEATHER_HUMIDITY_COLUMNS,
    WEATHER_FEATURE_COLUMNS,
    NO_WEATHER_FEATURE_COLUMNS,
    FULL_MULTIMODAL_COLUMNS,
    TOPO_SOLAR_COLUMNS,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

__all__ = [
    "DirectionalWeatherEngine",
    "SatelliteTabularizer",
    "MultiResolutionGridEngine",
    "FeatureMatrixBuilder",
    "TemporalAntiLeakageValidator",
    "SpatialAntiLeakageValidator",
    "TemporalAntiLeakageError",
    "SpatialAntiLeakageError",
    "TargetFeatureIsolationError",
    "PhysicsFallbackImputer",
    "TargetMatcher",
    "resolve_dataset_providers",
    "compute_geographic_proxies",
    "compute_solar_zenith_angle",
    "compute_solar_angles_vectorized",
    "compute_cyclical_time_features",
    "ALL_IR_BANDS",
    "BAND_COLS",
    "SUPPORTED_RESOLUTIONS_KM",
    "EXCLUSION_BUFFER_MAP",
    "DEFAULT_EXCLUSION_KM",
    "MAX_SEARCH_RADIUS_KM",
    "RESOLUTION_EXCLUSION_MAP",
    "PHYSICAL_BT_MIN_K",
    "PHYSICAL_BT_MAX_K",
    "KEY_COLUMNS",
    "SAT_BT_COLUMNS",
    "SAT_BTD_COLUMNS",
    "SAT_TEXTURE_COLUMNS",
    "SAT_TEMPORAL_COLUMNS",
    "WEATHER_PRESSURE_COLUMNS",
    "WEATHER_HUMIDITY_COLUMNS",
    "WEATHER_FEATURE_COLUMNS",
    "NO_WEATHER_FEATURE_COLUMNS",
    "FULL_MULTIMODAL_COLUMNS",
    "TOPO_SOLAR_COLUMNS",
    "ALL_FEATURE_COLUMNS",
    "TARGET_COLUMN",
]
