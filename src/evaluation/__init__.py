"""
Evaluation, Benchmarking & Nationwide Nowcasting Module (Phase 5)
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Exports:
- Statistical & Domain Metrics Engine (Continuous + PCD Thailand 5-tier AQI)
- Spatial-Temporal Cross-Validation Protocol (Regional Hold-out K-Fold)
- Real-time Multi-Resolution Map Generator & Visualizer (Parquet, GeoTIFF, Interactive Leaflet HTML, Static PNG)
- Multi-Resolution Benchmark Suite (3km, 2km, 1km, 500m across 4 Models + Ensemble)
"""

from src.evaluation.metrics import (
    compute_continuous_metrics,
    compute_aqi_classification_metrics,
    compute_comprehensive_metrics,
    assign_aqi_category,
    get_aqi_metadata,
    format_evaluation_summary_table,
    THAILAND_AQI_THRESHOLDS,
    AQI_CATEGORY_NAMES_EN,
    AQI_CATEGORY_NAMES_TH,
    AQI_COLOR_HEX,
    AQI_COLOR_NAMES,
)

from src.evaluation.spatial_cv import (
    RegionalHoldoutCV,
    SpatialTemporalEvaluator,
    assign_thailand_region,
    attach_region_column,
    THAILAND_REGIONS,
    REGION_LABELS_TH,
)

from src.evaluation.visualizer import DustMapVisualizer

from src.evaluation.benchmark_suite import (
    MultiResolutionBenchmarkSuite,
    RESOLUTION_NAMES,
)

from src.evaluation.plotter import EvaluationPlotter

from src.evaluation.shap_engine import (
    ShapAttributionEngine,
    TREE_EXPLAINER_KEYS,
)

__all__ = [
    # Metrics
    "compute_continuous_metrics",
    "compute_aqi_classification_metrics",
    "compute_comprehensive_metrics",
    "assign_aqi_category",
    "get_aqi_metadata",
    "format_evaluation_summary_table",
    "THAILAND_AQI_THRESHOLDS",
    "AQI_CATEGORY_NAMES_EN",
    "AQI_CATEGORY_NAMES_TH",
    "AQI_COLOR_HEX",
    "AQI_COLOR_NAMES",
    # Spatial CV
    "RegionalHoldoutCV",
    "SpatialTemporalEvaluator",
    "assign_thailand_region",
    "attach_region_column",
    "THAILAND_REGIONS",
    "REGION_LABELS_TH",
    # Visualization & Map Production
    "DustMapVisualizer",
    # Benchmark Suite
    "MultiResolutionBenchmarkSuite",
    "RESOLUTION_NAMES",
    # Plotter
    "EvaluationPlotter",
    # SHAP Attribution Engine
    "ShapAttributionEngine",
    "TREE_EXPLAINER_KEYS",
]
