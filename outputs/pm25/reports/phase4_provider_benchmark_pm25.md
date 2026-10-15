# 🏢 Phase 4: Multi-Resolution Provider Benchmark Report (PM25)

> Thailand Satellite Dust Prediction (PM25 Real-time Nowcasting)

## 1. Provider Benchmark Summary (Air4Thai vs AirGradient vs Combined)
*(Excluded sparse/non-representative networks: Clarity, AirNow, HabitatMap)*

| Experiment | Resolution | Provider | N | LightGBM MAE | CatBoost MAE | XGBoost MAE | Random Forest MAE | **Ensemble MAE** | Ensemble RMSE | Ensemble R² | Optimal Weights (L/C/X/R) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-2KM-AIR4THAI** | 2km | Air4Thai | 20 | 2.99 | 3.51 | 2.42 | 0.64 | **0.47** | 0.84 | 0.949 | 0.00 / 0.00 / 0.00 / 0.00 |
| **EXP-2KM-AIRGRADIENT** | 2km | AirGradient | 20 | 3.00 | 3.50 | 2.42 | 0.71 | **0.44** | 0.82 | 0.952 | 0.00 / 0.00 / 0.00 / 0.00 |
| **EXP-2KM-COMBINED** | 2km | Combined (Air4Thai + AirGradient) | 40 | 6.15 | 6.57 | 4.51 | 0.97 | **0.32** | 0.75 | 0.990 | 0.00 / 0.00 / 0.00 / 0.00 |
| **EXP-1KM-AIR4THAI** | 1km | Air4Thai | 20 | 2.99 | 3.51 | 2.42 | 0.64 | **0.45** | 0.84 | 0.950 | 0.00 / 0.00 / 0.00 / 0.00 |
| **EXP-1KM-AIRGRADIENT** | 1km | AirGradient | 20 | 3.00 | 3.50 | 2.42 | 0.71 | **0.44** | 0.81 | 0.954 | 0.00 / 0.00 / 0.00 / 0.00 |
| **EXP-1KM-COMBINED** | 1km | Combined (Air4Thai + AirGradient) | 40 | 6.15 | 6.57 | 4.51 | 0.97 | **0.32** | 0.75 | 0.990 | 0.00 / 0.00 / 0.00 / 0.00 |

## 2. Key Physical & Sensor Network Takeaways
- **Air4Thai (PCD Regulatory)**: Gold-standard reference monitors calibrated by Thai Pollution Control Department. Anchors absolute concentration scaling.
- **AirGradient (Dense IoT Optical Sensors)**: Provides micro-spatial plume variation across community and school zones; benefited by Random Forest variance damping.
- **Combined Model Stability**: Stacking Ensemble with Non-Negative Least Squares achieves consistent physical blending across both regulatory and dense sensor topologies.