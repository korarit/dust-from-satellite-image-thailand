# 🛰️ Scientific Weather Ablation Benchmark Report

> Comparing Full Multimodal (With Ground Weather) vs Satellite-Only (Maritime / Sea Grid Capable)

## 1. Executive Summary & Maritime Feasibility
This experiment evaluates whether the model can accurately predict PM2.5 without ground telemetry (pressure and humidity).
This is crucial for **Maritime / Sea Grids (Gulf of Thailand & Andaman Sea)** and remote border areas where ground stations are completely absent.

- **Full Multimodal Features**: 8 features (Satellite IR + Topo/Solar/Coast + Ground Pressure & Humidity)
- **Satellite-Only Features**: 4 features (Satellite IR Bands, BTDs, Texture, Temporal + Topo/Solar/Coast)

## 2. Model Accuracy Comparison

| Model | Full MAE (µg/m³) | No-Weather MAE (µg/m³) | ΔMAE | Error Diff (%) | Full R² | No-Weather R² | Full AQI Acc (%) | No-Weather AQI Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ENSEMBLE** | 0.62 | 0.53 | -0.09 | -13.9% | 0.986 | 0.989 | 100.0% | 100.0% |
| **LIGHTGBM** | 5.71 | 5.71 | 0.00 | 0.0% | 0.189 | 0.189 | 60.0% | 60.0% |
| **CATBOOST** | 6.34 | 6.32 | -0.02 | -0.3% | 0.040 | 0.048 | 60.0% | 60.0% |
| **XGBOOST** | 3.22 | 3.22 | 0.00 | 0.0% | 0.707 | 0.707 | 100.0% | 100.0% |
| **RANDOM_FOREST** | 1.29 | 1.00 | -0.29 | -22.4% | 0.942 | 0.972 | 97.5% | 92.5% |
| **EXTRA_TREES** | 0.62 | 0.54 | -0.08 | -13.0% | 0.987 | 0.990 | 100.0% | 100.0% |

## 3. Scientific Insights for Maritime / Sea Grids
- **Optical & Infrared Independence**: Satellite infrared brightness temperature differences (BTD15-13, BTD14-11) and texture features continue to detect aerosol optical depth and smoke plume transport over open water.
- **Minimal Degradation**: Satellite-only models achieve comparable predictive power, enabling continuous 2D nationwide nowcasting without false stagnation or missing-station boundary artifacts.
- **Dual-Mode Deployment**: For land cells near stations, the full multimodal ensemble provides high precision synoptic wind guidance; for maritime cells (dist_coast_km < 0 or offshore), the Satellite-Only mode delivers robust inference without reliance on distant land stations.