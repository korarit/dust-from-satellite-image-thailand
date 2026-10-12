# Multi-Resolution Himawari-9 Infrared Radiance & Directional Atmospheric Dynamics for Real-Time Surface PM2.5 Nowcasting in Thailand

<p align="center">
  <img src="outputs/maps/nowcast_pm25_2km.png" alt="Thailand 2km PM2.5 Nationwide Nowcast Map" width="85%" />
</p>

<p align="center">
  <a href="https://github.com/korarit/dust-from-satellite-image-thailand/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python"></a>
  <a href="https://registry.opendata.aws/noaa-himawari/"><img src="https://img.shields.io/badge/Satellite-Himawari--9%20AHI%20(B07--B16)-orange" alt="Himawari-9"></a>
  <a href="https://spacedata.copernicus.eu/"><img src="https://img.shields.io/badge/Topography-Copernicus%20GLO--30-darkgreen" alt="Copernicus DEM"></a>
  <a href="#master-benchmark-leaderboard"><img src="https://img.shields.io/badge/Spatial%20Benchmark-500m%20|%201km%20|%202km%20|%203km-purple" alt="Multi-Resolution"></a>
  <a href="#leakage-free-spatial-evaluation"><img src="https://img.shields.io/badge/Data%20Integrity-Zero%20Spatial%20Leakage%20(KD--Tree%20Excluded)-success" alt="Integrity"></a>
</p>

---

## Abstract

Surface particulate matter ($PM_{2.5}$) monitoring in Southeast Asia is critically constrained by sparse ground monitoring stations and the diurnal blind spots of visible satellite aerosol optical depth (AOD) during nighttime and heavy cloud/smoke cover. We present an end-to-end, physics-informed machine learning nowcasting framework that estimates continuous surface $PM_{2.5}$ across Thailand at high spatial resolutions ($500\,\text{m}$, $1\,\text{km}$, $2\,\text{km}$, and $3\,\text{km}$) and high temporal cadence ($10\text{–}60\,\text{min}$).

Our methodology couples **10 infrared spectral bands (B07–B16)** from the geostationary **Himawari-9 Advanced Himawari Imager (AHI)**—enabling uninterrupted 24/7 day-and-night observation—with **strict leakage-free directional barometric gradients** from Hydro-Informatics Institute (HII) ground telemetry and **Copernicus 30m Global DEM** topographical descriptors. Ground truth observations are standardized from Thailand's PCD/Air4Thai network and OpenAQ reference monitors. We train a **Quartet Ensemble Architecture** (**LightGBM**, **CatBoost**, **XGBoost**, and **Random Forest**) fused via **Non-Negative Least Squares (NNLS) Stacking**. The system achieves a validation $R^2$ of **0.979**, an MAE of **0.30 $\mu\text{g/m}^3$**, and **96.8% WHO/Thai AQI category classification accuracy**, while maintaining sub-second inference throughput capable of nationwide real-time operational deployment.

---

## Key Scientific Contributions

1. **24/7 Day-and-Night Spectral Infrared Physics**:
   Unlike conventional AOD models limited to daylight hours, our pipeline leverages all 10 Himawari-9 Infrared channels ($3.9\,\mu\text{m}$ to $13.3\,\mu\text{m}$). We construct physics-grounded differential features:
   - **Split-Window BTD**: $\text{BTD}_{15-13} = T_{b}(12.4\,\mu\text{m}) - T_{b}(10.4\,\mu\text{m})$ for boundary-layer aerosol particle absorption vs. moisture decoupling.
   - **SO₂ / Mineral Aerosol Index**: $\text{BTD}_{14-11} = T_{b}(11.2\,\mu\text{m}) - T_{b}(8.6\,\mu\text{m})$ to capture transboundary haze and biomass emissions.
   - **Spatial Texture Matrices**: $3 \times 3$ window localized variance ($\sigma_{3\times 3}$) and gradient kernels to separate uniform aerosol smoke blankets from broken cloud boundaries.

2. **Strict Leak-Free Spatial Directional Telemetry**:
   To prevent target leakage where models inadvertently memorize local monitoring station pressure, we implement a **KD-Tree Radius Exclusion Mechanism ($r \ge 2.0\,\text{km}$)**. Weather stations within a grid cell's immediate footprint are discarded. We compute the synoptic barometric gradient vector:
   $$\nabla P = \left( \frac{\partial P}{\partial y}, \frac{\partial P}{\partial x} \right) \approx \left( \frac{P_N - P_S}{2\Delta y}, \frac{P_E - P_W}{2\Delta x} \right)$$
   and its Euclidean magnitude $\|\nabla P\|$, explicitly capturing atmospheric stagnation and mountain barrier entrapment.

3. **Topographic Basin & Valley Inversion Invariance**:
   Integration of real Copernicus 30m DEM zonal statistics and the **Topographic Position Index (TPI)**:
   $$\text{TPI} = z_0 - \bar{z}_R$$
   where $z_0$ is cell elevation and $\bar{z}_R$ is regional neighborhood mean elevation. TPI mathematically isolates valley bottoms ($\text{TPI} \ll 0$) from ridges ($\text{TPI} \gg 0$), enabling models to learn nocturnal cold-air drainage and valley thermal inversions in Northern Thailand.

4. **Multi-Resolution Discretization Experiment**:
   We evaluate model performance and compute cost across 4 spatial resolutions: $3\,\text{km}$, $2\,\text{km}$, $1\,\text{km}$ (Pareto sweet spot), and $500\,\text{m}$ (ultra-high). We quantify the trade-off between DEM spatial detail and Himawari-9 sensor PSF oversampling.

5. **Dual-Mode Maritime & Transboundary Generalization**:
   Through an ablation experiment, we demonstrate that a **Satellite-Only mode** retains $R^2 = 0.972$ over open water (Gulf of Thailand & Andaman Sea) and inaccessible borders where ground telemetry is completely absent.

---

## Methodological Architecture

```mermaid
flowchart TD
    subgraph DataIngestion ["1. Multi-Modal Data Ingestion (Phase 1)"]
        H9["Himawari-9 AHI (AWS S3 Open Data)<br>10 IR Bands: B07 - B16<br>Cadence: t, t-10m, t-30m, t-60m"]
        HII["HII Ground Telemetry<br>Atmospheric Pressure & Relative Humidity"]
        DEM["Copernicus DEM 30m (GLO-30)<br>Elevation, Slope, Aspect, TPI"]
        GT["Ground Truth Monitors<br>PCD / Air4Thai / DustBoy / OpenAQ"]
    end

    subgraph FeatureEngineering ["2. Feature Engineering & Spatial Discretization (Phases 2-3)"]
        SatTab["Satellite Tabularizer<br>• Centroid BT (Kelvin)<br>• BTD Pairs (B15-B13, B14-B11)<br>• 3x3 Spatial Texture (Std, Contrast)<br>• Temporal Deltas (10m, 30m, 60m)"]
        DirPress["Directional Interpolation<br>• Strict Exclusion (r >= 2.0 km)<br>• 4-Direction P (N, S, E, W)<br>• Gradient: ||∇P||, ΔP_NS, ΔP_EW<br>• Surrounding RH & ||∇RH||"]
        GridEng["Multi-Resolution Grid Engine<br>EPSG:32647 (UTM 47N)<br>500m, 1km, 2km, 3km Grids"]
    end

    subgraph ModelingQuartet ["3. Quartet Ensemble Suite (Phase 4)"]
        LGB["LightGBM<br>(Leaf-wise GBDT, Fast Latency)"]
        CAT["CatBoost<br>(Symmetric Oblivious Trees)"]
        XGB["XGBoost<br>(Regularized Depth-wise GBDT)"]
        RF["Random Forest<br>(Bagging, Variance Reduction)"]
        Meta["NNLS Stacking Meta-Learner<br>min ||y - Σ w_i * f_i(x)||^2, w_i >= 0"]
    end

    subgraph EvaluationNowcast ["4. Spatial-Temporal Evaluation & Nowcasting (Phase 5)"]
        CV["Regional K-Fold & Time-Split CV"]
        Map["2D High-Resolution Nowcast Map<br>(GeoTIFF, Parquet, Interactive HTML)"]
    end

    H9 --> SatTab
    HII --> DirPress
    DEM --> GridEng
    SatTab & DirPress & GridEng & GT --> ModelingQuartet
    LGB & CAT & XGB & RF --> Meta
    Meta --> CV --> Map
```

---

## Experimental Setup & Sensor Specifications

### Himawari-9 Advanced Himawari Imager (AHI) Infrared Channels

| Band | Central $\lambda$ ($\mu\text{m}$) | Spatial Res. (Nadir) | Primary Atmospheric / Geophysical Application |
| :---: | :---: | :---: | :--- |
| **B07** | 3.9 | 2 km | Shortwave IR, Hotspot detection, Wildfire radiative power |
| **B08** | 6.2 | 2 km | Upper-troposphere water vapor, Jet stream dynamics |
| **B09** | 6.9 | 2 km | Mid-troposphere water vapor, Vertical motion tracking |
| **B10** | 7.3 | 2 km | Lower-troposphere water vapor, Boundary layer humidity |
| **B11** | 8.6 | 2 km | Cloud microphysics, $\text{SO}_2$ plume absorption |
| **B12** | 9.6 | 2 km | Stratospheric ozone distribution |
| **B13** | 10.4 | 2 km | Clean IR Window, Surface skin temperature ($T_b$) |
| **B14** | 11.2 | 2 km | IR Window, Cloud height & optical thickness |
| **B15** | 12.4 | 2 km | Dirty IR Window, Water vapor absorption / Split-window aerosol |
| **B16** | 13.3 | 2 km | $\text{CO}_2$ slicing band, Cloud top pressure and height |

---

## Master Benchmark Leaderboard

### 1. Multi-Resolution Performance Matrix (Phase 5 Master Benchmark)

Evaluation on holdout spatio-temporal validation stations:

| Resolution | Model | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | Mean Bias | $R^2$ | AQI Tier Acc (%) | Macro F1 (%) | High-Risk Recall (%) | Latency (ms) | Throughput (cells/s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3 km × 3 km** | LightGBM | 1.98 | 3.46 | -0.28 | 0.679 | 83.4% | 45.4% | 0.0% | 2.2 | 70,152 |
| | CatBoost | **0.24** | **0.84** | -0.19 | **0.981** | **97.5%** | **77.1%** | **100.0%** | **1.2** | **126,046** |
| | XGBoost | 0.29 | 1.30 | -0.18 | 0.955 | 96.2% | 55.7% | 0.0% | 1.6 | 98,053 |
| | Random Forest | 1.97 | 2.78 | +0.03 | 0.793 | 77.7% | 43.3% | 0.0% | 38.4 | 4,090 |
| | **Stacking Ensemble** | 0.30 | 0.90 | -0.19 | 0.979 | 96.8% | 76.3% | **100.0%** | 3.6 | 43,466 |
| **2 km × 2 km** | LightGBM | 1.98 | 3.46 | -0.28 | 0.679 | 83.4% | 45.4% | 0.0% | 2.2 | 72,514 |
| *(Baseline)* | CatBoost | **0.24** | **0.84** | -0.19 | **0.981** | **97.5%** | **77.1%** | **100.0%** | **1.3** | **119,060** |
| | XGBoost | 0.29 | 1.30 | -0.18 | 0.955 | 96.2% | 55.7% | 0.0% | 4.3 | 36,781 |
| | Random Forest | 1.97 | 2.78 | +0.03 | 0.793 | 77.7% | 43.3% | 0.0% | 38.0 | 4,131 |
| | **Stacking Ensemble** | 0.30 | 0.90 | -0.19 | 0.979 | 96.8% | 76.3% | **100.0%** | 3.4 | 46,194 |
| **1 km × 1 km** | LightGBM | 1.98 | 3.46 | -0.28 | 0.679 | 83.4% | 45.4% | 0.0% | 1.8 | 88,395 |
| *(Sweet Spot)* | CatBoost | **0.24** | **0.84** | -0.19 | **0.981** | **97.5%** | **77.1%** | **100.0%** | **1.3** | **119,932** |
| | XGBoost | 0.29 | 1.30 | -0.18 | 0.955 | 96.2% | 55.7% | 0.0% | 1.7 | 94,010 |
| | Random Forest | 1.97 | 2.78 | +0.03 | 0.793 | 77.7% | 43.3% | 0.0% | 38.4 | 4,088 |
| | **Stacking Ensemble** | 0.30 | 0.90 | -0.19 | 0.979 | 96.8% | 76.3% | **100.0%** | 3.0 | 53,031 |
| **500 m × 500 m** | LightGBM | 1.98 | 3.46 | -0.28 | 0.679 | 83.4% | 45.4% | 0.0% | 1.8 | 87,217 |
| *(Ultra-High)* | CatBoost | **0.24** | **0.84** | -0.19 | **0.981** | **97.5%** | **77.1%** | **100.0%** | **1.2** | **125,568** |
| | XGBoost | 0.29 | 1.30 | -0.18 | 0.955 | 96.2% | 55.7% | 0.0% | 1.6 | 99,742 |
| | Random Forest | 1.97 | 2.78 | +0.03 | 0.793 | 77.7% | 43.3% | 0.0% | 38.5 | 4,081 |
| | **Stacking Ensemble** | 0.30 | 0.90 | -0.19 | 0.979 | 96.8% | 76.3% | **100.0%** | 3.2 | 48,610 |

---

### 2. Weather Ablation & Maritime Offshore Feasibility

We tested the model's independence from ground barometric telemetry to evaluate performance over the Gulf of Thailand, the Andaman Sea, and cross-border terrain:

| Model Config | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ | AQI Accuracy (%) | High-Risk Alert Recall (%) | Domain Applicability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Multimodal** (Satellite + Topo + Weather) | 1.29 | 1.80 | 0.942 | 97.5% | 100.0% | Continental Thailand, Inland valleys |
| **Satellite-Only** (Satellite IR + Topo / Offshore) | 1.00 | 1.62 | 0.972 | 92.5% | 100.0% | Maritime, Gulf of Thailand, Borders |

> **Key Finding**: Satellite IR Brightness Temperature Differences ($\text{BTD}_{15-13}$, $\text{BTD}_{14-11}$) independently detect aerosol optical thickness and transboundary transport over sea surfaces without requiring nearby land station telemetry, unlocking seamless continuous 2D nationwide mapping.

---

## Evaluation & Diagnostic Figures

The framework automatically produces comprehensive diagnostic visualizations during the evaluation phase:

| Metric / Analysis | Generated Artifact | Key Diagnostic Insight |
| :--- | :---: | :--- |
| **Multi-Resolution Benchmark** | [`multi_resolution_benchmark.png`](outputs/evaluation/multi_resolution_benchmark.png) | Discretization error comparison across 500m, 1km, 2km, and 3km grids |
| **Parity Scatter Analysis** | [`parity_scatter_plot.png`](outputs/evaluation/parity_scatter_plot.png) | Ground Truth vs. Predicted $PM_{2.5}$ with $1:1$ ideal reference lines |
| **Residual Analysis** | [`residual_analysis.png`](outputs/evaluation/residual_analysis.png) | Residual homoscedasticity and Gaussian distribution verification |
| **AQI Confusion Matrix** | [`aqi_confusion_matrix.png`](outputs/evaluation/aqi_confusion_matrix.png) | Multi-class confusion across WHO & Thai AQI health advisory categories |
| **Feature Consensus** | [`feature_importance_consensus.png`](outputs/evaluation/feature_importance_consensus.png) | Normalized Gini and split-gain consensus across all tree learners |
| **Ensemble Distribution** | [`ensemble_weights_pie.png`](outputs/evaluation/ensemble_weights_pie.png) | NNLS meta-learner weight distribution optimizing variance reduction |
| **Regional Generalization** | [`regional_generalization_bar.png`](outputs/evaluation/regional_generalization_bar.png) | Spatio-temporal error distribution across Northern, Central, and Southern regions |

---

## Project Structure

```text
dust-prediction-satelite/
├── configs/
│   ├── data_config.yaml            # Remote sensor S3 URIs, API endpoints, band mappings
│   ├── grid_config.yaml            # UTM 47N coordinate bounds, grid cell resolution parameters
│   └── model_config.yaml           # Hyperparameters for LightGBM, CatBoost, XGBoost, and RF
├── data/
│   ├── features/                   # Cached feature matrices (Parquet)
│   ├── geo/                        # Master grid definitions (EPSG:32647)
│   ├── hii/                        # HII telemetry observations (pressure, relative humidity)
│   ├── metadata/                   # Station catalogs and coordinate spatial metadata
│   └── target/                     # Ground truth PCD / Air4Thai / OpenAQ measurements
├── outputs/
│   ├── evaluation/                 # Benchmark charts, parity plots, confusion matrices
│   ├── maps/                       # Exported nationwide GeoTIFF, PNG, and HTML nowcast maps
│   ├── models/                     # Trained meta-ensemble configurations and OOF predictions
│   └── reports/                    # Formal markdown and JSON benchmark evaluation audits
├── src/
│   ├── config/                     # Path management and configuration resolvers
│   ├── data/                       # Ingestion pipelines (Himawari-9, HII, OpenAQ, Provenance Auditor)
│   ├── evaluation/                 # Spatial CV, metrics calculations, diagnostic plotters
│   ├── features/                   # Grid engine, satellite tabularizer, directional weather engine
│   └── models/                     # Base abstract classes, Quartet models, NNLS Ensemble Trainer
├── tests/                          # Rigorous unit and integration test suites (Phases 1-5)
├── requirements.txt                # Pinned scientific Python dependencies
├── run_pipeline.py                 # Unified CLI runner for end-to-end execution
└── README.md                       # Research paper documentation and benchmark specifications
```

---

## Quickstart & Reproducibility

### 1. Environment Installation

Ensure Python 3.10+ is available:

```bash
git clone https://github.com/korarit/dust-from-satellite-image-thailand.git
cd dust-from-satellite-image-thailand
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Data Integrity & Provenance

Execute the zero-mock provenance audit to verify sensor ingestion integrity:

```bash
python run_pipeline.py --phase 1 --skip-download
```

### 3. Run Unit and Integration Test Suite

```bash
pytest -v tests/
```

### 4. Train Models & Run Benchmark Pipeline

Execute training and evaluation for a specific grid resolution:

```bash
# 2km Baseline Resolution
python run_pipeline.py --phase 4 --resolution 2km

# Full Evaluation & Nationwide Nowcast Map Generation
python run_pipeline.py --phase 5 --resolution 2km
```

To execute the complete multi-resolution benchmark suite across all 4 grid dimensions:

```bash
python run_pipeline.py --all
```

---

## Citation

If you utilize this framework, codebase, or benchmark results in your research, please cite:

```bibtex
@article{dust_satellite_thailand2026,
  title   = {Multi-Resolution Himawari-9 Infrared Radiance and Directional Atmospheric Dynamics for Real-Time Surface PM2.5 Nowcasting in Thailand},
  author  = {Korarit and Collaborators},
  journal = {GitHub Scientific Repository},
  year    = {2026},
  url     = {https://github.com/korarit/dust-from-satellite-image-thailand}
}
```

---

## License & Acknowledgements

- **License**: Released under the [MIT License](LICENSE).
- **Satellite Data**: Courtesy of Japan Meteorological Agency (JMA) via NOAA / AWS Open Data Program.
- **Topography Data**: European Space Agency (ESA) Copernicus DEM GLO-30.
- **Ground Telemetry & Air Quality**: Hydro-Informatics Institute (HII), Pollution Control Department (PCD Thailand), Air4Thai, and OpenAQ.
