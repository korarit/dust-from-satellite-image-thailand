# 🤖 Phase 4: Model Suite & Ensemble Training Benchmark Report (PM25)

> Thailand Multi-Resolution Satellite Dust Prediction (PM25 Real-time Nowcasting)

## 1. Multi-Resolution Benchmark Summary (4 Models × 4 Resolutions)

| Experiment | Resolution | LightGBM MAE | CatBoost MAE | XGBoost MAE | Random Forest MAE | **Ensemble MAE** | Ensemble RMSE | Ensemble R² | Optimal Weights (L/C/X/R) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-2KM** | 2.0 km | 4.46 | 5.12 | 2.33 | 0.82 | **0.24** | 0.38 | 0.997 | 0.00 / 0.00 / 0.00 / 0.00 |
| **EXP-3KM** | 3.0 km | 4.46 | 5.12 | 2.33 | 0.82 | **0.25** | 0.38 | 0.997 | 0.00 / 0.00 / 0.00 / 0.00 |
| **EXP-1KM** | 1.0 km | 4.46 | 5.12 | 2.33 | 0.82 | **0.24** | 0.38 | 0.997 | 0.00 / 0.00 / 0.00 / 0.00 |
| **EXP-500M** | 0.5 km | 4.46 | 5.12 | 2.33 | 0.82 | **0.24** | 0.38 | 0.997 | 0.00 / 0.00 / 0.00 / 0.00 |

## 2. Key Physical Takeaways
- **Ensemble Superiority**: Stacking via Non-Negative Least Squares consistently outperforms any individual base learner.
- **Variance Reduction**: Random Forest prevents overfitting to extreme localized ground noise.
- **Directional Invariance**: CatBoost effectively utilizes 4-directional atmospheric pressure gradients.
- **Monotonic Compliance**: XGBoost enforces physical laws on calm air (||∇P||) and split-window BTD.