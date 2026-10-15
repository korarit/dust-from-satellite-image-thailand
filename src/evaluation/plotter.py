"""
Phase 5: Scientific Evaluation Plotter & Visual Diagnostics
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Generates 7 Comprehensive Scientific Visualizations in outputs/evaluation/:
1. parity_scatter_plot.png: Observed vs Predicted PM2.5 Parity Plot (1:1 line, R², MAE, RMSE)
2. residual_analysis.png: Residual Errors vs Concentration & Error Distribution (Heteroscedasticity & Bias)
3. aqi_confusion_matrix.png: PCD Thailand 5-tier Health Warning Bands Confusion Matrix Heatmap
4. multi_resolution_benchmark.png: Resolution Accuracy Comparison (3k, 2k, 1k, 500m) & Pareto Frontier
5. regional_generalization_bar.png: Spatial Hold-out Generalization across 5 Geographic Regions
6. feature_importance_consensus.png: Top Atmospheric & Satellite Predictive Features Consensus
7. ensemble_weights_pie.png: Non-Negative Least Squares (NNLS) Quartet Weight Allocation

Strict Zero-Mock: 100% deterministic physical graphics, zero random calls.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.paths import EVALUATION_DIR, REPORTS_DIR, get_evaluation_dir, get_evaluation_csv_dir
from src.evaluation.metrics import (
    assign_aqi_category,
    compute_continuous_metrics,
    compute_aqi_classification_metrics,
    THAILAND_AQI_THRESHOLDS,
    AQI_COLOR_HEX,
    AQI_CATEGORY_NAMES_EN,
)

logger = logging.getLogger(__name__)

# Dark Theme Palette Configuration
BG_COLOR = "#0f172a"
PANEL_COLOR = "#1e293b"
GRID_COLOR = "#334155"
TEXT_COLOR = "#f8fafc"
MUTED_TEXT = "#94a3b8"
ACCENT_BLUE = "#38bdf8"
ACCENT_CYAN = "#06b6d4"


def set_custom_dark_style():
    """Applies clean, high-contrast dark theme styling for publication graphics."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": PANEL_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": MUTED_TEXT,
        "ytick.color": MUTED_TEXT,
        "text.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
    })


class EvaluationPlotter:
    """
    Generates publication-quality evaluation and benchmark diagnostic graphics
    saved to outputs/evaluation/.
    """

    def __init__(self, output_dir: Optional[Path] = None, pollutant: str = "pm25") -> None:
        self.pollutant = pollutant.lower()
        self.output_dir = output_dir or get_evaluation_dir(self.pollutant)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir = self.output_dir / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        set_custom_dark_style()
    def plot_parity_scatter(
        self,
        y_true: Union[np.ndarray, pd.Series, List[float]],
        y_pred: Union[np.ndarray, pd.Series, List[float]],
        model_name: str = "Stacking Ensemble (NNLS)",
        filename: Optional[str] = None,
        pollutant: str = "pm25",
    ) -> Path:
        """
        1. Observed vs Predicted Parity Scatter Plot.
        Features 1:1 Identity reference line, regression line, and statistical metrics badge.
        """
        pol = pollutant.lower()
        pol_name = pol.upper()
        if filename is None:
            filename = f"parity_scatter_plot_{pol}.png"

        y_t = np.asarray(y_true, dtype=np.float64).ravel()
        y_p = np.asarray(y_pred, dtype=np.float64).ravel()
        valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
        y_t, y_p = y_t[valid], y_p[valid]

        metrics = compute_continuous_metrics(y_t, y_p)
        cats = assign_aqi_category(y_t, pollutant=pol)
        point_colors = [AQI_COLOR_HEX[c] for c in cats]

        fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG_COLOR)
        ax.set_facecolor(PANEL_COLOR)

        # Identity Line (1:1)
        max_val = max(100.0, float(np.max([y_t.max(), y_p.max()])) * 1.1)
        ax.plot([0, max_val], [0, max_val], color="#ef4444", linestyle="--", linewidth=1.8, label="1:1 Identity Line", zorder=2)

        # Scatter points colored by PCD AQI category
        scatter = ax.scatter(y_t, y_p, c=point_colors, s=55, alpha=0.85, edgecolors="#ffffff", linewidth=0.5, zorder=3)

        # Linear regression trend line
        if len(y_t) > 2 and np.var(y_t) > 1e-6:
            coeffs = np.polyfit(y_t, y_p, deg=1)
            poly_x = np.linspace(0, max_val, 100)
            poly_y = np.polyval(coeffs, poly_x)
            ax.plot(poly_x, poly_y, color=ACCENT_CYAN, linestyle="-", linewidth=2.0,
                    label=f"Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}", zorder=4)

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel(f"Ground Truth Observed {pol_name} (µg/m³)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel(f"Model Predicted {pol_name} (µg/m³)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title(f"Parity Plot: Observed vs Predicted {pol_name}\n[{model_name}]", fontsize=13, fontweight="bold", color=ACCENT_BLUE, pad=12)
        ax.grid(True, alpha=0.4)

        # Stats Annotation Box
        stats_text = (
            f"N = {len(y_t)}\n"
            f"MAE = {metrics['mae']:.2f} µg/m³\n"
            f"RMSE = {metrics['rmse']:.2f} µg/m³\n"
            f"R² = {metrics['r2']:.3f}\n"
            f"Pearson r = {metrics['pearson_r']:.3f}\n"
            f"Mean Bias = {metrics['mean_bias']:.2f} µg/m³"
        )
        ax.text(
            0.04, 0.94, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#090d16", edgecolor=GRID_COLOR, alpha=0.9),
            color="#f1f5f9"
        )

        ax.legend(loc="lower right", facecolor="#090d16", edgecolor=GRID_COLOR, fontsize=10)
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Parity Scatter Plot -> %s", out_path)
        return out_path

    def plot_residual_analysis(
        self,
        y_true: Union[np.ndarray, pd.Series, List[float]],
        y_pred: Union[np.ndarray, pd.Series, List[float]],
        filename: Optional[str] = None,
        pollutant: str = "pm25",
    ) -> Path:
        """
        2. Residual Analysis: Error vs Concentration and Residual Histogram/KDE.
        Detects heteroscedasticity and evaluates zero-bias centering.
        """
        pol = pollutant.lower()
        pol_name = pol.upper()
        if filename is None:
            filename = f"residual_analysis_{pol}.png"

        y_t = np.asarray(y_true, dtype=np.float64).ravel()
        y_p = np.asarray(y_pred, dtype=np.float64).ravel()
        valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
        y_t, y_p = y_t[valid], y_p[valid]

        residuals = y_p - y_t
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)
        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL_COLOR)

        # Left: Residuals vs Observed Concentration
        ax1.axhline(0.0, color="#ef4444", linestyle="--", linewidth=1.5, zorder=2)
        ax1.axhline(mean_res, color=ACCENT_CYAN, linestyle=":", linewidth=1.5, label=f"Mean Bias ({mean_res:.2f})")
        ax1.scatter(y_t, residuals, color=ACCENT_BLUE, s=40, alpha=0.75, edgecolors="#ffffff", linewidth=0.4, zorder=3)
        ax1.set_xlabel(f"Observed {pol_name} (µg/m³)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Residual (Predicted - Observed) [µg/m³]", fontsize=11, fontweight="bold")
        ax1.set_title("Residuals vs Concentration", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
        ax1.grid(True, alpha=0.4)
        ax1.legend(facecolor="#090d16", edgecolor=GRID_COLOR)

        # Right: Residual Distribution Histogram & KDE
        sns.histplot(residuals, kde=True, color=ACCENT_CYAN, ax=ax2, bins=15, edgecolor="#ffffff", linewidth=0.5)
        ax2.axvline(0.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Zero Bias Reference")
        ax2.axvline(mean_res, color=ACCENT_BLUE, linestyle=":", linewidth=1.5, label=f"Mean ({mean_res:.2f} ± {std_res:.2f})")
        ax2.set_xlabel("Residual Error (µg/m³)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Frequency Density", fontsize=11, fontweight="bold")
        ax2.set_title("Residual Error Distribution", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
        ax2.grid(True, alpha=0.4)
        ax2.legend(facecolor="#090d16", edgecolor=GRID_COLOR)

        plt.suptitle(f"{pol_name} Residual Diagnostics & Error Distribution Analysis", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Residual Analysis -> %s", out_path)
        return out_path

    def plot_aqi_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series, List[float]],
        y_pred: Union[np.ndarray, pd.Series, List[float]],
        filename: Optional[str] = None,
        pollutant: str = "pm25",
    ) -> Path:
        """
        3. Confusion Matrix Heatmap for PCD Thailand 5-tier Health Warning Bands.
        """
        pol = pollutant.lower()
        pol_name = pol.upper()
        if filename is None:
            filename = f"aqi_confusion_matrix_{pol}.png"

        aqi_res = compute_aqi_classification_metrics(y_true, y_pred, pollutant=pol)
        cm = np.array(aqi_res["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=BG_COLOR)
        ax.set_facecolor(PANEL_COLOR)

        if pol == "pm10":
            short_labels = ["Very Good\n(0-50)", "Good\n(50-80)", "Moderate\n(80-120)", "Orange\n(120-180)", "Red\n(>180)"]
        else:
            short_labels = ["Very Good\n(0-15)", "Good\n(15-25)", "Moderate\n(25-37.5)", "Orange\n(37.5-75)", "Red\n(>75)"]

        # Annotation labels: count and percentage
        annot = np.empty_like(cm, dtype=object)
        n_total = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cnt = cm[i, j]
                pct = (cnt / n_total * 100.0) if n_total > 0 else 0.0
                annot[i, j] = f"{cnt}\n({pct:.1f}%)" if cnt > 0 else "0"

        cmap = sns.dark_palette("#38bdf8", as_cmap=True)
        sns.heatmap(
            cm, annot=annot, fmt="", cmap=cmap, cbar=True,
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=1.0, linecolor=GRID_COLOR, ax=ax,
            cbar_kws={"label": "Sample Count"}
        )

        ax.set_xlabel(f"Predicted {pol_name} AQI Health Band", fontsize=11, fontweight="bold", labelpad=10)
        ax.set_ylabel(f"True Ground Truth {pol_name} AQI Health Band", fontsize=11, fontweight="bold", labelpad=10)
        acc_val = aqi_res.get("accuracy", 0.0)
        f1_val = aqi_res.get("macro_f1", 0.0)
        recall_hr = aqi_res.get("high_risk_recall", 0.0)
        ax.set_title(
            f"Thailand PCD {pol_name} AQI 5-Tier Confusion Matrix\n"
            f"[Acc: {acc_val:.1f}% | Macro F1: {f1_val:.1f}% | High-Risk Alert Recall: {recall_hr:.1f}%]",
            fontsize=12, fontweight="bold", color=ACCENT_BLUE, pad=12
        )

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved AQI Confusion Matrix Heatmap -> %s", out_path)
        return out_path

    def plot_multi_resolution_benchmark(
        self,
        leaderboard_df: pd.DataFrame,
        filename: Optional[str] = None,
        pollutant: str = "pm25",
    ) -> Path:
        """
        4. Multi-Resolution Comparison: MAE Comparison Bar Chart and Latency vs Accuracy Pareto Frontier.
        """
        pol = pollutant.lower()
        fname = filename or f"multi_resolution_benchmark_{pol}.png"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG_COLOR)
        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL_COLOR)

        # Plot 1: MAE across Resolutions for Models
        models_order = ["LIGHTGBM", "CATBOOST", "XGBOOST", "RANDOM_FOREST", "Stacking Ensemble (NNLS)"]
        palette = ["#38bdf8", "#34d399", "#f59e0b", "#a78bfa", "#f43f5e"]

        plot_df = leaderboard_df.copy()
        sns.barplot(
            data=plot_df, x="resolution", y="mae", hue="model",
            palette="deep", ax=ax1, edgecolor="#ffffff", linewidth=0.5
        )
        ax1.set_title("PM2.5 MAE by Resolution & Model", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
        ax1.set_xlabel("Spatial Grid Resolution", fontsize=11, fontweight="bold")
        ax1.set_ylabel("MAE (µg/m³)", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.4)
        ax1.legend(title="Model", facecolor="#090d16", edgecolor=GRID_COLOR, fontsize=8)

        # Plot 2: Pareto Frontier (MAE vs Latency)
        ens_df = plot_df[plot_df["model"].str.contains("Stacking Ensemble", case=False)]
        other_df = plot_df[~plot_df["model"].str.contains("Stacking Ensemble", case=False)]

        ax2.scatter(other_df["latency_ms"], other_df["mae"], color=MUTED_TEXT, s=60, alpha=0.7, label="Base Learners", zorder=2)
        ax2.scatter(ens_df["latency_ms"], ens_df["mae"], color="#f43f5e", s=110, edgecolors="#ffffff", linewidth=1.5, label="Stacking Ensemble", zorder=3)

        for _, r in ens_df.iterrows():
            ax2.annotate(
                f"{r['resolution']}", (r['latency_ms'], r['mae']),
                textcoords="offset points", xytext=(8, 5),
                fontsize=9, color="#f8fafc", fontweight="bold"
            )

        ax2.set_title("Pareto Frontier: Accuracy vs Inference Latency", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
        ax2.set_xlabel("Inference Latency per Snapshot (ms)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("MAE (µg/m³) [Lower is Better]", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.4)
        ax2.legend(facecolor="#090d16", edgecolor=GRID_COLOR)

        plt.suptitle("Multi-Resolution Benchmark (3km, 2km, 1km, 500m) Diagnostics", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
        plt.tight_layout()

        out_path = self.output_dir / fname
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Multi-Resolution Benchmark Plot -> %s", out_path)
        return out_path

    def plot_regional_generalization(
        self,
        regional_results: Dict[str, Any],
        filename: Optional[str] = None,
        pollutant: str = "pm25",
    ) -> Path:
        """
        5. Regional Generalization: Spatial Hold-Out Cross-Validation performance across 5 regions.
        """
        pol = pollutant.lower()
        fname = filename or f"regional_generalization_bar_{pol}.png"
        by_reg = regional_results.get("by_region", {})
        regions = []
        maes = []
        rmses = []
        accs = []

        for reg_k, reg_data in by_reg.items():
            r_name = reg_k.capitalize()
            regions.append(r_name)
            m = reg_data.get("metrics", {})
            maes.append(m.get("mae", 0.0))
            rmses.append(m.get("rmse", 0.0))
            accs.append(m.get("aqi_accuracy", 0.0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOR)
        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL_COLOR)

        x = np.arange(len(regions))
        width = 0.35

        # Subplot 1: MAE and RMSE
        ax1.bar(x - width/2, maes, width, label="MAE (µg/m³)", color=ACCENT_BLUE, edgecolor="#ffffff", linewidth=0.5)
        ax1.bar(x + width/2, rmses, width, label="RMSE (µg/m³)", color="#f43f5e", edgecolor="#ffffff", linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(regions, fontsize=9)
        ax1.set_ylabel("Error (µg/m³)", fontsize=11, fontweight="bold")
        ax1.set_title("Regional Hold-Out Error (Unseen Regions)", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
        ax1.grid(True, alpha=0.4)
        ax1.legend(facecolor="#090d16", edgecolor=GRID_COLOR)

        # Subplot 2: AQI Tier Accuracy
        bars = ax2.bar(regions, accs, color="#34d399", edgecolor="#ffffff", linewidth=0.5, width=0.5)
        ax2.set_ylabel("AQI Tier Accuracy (%)", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, 100)
        ax2.set_title("Regional AQI Color Band Classification Accuracy", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
        ax2.grid(True, alpha=0.4)

        for b in bars:
            height = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2., height + 1.5, f"{height:.1f}%",
                     ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")

        plt.suptitle("Spatial Generalization: Regional Hold-Out Cross-Validation", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
        plt.tight_layout()

        out_path = self.output_dir / fname
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Regional Generalization Bar Chart -> %s", out_path)
        return out_path

    def plot_feature_importance_consensus(
        self,
        attribution_df: pd.DataFrame,
        top_n: int = 15,
        filename: Optional[str] = None,
        pollutant: str = "pm25",
    ) -> Path:
        """
        6. Top Predictive Atmospheric & Satellite Features Consensus across 4 Quartet Models.
        """
        pol = pollutant.lower()
        fname = filename or f"feature_importance_consensus_{pol}.png"
        df = attribution_df.head(top_n).sort_values("ensemble_importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
        ax.set_facecolor(PANEL_COLOR)

        bars = ax.barh(
            df["feature"], df["ensemble_importance"] * 100.0,
            color=ACCENT_BLUE, edgecolor="#ffffff", linewidth=0.6, alpha=0.9
        )

        ax.set_xlabel("Ensemble Normalized Importance (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Physical Predictive Feature", fontsize=11, fontweight="bold")
        ax.set_title(f"Top {top_n} Atmospheric & Satellite Predictive Features\n[Cross-Model Ensemble Consensus]",
                     fontsize=13, fontweight="bold", color=ACCENT_BLUE, pad=12)
        ax.grid(True, alpha=0.4)

        for b in bars:
            width = b.get_width()
            ax.text(width + 0.2, b.get_y() + b.get_height() / 2, f"{width:.1f}%",
                    va="center", ha="left", color="#f8fafc", fontsize=9, fontweight="bold")

        plt.tight_layout()
        out_path = self.output_dir / fname
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Feature Importance Consensus Plot -> %s", out_path)
        return out_path

    def plot_ensemble_weights_pie(
        self,
        weights: Dict[str, float],
        filename: Optional[str] = None,
        pollutant: str = "pm25",
    ) -> Path:
        """
        7. Donut / Pie Chart showing the Non-Negative Least Squares (NNLS) Quartet Weight Allocation.
        """
        pol = pollutant.lower()
        fname = filename or f"ensemble_weights_pie_{pol}.png"
        models = [k.upper().replace("_", " ") for k in weights.keys()]
        w_vals = [float(v) for v in weights.values()]

        # Ensure sum > 0
        if sum(w_vals) < 1e-6:
            w_vals = [1.0 / len(models)] * len(models)

        colors = ["#38bdf8", "#34d399", "#f59e0b", "#a78bfa"]

        fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        wedges, texts, autotexts = ax.pie(
            w_vals, labels=models, autopct="%1.1f%%",
            startangle=140, colors=colors[:len(models)],
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor=BG_COLOR, linewidth=2)
        )

        for text in texts:
            text.set_color(TEXT_COLOR)
            text.set_fontsize(11)
            text.set_fontweight("bold")
        for autotext in autotexts:
            autotext.set_color("#090d16")
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")

        ax.set_title("Stacking Ensemble Meta-Learner\n[Non-Negative Least Squares (NNLS) Weights]",
                     fontsize=13, fontweight="bold", color=ACCENT_BLUE, pad=16)

        plt.tight_layout()
        out_path = self.output_dir / fname
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Ensemble Weights Donut Chart -> %s", out_path)
        return out_path

    def export_all_evaluation_csvs(
        self,
        y_true: Union[np.ndarray, pd.Series, List[float]],
        y_pred: Union[np.ndarray, pd.Series, List[float]],
        leaderboard_df: pd.DataFrame,
        regional_results: Dict[str, Any],
        attribution_df: pd.DataFrame,
        ensemble_weights: Dict[str, float],
        pollutant: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Exports comprehensive evaluation dataframes as CSV files into outputs/{pollutant}/evaluation/csv/
        for tabular analytics, spreadsheet audits, and reporting:
        1. parity_predictions_{pol}.csv: Sample-by-sample observed, predicted, residual, and AQI tier.
        2. aqi_confusion_matrix_{pol}.csv: 5x5 confusion matrix counts across PCD Thailand AQI bands.
        3. multi_resolution_benchmark_{pol}.csv: Leaderboard across resolutions & models (MAE, RMSE, latency, etc.).
        4. regional_generalization_{pol}.csv: Regional hold-out cross validation metrics across Thailand regions.
        5. feature_importance_{pol}.csv: Ranked cross-model atmospheric and satellite feature importances.
        6. ensemble_weights_{pol}.csv: Non-Negative Least Squares (NNLS) base model weight allocations.
        """
        pol = (pollutant or self.pollutant).lower()
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        csv_files = {}

        # 1. Parity & Residuals Sample CSV
        y_t = np.asarray(y_true, dtype=np.float64).ravel()
        y_p = np.asarray(y_pred, dtype=np.float64).ravel()
        valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
        y_t, y_p = y_t[valid], y_p[valid]
        residuals = y_p - y_t
        obs_cats = assign_aqi_category(y_t, pollutant=pol)
        pred_cats = assign_aqi_category(y_p, pollutant=pol)

        parity_df = pd.DataFrame({
            "sample_index": np.arange(len(y_t)),
            f"observed_{pol}": y_t,
            f"predicted_{pol}": y_p,
            "residual_error": residuals,
            "absolute_error": np.abs(residuals),
            "observed_aqi_tier": obs_cats,
            "predicted_aqi_tier": pred_cats,
            "aqi_tier_match": (obs_cats == pred_cats).astype(int),
        })
        p_path = self.csv_dir / f"parity_predictions_{pol}.csv"
        parity_df.to_csv(p_path, index=False)
        csv_files["parity_predictions"] = p_path
        logger.info("Saved Evaluation CSV -> %s", p_path)

        # 2. AQI Confusion Matrix CSV
        aqi_res = compute_aqi_classification_metrics(y_t, y_p, pollutant=pol)
        cm = np.array(aqi_res["confusion_matrix"])
        tier_names = ["Tier_0_Very_Good", "Tier_1_Good", "Tier_2_Moderate", "Tier_3_Orange", "Tier_4_Red"]
        cm_df = pd.DataFrame(cm, index=[f"Actual_{t}" for t in tier_names], columns=[f"Pred_{t}" for t in tier_names])
        cm_path = self.csv_dir / f"aqi_confusion_matrix_{pol}.csv"
        cm_df.to_csv(cm_path)
        csv_files["aqi_confusion_matrix"] = cm_path
        logger.info("Saved Evaluation CSV -> %s", cm_path)

        # 3. Multi-Resolution Benchmark Leaderboard CSV
        if not leaderboard_df.empty:
            bench_path = self.csv_dir / f"multi_resolution_benchmark_{pol}.csv"
            leaderboard_df.to_csv(bench_path, index=False)
            csv_files["multi_resolution_benchmark"] = bench_path
            logger.info("Saved Evaluation CSV -> %s", bench_path)

        # 4. Regional Generalization CSV
        by_reg = regional_results.get("by_region", {})
        reg_rows = []
        for reg_k, reg_data in by_reg.items():
            m = reg_data.get("metrics", {})
            reg_rows.append({
                "region": reg_k,
                "region_name_th": reg_data.get("region_th", reg_k),
                "n_samples": m.get("n_samples", 0),
                "mae": m.get("mae", 0.0),
                "rmse": m.get("rmse", 0.0),
                "mean_bias": m.get("mean_bias", 0.0),
                "r2": m.get("r2", 0.0),
                "aqi_accuracy": m.get("aqi_accuracy", 0.0),
                "aqi_macro_f1": m.get("aqi_macro_f1", 0.0),
                "high_risk_recall": m.get("high_risk_recall", 0.0),
            })
        if reg_rows:
            reg_df = pd.DataFrame(reg_rows)
            reg_path = self.csv_dir / f"regional_generalization_{pol}.csv"
            reg_df.to_csv(reg_path, index=False)
            csv_files["regional_generalization"] = reg_path
            logger.info("Saved Evaluation CSV -> %s", reg_path)

        # 5. Feature Importance CSV
        if not attribution_df.empty:
            feat_path = self.csv_dir / f"feature_importance_{pol}.csv"
            attribution_df.to_csv(feat_path, index=False)
            csv_files["feature_importance"] = feat_path
            logger.info("Saved Evaluation CSV -> %s", feat_path)

        # 6. Ensemble Meta-Learner Weights CSV
        if ensemble_weights:
            w_rows = [{"model": k, "weight": float(v)} for k, v in ensemble_weights.items()]
            w_df = pd.DataFrame(w_rows)
            w_path = self.csv_dir / f"ensemble_weights_{pol}.csv"
            w_df.to_csv(w_path, index=False)
            csv_files["ensemble_weights"] = w_path
            logger.info("Saved Evaluation CSV -> %s", w_path)

        return csv_files

    def generate_all_evaluation_plots(
        self,
        y_true: Union[np.ndarray, pd.Series, List[float]],
        y_pred: Union[np.ndarray, pd.Series, List[float]],
        leaderboard_df: pd.DataFrame,
        regional_results: Dict[str, Any],
        attribution_df: pd.DataFrame,
        ensemble_weights: Dict[str, float],
        pollutant: str = "pm25",
    ) -> Dict[str, Path]:
        """Generates all 7 publication evaluation diagnostic plots with explicit pollutant suffix."""
        pol = pollutant.lower()
        p_parity = self.plot_parity_scatter(y_true, y_pred, pollutant=pol)
        p_res = self.plot_residual_analysis(y_true, y_pred, pollutant=pol)
        p_cm = self.plot_aqi_confusion_matrix(y_true, y_pred, pollutant=pol)
        p_multi = self.plot_multi_resolution_benchmark(leaderboard_df, pollutant=pol)
        p_reg = self.plot_regional_generalization(regional_results, pollutant=pol)
        p_feat = self.plot_feature_importance_consensus(attribution_df, pollutant=pol)
        p_w = self.plot_ensemble_weights_pie(ensemble_weights, pollutant=pol)

        # Also automatically export corresponding evaluation CSVs into outputs/{pol}/evaluation/csv/
        _ = self.export_all_evaluation_csvs(
            y_true=y_true,
            y_pred=y_pred,
            leaderboard_df=leaderboard_df,
            regional_results=regional_results,
            attribution_df=attribution_df,
            ensemble_weights=ensemble_weights,
            pollutant=pol,
        )

        return {
            "parity_scatter": p_parity,
            "residual_analysis": p_res,
            "aqi_confusion_matrix": p_cm,
            "multi_resolution_benchmark": p_multi,
            "regional_generalization": p_reg,
            "feature_importance_consensus": p_feat,
            "ensemble_weights_pie": p_w,
        }

    def plot_provider_comparison(
        self,
        summary_table: Union[pd.DataFrame, List[Dict[str, Any]]],
        filename: Optional[str] = None,
        pollutant: str = "pm25",
    ) -> Path:
        """
        Generates Multi-Panel Bar Comparison across Resolutions and Providers:
        - Panel 1: Ensemble MAE (µg/m³) for Air4Thai vs AirGradient vs Combined
        - Panel 2: Ensemble R² Coefficient of Determination across resolutions
        """
        pol = pollutant.lower()
        pol_name = pol.upper()
        fname = filename or f"provider_comparison_{pol}.png"
        df = pd.DataFrame(summary_table) if not isinstance(summary_table, pd.DataFrame) else summary_table.copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_COLOR)
        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL_COLOR)

        if df.empty or "provider" not in df.columns:
            logger.warning("Empty or invalid provider summary table. Skipping provider comparison plot.")
            plt.close(fig)
            return self.output_dir / fname

        # Grouped bar plots
        res_col = "resolution_name" if "resolution_name" in df.columns else "resolution_km"
        provider_order = [p for p in ["Air4Thai", "AirGradient"] if p in df["provider"].unique()]
        comb_names = [p for p in df["provider"].unique() if "combined" in str(p).lower()]
        provider_order += comb_names

        prov_palette = {
            "Air4Thai": "#38bdf8",     # Cyan/Blue
            "AirGradient": "#34d399",  # Emerald/Green
        }
        for cn in comb_names:
            prov_palette[cn] = "#f59e0b"  # Amber

        # 1. MAE
        sns.barplot(
            data=df, x=res_col, y="ensemble_mae", hue="provider",
            hue_order=provider_order, palette=prov_palette, ax=ax1,
            edgecolor="#ffffff", linewidth=0.5
        )
        ax1.set_title(f"Ensemble {pol_name} MAE by Resolution & Provider", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
        ax1.set_xlabel("Spatial Resolution", fontsize=11, fontweight="bold")
        ax1.set_ylabel(f"Ensemble MAE (µg/m³)", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.4)
        ax1.legend(title="Provider", facecolor="#090d16", edgecolor=GRID_COLOR, fontsize=9)

        # 2. R²
        sns.barplot(
            data=df, x=res_col, y="ensemble_r2", hue="provider",
            hue_order=provider_order, palette=prov_palette, ax=ax2,
            edgecolor="#ffffff", linewidth=0.5
        )
        ax2.set_title(f"Ensemble {pol_name} R² by Resolution & Provider", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
        ax2.set_xlabel("Spatial Resolution", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Coefficient of Determination (R²)", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.4)
        ax2.legend(title="Provider", facecolor="#090d16", edgecolor=GRID_COLOR, fontsize=9)

        plt.suptitle(f"Thailand Multi-Resolution {pol_name} Nowcasting: Ground Truth Provider Benchmark",
                     fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
        plt.tight_layout()

        out_path = self.output_dir / fname
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Provider Comparison Plot -> %s", out_path)

        # Export CSV
        csv_path = self.csv_dir / f"provider_benchmark_summary_{pol}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved Provider Benchmark CSV -> %s", csv_path)

        return out_path

    def plot_provider_parity_scatter(
        self,
        y_true: Union[np.ndarray, pd.Series, List[float]],
        y_pred: Union[np.ndarray, pd.Series, List[float]],
        providers: Union[np.ndarray, pd.Series, List[str]],
        pollutant: str = "pm25",
    ) -> Dict[str, Path]:
        """
        Generates individual and multi-panel parity scatter plots for each provider
        (Air4Thai, AirGradient, and Combined).
        """
        pol = pollutant.lower()
        pol_name = pol.upper()
        y_t = np.asarray(y_true, dtype=np.float64).ravel()
        y_p = np.asarray(y_pred, dtype=np.float64).ravel()
        p_arr = np.asarray(providers).ravel()

        unique_provs = [
            p for p in np.unique(p_arr)
            if str(p).lower() not in ("none", "nan", "unknown", "clarity", "airnow", "habitatmap")
        ]
        outputs = {}

        # 1. Individual parity plot per provider
        for prov in unique_provs:
            mask = p_arr == prov
            if np.sum(mask) >= 2:
                prov_clean = str(prov).lower().replace(" ", "_")
                fname = f"parity_scatter_{pol}_{prov_clean}.png"
                p_out = self.plot_parity_scatter(
                    y_t[mask], y_p[mask],
                    model_name=f"Stacking Ensemble ({prov} Only)",
                    filename=fname,
                    pollutant=pol,
                )
                outputs[prov] = p_out

        # 2. Combined overview multi-panel plot
        active_panels = [p for p in ["Air4Thai", "AirGradient"] if p in unique_provs] + ["Combined"]
        n_panels = len(active_panels)
        fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6), facecolor=BG_COLOR)
        if n_panels == 1:
            axes = [axes]

        valid_mask = (~np.isnan(y_t)) & (~np.isnan(y_p))
        if not np.any(valid_mask):
            plt.close(fig)
            return outputs

        max_val = max(100.0, float(np.nanmax(np.concatenate([y_t[valid_mask], y_p[valid_mask]]))) * 1.1)

        for ax, panel_name in zip(axes, active_panels):
            ax.set_facecolor(PANEL_COLOR)
            if panel_name == "Combined":
                mask = np.isin(p_arr, unique_provs) & valid_mask
                title_label = f"Combined ({' + '.join(unique_provs)})"
            else:
                mask = (p_arr == panel_name) & valid_mask
                title_label = panel_name

            sub_yt = y_t[mask]
            sub_yp = y_p[mask]
            if len(sub_yt) < 2:
                ax.set_title(f"{title_label} (N={len(sub_yt)} - Insufficient)", color=MUTED_TEXT)
                continue

            sub_m = compute_continuous_metrics(sub_yt, sub_yp)
            cats = assign_aqi_category(sub_yt, pollutant=pol)
            pt_colors = [AQI_COLOR_HEX[c] for c in cats]

            ax.plot([0, max_val], [0, max_val], color="#ef4444", linestyle="--", linewidth=1.6, label="1:1 Identity")
            ax.scatter(sub_yt, sub_yp, c=pt_colors, s=50, alpha=0.85, edgecolors="#ffffff", linewidth=0.5)

            if len(sub_yt) > 2 and np.var(sub_yt) > 1e-6:
                coeffs = np.polyfit(sub_yt, sub_yp, deg=1)
                poly_x = np.linspace(0, max_val, 100)
                poly_y = np.polyval(coeffs, poly_x)
                ax.plot(poly_x, poly_y, color=ACCENT_CYAN, linestyle="-", linewidth=1.8, label=f"Fit: {coeffs[0]:.2f}x + {coeffs[1]:.2f}")

            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_xlabel(f"Observed {pol_name} (µg/m³)", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"Predicted {pol_name} (µg/m³)", fontsize=10, fontweight="bold")
            ax.set_title(f"{title_label}\n[MAE: {sub_m['mae']:.2f} | R²: {sub_m['r2']:.3f} | N={len(sub_yt)}]",
                         fontsize=11, fontweight="bold", color=ACCENT_BLUE)
            ax.grid(True, alpha=0.35)
            ax.legend(loc="lower right", facecolor="#090d16", edgecolor=GRID_COLOR, fontsize=8)

        plt.suptitle(f"Provider-Partitioned Parity Comparison ({pol_name})", fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.98)
        plt.tight_layout()

        panel_path = self.output_dir / f"provider_parity_scatter_{pol}.png"
        plt.savefig(panel_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Multi-Panel Provider Parity Plot -> %s", panel_path)
        outputs["panel"] = panel_path
        return outputs


