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

from src.config.paths import EVALUATION_DIR, REPORTS_DIR, get_evaluation_dir
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

        return {
            "parity_scatter": p_parity,
            "residual_analysis": p_res,
            "aqi_confusion_matrix": p_cm,
            "multi_resolution_benchmark": p_multi,
            "regional_generalization": p_reg,
            "feature_importance_consensus": p_feat,
            "ensemble_weights_pie": p_w,
        }

