"""
SHAP Attribution Engine (Phase 5 Extension)
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Computes Shapley Additive exPlanations (SHAP) for the Stacking Ensemble:
- Per-model TreeExplainer SHAP values (LightGBM, CatBoost, XGBoost, HistGBR,
  Random Forest, Extra Trees) on a deterministic stride subsample
- Ensemble-consensus SHAP matrix: S_ensemble = sum_i w_i * S_model_i
  (valid because SHAP is additive w.r.t. the model output and the blend is a
  non-negative linear combination of base-model predictions)
- Directional consensus (signed mean SHAP) unlike split-gain importance
- Artifacts: CSV table + horizontal bar plot + beeswarm summary plot

Strict Zero-Mock: deterministic stride sampling (no np.random), all values
derived exclusively from fitted real observations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

logger = logging.getLogger(__name__)

# Dark Theme Palette (mirrors src.evaluation.plotter)
BG_COLOR = "#0f172a"
PANEL_COLOR = "#1e293b"
GRID_COLOR = "#334155"
TEXT_COLOR = "#f8fafc"
MUTED_TEXT = "#94a3b8"
ACCENT_BLUE = "#38bdf8"

# Tree-based learners supported by shap.TreeExplainer.
TREE_EXPLAINER_KEYS = (
    "lightgbm",
    "catboost",
    "xgboost",
    "hist_gbr",
    "random_forest",
    "extra_trees",
)

# Sole directional coding: positive mean SHAP -> pushes PM up, negative -> down.
DIVERGING_POS = "#f59e0b"
DIVERGING_NEG = "#38bdf8"


def _apply_dark_style() -> None:
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
        "axes.grid": False,
        "savefig.facecolor": BG_COLOR,
    })


class ShapAttributionEngine:
    """
    Computes ensemble-consensus SHAP attributions for a fitted
    ``DustStackingEnsemble`` (duck-typed: requires ``base_models``, ``weights``,
    ``feature_names_``, ``is_fitted_``).
    """

    def __init__(
        self,
        ensemble: Any,
        output_dir: Optional[Union[str, Path]] = None,
        pollutant: str = "pm25",
        max_samples: int = 5000,
        forest_max_samples: int = 1500,
    ) -> None:
        self.ensemble = ensemble
        self.pollutant = pollutant.lower()
        self.max_samples = int(max_samples)
        self.forest_max_samples = int(forest_max_samples)
        self.output_dir = Path(output_dir) if output_dir is not None else None

        # Populated by compute():
        self.consensus_shap_: Optional[np.ndarray] = None      # (n, features)
        self.X_sample_: Optional[pd.DataFrame] = None          # sampled inputs
        self.contributed_models_: List[str] = []
        self.attribution_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ utils
    def _stride_subsample(self, X: pd.DataFrame, cap: int) -> pd.DataFrame:
        """Deterministic even-stride subsample (zero randomness)."""
        if len(X) <= cap:
            return X.copy()
        indices = np.linspace(0, len(X) - 1, cap).astype(np.int64)
        return X.iloc[indices].reset_index(drop=True)

    def _prepare_matrix(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        f_names = list(getattr(self.ensemble, "feature_names_", None) or feature_cols)
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            names = f_names or [f"f{i}" for i in range(np.asarray(X).shape[1])]
            df = pd.DataFrame(np.asarray(X), columns=names)
        if f_names:
            missing = [c for c in f_names if c not in df.columns]
            if missing:
                raise ValueError(f"SHAP input missing %d feature columns: {missing[:5]}")
            df = df[f_names]
        return df

    # ---------------------------------------------------------------- compute
    def compute(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Runs per-model TreeExplainer SHAP on the deterministic subsample and
        returns the ensemble-consensus attribution table sorted by |SHAP| share.
        """
        if not getattr(self.ensemble, "is_fitted_", False):
            raise RuntimeError(f"{self.ensemble.__class__.__name__} must be fitted before SHAP.")

        feat_df = self._prepare_matrix(X, feature_cols)
        feature_names = list(feat_df.columns)

        weights: Dict[str, float] = getattr(self.ensemble, "weights", {})
        models: Dict[str, Any] = getattr(self.ensemble, "base_models", {})

        ensemble_values: Optional[np.ndarray] = None
        base_values: Optional[np.ndarray] = None
        per_model_abs: Dict[str, np.ndarray] = {}
        contributed: List[str] = []

        for m_key in TREE_EXPLAINER_KEYS:
            w = float(weights.get(m_key, 0.0))
            model = models.get(m_key)
            if w <= 0.0 or model is None:
                continue
            if not getattr(model, "is_fitted_", False) or getattr(model, "estimator", None) is None:
                logger.warning("SHAP: skip %s (not fitted or missing inner estimator).", m_key)
                continue

            cap = self.forest_max_samples if m_key in ("random_forest", "extra_trees") else self.max_samples
            X_sample = self._stride_subsample(feat_df, cap)
            logger.info("SHAP: computing TreeExplainer SHAP for %-18s (w=%.3f, n=%d rows)...",
                        m_key, w, len(X_sample))

            try:
                explainer = shap.TreeExplainer(model.estimator, model_output="raw")
            except Exception as exc:  # noqa: BLE001 - per-model isolation
                logger.warning("SHAP: TreeExplainer init failed for %s: %s", m_key, exc)
                continue

            try:
                sv = explainer.shap_values(X_sample, check_additivity=False)
            except TypeError:
                sv = explainer.shap_values(X_sample)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SHAP: shap_values failed for %s: %s", m_key, exc)
                continue

            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv, dtype=np.float64)
            if sv.ndim == 3:  # some explainers return (n, features, 1)
                sv = sv[:, :, 0]
            if sv.shape != (len(X_sample), len(feature_names)):
                logger.warning("SHAP: unexpected shape %s for %s; skipping.", sv.shape, m_key)
                continue

            ev = explainer.expected_value
            ev_arr = np.full(len(X_sample), float(np.ravel(ev)[0]), dtype=np.float64)

            w_sv = sv * w
            w_ev = ev_arr * w
            ensemble_values = w_sv if ensemble_values is None else ensemble_values + w_sv
            base_values = w_ev if base_values is None else base_values + w_ev
            per_model_abs[m_key] = np.mean(np.abs(sv), axis=0)
            contributed.append(m_key)

        if ensemble_values is None:
            raise RuntimeError(
                "SHAP: no tree-based contributed model available (all weights zero, "
                "not fitted, or explainer init failed)."
            )

        self.consensus_shap_ = ensemble_values
        self.X_sample_ = self._stride_subsample(feat_df, self.max_samples)
        self.contributed_models_ = contributed

        mean_abs = np.mean(np.abs(ensemble_values), axis=0)
        mean_signed = np.mean(ensemble_values, axis=0)
        total = float(mean_abs.sum())
        share = mean_abs / total if total > 1e-12 else np.zeros_like(mean_abs)

        rows: List[Dict[str, Any]] = []
        for i, fn in enumerate(feature_names):
            row: Dict[str, Any] = {
                "feature": fn,
                "mean_abs_shap": float(mean_abs[i]),
                "mean_signed_shap": float(mean_signed[i]),
                "shap_share": float(share[i]),
            }
            for m_key in contributed:
                row[f"{m_key}_mean_abs_shap"] = float(per_model_abs[m_key][i])
            rows.append(row)

        attrib = pd.DataFrame(rows).sort_values(
            by="mean_abs_shap", ascending=False
        ).reset_index(drop=True)
        attrib.insert(1, "rank", attrib.index + 1)
        self.attribution_ = attrib

        logger.info("SHAP: consensus computed over %d models | Top 3: %s",
                    len(contributed),
                    ", ".join(attrib["feature"].head(3).tolist()))
        return attrib

    # ----------------------------------------------------------------- export
    def plot_summary_bar(self, top_n: int = 20, filename: Optional[str] = None) -> Path:
        """Directional consensus bar: height = mean |SHAP|, color = dominant sign."""
        if self.attribution_ is None:
            raise RuntimeError("Call compute() before plotting.")
        _apply_dark_style()
        pol = self.pollutant
        fname = filename or f"shap_summary_bar_{pol}.png"
        out_path = (self.output_dir or Path("."))
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / fname

        top = self.attribution_.head(top_n).sort_values("mean_abs_shap", ascending=True)
        colors = [
            DIVERGING_POS if v >= 0.0 else DIVERGING_NEG
            for v in top["mean_signed_shap"].values
        ]

        fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
        ax.set_facecolor(PANEL_COLOR)
        bars = ax.barh(
            top["feature"], top["mean_abs_shap"],
            color=colors, edgecolor="#ffffff", linewidth=0.6, alpha=0.9,
        )
        ax.set_xlabel("Mean |SHAP| (µg/m³) - Ensemble-Weighted Consensus",
                      fontsize=11, fontweight="bold")
        ax.set_ylabel("Physical Predictive Feature", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Top {len(top)} SHAP Feature Attribution\n"
            f"[TreeExplainer Ensemble Consensus - Orange: raises PM2.5, Blue: lowers]",
            fontsize=13, fontweight="bold", color=ACCENT_BLUE, pad=12,
        )
        ax.grid(True, alpha=0.4)
        xmax = float(top["mean_abs_shap"].max()) or 1.0
        for b in bars:
            width = b.get_width()
            ax.text(width + xmax * 0.01, b.get_y() + b.get_height() / 2,
                    f"{width:.1f}", va="center", ha="left",
                    color=TEXT_COLOR, fontsize=9, fontweight="bold")
        ax.axvline(0.0, color=MUTED_TEXT, linewidth=0.8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved SHAP Summary Bar Plot -> %s", out_path)
        return out_path

    def plot_beeswarm(self, top_n: int = 20, filename: Optional[str] = None) -> Path:
        """Per-sample SHAP beeswarm of the ensemble-consensus value matrix."""
        if self.consensus_shap_ is None or self.X_sample_ is None:
            raise RuntimeError("Call compute() before plotting.")
        _apply_dark_style()
        pol = self.pollutant
        fname = filename or f"shap_beeswarm_{pol}.png"
        out_path = (self.output_dir or Path("."))
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / fname

        explanation = shap.Explanation(
            values=self.consensus_shap_,
            base_values=np.zeros(len(self.X_sample_)),
            data=self.X_sample_.values,
            feature_names=list(self.X_sample_.columns),
        )
        shap.plots.beeswarm(explanation, max_display=top_n, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.patch.set_facecolor(BG_COLOR)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved SHAP Beeswarm Plot -> %s", out_path)
        return out_path

    def export_csv(self, filename: Optional[str] = None) -> Path:
        """Writes the consensus attribution table as CSV."""
        if self.attribution_ is None:
            raise RuntimeError("Call compute() before exporting.")
        pol = self.pollutant
        fname = filename or f"shap_importance_{pol}.csv"
        out_path = (self.output_dir or Path("."))
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / fname
        self.attribution_.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info("Saved SHAP Attribution CSV -> %s", out_path)
        return out_path

    # ------------------------------------------------------------ entry point
    def compute_and_export(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_cols: Optional[List[str]] = None,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """
        One-call pipeline entry: compute -> CSV + summary bar + beeswarm.
        Returns a result dict with the attribution table and artifact paths.
        """
        attrib = self.compute(X, feature_cols=feature_cols)
        csv_path = self.export_csv()
        bar_path = self.plot_summary_bar(top_n=top_n)
        beeswarm_path: Optional[Path] = None
        try:
            beeswarm_path = self.plot_beeswarm(top_n=top_n)
        except Exception as exc:  # noqa: BLE001 - beeswarm is best-effort
            logger.warning("SHAP: beeswarm plot skipped (%s).", exc)

        result = {
            "attribution_df": attrib,
            "csv": csv_path,
            "summary_bar": bar_path,
            "beeswarm": beeswarm_path,
            "contributed_models": list(self.contributed_models_),
            "n_samples": int(len(self.X_sample_)) if self.X_sample_ is not None else 0,
        }
        logger.info("SHAP: export complete | contributed models: %s", result["contributed_models"])
        return result
