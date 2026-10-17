"""
Correlation clustering for greedy feature elimination (Step A-lite).

Groups model features whose absolute Pearson correlation >= --threshold
(union-find over pairs), so that elimination decisions in Step B are made
at the *cluster* level instead of dropping correlated partners one by one.

Usage (Colab, real matrix):
    python scripts/correlation_clustering.py \\
        --matrix /content/data/features/unified_pm25_matrix.parquet \\
        --features full --threshold 0.9 --max-rows 100000 \\
        --importance /content/outputs/pm25/evaluation/csv/cross_model_attribution_pm25.csv \\
        --out-dir /content/outputs/pm25/evaluation/csv

Outputs in --out-dir:
    correlation_clusters_<tag>.csv   (one row per cluster: members, keeper, drops)
    correlation_features_<tag>.csv   (one row per feature: cluster, keeper flag,
                                      max |r| inside cluster, importance)
    correlation_clusters_<tag>.md    (human-readable report)

Step B consumes the clusters CSV: drop a whole low-importance cluster at
once, or keep a single keeper per high-importance cluster.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_matrix_builder import (
    ALL_FEATURE_COLUMNS,
    NO_WEATHER_FEATURE_COLUMNS,
    FULL_MULTIMODAL_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlation clustering (Step A-lite).")
    p.add_argument("--matrix", required=True, help="Unified feature matrix parquet.")
    p.add_argument(
        "--features",
        default="full",
        choices=["full", "pure_satellite", "all_physical"],
        help="full: FULL_MULTIMODAL_COLUMNS (default). "
        "pure_satellite: NO_WEATHER_FEATURE_COLUMNS. "
        "all_physical: ALL_FEATURE_COLUMNS.",
    )
    p.add_argument("--threshold", type=float, default=0.9,
                   help="|r| >= threshold joins a cluster (default 0.9).")
    p.add_argument("--max-rows", type=int, default=100000,
                   help="Random subsample cap for corr computation (default 100000).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--importance", default=None,
                   help="Optional CSV with columns [feature, ensemble_importance].")
    p.add_argument("--out-dir", default=".",
                   help="Directory for output CSV/md reports.")
    p.add_argument("--tag", default=None,
                   help="Filename tag (default: derived from --features).")
    return p.parse_args()


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_importance(path: str) -> dict:
    imp = pd.read_csv(path)
    if "feature" not in imp.columns:
        raise ValueError(f"Importance CSV {path} needs a 'feature' column.")
    val_col = (
        "ensemble_importance" if "ensemble_importance" in imp.columns
        else "importance" if "importance" in imp.columns
        else None
    )
    if val_col is None:
        raise ValueError(f"Importance CSV {path} needs 'ensemble_importance' or 'importance'.")
    return dict(zip(imp["feature"].astype(str), imp[val_col].astype(float)))


def main() -> None:
    args = parse_args()
    tag = args.tag or args.features
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = {
        "full": list(FULL_MULTIMODAL_COLUMNS),
        "pure_satellite": list(NO_WEATHER_FEATURE_COLUMNS),
        "all_physical": list(ALL_FEATURE_COLUMNS),
    }[args.features]

    df = pd.read_parquet(args.matrix)
    cols = [c for c in schema if c in df.columns]
    missing = [c for c in schema if c not in df.columns]
    print(f"Matrix rows: {len(df)} | schema features: {len(schema)} "
          f"| present: {len(cols)} | missing: {len(missing)}")
    if missing:
        print(f"  missing cols (skipped): {missing}")
    if not cols:
        raise ValueError("No schema feature columns found in matrix.")

    X = df[cols].copy()
    if len(X) > args.max_rows:
        X = X.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"  subsampled to {len(X)} rows (seed={args.seed}) for corr.")

    numeric = X.select_dtypes(include=[np.number])
    const_cols = [c for c in numeric.columns if float(numeric[c].std()) == 0.0]
    work = numeric.drop(columns=const_cols)

    corr = work.corr(method="pearson", min_periods=10).abs().fillna(0.0)

    uf = UnionFind(list(work.columns))
    feats = list(work.columns)
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            if float(corr.iloc[i, j]) >= args.threshold:
                uf.union(feats[i], feats[j])

    groups: dict = {}
    for f in feats:
        groups.setdefault(uf.find(f), []).append(f)
    # Deterministic order: biggest cluster first, then by member name.
    clusters = sorted(groups.values(), key=lambda m: (-len(m), m[0]))

    importance = load_importance(args.importance) if args.importance else {}

    feat_rows = []
    clus_rows = []
    for cid, members in enumerate(clusters, start=1):
        sub = corr.loc[members, members]
        # Keeper: highest ensemble importance if available, else medoid
        # (member with the highest mean |r| to the rest of the cluster).
        if importance:
            keeper = max(members, key=lambda m: (float(importance.get(m, 0.0)), m))
        else:
            mean_r = sub.mean(axis=1)
            keeper = str(mean_r.idxmax())
        for m in members:
            others = [o for o in members if o != m]
            max_r = float(sub.loc[m, others].max()) if others else 0.0
            feat_rows.append({
                "feature": m,
                "cluster_id": cid,
                "cluster_size": len(members),
                "is_keeper": m == keeper,
                "max_abs_corr_in_cluster": round(max_r, 4),
                "ensemble_importance": round(float(importance.get(m, 0.0)), 6)
                if importance else "",
            })
        drops = sorted([m for m in members if m != keeper])
        clus_rows.append({
            "cluster_id": cid,
            "cluster_size": len(members),
            "keeper": keeper,
            "drop_candidates": ";".join(drops),
            "members": ";".join(sorted(members)),
        })

    feat_df = pd.DataFrame(feat_rows)
    clus_df = pd.DataFrame(clus_rows)

    feat_csv = out_dir / f"correlation_features_{tag}.csv"
    clus_csv = out_dir / f"correlation_clusters_{tag}.csv"
    feat_df.to_csv(feat_csv, index=False)
    clus_df.to_csv(clus_csv, index=False)

    n_single = int((feat_df["cluster_size"] == 1).sum())
    n_multi = len(clus_df) - int((clus_df["cluster_size"] == 1).sum())
    n_drops = int((~feat_df["is_keeper"]).sum())

    md = [
        f"# Correlation Clustering Report ({tag})",
        "",
        f"- Matrix: `{args.matrix}` (rows used: {len(X)})",
        f"- Threshold: |r| >= {args.threshold} | features clustered: {len(feats)} "
        f"| constant cols (auto-drop): {const_cols if const_cols else 'none'}",
        f"- Importance source: `{args.importance}`" if args.importance else "- Importance source: none (keeper = cluster medoid)",
        f"- Clusters: {len(clus_df)} ({n_single} singletons, {n_multi} multi-member) | "
        f"drop candidates: {n_drops}",
        "",
        "## Multi-member clusters (elimination order: lowest keeper importance first)",
        "",
        "| Cluster | Size | Keeper | Drop candidates |",
        "| :---: | :---: | :--- | :--- |",
    ]
    for _, r in clus_df[clus_df["cluster_size"] > 1].iterrows():
        md.append(f"| {int(r['cluster_id'])} | {int(r['cluster_size'])} | `{r['keeper']}` | `{r['drop_candidates']}` |")
    if n_multi == 0:
        md.append("_No multi-member clusters at this threshold._")
    md += [
        "",
        "## Constant (zero-variance) columns — drop outright, no retrain needed",
        "",
        ("`" + "`, `".join(const_cols) + "`") if const_cols else "_None._",
        "",
        "## Suggested Step-B order",
        "",
        "1. Drop constant columns outright.",
        "2. For each multi-member cluster (largest first): keep the keeper, "
        "drop the rest together in ONE retrain (they carry the same signal).",
        "3. Then greedy single-feature elimination on remaining low-importance "
        "singletons, gated on OOT R².",
    ]
    md_path = out_dir / f"correlation_clusters_{tag}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Clusters: {len(clus_df)} ({n_multi} multi-member) | "
          f"drop candidates: {n_drops} | constant: {len(const_cols)}")
    print(f"Wrote:\n  {feat_csv}\n  {clus_csv}\n  {md_path}")


if __name__ == "__main__":
    main()
