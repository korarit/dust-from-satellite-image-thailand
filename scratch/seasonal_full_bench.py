"""
Seasonal full-shot bench (scratch/dev only, does NOT touch the legacy full-year suite).

  S-train: S1 = months 1-6, S2 = months 7-12  (wider N for training)
  S-bench: S1 = months 1-3 (peak haze), S2 = months 7-9 (wet trough, 2026 data ends ~Sep)

Matrix (all at once, no gradual expansion):
  pollutant : pm25, pm10
  resolution: 3.0 / 2.0 / 1.0 / 0.5 km
  feature   : full (multimodal) / pure_satellite
  provider  : Air4Thai / AirGradient / Combined (+ cross-provider swap)
  season    : S1 / S2

Anti-overfit guards (reuse legacy mechanisms, frozen):
  - Single 2km Air4Thai tuned param set reused for every season/res/provider
    (ModelTrainer._tuned_override). NO per-season retuning.
  - GroupKFold by station_code (trainer.train_suite derives groups automatically).
  - Strict Out-of-Time: train 2024-2025 seasonal -> test 2026 same-season months.
  - Isolated checkpoints under outputs/<pol>/models/seasonal/ (never overwrite legacy).
  - Overfit gate per cell: weighted seasonal OOT-MAE must beat legacy baseline on the
    same months AND the (OOT-OOF) gap must not widen vs baseline.

Usage:
  python scratch/seasonal_full_bench.py
  python scratch/seasonal_full_bench.py --pollutants pm25 --cv 3 --per-model-checkpoint
  python scratch/seasonal_full_bench.py --dry-run   # wiring check only, no training
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import (
    FEATURES_DIR,
    TARGET_DIR,
    ensure_directories,
    get_models_dir,
    get_reports_dir,
)
from src.features import (
    FULL_MULTIMODAL_COLUMNS,
    NO_WEATHER_FEATURE_COLUMNS,
)
from src.features.grid_engine import MultiResolutionGridEngine
from src.models import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("SeasonalFullBench")

RESOLUTIONS = [3.0, 2.0, 1.0, 0.5]
FEATURE_SETS = {
    "full": list(FULL_MULTIMODAL_COLUMNS),
    "pure_satellite": list(NO_WEATHER_FEATURE_COLUMNS),
}
PROVIDERS = ["Air4Thai", "AirGradient"]

# S-train (wide, keeps N) vs S-bench (narrow peak/trough slices of 2026 test).
SEASONS = {
    "S1": {"train_months": [1, 2, 3, 4, 5, 6], "bench_months": [1, 2, 3]},
    "S2": {"train_months": [7, 8, 9, 10, 11, 12], "bench_months": [7, 8, 9]},
}


def _month_of(df: pd.DataFrame) -> pd.Series:
    for col in ("timestamp", "timestamp_utc"):
        if col in df.columns:
            return pd.to_datetime(df[col], utc=True, errors="coerce").dt.month
    raise KeyError("No timestamp/timestamp_utc column for seasonal split")


def _filter_months(df: pd.DataFrame, months: List[int]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[_month_of(df).isin(months)].copy().reset_index(drop=True)


def _load_unified(pol: str) -> pd.DataFrame:
    for cand in (
        FEATURES_DIR / f"unified_{pol}_matrix.parquet",
        PROJECT_ROOT / "data" / "features" / f"unified_{pol}_matrix.parquet",
    ):
        if cand.exists() and cand.stat().st_size > 0:
            logger.info("[%s] loading %s", pol, cand)
            return pd.read_parquet(cand)
    raise FileNotFoundError(f"Unified matrix for {pol} not found under {FEATURES_DIR}")


def _split_oot(df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for col in ("timestamp", "timestamp_utc"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            cut = pd.to_datetime(split_date, utc=True)
            return (
                df[ts < cut].copy().reset_index(drop=True),
                df[ts >= cut].copy().reset_index(drop=True),
            )
    return df.copy(), pd.DataFrame()


def _preflight_ground_truth(suffix: str) -> None:
    """Fail-fast data availability gate: seasonal training needs real month coverage."""
    try:
        tr = pd.read_parquet(TARGET_DIR / f"ground_truth_train{suffix}_2024_2025.parquet")
        te = pd.read_parquet(TARGET_DIR / f"ground_truth_test{suffix}_2026.parquet")
    except Exception as exc:
        raise FileNotFoundError(
            "Ground-truth train/test parquet missing. Run Phase 1 full ingestion first "
            "(python run_pipeline.py --phase 1 --full). "
            f"Original error: {exc}"
        )
    tr_m = pd.to_datetime(tr["timestamp_utc"], utc=True, errors="coerce").dt.month.value_counts().sort_index() if not tr.empty else pd.Series(dtype=int)
    te_m = pd.to_datetime(te["timestamp_utc"], utc=True, errors="coerce").dt.month.value_counts().sort_index() if not te.empty else pd.Series(dtype=int)
    logger.warning("GROUND-TRUTH COVERAGE train months=%s test months=%s", dict(tr_m), dict(te_m))
    missing_train_s1 = [m for m in (1, 2, 3, 4, 5, 6) if m not in set(tr_m.index)]
    missing_train_s2 = [m for m in (7, 8, 9, 10, 11, 12) if m not in set(tr_m.index)]
    if missing_train_s1 or missing_train_s2:
        raise RuntimeError(
            "BLOCKED: train ground-truth has no seasonal coverage. "
            f"Missing S1 train months={missing_train_s1}, missing S2 train months={missing_train_s2}. "
            f"Got train={dict(tr_m)}, test={dict(te_m)}. "
            "Action: run Phase 1 full ingestion across 2024-2025 "
            "(python run_pipeline.py --phase 1 --full --start-date 2024-01-01 --end-date 2026-09-30) "
            "to populate satellite_stations + ground_truth before seasonal training."
        )


def run_pollutant(
    pol: str,
    split_date: str,
    cv: int,
    per_model_checkpoint: bool,
    dry_run: bool,
) -> pd.DataFrame:
    tgt = "observed_pm10" if pol == "pm10" else "observed_pm25"
    _preflight_ground_truth("" if pol == "pm25" else f"_{pol}")
    df = _load_unified(pol)
    if tgt not in df.columns or df[tgt].isna().all():
        raise ValueError(f"[{pol}] target column {tgt} missing/all-NaN in unified matrix")

    train_all, test_all = _split_oot(df, split_date)
    logger.info("[%s] OOT split: train=%d test=%d", pol, len(train_all), len(test_all))
    del df
    gc.collect()

    res_engine = MultiResolutionGridEngine()

    rows: List[Dict] = []
    for season_tag, spec in SEASONS.items():
        tr_season = _filter_months(train_all, spec["train_months"])
        te_bench = _filter_months(test_all, spec["bench_months"])
        n_groups = (
            tr_season["station_code"].nunique()
            if "station_code" in tr_season.columns else tr_season["grid_id"].nunique()
            if "grid_id" in tr_season.columns else 0
        )
        logger.info(
            "[%s %s] train_months=%s N=%d (stations/cells=%s) | bench_months=%s N_test=%d",
            pol, season_tag, spec["train_months"], len(tr_season), n_groups,
            spec["bench_months"], len(te_bench),
        )
        if dry_run:
            rows.append({
                "pollutant": pol, "season": season_tag,
                "n_train": len(tr_season), "n_test_bench": len(te_bench),
                "n_groups": int(n_groups), "status": "dry-run",
            })
            continue
        if len(tr_season) < 10 or len(te_bench) < 4:
            logger.warning("[%s %s] too few samples, skipping", pol, season_tag)
            rows.append({
                "pollutant": pol, "season": season_tag,
                "n_train": len(tr_season), "n_test_bench": len(te_bench),
                "status": "skipped-sparse",
            })
            continue

        train_by_res = {
            r: res_engine.adapt_feature_matrix_for_resolution(tr_season, r) for r in RESOLUTIONS
        }
        test_by_res = {
            r: res_engine.adapt_feature_matrix_for_resolution(te_bench, r) for r in RESOLUTIONS
        }

        # Per-season trainer: isolated checkpoints + tagged reports (never
        # overwrites legacy full-year files).
        trainer = ModelTrainer(
            output_dir=get_models_dir(pol) / "seasonal" / season_tag,
            pollutant=pol,
            target_col=tgt,
        )
        trainer.output_dir.mkdir(parents=True, exist_ok=True)
        prov_res = trainer.run_provider_partitioned_training(
            train_matrices_by_res=train_by_res,
            test_matrices_by_res=test_by_res,
            providers=list(PROVIDERS),
            feature_sets=dict(FEATURE_SETS),
            include_combined=True,
            cross_provider_eval=True,
            save_checkpoints=True,
            cv=cv,
            per_model_checkpoint=per_model_checkpoint,
            report_tag=season_tag,
        )
        for rec in prov_res.get("summary_table", []):
            rows.append({
                "pollutant": pol, "season": season_tag,
                "train_months": "-".join(map(str, spec["train_months"])),
                "bench_months": "-".join(map(str, spec["bench_months"])),
                "n_train": len(tr_season), "n_test_bench": len(te_bench),
                **rec,
            })
        del train_by_res, test_by_res, prov_res
        gc.collect()

    out_dir = get_reports_dir(pol)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / f"seasonal_full_bench_{pol}.csv", index=False)
    with open(out_dir / f"seasonal_full_bench_{pol}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info("[%s] wrote %d rows -> %s", pol, len(out_df), out_dir)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Seasonal full-shot bench (scratch, legacy-safe)")
    ap.add_argument("--pollutants", default="pm25,pm10")
    ap.add_argument("--split-date", default="2026-01-01")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--per-model-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-per-model-checkpoint", dest="per_model_checkpoint", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_directories()
    all_rows = []
    for pol in [p.strip().lower() for p in args.pollutants.split(",") if p.strip()]:
        all_rows.append(run_pollutant(
            pol,
            split_date=args.split_date,
            cv=args.cv,
            per_model_checkpoint=args.per_model_checkpoint,
            dry_run=args.dry_run,
        ))
    if len(all_rows) > 1:
        combo = pd.concat(all_rows, ignore_index=True)
        logger.info("TOTAL seasonal cells: %d", len(combo))
    logger.info("Done. Legacy suite untouched; seasonal checkpoints under models/<pol>/seasonal/")


if __name__ == "__main__":
    main()
