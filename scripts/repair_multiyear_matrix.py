"""
Repair script for the multi-year unified matrix (data fix, NOT a schema change).

Background: the multi-year enrichment path never ran DirectionalWeatherEngine,
merged HII telemetry with mismatched station codes (HII internal codes like
ABRT/ACRU vs OPENAQ_* matrix codes -> 0% match), skipped tpi_index/relief_m,
and fit the imputer on the whole frame. Result: humidity locked at 70.0,
pressure_grad_mag constant, p_north/south/east/west identical copies,
imputed_flag == 1 everywhere.

What this script does (in order):
  0. Provenance audit of the existing pressure_mean_surround (HII agreement +
     elevation correlation) -> report only, never trusted blindly.
  1. HII station mapping by COORDINATES (haversine nearest within
     --max-match-km). Code systems are disjoint, so name matching is hopeless.
  2. tz-naive + floor-hour normalization on BOTH sides before the merge
     (the old `if "timestamp" in df` branch never fired).
  3. Fail-loud merge match-rate gate (default 0.8 of matchable rows).
  4. tpi_index / relief_m from master-grid terrain around each station
     (the grid file itself has no TPI/relief columns).
  5. Split-aware re-imputation: fit PhysicsFallbackImputer on train only,
     transform everything (old code fit on the whole frame = leak).
  6. Audit report (JSON + md): before/after per-column stats, match rates,
     mapping table, HII coverage.

What it deliberately does NOT do:
  - No p_north/south/east/west fabrication (7 HII stations cannot drive
    directional quadrants nationwide; that needs gridded wind, separate job).
  - No schema/column drops (elimination is Step B, decided on OOT numbers).

Usage (Colab):
    python scripts/repair_multiyear_matrix.py \\
        --matrix /content/drive/MyDrive/dust_project/data/features/unified_pm25_matrix.parquet \\
        --hii-dir /content/drive/MyDrive/dust_project/data/hii \\
        --master-grid /content/drive/MyDrive/dust_project/data/geo/thailand_2km_master_grid.parquet \\
        --out /content/drive/MyDrive/dust_project/data/features/unified_pm25_matrix_repaired.parquet \\
        --out-dir /content/drive/MyDrive/dust_project/data/features/repair_audit
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_matrix_builder import PhysicsFallbackImputer  # noqa: E402

EARTH_R_KM = 6371.0
PRESSURE_CANDIDATES = ["pressure", "press", "value", "val"]
HUMIDITY_CANDIDATES = ["humidity", "humid", "rh", "value", "val"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair the multi-year unified matrix.")
    p.add_argument("--matrix", required=True, help="Input unified matrix parquet.")
    p.add_argument("--hii-dir", required=True, help="Directory with hii_pressure_*.parquet / hii_humidity_*.parquet.")
    p.add_argument("--master-grid", default=None,
                   help="Master grid parquet with centroid_lat/centroid_lon/elevation_m (for TPI/relief). Skipped if absent.")
    p.add_argument("--out", required=True, help="Output repaired matrix parquet.")
    p.add_argument("--out-dir", default="repair_audit", help="Directory for audit JSON/md.")
    p.add_argument("--split-date", default="2026-01-01",
                   help="Train/test cutoff; imputer is fit on train only.")
    p.add_argument("--max-match-km", type=float, default=50.0,
                   help="Max HII->station mapping distance (default 50). Beyond -> NaN/imputed.")
    p.add_argument("--min-match-rate", type=float, default=0.8,
                   help="Fail if matched rows / matchable rows < this (default 0.8).")
    p.add_argument("--station-key", default=None,
                   help="Matrix station id column (default: station_code else location_id else grid_id).")
    return p.parse_args()


def norm_hour(s: pd.Series) -> pd.Series:
    """tz-naive + floor-hour, applied identically to both merge sides."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t.dt.tz_convert(None).dt.floor("h")


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def pick_station_key(df: pd.DataFrame, override=None) -> str:
    if override and override in df.columns:
        return override
    for c in ("station_code", "location_id", "grid_id"):
        if c in df.columns:
            return c
    raise ValueError("No station id column (station_code/location_id/grid_id) in matrix.")


def pick_ts_col(df: pd.DataFrame) -> str:
    for c in ("timestamp_utc", "timestamp"):
        if c in df.columns:
            return c
    raise ValueError("No timestamp column in matrix.")


def pick_latlon(df: pd.DataFrame):
    lat = "latitude" if "latitude" in df.columns else ("lat" if "lat" in df.columns else None)
    lon = "longitude" if "longitude" in df.columns else ("lon" if "lon" in df.columns else None)
    if lat is None or lon is None:
        raise ValueError("No lat/lon columns in dataframe.")
    return lat, lon


def load_hii_series(hii_dir: Path, kind: str):
    """Returns (values df [hii_station, ts_hour, val], stations df, ts_min, ts_max)."""
    files = sorted(hii_dir.glob(f"hii_{kind}_*.parquet"))
    if not files:
        return None, None, None, None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cands = PRESSURE_CANDIDATES if kind == "pressure" else HUMIDITY_CANDIDATES
    vcol = next((c for c in cands if c in df.columns), None)
    if vcol is None or "station_code" not in df.columns:
        raise ValueError(f"HII {kind} files lack value/station_code cols: {list(df.columns)}")
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    lat_c = "latitude" if "latitude" in df.columns else ("lat" if "lat" in df.columns else None)
    lon_c = "longitude" if "longitude" in df.columns else ("lon" if "lon" in df.columns else None)
    stations = None
    if lat_c and lon_c:
        stations = df.groupby("station_code")[[lat_c, lon_c]].median().reset_index()
        stations.columns = ["hii_station", "hii_lat", "hii_lon"]
    df = df[["station_code", ts_col, vcol]].copy()
    df["ts_hour"] = norm_hour(df[ts_col])
    df = df.dropna(subset=["ts_hour"]).drop_duplicates(["station_code", "ts_hour"])
    vals = df[["station_code", "ts_hour", vcol]].rename(
        columns={"station_code": "hii_station", vcol: "hii_value"})
    if stations is not None and not vals.empty:
        # Dead stations (coords known but zero valid hourly rows) must not
        # receive matrix stations in the mapping step.
        live = set(vals["hii_station"].astype(str).unique().tolist())
        stations = stations[stations["hii_station"].astype(str).isin(live)].copy()
    return vals, stations, vals["ts_hour"].min(), vals["ts_hour"].max()


def build_station_mapping(stn_coords: pd.DataFrame, skey: str,
                          mlat_c: str, mlon_c: str,
                          hii_stations, max_km: float) -> pd.DataFrame:
    """Nearest-HII-station map per matrix station (coordinate based).

    Only live HII stations (with >= 1 valid hourly row) are candidates;
    the caller passes one variable's station table at a time.
    """
    mapping = []
    if hii_stations is not None and len(hii_stations):
        m_lat = stn_coords[mlat_c].to_numpy(dtype=float)
        m_lon = stn_coords[mlon_c].to_numpy(dtype=float)
        for _, hrow in hii_stations.iterrows():
            d = haversine_km(m_lat, m_lon, float(hrow["hii_lat"]), float(hrow["hii_lon"]))
            j = int(np.argmin(d))
            mapping.append({
                "matrix_station": str(stn_coords[skey].iloc[j]),
                "hii_station": str(hrow["hii_station"]),
                "distance_km": round(float(d[j]), 3),
            })
    map_df = pd.DataFrame(mapping)
    if not map_df.empty:
        map_df = map_df[map_df["distance_km"] <= max_km].copy()
    return map_df


def col_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    return {
        "nan_rate": round(float(s.isna().mean()), 6),
        "nunique": int(s.nunique()),
        "mean": round(float(s.mean()), 6) if s.notna().any() else None,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = Path(args.matrix)
    mat = pd.read_parquet(matrix_path)
    print(f"Matrix: {matrix_path} | rows={len(mat)}")
    skey = pick_station_key(mat, args.station_key)
    ts_col = pick_ts_col(mat)
    mlat_c, mlon_c = pick_latlon(mat)
    print(f"  station_key={skey} ts_col={ts_col} latlon=({mlat_c},{mlon_c})")

    audit: dict = {
        "matrix": str(matrix_path),
        "rows": int(len(mat)),
        "station_key": skey,
        "split_date": args.split_date,
        "max_match_km": args.max_match_km,
    }

    # ---- Step 0: provenance of the existing pressure column ----
    prov = {}
    if "pressure_mean_surround" in mat.columns:
        prov["before"] = col_stats(mat["pressure_mean_surround"])
        if "elevation_m" in mat.columns:
            pe = pd.to_numeric(mat["pressure_mean_surround"], errors="coerce")
            ev = pd.to_numeric(mat["elevation_m"], errors="coerce")
            if float(pe.std()) > 0 and float(ev.std()) > 0:
                r = pe.corr(ev)
                prov["corr_with_elevation"] = round(float(r), 4) if pd.notna(r) else None
            else:
                prov["corr_with_elevation"] = None
                prov["corr_note"] = "constant column(s); correlation undefined"
    audit["provenance_pressure_mean_surround"] = prov

    # ---- Step 1: coordinate mapping HII -> matrix stations ----
    hii_dir = Path(args.hii_dir)
    p_vals, p_stn, p_min, p_max = load_hii_series(hii_dir, "pressure")
    h_vals, h_stn, h_min, h_max = load_hii_series(hii_dir, "humidity")
    audit["hii"] = {
        "pressure_rows": int(len(p_vals)) if p_vals is not None else 0,
        "humidity_rows": int(len(h_vals)) if h_vals is not None else 0,
        "pressure_stations": int(p_stn["hii_station"].nunique()) if p_stn is not None else 0,
        "humidity_stations": int(h_stn["hii_station"].nunique()) if h_stn is not None else 0,
    }

    mat["_ts_hour"] = norm_hour(mat[ts_col])
    stn_coords = mat.groupby(skey)[[mlat_c, mlon_c]].median().reset_index()
    # Per-variable mapping: a station live for pressure may be dead for
    # humidity and vice versa.
    map_p = build_station_mapping(stn_coords, skey, mlat_c, mlon_c, p_stn, args.max_match_km)
    map_h = build_station_mapping(stn_coords, skey, mlat_c, mlon_c, h_stn, args.max_match_km)
    audit["mapping"] = {
        "matrix_stations": int(stn_coords[skey].nunique()),
        "mapped_pressure_within_km": int(map_p["matrix_station"].nunique()) if not map_p.empty else 0,
        "mapped_humidity_within_km": int(map_h["matrix_station"].nunique()) if not map_h.empty else 0,
    }
    print(f"  HII->matrix mapped (pressure): {audit['mapping']['mapped_pressure_within_km']}/"
          f"{audit['mapping']['matrix_stations']} within {args.max_match_km} km")
    print(f"  HII->matrix mapped (humidity): {audit['mapping']['mapped_humidity_within_km']}/"
          f"{audit['mapping']['matrix_stations']} within {args.max_match_km} km")

    # ---- Step 2: merge (overwrite matched cells with real HII values) ----
    before_p = col_stats(mat["pressure_mean_surround"]) if "pressure_mean_surround" in mat.columns else None
    before_h = col_stats(mat["humidity_mean_surround"]) if "humidity_mean_surround" in mat.columns else None
    n_over_p = n_over_h = 0
    match_info = {}
    matched_idx: dict = {}
    maps = {"pressure": map_p, "humidity": map_h}
    for kind, vals, col in (("pressure", p_vals, "pressure_mean_surround"),
                            ("humidity", h_vals, "humidity_mean_surround")):
        map_df = maps[kind]
        if vals is None:
            match_info[kind] = {"status": "no_hii_files"}
            continue
        if map_df.empty:
            match_info[kind] = {"status": "no stations within max_match_km; HII merge skipped"}
            print(f"  WARNING: HII {kind}: no stations within max_match_km; merge skipped.")
            continue
        mat["_hii_station"] = mat[skey].astype(str).map(
            dict(zip(map_df["matrix_station"], map_df["hii_station"])))
        in_radius = mat["_hii_station"].notna()
        # matchable = in radius AND timestamp inside HII coverage
        cov = in_radius & mat["_ts_hour"].notna() & \
            (mat["_ts_hour"] >= vals["ts_hour"].min()) & (mat["_ts_hour"] <= vals["ts_hour"].max())
        key = mat.loc[cov, ["_hii_station", "_ts_hour"]].copy()
        key["_row"] = np.arange(len(mat))[cov.to_numpy()]
        merged = key.merge(vals, left_on=["_hii_station", "_ts_hour"],
                           right_on=["hii_station", "ts_hour"], how="left")
        hit = merged["hii_value"].notna()
        match_rate = float(hit.mean()) if len(merged) else 0.0
        match_info[kind] = {
            "matchable_rows": int(len(merged)),
            "matched_rows": int(hit.sum()),
            "match_rate": round(match_rate, 4),
        }
        print(f"  HII {kind}: matched {hit.sum()}/{len(merged)} (rate={match_rate:.3f})")
        if match_rate < args.min_match_rate:
            # Miss-pattern forensics to stdout + partial audit file, then fail.
            rep = {"variable": kind, "match_rate": round(match_rate, 4),
                   "threshold": args.min_match_rate,
                   "matchable_rows": int(len(merged)),
                   "matched_rows": int(hit.sum())}
            try:
                m_all = merged.copy()
                m_all["_hit"] = hit.to_numpy()
                m_all["_month"] = m_all["_ts_hour"].dt.to_period("M").astype(str)
                by_stn = (m_all.groupby("_hii_station")["_hit"]
                          .agg(["mean", "size"]).sort_values("mean"))
                rep["worst_stations"] = [
                    {"hii_station": str(i), "match_rate": round(float(r["mean"]), 3),
                     "matchable": int(r["size"])}
                    for i, r in by_stn.head(20).iterrows()]
                by_mo = (m_all.groupby("_month")["_hit"]
                         .agg(["mean", "size"]).sort_values("_month"))
                rep["by_month"] = [
                    {"month": str(i), "match_rate": round(float(r["mean"]), 3),
                     "matchable": int(r["size"])}
                    for i, r in by_mo.iterrows()]
            except Exception as exc:
                rep["breakdown_error"] = str(exc)
            audit["match"] = match_info
            audit["gate_failure"] = rep
            audit["status"] = "failed_gate"
            fail_path = out_dir / "repair_audit_failed.json"
            with open(fail_path, "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2, default=str)
            print(f"  worst stations (lowest match): {rep.get('worst_stations', [])[:5]}")
            print(f"  by month: {rep.get('by_month', [])}")
            print(f"  partial audit -> {fail_path}")
            raise SystemExit(
                f"FAIL-LOUD: HII {kind} match rate {match_rate:.3f} < "
                f"--min-match-rate {args.min_match_rate}. Refusing to write a "
                f"half-repaired matrix. Inspect keys/coverage and retry.")
        if col not in mat.columns:
            mat[col] = np.nan
        idx = merged.loc[hit, "_row"].to_numpy()
        vals_to_write = merged.loc[hit, "hii_value"].to_numpy(dtype=np.float32)
        mat.loc[mat.index[idx], col] = vals_to_write
        if kind == "pressure":
            n_over_p = int(hit.sum())
        else:
            n_over_h = int(hit.sum())
        if col not in matched_idx:
            matched_idx[col] = set()
        matched_idx[col].update(idx.tolist())
    mat = mat.drop(columns=["_hii_station"], errors="ignore")
    # Unmatched cells keep STALE values (never NaN), so the imputer cannot see
    # them. NaN them out explicitly -> train-median imputation + imputed_flag=1
    # (honest "no observation" instead of silently trusted junk). Only applies
    # where a merge actually ran (vals present); without HII files at all we
    # leave existing values untouched.
    n_nanned = {}
    for col, idx_set in matched_idx.items():
        if col not in mat.columns:
            continue
        idx_arr = np.array(sorted(idx_set), dtype=np.int64)
        keep = np.zeros(len(mat), dtype=bool)
        keep[idx_arr[idx_arr < len(mat)]] = True
        n_new_nan = int((~keep).sum())
        mat.loc[~keep, col] = np.nan
        n_nanned[col] = n_new_nan
        print(f"  {col}: {n_new_nan} unmatched cells set to NaN for re-imputation.")
    audit["match"] = match_info
    for kind in ("pressure", "humidity"):
        info = match_info.get(kind, {})
        if isinstance(info, dict) and "matched_rows" in info:
            info["coverage_of_all_rows"] = round(info["matched_rows"] / max(len(mat), 1), 4)
    total_over = (match_info.get("pressure", {}).get("matched_rows", 0)
                  if isinstance(match_info.get("pressure"), dict) else 0)
    print(f"  overall HII pressure coverage: {total_over}/{len(mat)} rows "
          f"({total_over / max(len(mat), 1):.3f})")
    audit["overwritten_cells"] = {"pressure_mean_surround": n_over_p,
                                   "humidity_mean_surround": n_over_h}
    audit["naned_for_reimpute"] = n_nanned
    if before_p:
        audit["pressure_mean_surround_before"] = before_p
        audit["pressure_mean_surround_after"] = col_stats(mat["pressure_mean_surround"])
    if before_h:
        audit["humidity_mean_surround_before"] = before_h
        audit["humidity_mean_surround_after"] = col_stats(mat["humidity_mean_surround"])

    # ---- Step 4: tpi_index / relief_m from master-grid terrain ----
    topo_note = "skipped (no --master-grid)"
    if args.master_grid and Path(args.master_grid).exists():
        try:
            from scipy.spatial import cKDTree
            from src.features.grid_engine import MultiResolutionGridEngine
            grid = pd.read_parquet(args.master_grid)
            need = {"centroid_lat", "centroid_lon", "elevation_m"}
            if need.issubset(grid.columns) and "elevation_m" in mat.columns:
                eng = MultiResolutionGridEngine()
                gx, gy = eng.to_utm.transform(
                    grid["centroid_lon"].to_numpy(dtype=float),
                    grid["centroid_lat"].to_numpy(dtype=float))
                gtree = cKDTree(np.column_stack([gx / 1000.0, gy / 1000.0]))
                uniq = mat[[mlat_c, mlon_c]].copy()
                uniq["_ux"], uniq["_uy"] = eng.to_utm.transform(
                    uniq[mlon_c].to_numpy(dtype=float), uniq[mlat_c].to_numpy(dtype=float))
                ucoords = np.column_stack([uniq["_ux"].to_numpy() / 1000.0,
                                           uniq["_uy"].to_numpy() / 1000.0])
                gelev = grid["elevation_m"].to_numpy(dtype=float)
                # TPI radius 6km / relief radius 2km (same as grid engine defaults).
                # Neighbor stats via unique-coordinate lookup (stations share cells).
                nbr6 = gtree.query_ball_point(ucoords, r=6.0)
                nbr2 = gtree.query_ball_point(ucoords, r=2.0)
                # Vectorized equivalent: station elev minus mean grid elev in 6km.
                selev = pd.to_numeric(
                    mat["elevation_m"], errors="coerce").to_numpy(dtype=float)
                # per-row via unique-coordinate lookup table
                ukey = list(zip(uniq[mlat_c].to_numpy(), uniq[mlon_c].to_numpy()))
                lut_mean6, lut_rel2 = {}, {}
                for i, k in enumerate(ukey):
                    if k not in lut_mean6:
                        n6, n2 = nbr6[i], nbr2[i]
                        lut_mean6[k] = float(np.nanmean(gelev[n6])) if len(n6) else np.nan
                        lut_rel2[k] = (float(np.nanmax(gelev[n2]) - np.nanmin(gelev[n2]))
                                       if len(n2) > 1 else 0.0)
                mean6 = np.array([lut_mean6[k] for k in
                                  zip(mat[mlat_c].to_numpy(), mat[mlon_c].to_numpy())],
                                 dtype=float)
                relv = np.array([lut_rel2[k] for k in
                                 zip(mat[mlat_c].to_numpy(), mat[mlon_c].to_numpy())],
                                dtype=float)
                mat["tpi_index"] = np.where(np.isnan(mean6) | np.isnan(selev), 0.0,
                                            (selev - mean6)).astype(np.float32)
                mat["relief_m"] = np.where(np.isnan(relv), 0.0, relv).astype(np.float32)
                topo_note = (f"computed from {args.master_grid} "
                             f"(TPI r=6km, relief r=2km)")
            else:
                topo_note = "skipped (grid lacks centroid_lat/centroid_lon/elevation_m)"
        except Exception as exc:
            topo_note = f"skipped (error: {exc})"
    audit["topo"] = topo_note
    print(f"  topo: {topo_note}")

    # ---- Step 5: split-aware re-imputation ----
    split_dt = pd.to_datetime(args.split_date, utc=True)
    ts_for_split = pd.to_datetime(mat["_ts_hour"], utc=True)
    train_mask = ts_for_split < split_dt
    imp = PhysicsFallbackImputer()
    if train_mask.any():
        imp.fit(mat[train_mask])
        audit["imputer_fit_rows"] = int(train_mask.sum())
    else:
        imp.fit(mat)
        audit["imputer_fit_rows"] = int(len(mat))
        audit["imputer_fit_note"] = "no rows before split-date; fit on all rows"
    mat = imp.transform(mat)
    audit["imputer_fitted_cols"] = len(imp.fitted_stats_)

    # ---- Step 6: write outputs ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drop_tmp = [c for c in ("_ts_hour", "_hii_station") if c in mat.columns]
    mat = mat.drop(columns=drop_tmp)
    mat.to_parquet(out_path, index=False)

    watched = ["pressure_mean_surround", "humidity_mean_surround", "tpi_index",
               "relief_m", "imputed_flag", "pressure_grad_mag",
               "p_north", "bt_b13", "elevation_m"]
    audit["columns_after"] = {c: col_stats(mat[c]) for c in watched if c in mat.columns}

    audit_path = out_dir / "repair_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    md = [
        "# Matrix Repair Audit",
        "",
        f"- Input: `{matrix_path}` ({audit['rows']} rows)",
        f"- Output: `{out_path}`",
        f"- Split date (imputer fit on train only): `{args.split_date}` "
        f"(fit rows: {audit['imputer_fit_rows']})",
        f"- HII: {audit['hii']['pressure_stations']} pressure stations "
        f"({audit['hii']['pressure_rows']} rows), "
        f"{audit['hii']['humidity_stations']} humidity stations "
        f"({audit['hii']['humidity_rows']} rows)",
        f"- Mapping pressure: {audit['mapping']['mapped_pressure_within_km']}/"
        f"{audit['mapping']['matrix_stations']} matrix stations within "
        f"{args.max_match_km} km",
        f"- Mapping humidity: {audit['mapping']['mapped_humidity_within_km']}/"
        f"{audit['mapping']['matrix_stations']} matrix stations within "
        f"{args.max_match_km} km",
        f"- Overwritten cells: pressure={n_over_p}, humidity={n_over_h}",
        f"- NaNed for re-impute (stale values, imputed_flag=1): {n_nanned}",
        f"- Topo: {topo_note}",
        "",
        "## Match rates (fail-loud gate)",
        "",
        "```json",
        json.dumps(match_info, indent=2),
        "```",
        "",
        "## Column stats after repair",
        "",
        "| Column | NaN rate | nunique | mean |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for c, s in audit["columns_after"].items():
        md.append(f"| `{c}` | {s['nan_rate']} | {s['nunique']} | {s['mean']} |")
    for _tag, _map in (("pressure", map_p), ("humidity", map_h)):
        if not _map.empty:
            md += ["", f"## Station mapping (matrix -> HII, {_tag})",
                   "", "| Matrix station | HII station | km |",
                   "| :--- | :--- | :---: |"]
            for _, r in _map.iterrows():
                md.append(f"| `{r['matrix_station']}` | `{r['hii_station']}` | {r['distance_km']} |")
    md += ["",
           "## Still dead by design (see Step-B elimination, NOT fabricated here)",
           "",
           "- `p_north/south/east/west`, `delta_p_*`, `pressure_grad_mag`, "
           "`humidity_std/grad`: sparse HII stations cannot drive directional "
           "quadrants nationwide; left untouched for schema elimination.",
           ""]
    md_path = out_dir / "repair_audit.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote:\n  {out_path}\n  {audit_path}\n  {md_path}")


if __name__ == "__main__":
    main()
