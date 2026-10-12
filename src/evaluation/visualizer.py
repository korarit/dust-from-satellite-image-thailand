"""
Phase 5: Real-time Multi-Resolution Nowcast Map Generator & Visualizer
Thailand Dust Prediction Satellite (Multi-Resolution PM2.5 Nowcasting)

Implements:
1. Multi-Resolution Nowcast Map Production:
   - Discretizes model predictions across grid cells into Thailand PCD 5-tier AQI colors.
2. Deliverable 1: Apache Parquet Dataset:
   - Stores (grid_id, lat, lon, utm_x, utm_y, pm25_pred, aqi_category, aqi_color, aqi_label_th, aqi_label_en, timestamp)
3. Deliverable 2: High-Resolution GeoTIFF Raster:
   - Exports raster with georeferencing metadata (EPSG:32647 UTM 47N or EPSG:4326 WGS84) via rasterio.
4. Deliverable 3: Standalone Interactive Web Map (HTML + Leaflet):
   - Interactive zooming, panning, color-coded cells, popup inspection, and floating PCD AQI legend.
5. Deliverable 4: Publication-Quality Static PNG Map:
   - High-DPI Matplotlib visualization with Thailand coordinate frame, AQI colorbar, and distribution stats.
6. Strict Zero-Mock: 100% deterministic physical rendering, zero random calls.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    import rasterio
    from rasterio.transform import from_origin, from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from src.config.paths import MAPS_DIR, REPORTS_DIR
from src.evaluation.metrics import (
    assign_aqi_category,
    get_aqi_metadata,
    THAILAND_AQI_THRESHOLDS,
    AQI_COLOR_HEX,
    AQI_CATEGORY_NAMES_EN,
    AQI_CATEGORY_NAMES_TH,
)

logger = logging.getLogger(__name__)


class DustMapVisualizer:
    """
    Produces multi-format deliverables for Thailand PM2.5 Nowcasting maps:
    Parquet, GeoTIFF, Interactive Leaflet HTML, and Static PNG.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or MAPS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def attach_aqi_presentation_fields(
        self,
        df: pd.DataFrame,
        pred_col: str = "pm25_pred",
        timestamp_utc: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Enriches prediction DataFrame with PCD category indices, hex colors,
        and bilingual descriptions.
        """
        result = df.copy()
        if pred_col not in result.columns:
            raise KeyError(f"Prediction column '{pred_col}' not found in DataFrame.")

        preds = result[pred_col].values
        cats = assign_aqi_category(preds)
        result["aqi_category"] = cats

        colors = [AQI_COLOR_HEX[c] for c in cats]
        labels_en = [AQI_CATEGORY_NAMES_EN[c] for c in cats]
        labels_th = [AQI_CATEGORY_NAMES_TH[c] for c in cats]

        result["aqi_color"] = colors
        result["aqi_label_en"] = labels_en
        result["aqi_label_th"] = labels_th

        # Standardize lat/lon coordinates
        if "lat" not in result.columns and "centroid_lat" in result.columns:
            result["lat"] = result["centroid_lat"]
        if "lon" not in result.columns and "centroid_lon" in result.columns:
            result["lon"] = result["centroid_lon"]

        # If still missing lat/lon but utm_x/utm_y exist, convert from UTM 47N to WGS84
        if ("lat" not in result.columns or "lon" not in result.columns) and ("utm_x" in result.columns and "utm_y" in result.columns):
            try:
                import pyproj
                transformer = pyproj.Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
                lons, lats = transformer.transform(result["utm_x"].values, result["utm_y"].values)
                result["lat"] = lats
                result["lon"] = lons
            except Exception as e:
                logger.warning("Could not convert UTM to Lat/Lon: %s", e)

        if timestamp_utc is not None and "timestamp_utc" not in result.columns:
            result["timestamp_utc"] = str(timestamp_utc)

        return result

    def export_parquet(
        self,
        df: pd.DataFrame,
        filename: str = "nowcast_pm25_map.parquet",
    ) -> Path:
        """Exports nowcast map table to Apache Parquet."""
        out_path = self.output_dir / filename
        df.to_parquet(out_path, index=False)
        logger.info("Saved Nowcast Map Parquet -> %s (%d cells)", out_path, len(df))
        return out_path

    def export_geotiff(
        self,
        df: pd.DataFrame,
        filename: str = "nowcast_pm25_map.tif",
        pred_col: str = "pm25_pred",
        resolution_m: float = 2000.0,
        crs: str = "EPSG:32647",
    ) -> Optional[Path]:
        """
        Rasters the grid predictions and exports a georeferenced single-band GeoTIFF.
        Uses UTM Zone 47N (EPSG:32647) coordinates if present, otherwise WGS84 (EPSG:4326).
        """
        if not HAS_RASTERIO:
            logger.warning("rasterio is not installed. Skipping GeoTIFF export.")
            return None

        out_path = self.output_dir / filename

        # Prefer UTM coordinates for regular metric grid
        if "utm_x" in df.columns and "utm_y" in df.columns:
            x_vals = df["utm_x"].values
            y_vals = df["utm_y"].values
            used_crs = crs
            res = float(resolution_m)
        elif "lon" in df.columns and "lat" in df.columns:
            x_vals = df["lon"].values
            y_vals = df["lat"].values
            used_crs = "EPSG:4326"
            res = float(resolution_m) / 111000.0  # Approx degrees
        else:
            logger.error("Neither UTM nor Lat/Lon coordinates found in DataFrame for GeoTIFF.")
            return None

        min_x, max_x = float(np.min(x_vals)), float(np.max(x_vals))
        min_y, max_y = float(np.min(y_vals)), float(np.max(y_vals))

        width = max(1, int(np.round((max_x - min_x) / res)) + 1)
        height = max(1, int(np.round((max_y - min_y) / res)) + 1)

        raster = np.full((height, width), -9999.0, dtype=np.float32)

        # Place grid values into raster using rounded indexing to prevent floating truncations
        col_indices = np.clip(np.round((x_vals - min_x) / res).astype(int), 0, width - 1)
        row_indices = np.clip(np.round((max_y - y_vals) / res).astype(int), 0, height - 1)
        vals = df[pred_col].values.astype(np.float32)

        for r, c, val in zip(row_indices, col_indices, vals):
            raster[r, c] = val

        # from_origin expects upper-left pixel corner (min_x - res/2, max_y + res/2) rather than centroid
        transform = from_origin(min_x - res / 2.0, max_y + res / 2.0, res, res)

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=used_crs,
            transform=transform,
            nodata=-9999.0,
            compress="deflate",
        ) as dst:
            dst.write(raster, 1)

        logger.info("Saved Nowcast GeoTIFF -> %s (Size: %dx%d, Res: %.1f m, CRS: %s)",
                    out_path, width, height, resolution_m, used_crs)
        return out_path

    def export_interactive_html(
        self,
        df: pd.DataFrame,
        filename: str = "nowcast_pm25_map.html",
        pred_col: str = "pm25_pred",
        title: str = "Thailand Real-time PM2.5 Nowcast Map",
    ) -> Path:
        """
        Generates a standalone, self-contained interactive Leaflet web map.
        Requires zero local JS dependencies (uses Leaflet via standard CDN).
        """
        out_path = self.output_dir / filename

        plot_df = df.copy()
        if "lat" not in plot_df.columns and "centroid_lat" in plot_df.columns:
            plot_df["lat"] = plot_df["centroid_lat"]
        if "lon" not in plot_df.columns and "centroid_lon" in plot_df.columns:
            plot_df["lon"] = plot_df["centroid_lon"]
        if ("lat" not in plot_df.columns or "lon" not in plot_df.columns) and ("utm_x" in plot_df.columns and "utm_y" in plot_df.columns):
            try:
                import pyproj
                transformer = pyproj.Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
                lons, lats = transformer.transform(plot_df["utm_x"].values, plot_df["utm_y"].values)
                plot_df["lat"] = lats
                plot_df["lon"] = lons
            except Exception as e:
                logger.warning("Could not convert UTM to Lat/Lon: %s", e)

        # Ensure lat and lon are present
        if "lat" not in plot_df.columns or "lon" not in plot_df.columns:
            logger.error("Latitude and Longitude required for HTML map.")
            return out_path

        # Subsample if dataset is extraordinarily large for responsive browser DOM
        if len(plot_df) > 5000:
            plot_df = plot_df.iloc[::max(1, len(plot_df) // 3000)].copy()

        # Center point
        mean_lat = float(plot_df["lat"].mean()) if not plot_df.empty else 13.75
        mean_lon = float(plot_df["lon"].mean()) if not plot_df.empty else 100.5

        # Format GeoJSON-like points
        markers = []
        for _, row in plot_df.iterrows():
            pm_val = round(float(row.get(pred_col, 0.0)), 1)
            cat_idx = int(row.get("aqi_category", 0))
            meta = get_aqi_metadata(cat_idx)
            markers.append({
                "lat": round(float(row["lat"]), 5),
                "lon": round(float(row["lon"]), 5),
                "pm": pm_val,
                "color": meta["color_hex"],
                "tier_en": meta["label_en"],
                "tier_th": meta["label_th"],
                "gid": str(row.get("grid_id", "N/A")),
                "elev": round(float(row.get("elevation_m", 0.0)), 1) if "elevation_m" in row else None,
            })

        markers_json = json.dumps(markers)

        # Statistics for header cards
        all_pm = df[pred_col].dropna().values
        min_pm = round(float(np.min(all_pm)), 1) if len(all_pm) > 0 else 0.0
        max_pm = round(float(np.max(all_pm)), 1) if len(all_pm) > 0 else 0.0
        mean_pm = round(float(np.mean(all_pm)), 1) if len(all_pm) > 0 else 0.0
        high_risk_count = int(np.sum(all_pm >= 37.6)) if len(all_pm) > 0 else 0

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{
            margin: 0; padding: 0; width: 100%; height: 100%;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #111827; color: #f3f4f6;
        }}
        #map {{ width: 100%; height: 100%; }}
        .header-panel {{
            position: absolute; top: 15px; left: 60px; z-index: 1000;
            background: rgba(17, 24, 39, 0.92); backdrop-filter: blur(8px);
            padding: 14px 20px; border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);
            max-width: 480px;
        }}
        .header-panel h1 {{
            margin: 0 0 6px 0; font-size: 17px; font-weight: 700; color: #38bdf8;
            display: flex; align-items: center; gap: 8px;
        }}
        .header-panel p {{ margin: 0 0 10px 0; font-size: 12px; color: #9ca3af; }}
        .kpi-row {{ display: flex; gap: 10px; margin-top: 6px; }}
        .kpi-card {{
            flex: 1; background: rgba(31, 41, 55, 0.8); padding: 6px 10px;
            border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); text-align: center;
        }}
        .kpi-val {{ font-size: 15px; font-weight: 700; color: #f9fafb; }}
        .kpi-lbl {{ font-size: 10px; color: #9ca3af; text-transform: uppercase; }}
        .legend-panel {{
            position: absolute; bottom: 25px; right: 20px; z-index: 1000;
            background: rgba(17, 24, 39, 0.92); backdrop-filter: blur(8px);
            padding: 12px 16px; border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);
            font-size: 11px;
        }}
        .legend-title {{ font-weight: 700; margin-bottom: 8px; font-size: 12px; color: #e5e7eb; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.3); }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="header-panel">
        <h1>🛰️ Thailand PM2.5 Satellite Nowcast</h1>
        <p>Ensemble Quartet (Himawari-9 IR + HII Directional Pressure + DEM 30m)</p>
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-val">{mean_pm}</div><div class="kpi-lbl">Mean µg/m³</div></div>
            <div class="kpi-card"><div class="kpi-val">{min_pm} - {max_pm}</div><div class="kpi-lbl">Min - Max</div></div>
            <div class="kpi-card"><div class="kpi-val" style="color: {'#ef4444' if high_risk_count > 0 else '#10b981'};">{high_risk_count}</div><div class="kpi-lbl">Alert Cells (≥37.6)</div></div>
        </div>
    </div>
    <div class="legend-panel">
        <div class="legend-title">PCD Thailand PM2.5 Standards</div>
        <div class="legend-item"><div class="legend-color" style="background:#00b0f0;"></div><span>0 - 15.0 µg/m³ (ดีมาก / Very Good)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#92d050;"></div><span>15.1 - 25.0 µg/m³ (ดี / Good)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#ffff00;"></div><span>25.1 - 37.5 µg/m³ (ปานกลาง / Moderate)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#ff7c00;"></div><span>37.6 - 75.0 µg/m³ (เริ่มมีผลกระทบ / Orange)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#ff0000;"></div><span>> 75.0 µg/m³ (มีผลกระทบ / Hazardous)</span></div>
    </div>
    <script>
        const map = L.map('map', {{ zoomControl: false }}).setView([{mean_lat}, {mean_lon}], 6);
        L.control.zoom({{ position: 'topright' }}).addTo(map);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; CartoDB & OpenStreetMap',
            maxZoom: 16
        }}).addTo(map);

        const markers = {markers_json};
        markers.forEach(m => {{
            const circle = L.circleMarker([m.lat, m.lon], {{
                radius: 7,
                fillColor: m.color,
                color: '#ffffff',
                weight: 0.8,
                opacity: 0.9,
                fillOpacity: 0.85
            }}).addTo(map);

            let popupContent = `
                <div style="font-family: sans-serif; font-size: 12px; color: #111;">
                    <div style="font-weight: bold; font-size: 14px; margin-bottom: 4px;">
                        PM2.5: <span style="color: ${{m.color === '#ffff00' ? '#ca8a04' : m.color}};">${{m.pm}} µg/m³</span>
                    </div>
                    <div><b>Status:</b> ${{m.tier_th}} (${{m.tier_en}})</div>
                    <div><b>Grid ID:</b> ${{m.gid}}</div>
                    <div><b>Coords:</b> ${{m.lat.toFixed(4)}}, ${{m.lon.toFixed(4)}}</div>
                    ${{m.elev !== null ? `<div><b>Elevation:</b> ${{m.elev}} m</div>` : ''}}
                </div>
            `;
            circle.bindPopup(popupContent);
        }});
    </script>
</body>
</html>"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_template)

        logger.info("Saved Interactive Nowcast Map HTML -> %s", out_path)
        return out_path

    def export_static_png(
        self,
        df: pd.DataFrame,
        filename: str = "nowcast_pm25_map.png",
        pred_col: str = "pm25_pred",
        title: str = "Thailand Multi-Resolution PM2.5 Nowcast",
    ) -> Path:
        """
        Generates publication-quality static PNG visualization of PM2.5 across Thailand.
        """
        out_path = self.output_dir / filename

        plot_df = df.copy()
        if "lat" not in plot_df.columns and "centroid_lat" in plot_df.columns:
            plot_df["lat"] = plot_df["centroid_lat"]
        if "lon" not in plot_df.columns and "centroid_lon" in plot_df.columns:
            plot_df["lon"] = plot_df["centroid_lon"]
        if ("lat" not in plot_df.columns or "lon" not in plot_df.columns) and ("utm_x" in plot_df.columns and "utm_y" in plot_df.columns):
            try:
                import pyproj
                transformer = pyproj.Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
                lons, lats = transformer.transform(plot_df["utm_x"].values, plot_df["utm_y"].values)
                plot_df["lat"] = lats
                plot_df["lon"] = lons
            except Exception as e:
                logger.warning("Could not convert UTM to Lat/Lon: %s", e)

        if "lat" not in plot_df.columns or "lon" not in plot_df.columns:
            logger.error("Latitude and Longitude required for Static PNG map.")
            return out_path

        fig = plt.figure(figsize=(12, 10), facecolor="#0f172a")
        gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])

        ax_map = fig.add_subplot(gs[0, 0])
        ax_map.set_facecolor("#1e293b")

        # Discretized PCD colormap
        cmap = ListedColormap(AQI_COLOR_HEX)
        bounds = [0.0, 15.0, 25.0, 37.5, 75.0, 150.0]
        norm = BoundaryNorm(bounds, cmap.N)

        lats = plot_df["lat"].values
        lons = plot_df["lon"].values
        preds = np.clip(plot_df[pred_col].values, 0.0, 150.0)

        scatter = ax_map.scatter(
            lons, lats,
            c=preds,
            cmap=cmap,
            norm=norm,
            s=22,
            edgecolors="none",
            alpha=0.9
        )

        ax_map.set_xlim(97.0, 106.0)
        ax_map.set_ylim(5.5, 20.5)
        ax_map.set_title(title, fontsize=14, fontweight="bold", color="#38bdf8", pad=12)
        ax_map.set_xlabel("Longitude (°E)", fontsize=10, color="#94a3b8")
        ax_map.set_ylabel("Latitude (°N)", fontsize=10, color="#94a3b8")
        ax_map.tick_params(colors="#94a3b8")
        for spine in ax_map.spines.values():
            spine.set_color("#334155")

        # Subplot: Distribution Histogram & AQI Tier Counts
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_dist.set_facecolor("#1e293b")

        cats = assign_aqi_category(plot_df[pred_col].values)
        cat_counts = [int(np.sum(cats == i)) for i in range(5)]

        bars = ax_dist.barh(
            range(5),
            cat_counts,
            color=AQI_COLOR_HEX,
            edgecolor="#ffffff",
            linewidth=0.5
        )
        ax_dist.set_yticks(range(5))
        ax_dist.set_yticklabels(AQI_CATEGORY_NAMES_EN, fontsize=9, color="#e2e8f0")
        ax_dist.invert_yaxis()
        ax_dist.set_title("Grid Cell Distribution", fontsize=12, fontweight="bold", color="#38bdf8", pad=12)
        ax_dist.set_xlabel("Number of Grid Cells", fontsize=10, color="#94a3b8")
        ax_dist.tick_params(colors="#94a3b8")
        for spine in ax_dist.spines.values():
            spine.set_color("#334155")

        for b, count in zip(bars, cat_counts):
            width = b.get_width()
            ax_dist.text(width + max(1, 0.02 * max(cat_counts, default=1)), b.get_y() + b.get_height() / 2,
                         f"{count}", va="center", ha="left", color="#f8fafc", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Saved Publication PNG Map -> %s", out_path)
        return out_path

    def generate_all_deliverables(
        self,
        grid_predictions_df: pd.DataFrame,
        base_filename_stem: str = "nowcast_pm25_map",
        pred_col: str = "pm25_pred",
        resolution_m: float = 2000.0,
        timestamp_utc: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Path]:
        """
        Executes complete deliverables suite:
        Parquet dataset, GeoTIFF raster, Interactive Leaflet HTML, and Static PNG.
        """
        enriched_df = self.attach_aqi_presentation_fields(
            grid_predictions_df, pred_col=pred_col, timestamp_utc=timestamp_utc
        )

        p_parquet = self.export_parquet(enriched_df, f"{base_filename_stem}.parquet")
        p_tif = self.export_geotiff(enriched_df, f"{base_filename_stem}.tif", pred_col=pred_col, resolution_m=resolution_m)
        p_html = self.export_interactive_html(enriched_df, f"{base_filename_stem}.html", pred_col=pred_col)
        p_png = self.export_static_png(enriched_df, f"{base_filename_stem}.png", pred_col=pred_col)

        return {
            "parquet": p_parquet,
            "geotiff": p_tif,
            "interactive_html": p_html,
            "static_png": p_png,
        }
