import io
import base64
from typing import List, Tuple, Dict, Any, Optional

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_leaflet as dl
import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import cv2
import json

parquet_file: str = "output.parquet"
df = pd.read_parquet(parquet_file)

points: List[Tuple[float, float, float]] = list(df.itertuples(index=False, name=None))

lat_center: float = sum(lat for lat, _, _ in points) / len(points)
lng_center: float = sum(lng for _, lng, _ in points) / len(points)

global_vmin: float = min(val for _, _, val in points)
global_vmax: float = max(val for _, _, val in points)

def value_to_color(v: float, cmap_name: str, vmin: float, vmax: float) -> str:
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(v))
    return mcolors.to_hex(rgba)

zoom_to_h3_profiles: Dict[str, Dict[range, int]] = {
    "low": {range(0, 7): 3, range(7, 9): 4, range(9, 11): 5, range(11, 13): 6, range(13, 15): 7, range(15, 17): 8, range(17, 19): 9},
    "medium": {range(0, 7): 4, range(7, 9): 5, range(9, 11): 6, range(11, 13): 7, range(13, 15): 8, range(15, 17): 9, range(17, 19): 10},
    "high": {range(0, 7): 5, range(7, 9): 6, range(9, 11): 7, range(11, 13): 8, range(13, 15): 9, range(15, 17): 10, range(17, 19): 11}
}

def zoom_to_resolution(zoom: int, profile: str = "low") -> int:
    profile_map = zoom_to_h3_profiles.get(profile, zoom_to_h3_profiles["low"])
    for zr, res in profile_map.items():
        if zoom in zr:
            return res
    return 2

all_resolutions: List[int] = sorted(set(res for profile in zoom_to_h3_profiles.values() for res in profile.values()))

precomputed_h3: Dict[int, Dict[str, List[float]]] = {}
for res in all_resolutions:
    cell_values: Dict[str, List[float]] = {}
    for lat, lng, val in points:
        cell = h3.latlng_to_cell(lat, lng, res)
        cell_values.setdefault(cell, []).append(val)
    precomputed_h3[res] = cell_values

def cmap_label(cmap_name: str, invert: bool = False) -> html.Div:
    if invert:
        cmap_name += "_r"
    cmap = plt.get_cmap(cmap_name)
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(2, 0.2))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return html.Div([html.Img(src="data:image/png;base64," + data, style={"width": "100%", "height": "20px"})])

def filter_points(points, kernel_size=3):
    if not points:
        return []

    lats = [lat for lat, _, _ in points]
    lngs = [lng for _, lng, _ in points]
    lat_min, lat_max = min(lats), max(lats)
    lng_min, lng_max = min(lngs), max(lngs)

    width = 500
    height = max(1, round(width * (lat_max - lat_min) / (lng_max - lng_min + 1e-12)))

    mask = np.zeros((height, width), dtype=np.uint8)
    for lat, lng, _ in points:
        x = int((lng - lng_min) / (lng_max - lng_min + 1e-12) * (width - 1))
        y = int((lat_max - lat) / (lat_max - lat_min + 1e-12) * (height - 1))
        mask[y, x] = 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    filtered = []
    for lat, lng, val in points:
        x = int((lng - lng_min) / (lng_max - lng_min + 1e-12) * (width - 1))
        y = int((lat_max - lat) / (lat_max - lat_min + 1e-12) * (height - 1))
        if opened[y, x] > 0:
            filtered.append((lat, lng, val))
    return filtered

app: dash.Dash = dash.Dash(__name__)
initial_cmap_names: List[str] = ["RdYlGn", "coolwarm", "viridis", "plasma"]

app.layout = html.Div([
    dl.Map(
        center=[lat_center, lng_center],
        zoom=7,
        style={"width": "100%", "height": "100vh"},
        id="map",
        children=[
            dl.TileLayer(id="basemap"),
            dl.LayerGroup(id="hex-layer")
        ]
    ),
    html.Div(id="color-legend", className="color-legend"),
    html.Div(id="layer-name", className="layer-name"),
    html.Div([
        html.Div([
            html.Div("Basemap", className="panel-subtitle", style={"width": "27%"}),
            html.Div(
                dcc.Dropdown(
                    id="basemap-dropdown",
                    options=[
                        {"label": "Default (OSM)", "value": "osm"},
                        {"label": "Satellite (ESRI)", "value": "sat"}
                    ],
                    value="osm",
                    clearable=False,
                    searchable=False,
                    style={
                        "border": "1px solid transparent",
                        "box-shadow": "none",
                        "font-family": "Roboto, sans-serif",
                        "font-size": "14px",
                        "width": "100%",
                        "text-align": "left"
                    }
                ),
                style={"width": "73%"}
            )
        ], className="panel-section", style={"display": "flex", "align-items": "center", "gap": "10px"}),

        html.Div([
            html.Div("Colormap", className="panel-subtitle", style={"width": "25%", "margin-top": "-5px"}),
            html.Div(
                dcc.Dropdown(
                    id="colormap-dropdown",
                    options=[{"label": cmap_label(name), "value": name} for name in initial_cmap_names],
                    value="RdYlGn",
                    clearable=False,
                    searchable=False,
                    style={
                        "border": "1px solid transparent",
                        "box-shadow": "none",
                        "font-family": "Roboto, sans-serif",
                        "font-size": "14px",
                        "width": "100%",
                        "text-align": "left"
                    }
                ),
                style={"width": "75%"}
            )
        ], className="panel-section", style={"display": "flex", "align-items": "center", "gap": "10px"}),

        html.Div(
            html.Button("Invert colormap", id="invert-btn", n_clicks=0, className="panel-btn full-btn"),
            style={"margin-top": "-10px"}
        ),

        html.Div([
            html.Div("Color by", className="panel-subtitle", style={"width": "27%"}),
            html.Div(
                dcc.Dropdown(
                    id="agg-method",
                    options=[
                        {"label": "Minimum", "value": "min"},
                        {"label": "Maximum", "value": "max"},
                        {"label": "Mean", "value": "mean"},
                        {"label": "Count", "value": "count"},
                    ],
                    value="min",
                    clearable=False,
                    style={
                        "border": "1px solid transparent",
                        "box-shadow": "none",
                        "font-family": "Roboto, sans-serif",
                        "font-size": "14px",
                        "width": "100%",
                        "text-align": "left"
                    }
                ),
                style={"width": "73%"}
            )
        ], className="panel-section",
            style={"display": "flex", "align-items": "center", "gap": "10px", "margin-bottom": "2px"}),

        html.Div([
            html.Div("Opacity", className="panel-subtitle", style={"width": "22%"}),
            html.Div(
                dcc.Slider(
                    id="opacity-slider",
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.7,
                    marks={0: "0", 0.5: "0.5", 1: "1"},
                    tooltip={"placement": "bottom", "always_visible": False},
                    className="slider-medium",
                ),
                style={"width": "78%", "max-width": "150px", "margin-top": "20px"}
            )
        ], className="panel-section",
            style={"display": "flex", "align-items": "center", "gap": "10px", "margin-bottom": "2px"}),

        html.Div([
            html.Div("Scale", className="panel-subtitle", style={"width": "27%"}),
            html.Div(
                dcc.Dropdown(
                    id="scale-mode",
                    options=[
                        {"label": "Fixed resolution", "value": "fixed"},
                        {"label": "Based on zoom", "value": "zoom"},
                    ],
                    value="zoom",
                    clearable=False,
                    style={
                        "border": "1px solid transparent",
                        "box-shadow": "none",
                        "font-family": "Roboto, sans-serif",
                        "font-size": "14px",
                        "width": "100%",
                        "text-align": "left"
                    }
                ),
                style={"width": "73%"}
            )
        ], className="panel-section",
            style={"display": "flex", "align-items": "center", "gap": "10px", "margin-bottom": "5px"}),

        html.Div([
            html.Div("Resolution", className="panel-subtitle", style={"text-align": "left", "width": "100%"}),
            html.Div(children=[
                html.Button("Low", id="btn-low", className="panel-btn"),
                html.Button("Medium", id="btn-medium", className="panel-btn"),
                html.Button("High", id="btn-high", className="panel-btn")
            ], className="btn-group"),
            dcc.Store(id="zoom-profile", data="low")
        ], id="zoom-profile-container", style={"margin-top": "10px", "gap":"1-px","margin-bottom": "10px"}),

        html.Div([
            html.Div("Resolution", className="panel-subtitle", style={"text-align": "left", "width": "100%"}),
            html.Div(
                dcc.Slider(
                    id="fixed-res",
                    min=min(all_resolutions),
                    max=max(all_resolutions),
                    step=1,
                    marks={res: str(res) for res in all_resolutions},
                    value=7,
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False
                ),
                style={"width": "100%"}
            )
        ], id="fixed-res-container", className="hidden",
            style={"width": "100%", "max-width": "100%", "box-sizing": "border-box", "gap":"10px","margin-bottom": "10px"}),

        html.Div([
            html.Div([
                html.Div("Range of values", className="panel-subtitle"),
                html.Div([
                    html.Div([
                        dcc.Input(
                            id="min-input",
                            type="number",
                            value=round(global_vmin, 2),
                            step=0.001,
                            className="input-small"
                        )
                    ], className="input-container"),
                    html.Div("-", style={"alignSelf": "center"}),
                    html.Div([
                        dcc.Input(
                            id="max-input",
                            type="number",
                            value=round(global_vmax, 2),
                            step=0.001,
                            className="input-small"
                        )
                    ], className="input-container")
                ], className="input-group",
                    style={"display": "flex", "align-items": "center", "margin-left": "15px", "gap": "5px",
                           "margin-top": "5px"})
            ], className="panel-section", style={"display": "flex", "align-items": "center", "margin-bottom": "5px"}),
            html.Div([
                dcc.RangeSlider(
                    id="threshold-slider",
                    min=round(global_vmin, 2),
                    max=round(global_vmax, 2),
                    step=0.001,
                    value=[round(global_vmin, 2), round(global_vmax, 2)],
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks={float(f"{v:.3f}"): f"{v:.2f}" for v in np.linspace(global_vmin, global_vmax, 5)}
                )
            ], className="range-slider-container", style={"margin-top": "10px"})
        ]),

        dcc.Store(id="download-counter", data=0),
        html.Div([
            html.Button(
                "Download Hex Layer (GeoJSON)",
                id="download-btn",
                className="panel-btn full-btn",
                style={"margin-top": "10px"}
            ),
            dcc.Download(id="download-geojson")
        ]),

        html.Div(className="separator"),

        html.Div([
            html.Span("►", id="morph-toggle-icon", style={"margin-right": "5px", "margin-bottom":"10px", "cursor": "pointer"}),
            html.Div("Filter points (MORPH_OPEN)", className="panel-subtitle panel-subtitle-bold")
        ], id="morph-toggle-header", style={"display": "flex", "align-items": "center", "margin-bottom": "-5px", "cursor": "pointer"}),
        html.Div([
            dcc.Checklist(
                id="filter-checkbox",
                options=[{"label": "Enable MORPH_OPEN", "value": "filter"}],
                value=[],
                className="checklist-small",
                style={"margin-bottom": "-2px", "font-weight": "bold", "font-family": "Roboto, sans-serif"}
            ),
            html.Div([
                html.Div([
                    html.Div("Kernel size:",
                             style={
                                 "font-weight": "bold",
                                 "font-family": "Roboto, sans-serif",
                                 "font-size": "14px",
                                 "margin-right": "8px",
                                 "display": "flex",
                                 "align-items": "center"
                             }
                             ),
                    dcc.Dropdown(
                        id="kernel-dropdown",
                        options=[{"label": f"{i}x{i}", "value": i} for i in [3, 5, 7]],
                        value=3,
                        clearable=False,
                        style={
                            "border": "1px solid transparent",
                            "box-shadow": "none",
                            "font-family": "Roboto, sans-serif",
                            "font-size": "14px",
                            "width": "75px"
                        }
                    )
                ], style={"display": "flex", "align-items": "center", "gap": "6px"}),
            ], style={"display": "flex", "flex-direction": "column", "width": "100%"})
        ], id="filter-options-container", style={"display": "none", "flex-direction": "column", "padding-left": "15px"})
    ], className="control-panel")
])



@app.callback(
    Output("colormap-dropdown", "options"),
    Input("invert-btn", "n_clicks")
)
def update_dropdown(n_clicks: int) -> List[Dict[str, Any]]:
    invert: bool = n_clicks % 2 == 1
    return [{"label": cmap_label(name, invert), "value": name} for name in initial_cmap_names]


@app.callback(
    Output("basemap", "url"),
    Input("basemap-dropdown", "value")
)
def update_basemap(basemap: str) -> str:
    if basemap == "sat":
        return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    return "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"


@app.callback(
    Output("zoom-profile-container", "style"),
    Output("fixed-res-container", "style"),
    Input("scale-mode", "value")
)
def toggle_resolution_containers(scale_mode: str):
    if scale_mode == "zoom":
        return {"display": "block", "margin-bottom": "10px"}, {"display": "none"}
    return {"display": "none"}, {"display": "block", "margin-bottom": "10px"}


@app.callback(
    Output("layer-name", "children"),
    [
        Input("agg-method", "value"),
        Input("scale-mode", "value"),
        Input("fixed-res", "value"),
        Input("zoom-profile", "data"),
        Input("map", "zoom")
    ]
)
def update_layer_name(agg_method, scale_mode, fixed_res, zoom_profile, zoom):
    zoom = zoom or 7
    show_res = fixed_res if scale_mode == "fixed" else zoom_to_resolution(zoom, zoom_profile)
    method_label = {"min": "Minimum", "max": "Maximum", "mean": "Mean", "count": "Count"}[agg_method]
    return f"Hexagon resolution: {show_res} (Aggregation method: {method_label})"

@app.callback(
    [Output("btn-low", "className"),
     Output("btn-medium", "className"),
     Output("btn-high", "className"),
     Output("zoom-profile", "data")],
    [Input("btn-low", "n_clicks"),
     Input("btn-medium", "n_clicks"),
     Input("btn-high", "n_clicks")],
    prevent_initial_call=False
)
def update_zoom_profile(n_low, n_medium, n_high):
    ctx = dash.callback_context
    classes = ["panel-btn", "panel-btn", "panel-btn"]
    profile = "low"

    if ctx.triggered:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "btn-low":
            classes[0] = "panel-btn active-btn"
            profile = "low"
        elif triggered_id == "btn-medium":
            classes[1] = "panel-btn active-btn"
            profile = "medium"
        elif triggered_id == "btn-high":
            classes[2] = "panel-btn active-btn"
            profile = "high"

    return classes + [profile]


@app.callback(
    Output("fixed-res", "value"),
    Input("scale-mode", "value"),
    Input("map", "zoom"),
    Input("zoom-profile", "data"),
    prevent_initial_call=True
)
def set_fixed_default(scale_mode: str, zoom: Optional[int], zoom_profile: str) -> int:
    zoom = zoom or 7
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("scale-mode") and scale_mode == "fixed":
        return zoom_to_resolution(zoom, zoom_profile)
    return dash.no_update


@app.callback(
    [Output("filter-options-container", "style"),
     Output("morph-toggle-icon", "children")],
    Input("morph-toggle-header", "n_clicks"),
    prevent_initial_call=True
)
def toggle_morph(n_clicks):
    if n_clicks is None:
        return dash.no_update, dash.no_update
    if n_clicks % 2 == 1:
        return {"display": "flex", "flex-direction": "column", "padding-left": "15px"}, "▼"
    return {"display": "none"}, "►"


@app.callback(
    [Output("threshold-slider", "value"),
     Output("min-input", "value"),
     Output("max-input", "value")],
    [Input("threshold-slider", "value"),
     Input("min-input", "value"),
     Input("max-input", "value")]
)
def sync_range(slider_range, min_val, max_val):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "threshold-slider":
        return slider_range, slider_range[0], slider_range[1]
    elif trigger_id in ["min-input", "max-input"]:
        if min_val is None or max_val is None:
            raise dash.exceptions.PreventUpdate
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        return [min_val, max_val], min_val, max_val
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("download-geojson", "data"),
    Output("download-counter", "data"),
    Input("download-btn", "n_clicks"),
    State("download-counter", "data"),
    State("map", "zoom"),
    State("agg-method", "value"),
    State("threshold-slider", "value"),
    State("scale-mode", "value"),
    State("fixed-res", "value"),
    State("zoom-profile", "data"),
    State("filter-checkbox", "value"),
    State("kernel-dropdown", "value"),
    prevent_initial_call=True
)
def download_hex_layer(n_clicks, counter, zoom, agg_method, threshold,
                       scale_mode, fixed_res, zoom_profile,
                       filter_value, kernel_size):
    counter += 1
    zoom = zoom or 7
    show_res = fixed_res if scale_mode == "fixed" else zoom_to_resolution(zoom, zoom_profile)

    if "filter" in filter_value:
        filtered_points = filter_points(points, kernel_size=kernel_size)
        cell_values = {}
        for lat, lng, val in filtered_points:
            cell = h3.latlng_to_cell(lat, lng, show_res)
            cell_values.setdefault(cell, []).append(val)
    else:
        cell_values = precomputed_h3.get(show_res, {})

    min_thresh, max_thresh = threshold
    display_values = {}
    for cell, vals in cell_values.items():
        filtered_vals = [v for v in vals if min_thresh <= v <= max_thresh]
        if not filtered_vals:
            continue
        if agg_method == "count":
            val = len(filtered_vals)
        else:
            val = {"min": np.min, "max": np.max, "mean": np.mean}[agg_method](filtered_vals)
        display_values[cell] = val

    features = []
    for cell, val in display_values.items():
        boundary = list(h3.cell_to_boundary(cell))
        coords = [[lng, lat] for lat, lng in boundary + [boundary[0]]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"value": val, "cell": cell}
        })

    geojson = {"type": "FeatureCollection", "features": features}

    filename = f"hex_layer_{counter}.geojson"
    return dcc.send_string(json.dumps(geojson), filename), counter


@app.callback(
    Output("hex-layer", "children"),
    [
        Input("map", "zoom"),
        Input("colormap-dropdown", "value"),
        Input("agg-method", "value"),
        Input("invert-btn", "n_clicks"),
        Input("threshold-slider", "value"),
        Input("scale-mode", "value"),
        Input("fixed-res", "value"),
        Input("zoom-profile", "data"),
        Input("filter-checkbox", "value"),
        Input("kernel-dropdown", "value"),
        Input("opacity-slider", "value")
    ]
)
def update_layer(
    zoom, cmap_name, agg_method, n_clicks, threshold, scale_mode,
    fixed_res, zoom_profile, filter_value, kernel_size, opacity
):
    zoom = zoom or 7
    show_res = fixed_res if scale_mode == "fixed" else zoom_to_resolution(zoom, zoom_profile)
    invert = n_clicks % 2 == 1
    cmap_to_use = cmap_name + "_r" if invert else cmap_name

    if "filter" in filter_value:
        filtered_points = filter_points(points, kernel_size=kernel_size)
        cell_values = {}
        for lat, lng, val in filtered_points:
            cell = h3.latlng_to_cell(lat, lng, show_res)
            cell_values.setdefault(cell, []).append(val)
    else:
        cell_values = precomputed_h3.get(show_res, {})

    min_thresh, max_thresh = threshold
    display_values = {}
    for cell, vals in cell_values.items():
        filtered_vals = [v for v in vals if min_thresh <= v <= max_thresh]
        if not filtered_vals:
            continue
        if agg_method == "count":
            val = len(filtered_vals)
        else:
            val = {"min": np.min, "max": np.max, "mean": np.mean}[agg_method](filtered_vals)
        display_values[cell] = val

    if display_values:
        if agg_method == "count":
            vmin, vmax = 0, max(display_values.values())
        else:
            vmin, vmax = min(display_values.values()), max(display_values.values())
        if vmin == vmax:
            vmax = vmin + 1
    else:
        vmin, vmax = 0, 1

    polygons = []
    for cell, val in display_values.items():
        fill_color = (mcolors.to_hex(plt.get_cmap(cmap_to_use)((val - vmin) / (vmax - vmin)))
                      if agg_method == "count" else value_to_color(val, cmap_to_use, vmin, vmax))
        boundary = list(h3.cell_to_boundary(cell))
        coords = [[lat, lng] for lat, lng in boundary + [boundary[0]]]
        popup_label = "Count: " if agg_method == "count" else "Value: "
        val_str = str(val) if agg_method == "count" else f"{val:.8f}"
        polygons.append(
            dl.Polygon(
                id=f"hex-{cell}-{agg_method}-{cmap_to_use}-{opacity}",
                positions=coords,
                color="black",
                weight=1,
                fillColor=fill_color,
                fillOpacity=opacity,
                children=[dl.Popup(html.Div([html.Strong(popup_label), val_str]))]
            )
        )
    return polygons


@app.callback(
    Output("color-legend", "children"),
    [
        Input("colormap-dropdown", "value"),
        Input("invert-btn", "n_clicks"),
        Input("agg-method", "value"),
        Input("threshold-slider", "value"),
        Input("scale-mode", "value"),
        Input("fixed-res", "value"),
        Input("zoom-profile", "data"),
        Input("map", "zoom"),
        Input("filter-checkbox", "value"),
        Input("kernel-dropdown", "value"),
    ]
)
def update_legend(
    cmap_name, n_clicks, agg_method, threshold,
    scale_mode, fixed_res, zoom_profile, zoom,
    filter_value, kernel_size
):
    invert = n_clicks % 2 == 1
    cmap_to_use = cmap_name + "_r" if invert else cmap_name
    zoom = zoom or 7
    show_res = fixed_res if scale_mode == "fixed" else zoom_to_resolution(zoom, zoom_profile)

    if "filter" in filter_value:
        filtered_points = filter_points(points, kernel_size=kernel_size)
        cell_values = {}
        for lat, lng, val in filtered_points:
            cell = h3.latlng_to_cell(lat, lng, show_res)
            cell_values.setdefault(cell, []).append(val)
    else:
        cell_values = precomputed_h3.get(show_res, {})

    min_thresh, max_thresh = threshold
    display_values = {}
    for cell, vals in cell_values.items():
        filtered_vals = [v for v in vals if min_thresh <= v <= max_thresh]
        if not filtered_vals:
            continue
        if agg_method == "count":
            val = len(filtered_vals)
        else:
            val = {"min": np.min, "max": np.max, "mean": np.mean}[agg_method](filtered_vals)
        display_values[cell] = val

    if display_values:
        if agg_method == "count":
            vmin, vmax = 0, max(display_values.values())
        else:
            vmin, vmax = min(display_values.values()), max(display_values.values())
        if vmin == vmax:
            vmax = vmin + 1
    else:
        vmin, vmax = 0, 1

    cmap = plt.get_cmap(cmap_to_use)
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(3, 0.3))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    values = np.linspace(vmin, vmax, 5)
    labels = [str(int(v)) if agg_method == "count" else f"{v:.2f}" for v in values]

    return html.Div([
        html.Div("Color scale", style={"font-weight": "bold", "margin-bottom": "5px"}),
        html.Img(src="data:image/png;base64," + data, className="legend-bar"),
        html.Div([html.Span(label) for label in labels], className="legend-labels")
    ])


if __name__ == "__main__":
    app.run(debug=True)
