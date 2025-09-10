import dash
from dash import html, dcc, Output, Input, State
import dash_leaflet as dl
import dask.dataframe as dd
import geopandas as gpd
from shapely.geometry import Point, box
import json
import io, zipfile, tempfile
from pathlib import Path
import rasterio
from rasterio.mask import mask
from cdse_aoi_downloader import download_image
import base64

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label("Choose data source:"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[
                    {"label": "Sentinel-2", "value": "sentinel2"},
                    {"label": "Landsat", "value": "landsat"}
                ],
                placeholder="Select dataset",
                clearable=False
            ),
            html.Label("Upload AOI (GeoJSON):", style={"marginTop": "10px"}),
            dcc.Upload(
                id='upload-geojson',
                children=html.Div(['Drag or ', html.A('select file')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                       'textAlign': 'center'},
                multiple=False
            ),
            html.Div(id='upload-status', style={"marginTop": "5px", "fontWeight": "bold"}),
            html.Button("Submit", id="load-button", n_clicks=0, style={'marginTop': '10px', 'display': 'block'}),
        ], style={'flex': '1', 'marginRight': '10px'}),

        html.Div([
            html.Label("Select bands:", style={"marginTop": "10px"}),
            html.Fieldset([
                dcc.Checklist(
                    id="bands-checklist",
                    options=[
                        {"label": "B01 (Coastal aerosol)", "value": "B01"},
                        {"label": "B02 (Blue)", "value": "B02"},
                        {"label": "B03 (Green)", "value": "B03"},
                        {"label": "B04 (Red)", "value": "B04"},
                        {"label": "B05 (Vegetation Red Edge)", "value": "B05"},
                        {"label": "B06 (Vegetation Red Edge)", "value": "B06"},
                        {"label": "B07 (Vegetation Red Edge)", "value": "B07"},
                        {"label": "B08 (NIR)", "value": "B08"},
                        {"label": "B8A (Vegetation Red Edge)", "value": "B8A"},
                        {"label": "B09 (Water vapor)", "value": "B09"},
                        {"label": "B10 (SWIR - Cirrus)", "value": "B10"},
                        {"label": "B11 (SWIR)", "value": "B11"},
                        {"label": "B12 (SWIR)", "value": "B12"}
                    ],
                    value=[],
                    inline=False
                )
            ], id="bands-fieldset", disabled=True),  # Initially disabled

            html.Button("Download images", id="download-images-btn", n_clicks=0,
                        style={'marginTop': '10px', 'display': 'block'}, disabled=True),
            dcc.Download(id="download-zip"),
        ], style={'flex': '1', 'marginLeft': '10px'}),
    ], style={'display': 'flex', 'marginBottom': '20px'}),

    # Map
    dl.Map(
        id="map",
        children=[dl.TileLayer()],
        style={'width': '100%', 'height': '600px'},
        center=[0, 0],
        zoom=2
    ),
    dcc.Store(id="bboxes-store"),
    dcc.Store(id="bboxes-latest-store"),
])


def parse_uploaded_geojson(contents):
    """
    Parse the uploaded GeoJSON file and return a GeoDataFrame.

    Args:
        contents (str): Base64-encoded contents of uploaded file.

    Returns:
        gpd.GeoDataFrame or None: GeoDataFrame of AOI features or None on failure.
    """
    if contents is None:
        return None
    try:
        content_type, content_string = contents.split(',')
        decoded = io.BytesIO(base64.b64decode(content_string))
        geojson_dict = json.load(decoded)
        return gpd.GeoDataFrame.from_features(geojson_dict.get("features", [geojson_dict]), crs="EPSG:4326")
    except Exception as e:
        print("GeoJSON parsing error:", e)
        return None


def bbox_to_geojson_file(bbox, out_path):
    """
    Convert a bounding box to a GeoJSON file.

    Args:
        bbox (tuple): Bounding box (min_lon, min_lat, max_lon, max_lat)
        out_path (Path): Output file path

    Returns:
        Path: Path to saved GeoJSON file
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            },
            "properties": {}
        }]
    }
    out_path.write_text(json.dumps(geojson))
    return out_path


def create_bboxes_global_grid(df_all, aoi_gdf, step_deg=0.18):
    """
    Generate bounding boxes from dataset points within the AOI.

    Args:
        df_all (dd.DataFrame): Dataset with 'Lon', 'Lat', 'Date', 'id', 'dataset'
        aoi_gdf (gpd.GeoDataFrame): AOI polygon
        step_deg (float): Grid step size in degrees

    Returns:
        list: List of dictionaries with bbox info: id, bbox, dates, dataset
    """
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    df = df_all[(df_all['Lon'] >= minx) & (df_all['Lon'] <= maxx) &
                (df_all['Lat'] >= miny) & (df_all['Lat'] <= maxy)]
    df['geometry'] = df.apply(lambda row: Point(row['Lon'], row['Lat']), axis=1, meta=('geometry', 'object'))
    gdf = gpd.GeoDataFrame(df.compute(), geometry='geometry', crs="EPSG:4326")
    points_in_aoi = gdf[gdf.geometry.within(aoi_gdf.geometry.iloc[0])]
    points_grouped = points_in_aoi.groupby('id').agg({
        'Date': lambda x: sorted(list(x)),
        'dataset': 'first',
        'geometry': 'first'
    }).reset_index()
    bboxes_list = []
    half_size = step_deg / 2
    for _, row in points_grouped.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        min_lon, max_lon = lon - half_size, lon + half_size
        min_lat, max_lat = lat - half_size, lat + half_size
        bboxes_list.append({
            "id": row['id'],
            "bbox": (min_lon, min_lat, max_lon, max_lat),
            "dates": row['Date'],
            "dataset": row['dataset']
        })
    return bboxes_list


def create_bbox_layers(bboxes_data):
    """
    Create Dash Leaflet Rectangle layers for visualizing bounding boxes.

    Args:
        bboxes_data (list): List of bbox dictionaries

    Returns:
        list: List of dl.Rectangle layers
    """
    rectangles = []
    for item in bboxes_data:
        min_lon, min_lat, max_lon, max_lat = item["bbox"]
        popup_content = html.Div([
            html.B(f"ID: {item.get('id', 'unknown')}"), html.Br(),
            html.B(f"Dataset: {item.get('dataset', 'unknown')}"), html.Br(),
            html.B("Bounding Box:"),
            html.Div(f"Min Lon: {min_lon}, Min Lat: {min_lat}", style={"marginLeft": "10px"}),
            html.Div(f"Max Lon: {max_lon}, Max Lat: {max_lat}", style={"marginLeft": "10px"}),
            html.B("Dates:"),
            html.Div([html.Div(str(date)) for date in item.get("dates", [])],
                     style={"maxHeight": "100px", "overflowY": "auto",
                            "border": "1px solid #ccc", "padding": "5px",
                            "backgroundColor": "#f9f9f9", "borderRadius": "5px"})
        ], style={"fontSize": "12px", "lineHeight": "1.2em"})
        rectangles.append(
            dl.Rectangle(
                id=f"bbox-{item.get('id', '')}",
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color='darkgreen',
                weight=1,
                fillColor='green',
                fillOpacity=0.5,
                children=[dl.Popup(popup_content)]
            )
        )
    return rectangles


@app.callback(
    Output("map", "children"),
    Output("bboxes-store", "data"),
    Output("bboxes-latest-store", "data"),
    Input("load-button", "n_clicks"),
    State("dataset-dropdown", "value"),
    State("upload-geojson", "contents")
)
def update_map(n_clicks, dataset_choice, uploaded_contents):
    """
    Load AOI, dataset points, create bounding boxes and display on map.

    Returns:
        layers (list), bboxes_data (list), bboxes_latest_data (list)
    """
    if n_clicks == 0 or not dataset_choice or not uploaded_contents:
        return [dl.TileLayer()], [], []
    aoi_gdf = parse_uploaded_geojson(uploaded_contents)
    if aoi_gdf is None:
        return [dl.TileLayer()], [], []
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    scale = 1.35
    dx = (maxx - minx) * (scale - 1) / 2
    dy = (maxy - miny) * (scale - 1) / 2
    expanded_box = box(minx - dx, miny - dy, maxx + dx, maxy + dy)
    expanded_gdf = gpd.GeoDataFrame(geometry=[expanded_box], crs="EPSG:4326")
    if dataset_choice == "sentinel2":
        df_all = dd.read_parquet("sentinel2.parquet", engine="pyarrow").assign(dataset="Sentinel-2")
    elif dataset_choice == "landsat":
        df_all = dd.read_parquet("landsat.parquet", engine="pyarrow").assign(dataset="Landsat")
    else:
        return [dl.TileLayer()], [], []
    bboxes_data = create_bboxes_global_grid(df_all, aoi_gdf)
    bbox_layers = create_bbox_layers(bboxes_data)
    bboxes_latest_data = [
        {"id": item["id"], "bbox": item["bbox"], "latest_date": max(item["dates"]) if item.get("dates") else None}
        for item in bboxes_data
    ]
    layers = [dl.TileLayer(), dl.LayerGroup(bbox_layers),
              dl.GeoJSON(data=expanded_gdf.__geo_interface__, style={"color": "red", "fill": False, "weight": 1})]
    return layers, bboxes_data, bboxes_latest_data


@app.callback(
    Output('upload-geojson', 'children'),
    Input('upload-geojson', 'contents'),
    State('upload-geojson', 'filename')
)
def update_upload_text(contents, filename):
    """
    Update text in upload component to show uploaded filename.
    """
    if contents:
        return html.Div([f"Uploaded file: {filename}"],
                        style={"color": "green", "lineHeight": "60px", 'fontWeight': 'bold'})
    return html.Div(['Drag or ', html.A('select file')])


@app.callback(
    Output("bands-fieldset", "disabled"),
    Output("download-images-btn", "disabled"),
    Input("dataset-dropdown", "value"),
    Input("bboxes-store", "data")
)
def toggle_sentinel_options(dataset_choice, bboxes_data):
    """
    Enable or disable Sentinel-2 bands options based on dataset selection and AOI availability.
    """
    if dataset_choice == "sentinel2" and bboxes_data:
        return False, False
    else:
        return True, True


@app.callback(
    Output("download-zip", "data"),
    Input("download-images-btn", "n_clicks"),
    State("dataset-dropdown", "value"),
    State("bboxes-store", "data"),
    State("bands-checklist", "value"),
    prevent_initial_call=True
)
def download_images(n_clicks, dataset_choice, bboxes_data_full, selected_bands):
    """
    Download selected Sentinel-2 images for each bbox and return as zip.
    """
    if dataset_choice != "sentinel2" or not bboxes_data_full or not selected_bands:
        return None
    tmpdir = Path(tempfile.mkdtemp())
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i, item in enumerate(bboxes_data_full):
            bbox = item["bbox"]
            bbox_id = item.get("id", f"bbox_{i}")
            if not item.get("dates"):
                continue
            sorted_dates = sorted(item["dates"], reverse=True)
            success = False
            for date in sorted_dates:
                fixed_date = str(date).replace("_", "-")
                start = f"{fixed_date}T00:00:00Z"
                end = f"{fixed_date}T23:59:59Z"
                aoi_path = tmpdir / f"{bbox_id}.geojson"
                bbox_to_geojson_file(bbox, aoi_path)
                try:
                    out_file = download_image(
                        aoi_path=aoi_path,
                        start=start,
                        end=end,
                        out_dir=tmpdir,
                        bands=selected_bands,
                        res=10
                    )
                    if out_file and out_file.exists():
                        print(f"[INFO] Pobrano obraz {out_file.name} dla bbox {bbox_id}, data {fixed_date}")
                        with rasterio.open(out_file) as src:
                            aoi_gdf = gpd.read_file(aoi_path).to_crs(src.crs)
                            if not aoi_gdf.intersects(gpd.GeoSeries([box(*src.bounds)], crs=src.crs)).any():
                                print(f"[WARN] AOI nie nachodzi na obraz {out_file.name} -> pomijam")
                                continue

                            out_image, out_transform = mask(src, aoi_gdf.geometry, crop=True)
                            nodata_val = src.nodata if src.nodata is not None else 0
                            nodata_ratio = (out_image == nodata_val).sum() / out_image.size
                            if nodata_ratio > 0.1:
                                print(f"[WARN] Za dużo nodata ({nodata_ratio:.2%}) w {out_file.name} -> pomijam")
                                continue

                            meta = src.meta.copy()
                            meta.update({
                                "driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform
                            })
                            cropped_path = tmpdir / f"{bbox_id}_{fixed_date}.tif"
                            with rasterio.open(cropped_path, "w", **meta) as dst:
                                dst.write(out_image)

                            print(f"[OK] Zapisano przycięty obraz: {cropped_path.name}")
                            zf.write(cropped_path, arcname=cropped_path.name)
                            success = True
                            break
                except Exception as e:
                    print(f"[ERROR] Pobieranie/obróbka bbox {bbox_id}, data {fixed_date} nie powiodła się: {e}")
                    continue

    zip_buf.seek(0)
    return dcc.send_bytes(zip_buf.getvalue(), "bboxes_images.zip")


if __name__ == "__main__":
    app.run(debug=True)
