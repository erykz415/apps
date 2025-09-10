# Hexagonal Map Visualization App

This project is an **interactive web application** built with [Dash](https://dash.plotly.com/), [dash-leaflet](https://dash-leaflet.herokuapp.com/), and [H3](https://h3geo.org/).  
It visualizes point-based geospatial data aggregated into **hexagonal cells** at different resolutions, with interactive controls for colormaps, aggregation, filtering, and exporting results as **GeoJSON**.
[app.py](../h3-app/app.py)
---

## Features

- **Interactive Map** (Leaflet-based)
  - Switch between **OSM** (default) and **Satellite** basemaps.  
  - Dynamic hexagon rendering depending on zoom or fixed resolution.  

- **Colormap Controls**
  - Choose from multiple colormaps (`RdYlGn`, `coolwarm`, `viridis`, `plasma`).  
  - Invert colormap with one click.  
  - Adjustable polygon opacity.  

- **Aggregation Methods**
  - Aggregate values per hexagon by:
    - Minimum  
    - Maximum  
    - Mean  
    - Count  

- **Resolution Control**
  - **Zoom-based resolution** (profiles: *low*, *medium*, *high*).  
  - **Fixed resolution** (manual slider).  

- **Value Range Filtering**
  - Set custom min/max thresholds.  
  - Interactive range slider linked to numeric inputs.  

- **Point Filtering (MORPH_OPEN)**
  - Optional morphological filtering to remove isolated points.  
  - Adjustable kernel size (3x3, 5x5, 7x7).  

- **Export**
  - Download the current hex layer as **GeoJSON** for use in GIS or other mapping tools.  

- **Legend & Layer Info**
  - Auto-updating color legend based on data range.  
  - Display of current resolution and aggregation method.  

---

## Project Structure
```python

h3-app/
├── assets
│ └── styles.css
├── app.py
├── output.parquet
├── README.md
├── requirements.txt
└── tiff_to_parquet.py
```

## Installation

1. Clone the repository

2. Create a virtual environment (optional but recommended):

3. Install dependencies (if necessary): `pip install -r requirements.txt`

## Usage
1. Make sure you have a output.parquet file with 3 columns:
```latitude, longitude, value```

2. Run the application:
`python app.py`

3. Open your browser at:
```http://127.0.0.1:8050/```
4. Use the control panel to adjust:
   * Basemap
   * Colormap & inversion
   * Aggregation method
   * Scale mode (zoom-based or fixed resolution)
   * Value thresholds
   * Filtering options


