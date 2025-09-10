#!/usr/bin/env python3
"""
CDSE AOI downloader

Features:
1) PROCESS mode (default): Clips and downloads AOI-cropped GeoTIFFs via Sentinel Hub Process API.
2) SAFE mode (--mode safe): Downloads original .SAFE (zipped) products via OData.

Requirements:
    pip install sentinelhub geopandas shapely requests tqdm

Environment:
    # Sentinel Hub (Process API)
    CDSE_SH_CLIENT_ID
    CDSE_SH_CLIENT_SECRET

    # OData (SAFE downloads only)
    CDSE_USERNAME
    CDSE_PASSWORD
"""

import os
import sys
import json
import argparse
import datetime as dt
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Geospatial
import geopandas as gpd
from shapely.geometry import shape, mapping

# Sentinel Hub
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    Geometry,
    bbox_to_dimensions,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    filter_times,
)

# ---------------------------
# Utilities
# ---------------------------

def load_aoi_geojson(aoi_path: Path) -> Geometry:
    """Load AOI geometry from GeoJSON file (assumed WGS84)."""
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        raise ValueError("AOI file has no CRS. Please ensure it is WGS84 (EPSG:4326).")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    geom = gdf.union_all()
    return Geometry(geom, crs=CRS.WGS84)

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def build_sh_config() -> SHConfig:
    """Build SHConfig for Copernicus Data Space Ecosystem."""
    cfg = SHConfig()
    cfg.sh_client_id = "sh-7a1d1c50-e725-45be-939b-4d64f1b5879c"
    cfg.sh_client_secret = "iaXtSaktdaA7eH62nGq4tX1yMcCptrwk"
    # Official CDSE endpoints:
    cfg.sh_base_url = "https://sh.dataspace.copernicus.eu"
    cfg.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise RuntimeError("Missing CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET environment variables.")
    return cfg

def parse_bands(bands_csv: str) -> List[str]:
    bands = [b.strip() for b in bands_csv.split(",") if b.strip()]
    if not bands:
        raise ValueError("No bands provided.")
    return bands

def build_evalscript(bands: List[str]) -> str:
    """
    Build Evalscript V3 for Process API.
    Returns float32 stack of requested bands + dataMask as last band.
    """
    bands_list = ", ".join([f'"{b}"' for b in bands])
    n_bands = len(bands) + 1  # +1 for dataMask
    return f"""//VERSION=3
    function setup() {{
    return {{
        input: [{{
        bands: [{bands_list}, "dataMask"]
        }}],
        output: {{
        bands: {n_bands},
        sampleType: "FLOAT32"
        }}
    }};
    }}

    function evaluatePixel(s) {{
    return [{", ".join([f"s.{b}" for b in bands])}, s.dataMask];
    }}
    """

# ---------------------------
# PROCESS MODE (Sentinel Hub)
# ---------------------------

def search_s2_items(
    cfg: SHConfig,
    geometry: Geometry,
    start: str,
    end: str,
    max_cloud: float,
    limit: int,
    best_only: bool = False
) -> Tuple[List[dt.datetime], List[dict]]:
    """
    Search Sentinel-2 L2A items over AOI and time range with cloud filter.
    Returns list of timestamps (either all or best one only).
    """
    catalog = SentinelHubCatalog(config=cfg)

    time_interval = (start, end)
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        geometry=geometry,
        time=time_interval,
        filter=f"eo:cloud_cover <= {max_cloud}",
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
        limit=limit,
    )

    items = list(search_iterator)
    if not items:
        return [], []

    if best_only:
        # pick scene with lowest cloud cover
        best_item = min(items, key=lambda i: i["properties"].get("eo:cloud_cover", 100))
        ts = dt.datetime.fromisoformat(best_item["properties"]["datetime"].replace("Z", ""))
        return [ts], [best_item]

    # default: all timestamps
    all_timestamps = search_iterator.get_timestamps()
    time_diff = dt.timedelta(hours=1)
    unique_acq = filter_times(all_timestamps, time_difference=time_diff)
    return unique_acq, items

def call_process_api_direct(client_id: str, client_secret: str, payload: dict) -> bytes:
    """Direct call to CDSE Process API without sentinelhub-py"""
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    proc_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    # Get token
    tok = requests.post(
        token_url,
        data={"grant_type":"client_credentials", "client_id":client_id, "client_secret":client_secret},
        timeout=60,
    )
    tok.raise_for_status()
    access = tok.json()["access_token"]

    # Send Process API request
    r = requests.post(
        proc_url,
        headers={"Authorization": f"Bearer {access}"},
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    return r.content

def process_download(
    cfg: SHConfig,
    geometry: Geometry,
    timestamps: List[dt.datetime],
    bands: List[str],
    resolution: float,
    out_dir: Path,
) -> None:
    """
    For each timestamp, request AOI-clipped GeoTIFF with requested bands (+dataMask) via CDSE Process API.
    """
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    # Step 1: get OAuth token
    r = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": cfg.sh_client_id,
            "client_secret": cfg.sh_client_secret,
        },
        timeout=30,
    )
    r.raise_for_status()
    token = r.json()["access_token"]

    # Step 2: prepare request headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Step 3: bounding box + CRS + resolution
    geom_geojson = mapping(geometry.geometry)
    crs_uri = "http://www.opengis.net/def/crs/EPSG/0/4326"

    # Step 4: Evalscript
    evalscript = build_evalscript(bands)

    for ts in tqdm(timestamps, desc="Downloading (Process API direct)"):
        payload = {
            "input": {
                "bounds": {
                    "geometry": geom_geojson,
                    "properties": {"crs": crs_uri},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": ts.replace(microsecond=0).isoformat() + "Z",
                                "to": (ts + dt.timedelta(hours=1)).replace(microsecond=0).isoformat() + "Z",
                            }
                        },
                        "processing": {
                            "resolution": resolution
                        }
                    }
                ]
            },
            "output": {
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/tiff"}
                    }
                ]
            },
            "evalscript": evalscript
        }

        response = requests.post(
            process_url,
            headers=headers,
            json=payload,
            timeout=300
        )
        if response.status_code == 400:
            print(f"❌ Bad request for date {ts.date()}: {response.text}")
            continue
        response.raise_for_status()

        out_name = f"S2L2A_{ts.strftime('%Y%m%dT%H%M%S')}_{resolution:.0f}m_{'-'.join(bands)}.tif"
        with open(out_dir / out_name, "wb") as f:
            f.write(response.content)

# ---------------------------
# SAFE MODE (OData)
# ---------------------------

OAUTH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_SEARCH = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ODATA_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

def get_cdse_token_via_password() -> str:
    """
    Get OAuth access token using Resource Owner Password (client_id=cdse-public).
    Recommended only for scripts where environment variables are used safely.
    """
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD for OData SAFE downloads.")

    data = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    resp = requests.post(OAUTH_TOKEN_URL, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["access_token"]

def find_product_id_by_name(product_name: str, token: str) -> Optional[str]:
    """
    Use OData to find product Id by exact Name (e.g. 'S2A_MSIL2A_...SAFE').
    """
    params = {"$filter": f"Name eq '{product_name}'", "$select": "Id,Name"}
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(ODATA_SEARCH, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    values = r.json().get("value", [])
    if not values:
        return None
    return values[0]["Id"]

def download_safe_product(product_id: str, token: str, out_zip: Path) -> None:
    """
    Stream download .SAFE ZIP for given product Id.
    """
    url = ODATA_DOWNLOAD.format(product_id=product_id)
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        with open(out_zip, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_zip.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def safe_download_from_catalog_ids(
    cfg: SHConfig,
    geometry: Geometry,
    start: str,
    end: str,
    max_cloud: float,
    limit: int,
    out_dir: Path,
) -> None:
    """
    1) Use SH Catalog to find product names intersecting AOI (faster/easier with CQL2 and geometry).
    2) Resolve each product name to OData product Id.
    3) Download ZIP via OData.
    """
    token = get_cdse_token_via_password()
    catalog = SentinelHubCatalog(config=cfg)
    time_interval = (start, end)
    search_it = catalog.search(
        DataCollection.SENTINEL2_L2A,
        geometry=geometry,
        time=time_interval,
        filter=f"eo:cloud_cover <= {max_cloud}",
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
        limit=limit,
    )
    items = list(search_it)
    if not items:
        print("No matching products found for SAFE download.")
        return

    for item in items:
        name = item["id"]  # e.g. 'S2A_MSIL2A_...SAFE'
        pid = find_product_id_by_name(name, token)
        if not pid:
            print(f"Product not found in OData by name: {name}")
            continue
        out_zip = out_dir / f"{name}.zip"
        download_safe_product(pid, token, out_zip)

def download_image(aoi_path: Path, start: str, end: str, out_dir: Path,
                   bands=None, res=10):
    cfg = build_sh_config()
    geometry = load_aoi_geojson(Path(aoi_path))

    if bands is None:
        bands = ["B04", "B03", "B02"]

    if isinstance(bands, list):
        bands = ",".join(bands)

    bands = parse_bands(bands)

    timestamps, _ = search_s2_items(cfg, geometry, start, end, 20, 10, best_only=True)
    if not timestamps:
        return None

    process_download(cfg, geometry, timestamps, bands, res, out_dir)

    out_file = next(out_dir.glob("*.tif"))
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Download Copernicus data for an AOI.")
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON (WGS84).")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--max-cloud", type=float, default=20.0, help="Max cloud cover percent (S2).")
    parser.add_argument("--bands", default="B02,B03,B04,B08", help="Comma-separated S2 bands (Process mode).")
    parser.add_argument("--res", type=float, default=10.0, help="Target resolution in meters (Process mode).")
    parser.add_argument("--limit", type=int, default=50, help="Max number of items to consider.")
    parser.add_argument("--mode", choices=["process", "safe"], default="process", help="process=GeoTIFF clipped; safe=original .SAFE zip")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_outdir(out_dir)

    # Build config for both modes (Catalog + Process need SH credentials)
    cfg = build_sh_config()

    # Load AOI
    geometry = load_aoi_geojson(Path(args.aoi))

    if args.mode == "process":
        bands = parse_bands(args.bands)
        timestamps, _ = search_s2_items(cfg, geometry, args.start, args.end, args.max_cloud, args.limit, best_only=True)
        if not timestamps:
            print("No acquisitions match your filters.")
            sys.exit(0)
        process_download(cfg, geometry, timestamps, bands, args.res, out_dir)
        print(f"Done. Saved {len(timestamps)} GeoTIFF(s) to: {out_dir.resolve()}")

    else:  # safe
        safe_download_from_catalog_ids(cfg, geometry, args.start, args.end, args.max_cloud, args.limit, out_dir)
        print(f"Done. SAFE products (if any) saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()