import rasterio
import numpy as np
import pandas as pd


def tiff_to_parquet(tiff_path: str, output_path: str, value_column: str = "value") -> None:
    """
    Convert a single-band GeoTIFF raster file into a Parquet file with point coordinates and values.

    Args:
        tiff_path (str): Path to the input GeoTIFF file.
        output_path (str): Destination path for the output Parquet file.
        value_column (str, optional): Name of the value column in the output Parquet.
            Defaults to "value".

    Returns:
        None: Writes a Parquet file containing latitude, longitude, and raster values.
    """
    with rasterio.open(tiff_path) as src:
        data: np.ma.MaskedArray = src.read(1, masked=True)
        transform = src.transform

    if np.isscalar(data.mask) or data.mask.ndim == 0:
        mask: np.ndarray = ~np.isnan(data)
    else:
        mask = ~data.mask

    rows, cols = np.nonzero(mask)
    values: np.ndarray = data.data[rows, cols] if hasattr(data, "data") else data[rows, cols]

    xs, ys = rasterio.transform.xy(transform, rows, cols)

    df: pd.DataFrame = pd.DataFrame({
        "lat": ys,
        "lng": xs,
        value_column: values.astype(float)
    })

    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")


tiff_path = "file.tiff"
tiff_to_parquet(tiff_path, "output.parquet")
