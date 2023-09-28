import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import box
from deepforest import main


def _crop_box(crop_center, crop_size):
    """define square crop box by crop center and size

    Args:
        crop_center (Tuple): center coordinates of all boxes
        crop_size (int, float): size of the crop

    Returns:
        tuple: Bounds of the crop box in (minx, miny, maxx, maxy)
    """
    return (
        crop_center[0] - crop_size / 2.0,
        crop_center[1] - crop_size / 2.0,
        crop_center[0] + crop_size / 2.0,
        crop_center[1] + crop_size / 2.0,
    )  # minx, miny, maxx, maxy


def _to_gdf(df_prediction, crs=None):
    """Convert df to gdf"""
    geoms = box(**df_prediction[["xmin", "ymin", "xmax", "ymax"]])

    return gpd.GeoDataFrame(
        df_prediction.drop(["xmin", "ymin", "xmax", "ymax"], axis=1),
        geometry=geoms,
        crs=crs,
    )


def _to_image_geo(row, image):
    """Convert one row of image coordinates to georeferenced coordinates of that image.
    One should use apply_func to apply this function to all rows of a df

    Args:
        row (dataframe): one row of pandas.dataframe with cols: ["xmin", "ymin", "xmax", "ymax"]
        image (DataArray): original image

    Returns:
        row: row with a corrected reference
    """
    row["xmin"] = image["x"].isel({"x": int(row["xmin"])}).values.tolist()
    row["xmax"] = (
        image["x"]
        .isel({"x": np.min((int(row["xmax"]), image["x"].shape[0] - 1))})
        .values.tolist()
    )
    row["ymin"] = image["y"].isel({"y": int(row["ymin"])}).values.tolist()
    row["ymax"] = (
        image["y"]
        .isel({"y": np.min((int(row["ymax"]), image["y"].shape[0] - 1))})
        .values.tolist()
    )
    return row


def _to_numpy(img_da):
    """Convert a DataArray image to numpy. Apply rotation and flip to align the pixel coordinates with geo coordinates"""
    img_np = img_da.transpose("y", "x", "band").to_numpy()
    return img_np


def make_crops(image, crop_centers, crop_sizes):
    """Split big image into smaller sqaure crops

    Args:
        image (xarray.DataArray): input image
        crop_centers (list of tuples): center coords of each crop
        crop_sizes (list or list of floats): sizes of each crop

    Returns:
        list of xarray.DataArray: list of image crops
    """
    if isinstance(crop_sizes, int):
        crop_sizes = [crop_sizes] * len(crop_centers)

    assert len(crop_centers) == len(crop_sizes)

    list_crops = []
    for center, size in zip(crop_centers, crop_sizes):
        cbox = _crop_box(center, size)
        list_crops.append(image.rio.clip_box(*cbox))
    return list_crops


def predict_crops(image_crops, tile_size, overlap_size):
    """
    Run prediction on a list of image crops

    Args:
        image_crops (list of xarray.DataArray): list of image crops
        tile_size (int): tile sizes
        overlap_size (float): overlap sizes

    Returns:
        geopandas.GeoDataFrame: geodataframe of all pridicted polygons, in the with the same image crs
    """
    model = main.deepforest()
    model.use_release()

    df_predict = None
    for img in image_crops:
        df_predict_one = model.predict_tile(
            image=_to_numpy(img),
            patch_size=tile_size,
            patch_overlap=overlap_size,
        )

        # Covert pixel coords to geo coords
        df_predict_one = df_predict_one.apply(_to_image_geo, axis=1, image=img)

        if df_predict is None:
            df_predict = df_predict_one
        else:
            df_predict = pd.concat([df_predict, df_predict_one], axis=0)

    df_predict = df_predict.reset_index(drop=True)

    gdf_predict = _to_gdf(df_predict, image_crops[0].rio.crs)

    return gdf_predict


def predict_experiments(image_crops, list_tile_sizes, list_overlap_sizes):
    """
    Run prediction on image crops, exhaustly search all possible combinations given in tile sizes and overlap sizes.

    Args:
        image_crops (list of xarray.DataArray): list of image crops
        list_tile_sizes (list): list of tile sizes
        list_overlap_sizes (list): list of overlap sizes

    Returns:
        dict: dictionary of all experiments
    """
    search_tile_sizes, search_overlap_sizes = np.meshgrid(
        list_tile_sizes, list_overlap_sizes
    )

    results = dict()
    for tile_size, overlap_size in zip(
        search_tile_sizes.flatten(), search_overlap_sizes.flatten()
    ):
        key = f"tile{tile_size}_overlap{overlap_size:.2f}"
        prediction = predict_crops(image_crops, tile_size, overlap_size)

        results[key] = prediction

    return results
