import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import shapely
import xarray as xr

from rioxarray.exceptions import NoDataInBounds
from sklearn.preprocessing import LabelEncoder


class ObjectExtractor:
    """
    Extract geocoded objects from one or more raster datasets using
    the provided contour boundaries.

    Args:
        images: raster images where to extract the objects from,
            provided as a list of paths or xarray DataArray objects.
        geoms: geometries of the objects, provided as a list of polygons,
            points, bounds or as a geopandas GeoSeries. If points are
            provided, the `buffer` argument must be provided. Geometries
            are expexcted to be in the same CRS as the images. Indices
            should match the ones of `classes`.
        labels (optional): object labels, provided as a list or as a
            pandas Series. Indices should match the ones of `objects`.
        buffer (optional): buffer width applied to the geometries, in local
            CRS units.
        min_pixel_size (optional): side (in pixel) of the smallest cutout
            (smaller cutouts will be discarded).
        max_pixel_size (optional):  side (in pixel) of the largest cutout
            (larger cutouts will be discarded).
        pixel_size (optional): cutouts will be padded to a square image
            of the given side (in pixel). If provided, it should be greater
            than or equal to max_pixel_size.
        min_sample_size (optional): classes with less samples than the given
            amount will be discarded.
        encode_labels (optional): whether to normalize and convert
            non-numerical labels to numerical labels.
    """
    def __init__(
            self,
            images,
            geoms,
            labels=None,
            buffer=None,
            min_pixel_size=None,
            max_pixel_size=None,
            pixel_size=None,
            min_sample_size=None,
            encode_labels=True,
    ):
        self.images = images
        labels = np.zeros(len(geoms), dtype=int) if labels is None else labels
        geometry = _normalize_geoms(geoms, buffer)
        self.labels = gpd.GeoDataFrame({'label': labels}, geometry=geometry)

        self.min_pixel_size = min_pixel_size
        self.max_pixel_size = max_pixel_size
        self.pixel_size = pixel_size
        self.min_sample_size = min_sample_size

        self.encode_labels = encode_labels

    def _generate_cutouts_for_image(self, image):
        """ Extract all cutouts from a given image. """

        if not isinstance(image, xr.DataArray):
            image = rioxarray.open_rasterio(image, masked=True)

        if self.labels.crs is None:
            # if labels don't have a CRS, assume it's the correct one
            self.labels.set_crs(image.rio.crs)
        else:
            assert self.labels.crs == image.rio.crs

        # select relevant objects
        xmin, ymin, xmax, ymax = image.rio.bounds()
        labels_clip = self.labels.cx[xmin:xmax, ymin:ymax]
        labels_clip = labels_clip[~labels_clip.is_empty]

        for el in labels_clip.itertuples():

            try:
                # First clip with bounding box:
                # it is more efficient to first do the cropping based
                # on the bounds and then use the actual geometry
                image_box = image.rio.clip_box(*el.geometry.bounds)
                image_clip = image_box.rio.clip([el.geometry], drop=True)
            except NoDataInBounds:
                # some polygons may have very small intersections, which
                # results in an empty array here - we drop these.
                continue

            # drop too large and too small cutouts
            is_too_large = _is_too_large(image_clip, self.max_pixel_size) \
                if self.max_pixel_size is not None else False
            is_too_small = _is_too_small(image_clip, self.min_pixel_size) \
                if self.min_pixel_size is not None else False
            if is_too_large or is_too_small:
                continue

            cutout = _pad_image(image_clip, self.pixel_size) \
                if self.pixel_size is not None else image_clip

            affine_params = _get_affine_params_dict(cutout.rio.transform())
            cutout = _to_np_array(cutout)

            yield el.Index, el.label, affine_params, cutout

    def get_cutouts(self):
        """
        Load and return cutouts extracted from all images.

        Returns:
            labels: labels, as a pd.Series with shape (ncutouts,)
            affine_params: affine transformation parameters, as a pd.DataFrame
                with shape (ncutouts, 6). We use the affine naming for the
                transformation params: https://affine.readthedocs.io/en/latest/
            crs: coordinate reference system employed, as a pyproj.CRS object.
            cutouts: sequence of cutouts, as a np.ndarray with shape
                (ncutouts, ny, nx, 3)
        """

        results = []
        for image in self.images:
            cutout_iter = self._generate_cutouts_for_image(image)
            results.extend(list(cutout_iter))

        idxs, labels, affine_params, cutouts = list(zip(*results))

        labels = pd.Series(name='label', data=labels, index=idxs)
        affine_params = pd.DataFrame(data=affine_params, index=idxs)
        cutouts = np.stack(cutouts)

        labels, affine_params, cutouts = _remove_small_classes(
            labels, affine_params, cutouts, self.min_sample_size
        )

        # convert class labels to categorical values
        if self.encode_labels:
            le = LabelEncoder()
            le.fit(labels)
            labels = pd.Series(
                name='label', data=le.transform(labels), index=labels.index
            )
        return labels, affine_params, self.labels.crs, cutouts


def _normalize_geoms(geoms, buffer=None):
    """
    Normalize `geoms` accounting for the following possible input types:

    * list of polygons
    * list of points (with a buffer)
    * list of bounds
    * geopandas GeoSeries

    Always return a GeoSeries with polygons
    """
    if not isinstance(geoms, gpd.GeoSeries):
        el = geoms[0]
        if isinstance(el, shapely.Geometry):
            geometry = gpd.GeoSeries(geoms)
        elif len(el) == 2:
            # (x1, y1), (x2, y2), ...
            geometry = gpd.GeoSeries.from_xy(*list(zip(*geoms)))
        elif len(el) == 4:
            # (minx1, miny1, maxx1, maxy1), (minx2, min2, maxx2, maxy2), ...
            polygons = [shapely.Polygon.from_bounds(*bb) for bb in geoms]
            geometry = gpd.GeoSeries(polygons)
    else:
        geometry = geoms

    if buffer is not None:
        geometry = geometry.buffer(buffer)

    if (geometry.geom_type == 'Point').any():
        raise ValueError(
            'With Point geometries, the `buffer` parameter must be provided'
        )
    return geometry


def _to_np_array(data_array):
    """
    Extract image data from the DataArray, standardizing the
    axis ordering: (y, x, band).
    """
    if data_array.ndim == 3:
        return data_array.transpose('y', 'x', 'band').data
    elif data_array.ndim == 2:
        return data_array.transpose('y', 'x').data
    else:
        raise ValueError('Input data shape not understood.')


def _get_affine_params_dict(transform):
    """ Extract Affine transform parameters to a dictionary. """
    return {el: getattr(transform, el) for el in 'abcdef'}


def _remove_small_classes(labels, affine_params, cutouts, min_sample_size):
    """
    Balance class compositions by dropping classes with few samples.
    """
    labels_unique, counts = np.unique(labels, return_counts=True)
    labels_to_keep = labels_unique[counts >= min_sample_size]
    mask = np.isin(labels, labels_to_keep)
    return (
        labels[mask],
        affine_params[mask],
        cutouts[mask],
    )


def _is_too_large(data, max_pixel_size):
    if max(data.rio.shape) > max_pixel_size:
        return True
    return False


def _is_too_small(data, min_pixel_size):
    if min(data.rio.shape) <= min_pixel_size:
        return True
    return False


def _pad_image(data, pixel_size):
    """ Pad each image to shape (pixel_size, pixel_size) """

    pad_width_1 = (pixel_size - data.rio.width) // 2
    pad_width_2 = pixel_size - data.rio.width - pad_width_1
    pad_height_1 = (pixel_size - data.rio.height) // 2
    pad_height_2 = pixel_size - data.rio.height - pad_height_1
    padded = data.pad(
        x=(pad_width_1, pad_width_2),
        y=(pad_height_1, pad_height_2),
        mode='constant',
        constant_values=0,
    )
    return padded.rio.transform(recalc=True)
