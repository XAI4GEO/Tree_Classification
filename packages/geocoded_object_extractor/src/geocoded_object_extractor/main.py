import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
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
        self.labels = gpd.GeoDataFrame({'label': labels, 'geometry': geometry})

        self.min_pixel_size = min_pixel_size
        self.max_pixel_size = max_pixel_size
        self.pixel_size = pixel_size
        self.min_sample_size = min_sample_size

        self.encode_labels = encode_labels

        self.crs = None

    def _generate_cutouts_for_image(self, image):
        """ Extract all cutouts from a given image. """

        if not isinstance(image, xr.DataArray):
            image = rioxarray.open_rasterio(image)

        # If the image has no CRS, assume it's WGS84.
        # This is an assumption made of pure raster, such as png data.
        # It is okay of no georeferencing based computation, such as distance
        if image.rio.crs is None:
                image = image.rio.write_crs('4326')

        # check CRSs are consistent
        if self.crs is None:
            if image.rio.crs is not None:
                self.crs = pyproj.CRS.from_wkt(image.rio.crs.to_wkt())
        else:
            # all rasters must have the same CRS
            assert self.crs == image.rio.crs
        labels = self.labels
        if labels.crs is None:
            # if labels don't have a CRS, assume it's the correct one
            labels = labels.set_crs(self.crs)
        else:
            # vector and rasters must have the same CRS
            assert labels.crs == self.crs

        # select relevant objects
        xmin, ymin, xmax, ymax = image.rio.bounds()
        labels_clip = labels.cx[xmin:xmax, ymin:ymax]
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

            data = _to_np_array(image_clip)

            yield el.Index, el.label, image_clip.rio.transform(), data

    def get_cutouts(self):
        """
        Load and return cutouts extracted from all images.

        Returns:
            labels: labels, as a pd.Series with shape (ncutouts,)
            transform_params: affine transformation parameters, as a
                pd.DataFrame with shape (ncutouts, 6). We use the affine naming
                for the 2D transformation parameters (a, b, c, d, e, f):
                https://affine.readthedocs.io/en/latest/
            crs: coordinate reference system employed, as a pyproj.CRS object.
            cutouts: sequence of cutouts, as a np.ndarray with shape
                (ncutouts, ny, nx, nbands)
        """

        results = []
        for image in self.images:
            cutout_iter = self._generate_cutouts_for_image(image)
            results.extend(list(cutout_iter))

        idxs, labels, transforms, cutout_list = list(zip(*results))

        idxs = np.array(idxs)
        labels = np.array(labels)
        transforms = np.array(transforms)
        # cutouts have different shapes so far!
        cutouts = np.empty(len(cutout_list), dtype=object)
        cutouts[:] = cutout_list[:]

        if self.min_sample_size is not None:
            idxs, labels, transforms, cutouts = _remove_small_classes(
                idxs, labels, transforms, cutouts, self.min_sample_size
            )

        idxs, labels, transforms, cutouts = _drop_duplicates(
            idxs, labels, transforms, cutouts
        )

        if self.encode_labels:
            labels = _encode_labels(labels)

        transforms, cutouts = _pad_data_and_stack(
            transforms, cutouts, self.pixel_size
        )

        labels = pd.Series(name='label', data=labels, index=idxs)
        transform_params = pd.DataFrame(
            data=transforms, columns=list('abcdefghi'), index=idxs
        )
        return labels, transform_params, self.crs, cutouts


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


def _remove_small_classes(
        idxs, labels, transforms, cutouts, min_sample_size
):
    """
    Balance class compositions by dropping classes with few samples.
    """
    labels_unique, counts = np.unique(labels, return_counts=True)
    labels_to_keep = labels_unique[counts >= min_sample_size]
    mask = np.isin(labels, labels_to_keep)
    return (
        idxs[mask],
        labels[mask],
        transforms[mask],
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


def _drop_duplicates(idxs, labels, transforms, cutouts):
    """ Drop elements with the same indices """
    idxs_unique, counts = np.unique(idxs, return_counts=True)
    mask = np.ones(len(idxs), dtype=bool)
    for idx in idxs_unique[counts > 1]:
        duplicates, = np.nonzero(idxs == idx)
        # among the cutouts with duplicate idx, keep the largest
        # to_keep = np.argmax([c.shape[0]*c.shape[1] for c in cutouts[duplicates]])
        # among the cutouts with duplicate idx, keep the first
        mask[duplicates] = False
        mask[duplicates[0]] = True
    return (
        idxs[mask],
        labels[mask],
        transforms[mask],
        cutouts[mask],
    )


def _pad_data_and_update_transform(data, transform, pixel_size):
    """
    Pad data to shape (pixel_size, pixel_size), updating the
    affine transformation accordingly.
    """
    height, width, *_ = data.shape
    pad_top = (pixel_size - height) // 2
    pad_bottom = pixel_size - height - pad_top
    pad_left = (pixel_size - width) // 2
    pad_right = pixel_size - width - pad_left

    data_pad = np.pad(
        data,
        pad_width=[
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0),
        ],
        mode='constant',
        constant_values=0,
    )

    # determine new coordinates of top-left corner (c and f params)
    c_pad, f_pad = affine.Affine(*transform) * (-pad_left, -pad_top)
    transform_pad = transform
    transform_pad[2] = c_pad
    transform_pad[5] = f_pad
    return data_pad, transform_pad


def _pad_data_and_stack(transforms, cutouts, pixel_size=None):
    if pixel_size is None:
        pixel_size = max(max(*c.shape) for c in cutouts)

    cutout_ = cutouts[0]
    cutouts_pad = np.zeros(
        (len(cutouts), pixel_size, pixel_size, cutout_.shape[-1]),
        dtype=cutout_.dtype
    )
    transforms_pad = np.zeros_like(transforms)
    for ncutout, (transform, cutout) in enumerate(zip(transforms, cutouts)):
        cutout_pad, transform_pad = _pad_data_and_update_transform(
            cutout, transform, pixel_size
        )
        cutouts_pad[ncutout, ...] = cutout_pad[...]
        transforms_pad[ncutout] = transform_pad[:]
    return transforms_pad, cutouts_pad


def _encode_labels(labels):
    """ Convert class labels to categorical (numeric) values """
    le = LabelEncoder()
    le.fit(labels)
    return le.transform(labels)
