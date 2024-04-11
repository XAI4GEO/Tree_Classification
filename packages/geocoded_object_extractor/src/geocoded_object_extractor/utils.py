import pathlib

import affine
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import skimage.transform
import xarray as xr


def augment_data(labels, cutouts):
    """
    Augment data by rotation (90, 180 and 270 degrees) and mirroring.

    Args:
        labels: labels, as a pd.Series with shape (ncutouts,)
        cutouts: sequence of cutouts, as a np.ndarray with shape
                (ncutouts, ny, nx, nbands)

    Returns:
        labels_aug: labels of the augmented dataset
        cutouts_aug: augmented dataset
    """
    idxs, labels, cutouts = _augment_data(labels.index, labels.values, cutouts)
    labels = pd.Series(name='label', data=labels, index=idxs)
    return labels, cutouts


def _flip(img):
    # horizontal flip
    return img[::-1, :]


def _rotate(img, angle, resize=False):
    # for rotation by a multiple of 90 degrees, we actually
    # don't have to interpolate (order 0), preserving int data types
    order = 0 if angle in [90, 180, 270] else None
    return skimage.transform.rotate(
        img,
        angle,
        order=order,
        resize=resize,
        mode='constant',
        cval=0,
    )


def _augment_data(idxs, labels, cutouts):
    """
    Augment data by performing 90 degree rotations and flipping to
    the images for all classes.
    """
    idxs_add = []
    labels_add = []
    cutouts_add = []
    for idx, label, cutout in zip(idxs, labels, cutouts):
        # Flip and add image
        flipped_cutout = _flip(cutout)
        cutouts_add.append(flipped_cutout)
        labels_add.append(label)
        idxs_add.append(idx)

        # Rotate image by three angles and flip each image
        for angle in [90, 180, 270]:
            rotated_cutout = _rotate(cutout, angle)
            cutouts_add.append(rotated_cutout)
            labels_add.append(label)
            idxs_add.append(idx)
            flipped_cutout = _flip(rotated_cutout)
            cutouts_add.append(flipped_cutout)
            labels_add.append(label)
            idxs_add.append(idx)

    cutouts = np.concatenate([cutouts, cutouts_add])
    idxs = np.concatenate([idxs, idxs_add])
    labels = np.concatenate([labels, labels_add])

    return idxs, labels, cutouts


def write_cutouts(
        cutouts, crs, transform_params, labels=None, outdir='./',
        overwrite=False
):
    """
    Write cutouts extracted using the ObjectExtractor to disk as a set
    of GeoTIFF files. If labels are provided, these are stored as a "LABEL"
    attribute in the GeoTIFFs.

    Args:
        cutouts: sequence of cutouts, as a np.ndarray with shape
            (ncutouts, ny, nx, nbands)
        crs: coordinate reference system employed, as a pyproj.CRS object.
        transform_params: affine transformation parameters, as a
            pd.DataFrame with shape (ncutouts, 6). We use the affine naming
            for the 2D transformation parameters (a, b, c, d, e, f):
            https://affine.readthedocs.io/en/latest/
        labels (optional): labels, as a pd.Series with shape (ncutouts,)
        outdir (optional): output directory
        overwrite (optional): if files already exist, overwrite them
    """
    dir = pathlib.Path(outdir)
    dir.mkdir(exist_ok=True, parents=True)

    for nel, params in enumerate(transform_params.itertuples()):
        path = dir / f'cutout-{nel+1:05d}.tif'
        cutout = cutouts[nel]
        transform = _get_transform_from_params(params)
        attrs = {'ID': params.Index}
        if labels is not None:
            attrs['LABEL'] = labels[params.Index]
        _write_cutout(path, cutout, crs, transform, attrs, overwrite)


def _write_cutout(
        path, cutout, crs, transform, attrs=None, overwrite=False
):
    """ Write out cutout to GeoTIFF. """

    if cutout.ndim == 3:
        data = cutout.transpose(2, 0, 1)
        dims = ('band', 'y', 'x')
    elif cutout.ndim == 2:
        data = cutout
        dims = ('y', 'x')
    else:
        raise ValueError('Expected 2 or 3 dimensions for cutouts')

    da = xr.DataArray(data=data, dims=dims)
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)

    if attrs is not None:
        da = da.rio.set_attrs(attrs)

    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already exists. Set overwrite=True')

    da.rio.to_raster(path, driver="GTiff", compress="LZW")


def _get_transform_from_params(transform_params):
    kwargs = {p: getattr(transform_params, p) for p in 'abcdef'}
    return affine.Affine(**kwargs)
