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
                (ncutouts, ny, nx, 3)

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
    return skimage.transform.rotate(
        img,
        angle,
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
        cutouts, crs, affine_params, labels=None, outdir='./', overwrite=False
):
    """
    Write cutouts extracted using the ObjectExtractor to disk as a set
    of GeoTIFF files. If labels are provided, these are stored as a "LABEL"
    attribute in the GeoTIFFs.

    Args:
        cutouts: sequence of cutouts, as a np.ndarray with shape
            (ncutouts, ny, nx, 3)
        crs: coordinate reference system employed, as a pyproj.CRS object.
        affine_params: affine transformation parameters, as a pd.DataFrame
            with shape (ncutouts, 6). We use the affine naming for the
            transformation params: https://affine.readthedocs.io/en/latest/
        labels (optional): labels, as a pd.Series with shape (ncutouts,)
        outdir (optional): output directory
        overwrite (optional): if files already exist, overwrite them
    """
    dir = pathlib.Path(outdir)
    dir.mkdir(exist_ok=True, parents=True)

    for nel, params in enumerate(affine_params.itertuples()):
        path = dir / f'cutout-{nel:05d}.tif'
        cutout = cutouts[nel]
        label = labels[params.Index] if labels is not None else None
        affine_transform = _get_affine_transform_from_params(params)
        _write_cutout(path, cutout, crs, affine_transform, label, overwrite)


def _write_cutout(
        path, cutout, crs, affine_transform, label=None, overwrite=False
):
    """ Write out cutout to GeoTIFF. """
    da = xr.DataArray(data=cutout)
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(affine_transform)

    if label is not None:
        da = da.rio.set_attrs({'LABEL': label})

    if path.isfile() and not overwrite:
        raise FileExistsError(f'{path} already exists. Set overwrite=True')

    da.rio.to_raster(path, driver="GTiff", compress="LZW")


def _get_affine_transform_from_params(affine_params):
    kwargs = {p: getattr(affine_params, p) for p in 'abcdef'}
    return affine.Affine(**kwargs)
