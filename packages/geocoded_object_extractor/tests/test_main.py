import affine
import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from shapely import Point, Polygon
from geocoded_object_extractor.main import ObjectExtractor, _normalize_geoms


def _check(geometry):
    assert isinstance(geometry, gpd.GeoSeries)
    assert (geometry.geom_type == 'Polygon').all()


def test__normalize_geoms_with_points_without_buffer():
    coords = [Point(xi, xi) for xi in range(10)]
    with pytest.raises(ValueError):
        _normalize_geoms(coords)


def test__normalize_geoms_with_list_of_points():
    pts = [Point(xi, xi) for xi in range(10)]
    geometry = _normalize_geoms(pts, buffer=0.1)
    _check(geometry)


def test__normalize_geoms_with_list_of_coords():
    coords = [(xi, xi) for xi in range(10)]
    geometry = _normalize_geoms(coords, buffer=0.1)
    _check(geometry)


def test__normalize_geoms_with_list_of_polygons():
    coords = [
        Polygon([(0, 0), (1, 1), (2, 2), (0, 0)]),
        Polygon([(1, 1), (2, 2), (3, 3), (1, 1)])
    ]
    geometry = _normalize_geoms(coords)
    _check(geometry)


def test__normalize_geoms_with_list_of_bounds():
    coords = [(0, 0, 1, 1), (1, 1, 2, 2), (2, 2, 3, 3)]
    geometry = _normalize_geoms(coords)
    _check(geometry)


def test__normalize_geoms_with_geoseries():
    coords = [Point(xi, xi) for xi in range(10)]
    geoseries = gpd.GeoSeries(coords).buffer(0.1)
    geometry = _normalize_geoms(geoseries)
    assert (geometry == geoseries).all()
    _check(geometry)


def _setup_tiles():
    data_tile_1 = np.array([[
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
    ]])
    data_tile_2 = np.array([[
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]])
    transform_tile_1 = affine.Affine(a=1, b=0, c=-0.5, d=0, e=-1, f=5.5)
    transform_tile_2 = affine.Affine(a=1, b=0, c=2.5, d=0, e=-1, f=5.5)
    tile_1 = xr.DataArray(data=data_tile_1, dims=('band', 'y', 'x'))
    tile_1 = tile_1.rio.write_transform(transform_tile_1)
    tile_1 = tile_1.rio.write_crs('EPSG:4326')
    tile_2 = xr.DataArray(data=data_tile_2, dims=('band', 'y', 'x'))
    tile_2 = tile_2.rio.write_transform(transform_tile_2)
    tile_2 = tile_2.rio.write_crs('EPSG:4326')
    return tile_1, tile_2


def test_get_cutouts_handles_identical_objects_with_same_id():
    tiles = _setup_tiles()
    geoms = [(4, 2)]  # center of the "ones", in xy coord
    buffer = 1.8  # radius to create a 3x3 cutout - must be sqrt(2) < r < 1
    oe = ObjectExtractor(images=tiles, geoms=geoms, buffer=buffer)
    labels, transform_params, _, cutouts = oe.get_cutouts()
    assert len(labels) == len(geoms)
    assert len(transform_params) == len(geoms)
    assert cutouts.shape == (len(geoms), 3, 3, 1)
    assert np.all(cutouts == 1)


def test_get_cutouts_handles_different_objects_with_same_id():
    tiles = _setup_tiles()
    # for tile 1, the following coords and buffer will generate a cutout 3x2
    geoms = [(5, 2)]
    buffer = 1.8
    oe = ObjectExtractor(images=tiles, geoms=geoms, buffer=buffer)
    labels, transform_params, _, cutouts = oe.get_cutouts()
    assert len(labels) == len(geoms)
    assert len(transform_params) == len(geoms)
    assert cutouts.shape == (len(geoms), 3, 3, 1)
    assert np.allclose(
        np.squeeze(cutouts),
        np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ])
    )
