import geopandas as gpd
import pytest

from shapely import Point, Polygon
from geocoded_object_extractor.main import _normalize_geoms


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
