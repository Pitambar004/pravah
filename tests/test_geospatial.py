import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from pravah.gis.exposure import summarize_exposure


def test_projected_exposure_intersections():
    water = gpd.GeoDataFrame(geometry=[Polygon([(0,0),(10,0),(10,10),(0,10)])], crs="EPSG:32645")
    buildings = gpd.GeoDataFrame(geometry=[Polygon([(8,8),(12,8),(12,12),(8,12)]), Polygon([(15,15),(16,15),(16,16),(15,16)])], crs="EPSG:32645")
    roads = gpd.GeoDataFrame(geometry=[LineString([(-5,5),(15,5)])], crs="EPSG:32645")
    res = summarize_exposure(water, buildings, roads)
    assert res["water_area_m2"] == pytest.approx(100)
    assert res["building_footprints_intersecting"] == 1
    assert res["road_length_m_intersecting"] == pytest.approx(10)
    assert res["bridge_tagged_features_intersecting"] is None
