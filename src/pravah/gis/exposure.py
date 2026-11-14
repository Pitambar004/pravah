"""Exposure intersections from a model-derived water polygon (not damage)."""

import geopandas as gpd


def summarize_exposure(water: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame | None,
                       roads: gpd.GeoDataFrame | None,
                       bridges: gpd.GeoDataFrame | None = None,
                       projected_crs: str = "EPSG:32645") -> dict:
    if water.crs is None:
        raise ValueError("water needs an explicit CRS")
    w = water.to_crs(projected_crs)
    w = w[~w.geometry.is_empty & w.geometry.notna()]
    if w.empty:
        target = None
        area = 0.0
    else:
        target = w.geometry.union_all()
        area = float(target.area)

    def prepped(layer):
        if layer is None:
            return None
        if layer.crs is None:
            raise ValueError("OSM source missing CRS")
        frame = layer.to_crs(projected_crs)
        return frame[~frame.geometry.is_empty & frame.geometry.notna()]

    b, r, br = prepped(buildings), prepped(roads), prepped(bridges)
    if target is None:
        return {"water_area_m2": area,
                "building_footprints_intersecting": 0 if b is not None else None,
                "road_length_m_intersecting": 0.0 if r is not None else None,
                "bridge_tagged_features_intersecting": 0 if br is not None else None}
    # Distinct records only; exact asset dedup must be handled by OSM extraction.
    b_count = int(b.geometry.intersects(target).sum()) if b is not None else None
    road_length = float(r.geometry.intersection(target).length.sum()) if r is not None else None
    bridge_count = int(br.geometry.intersects(target).sum()) if br is not None else None
    return {"water_area_m2": area, "building_footprints_intersecting": b_count,
            "road_length_m_intersecting": road_length,
            "bridge_tagged_features_intersecting": bridge_count}
