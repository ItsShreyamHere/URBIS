"""People living in mapped slums (SDG 11.1.1 numerator) via the WorldPop stats API — no raster download.

WorldPop 2020 100 m unconstrained counts. Known bias: dasymetric models spread people by building
cover and tend to UNDER-count very dense slums, so treat the figure as a lower-bound estimate.
"""
import json
import time

import numpy as np
import rasterio
import requests
import shapely
from rasterio.features import shapes
from rasterio.warp import transform_geom

API = "https://api.worldpop.org/v1/services/stats"


def _json(text):
    # the API wraps its JSON in PHP deprecation notices; decode the first JSON object only
    return json.JSONDecoder().raw_decode(text[text.index("{\n"):])[0]


def _query(geom_4326, year=2020, attempts=3):
    """Population inside ONE polygon (the API rejects multi-geometries). Retries transient failures."""
    for i in range(attempts):
        try:
            return _query_once(geom_4326, year)
        except (requests.RequestException, ValueError) as e:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def _query_once(geom_4326, year):
    fc = dict(type="FeatureCollection",
              features=[dict(type="Feature", properties={}, geometry=shapely.geometry.mapping(geom_4326))])
    r = requests.post(API, data=dict(dataset="wpgppop", year=year, geojson=json.dumps(fc), runasync="false"),
                      timeout=300)
    r.raise_for_status()
    d = _json(r.text)
    for _ in range(100):                                   # may finish asynchronously
        if "data" in d or d.get("error"):
            break
        time.sleep(3)
        d = _json(requests.get(f"https://api.worldpop.org/v1/tasks/{d['taskid']}", timeout=60).text)
    if d.get("error") or "data" not in d:
        raise RuntimeError(f"WorldPop: {d.get('error_message', d)}")
    return float(d["data"]["total_population"])


def inside(polys, bbox):
    """Total population inside many polygons with 2 requests: pop(box) - pop(box with polygons as holes).
    ponytail: islands inside a slum polygon count as slum; fine at 10 m mapping resolution."""
    box = shapely.box(*bbox)
    u = shapely.unary_union([shapely.Polygon(p.exterior) for p in polys]).intersection(box.buffer(-1e-6))
    parts = [g for g in getattr(u, "geoms", [u]) if g.geom_type == "Polygon" and not g.is_empty]
    total = _query(box)
    if not parts:
        return 0.0, total
    outside = _query(shapely.Polygon(box.exterior, [p.exterior for p in parts]))
    return max(total - outside, 0.0), total


def slum_polygons(prob_path, threshold, min_ha=1.0):
    with rasterio.open(prob_path) as s:
        p = s.read(1)
        m = ((p != 255) & (p / 250 >= threshold)).astype(np.uint8)
        polys = [shapely.geometry.shape(g) for g, v in shapes(m, mask=m.astype(bool), transform=s.transform) if v]
        crs, res = s.crs, abs(s.transform.a)
    polys = [g.simplify(res, preserve_topology=True) for g in polys if g.area >= min_ha * 1e4]
    return [shapely.geometry.shape(transform_geom(crs, "EPSG:4326", shapely.geometry.mapping(g))) for g in polys]


def estimate(prob_path, threshold, aoi_bbox_4326):
    """dict(slum_pop, aoi_pop, slum_pop_share_pct). Network errors propagate to the caller."""
    slum, aoi = inside(slum_polygons(prob_path, threshold), aoi_bbox_4326)
    return dict(slum_pop=round(slum), aoi_pop=round(aoi), slum_pop_share_pct=round(100 * slum / max(aoi, 1), 1))
