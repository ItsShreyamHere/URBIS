"""URBIS-Priorities: turn the slum map + risk + access layers into a ranked list of settlements.

One row per settlement (connected slum area >= 0.5 ha, the URBIS definition minimum):
  people      dasymetric estimate: AOI slum population (WorldPop) shared out by roof area
  flood_pct   % of its area < 5 m above the nearest water (URBIS-Risk)
  heat_c      mean surface-temperature anomaly vs the city (URBIS-Risk)
  health_m / school_m   median straight-line distance to the nearest facility (URBIS-Access)
  no_road_pct % of its roofs more than 50 m from a drivable road (street access, URBIS-Access)
  need        mean of five 0-1 scores: flood_pct/100, heat_c/4, health_m/2000, school_m/2000, no_road_pct/100
  priority    people x need   -> rank 1 = most people facing the most compounded risk

The weights are deliberately simple and printed in the output so a city can argue with them.
Not a list for enforcement: see docs/MODEL_CARD.md (prohibited use).
"""
import csv
import json

import numpy as np
import rasterio
import shapely
from rasterio.features import shapes
from rasterio.warp import transform_geom
from scipy import ndimage

MIN_HA = 0.5
ROAD_M = 50       # same as access.ROAD_M
CONFIRM = 0.95    # same as patches.CONFIRM


def table(prob, threshold, grid, bld_cover=None, risk=None, access=None, slum_pop=None, confidence=None):
    m = np.nan_to_num(prob) >= threshold
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    px_ha = abs(grid.transform.a * grid.transform.e) / 1e4
    area = ndimage.sum_labels(m, lab, np.arange(1, n + 1)) * px_ha
    keep = np.nonzero(area >= MIN_HA)[0] + 1
    lab = np.where(np.isin(lab, keep), lab, 0)
    idx = keep
    roof = ndimage.sum_labels(bld_cover if bld_cover is not None else m.astype(np.float32), lab, idx)
    share = roof / max(roof.sum(), 1e-9)
    mean = lambda a: ndimage.mean(a, lab, idx) if a is not None else np.full(len(idx), np.nan)
    med = lambda a: ndimage.median(a, lab, idx) if a is not None else np.full(len(idx), np.nan)
    risk, access = risk or {}, access or {}
    hand = risk.get("hand")
    flood = mean((hand < 5).astype(np.float32)) * 100 if hand is not None else np.full(len(idx), np.nan)
    heat = mean(np.nan_to_num(risk["heat_anom"])) if "heat_anom" in risk else np.full(len(idx), np.nan)
    health, school = med(access.get("dist_health")), med(access.get("dist_school"))
    road = access.get("dist_road")
    w = bld_cover if bld_cover is not None else np.ones(lab.shape, np.float32)
    no_road = (ndimage.sum_labels(w * (road > ROAD_M), lab, idx) / np.maximum(ndimage.sum_labels(w, lab, idx), 1e-9)
               * 100 if road is not None and np.isfinite(road).any() else np.full(len(idx), np.nan))
    comps = np.stack([flood / 100, heat / 4, health / 2000, school / 2000, no_road / 100], 1)
    need = np.nanmean(np.clip(comps, 0, 1), 1)
    conf = (ndimage.maximum(np.nan_to_num(confidence, nan=-1), lab, idx) if confidence is not None
            else np.full(len(idx), -1.0))
    people = share * slum_pop if slum_pop is not None else np.full(len(idx), np.nan)
    cy, cx = np.array(ndimage.center_of_mass(m, lab, idx)).T
    xs, ys = grid.transform * (cx + 0.5, cy + 0.5)
    lon, lat = rasterio.warp.transform(grid.crs, "EPSG:4326", xs, ys)
    rows = [dict(id=int(i), lon=round(lo, 5), lat=round(la, 5), area_ha=round(float(a), 1),
                 people=None if np.isnan(p) else int(round(p)), flood_pct=_r(f), heat_c=_r(h, 2),
                 health_m=_r(hm, 0), school_m=_r(sm, 0), no_road_pct=_r(nr), need=_r(nd, 3),
                 confidence=None if c < 0 else round(float(c), 2), confirmed=bool(c >= CONFIRM),
                 priority=None if np.isnan(p * nd) else round(float(p * nd)))
            for i, lo, la, a, p, f, h, hm, sm, nr, nd, c in zip(idx, lon, lat, area[idx - 1], people, flood, heat,
                                                                health, school, no_road, need, conf)]
    key = (lambda r: -(r["priority"] or 0)) if slum_pop is not None else (lambda r: -r["area_ha"] * (r["need"] or 0))
    rows.sort(key=key)
    for k, r in enumerate(rows, 1):
        r["rank"] = k
    return rows, lab


def _r(v, nd=1):
    return None if v is None or np.isnan(v) else round(float(v), nd)


def write(rows, lab, grid, out_csv, out_geojson):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank"] + [k for k in rows[0] if k != "rank"]) if rows else None
        if w:
            w.writeheader(); w.writerows(rows)
    by_id = {r["id"]: r for r in rows}
    feats = []
    for g, v in shapes(lab.astype(np.int32), mask=lab > 0, transform=grid.transform):
        if int(v) in by_id:
            geom = shapely.geometry.shape(g).simplify(abs(grid.transform.a), preserve_topology=True)
            feats.append(dict(type="Feature", properties=by_id[int(v)],
                              geometry=transform_geom(grid.crs, "EPSG:4326", shapely.geometry.mapping(geom))))
    with open(out_geojson, "w") as f:
        json.dump(dict(type="FeatureCollection", features=feats), f)


if __name__ == "__main__":
    # self-check: two settlements; the bigger, flood-prone one ranks first
    from rasterio.transform import from_origin
    from .s2 import Grid
    g = Grid("EPSG:32643", from_origin(500000, 2000000, 10, 10), 40, 40)
    p = np.zeros((40, 40), np.float32); p[2:14, 2:14] = 0.9; p[25:37, 25:29] = 0.9
    hand = np.full((40, 40), 10, np.float32); hand[:20] = 1
    rows, lab = table(p, 0.5, g, risk=dict(hand=hand), slum_pop=1000)
    assert [r["area_ha"] for r in rows] == [1.4] and rows[0]["flood_pct"] == 100, rows   # 0.48 ha one dropped
    assert rows[0]["people"] == 1000
    road = np.full((40, 40), 100, np.float32); road[:, :8] = 0                   # half the big one is on a road
    rows, _ = table(p, 0.5, g, access=dict(dist_road=road))
    assert rows[0]["no_road_pct"] == 50.0, rows[0]
    rows, _ = table(p, 0.5, g, confidence=np.full((40, 40), 0.95, np.float32))
    assert rows[0]["confidence"] == 0.95 and rows[0]["confirmed"], rows[0]
    print("ok", rows[0])
