"""URBIS city profile: slum map -> people, flood/heat exposure, access to services, ranked settlements.

Every stage is optional and fails soft (network down, no water in the AOI, no facilities mapped): the
slum map is never lost because a supplementary layer could not be fetched.
"""
import numpy as np
import rasterio

import csv
import json

from . import access, patches, plan, population, priorities, risk, s2, unet


def _soft(name, fn, log):
    try:
        return fn()
    except Exception as e:  # supplementary layer: report and carry on
        log(f"warning: {name} skipped ({type(e).__name__}: {e})")
        return None


def run(prob_path, threshold, bbox, log=print):
    d = prob_path.parent
    r, grid = s2.load(d / "s2.tif")
    bbox = bbox or grid.lonlat_bbox()
    with rasterio.open(prob_path) as f:
        a = f.read(1)
    prob = np.where(a == 255, np.nan, a / 250).astype(np.float32)
    bld = None
    if (d / "bld.tif").exists():
        with rasterio.open(d / "bld.tif") as f:
            bld = f.read(1).astype(np.float32) / 250

    pop = _soft("population (WorldPop)", lambda: population.estimate(prob_path, threshold, bbox), log) or {}
    rk = _soft("flood/heat (Copernicus DEM, Landsat)", lambda: risk.layers(grid, r, log), log) or {}
    ac, counts = _soft("access (Overture places)", lambda: access.layers(grid, log), log) or ({}, {})
    del r

    slum, rest = prob >= threshold, prob < 0.2
    summ = dict(pop)
    if "hand" in rk:
        summ["flood_slum_pct"] = round(100 * float(np.mean(rk["hand"][slum] < risk.LOW_M)), 1) if slum.any() else None
        summ["flood_rest_pct"] = round(100 * float(np.mean(rk["hand"][rest] < risk.LOW_M)), 1)
    if "heat_anom" in rk and np.isfinite(rk["heat_anom"]).any():
        summ["heat_slum_c"] = round(float(np.nanmean(rk["heat_anom"][slum])), 2) if slum.any() else None
        summ["heat_rest_c"] = round(float(np.nanmean(rk["heat_anom"][rest])), 2)
    for k in ("health", "school"):
        v = ac.get(f"dist_{k}")
        if v is not None and np.isfinite(v).any():
            summ[f"{k}_slum_m"] = round(float(np.median(v[slum]))) if slum.any() else None
            summ[f"{k}_rest_m"] = round(float(np.median(v[rest])))
    v = ac.get("dist_road")
    if v is not None and np.isfinite(v).any():                     # share of roofs beyond road reach
        w = bld if bld is not None else np.ones(v.shape, np.float32)
        far = lambda m: round(100 * float((w[m] * (v[m] > access.ROAD_M)).sum() / max(w[m].sum(), 1e-9)), 1)
        summ["no_road_slum_pct"] = far(slum) if slum.any() else None
        summ["no_road_rest_pct"] = far(rest)
    summ["facilities"] = counts

    conf = unet.read_prob(d / "patch_score.tif") if (d / "patch_score.tif").exists() else None
    rows, lab = priorities.table(prob, threshold, grid, bld, rk, ac, pop.get("slum_pop"), conf)
    if conf is not None:
        summ["confirmed_settlements"] = sum(r["confirmed"] for r in rows)
        summ["confirmed_ha"] = round(sum(r["area_ha"] for r in rows if r["confirmed"]), 1)
    raw = next((unet.read_prob(d / f) for f in ("prob_screen.tif", "prob_raw.tif") if (d / f).exists()), prob)
    ok = ~np.isnan(raw)
    summ["screen_pct"] = round(100 * float((raw[ok] >= patches.SCREEN).mean()), 1) if ok.any() else None
    if rows:
        priorities.write(rows, lab, grid, d / "settlements.csv", d / "settlements.geojson")
        log(f"wrote {d / 'settlements.csv'} ({len(rows)} settlements ranked)")
    if pop.get("slum_pop") and ac:
        people = plan.people_raster(slum, bld, pop["slum_pop"])
        summ["plan"], sites = {}, []
        for k in ("health", "school"):
            if ac.get(f"dist_{k}") is not None:
                rs, unserved = plan.sites(people, ac[f"dist_{k}"], grid)
                summ["plan"][k] = dict(unserved=unserved, sites=rs)
                sites += [dict(kind=k, **r) for r in rs]
        if sites:
            with open(d / "plan.csv", "w", newline="") as f:
                w = csv.DictWriter(f, sites[0].keys()); w.writeheader(); w.writerows(sites)
            (d / "plan.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=[
                dict(type="Feature", properties=s_, geometry=dict(type="Point", coordinates=[s_["lon"], s_["lat"]]))
                for s_ in sites])))
            log(f"wrote {d / 'plan.csv'} (new {', '.join(summ['plan'])} sites, {plan.REACH_M} m reach)")
    return summ, rows
