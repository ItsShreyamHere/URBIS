"""URBIS command line.

  urbis map W,S,E,N [--place "Nairobi"]     slum map + people + flood/heat + access + ranked settlements
  urbis change W,S,E,N [--from 2019 --to 2024]   where settlements appeared / were cleared
  urbis calibrate runs/<x>/slum_prob.tif (--share 0.12 | --points pts.geojson)   adapt the threshold locally
  urbis adapt runs/<x>/slum_prob.tif --labels areas.geojson   fine-tune on the city's own labelled areas
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

from . import buildings, model, profile, report, s2

MAX_KM2 = 5000   # ponytail: one in-memory composite per run; tile the AOI if bigger cities are needed
W = Path(__file__).parent / "weights"
M = W / "no_mumbai"   # stage-1 stack trained without Mumbai: Mumbai (SRA) is the benchmark city, never trained on


def geocode(place):
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params=dict(q=place, format="json", limit=1),
                     headers={"User-Agent": "urbis-1 slum mapping (research)"}, timeout=30)
    r.raise_for_status()
    if not r.json():
        raise SystemExit(f"could not geocode {place!r}; pass a bbox instead")
    s, n, w, e = map(float, r.json()[0]["boundingbox"])
    return w, s, e, n


def aoi(ap, a):
    if not (a.place or a.bbox):
        ap.error("give a bbox (west,south,east,north) or --place")
    try:
        bbox = tuple(map(float, a.bbox.split(","))) if a.bbox else geocode(a.place)
    except ValueError:
        ap.error("bbox must be four numbers: west,south,east,north")
    if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        ap.error("bbox must be west,south,east,north")
    grid = s2.Grid.from_bbox(*bbox)
    km2 = grid.height * grid.width / 1e4
    if km2 > MAX_KM2:
        raise SystemExit(f"AOI is {km2:.0f} km2 (> {MAX_KM2}); pass a smaller bbox")
    name = a.place.lower().replace(" ", "_").replace(",", "") if a.place else \
        "bbox_" + "_".join(f"{v:.3f}" for v in bbox)
    print(f"AOI {bbox} -> {grid.width}x{grid.height} px ({km2:.0f} km2), {grid.crs}")
    return bbox, grid, name


def predict(comp, bld, out, tree_path, patch_check=True, booster=None, tree_w=None):
    """URBIS-1: tree model, averaged with the U-Net when its weights ship (combination chosen out-of-fold).
    booster / tree_w: locally adapted trees and their ensemble share (urbis adapt) in place of the shipped trees."""
    booster, meta = (booster, model.load(tree_path)[1]) if booster is not None else model.load(tree_path)
    shape = bld.with_name("shape.tif") if bld is not None and meta["buildings"] == 2 else None
    seg = M / "urbis1_seg.pt"
    if not seg.exists() or not meta["buildings"] or Path(tree_path).name != "urbis1.txt":
        model.predict(booster, comp, out, bld, shape)
        return meta["threshold"]
    from . import unet
    sm = json.loads(seg.with_suffix(".json").read_text())
    tree_out = out.with_name("prob_trees.tif")
    model.predict(booster, comp, tree_out, bld, shape)
    unet.predict(unet.load(seg), comp, out.with_name("prob_seg.tif"), bld)
    g, u = unet.read_prob(tree_out), unet.read_prob(out.with_name("prob_seg.tif"))
    from . import patches
    p = np.sqrt(g * u) if sm["combine"] == "ensemble_geo" else {"trees": g, "unet": u}.get(sm["combine"], patches.ensemble(g, u, tree_w))
    so = M / "s2only_urbis1.txt"
    if so.exists():
        # field screen: imagery-only trees decide where Overture has no footprints (urbis/patches.screen)
        from . import patches
        model.predict(model.load(so)[0], comp, out.with_name("prob_s2only.tif"))
        sc = patches.screen(p, unet.read_prob(out.with_name("prob_s2only.tif")), buildings.load(bld)[..., 1])
        unet.write_prob(sc, comp, out.with_name("prob_screen.tif"), "URBIS-1 field screen")
    pm = M / "urbis1_patch.txt"
    if patch_check and pm.exists() and sm["combine"] == "ensemble":
        # second stage (urbis/patches.py): patch scores written alongside; the map itself is the raw ensemble
        import lightgbm as lgb
        from . import patches
        unet.write_prob(p, comp, out.with_name("prob_raw.tif"), "URBIS-1 ensemble before the patch check")
        shp = bld.with_name("shape.tif")
        if not shp.exists():
            print(f"{buildings.fetch(s2.Grid.from_raster(comp), bld, only_shape=True)} footprints for shape (Overture Maps)")
        lab, X = patches.features(g, u, buildings.load(bld), buildings.load_shape(shp), s2.load(comp)[0])
        cut = json.loads((M / "patch_results.json").read_text())["cut"]
        score = lgb.Booster(model_file=str(pm)).predict(X) if len(X) else np.zeros(0)
        # ponytail: scores kept as information, no longer applied. Without the label-edge leak the check lowers
        # unseen-city AP (out of fold 0.557 -> 0.540, validation 0.604 -> 0.565; docs/VALIDATION.md)
        unet.write_prob(np.concatenate([[np.nan], score]).astype(np.float32)[lab], comp,
                        out.with_name("patch_score.tif"), "URBIS-1 patch check score (value/250)")
        print(f"patch check (not applied): {int((score < cut).sum())} of {len(score)} candidate patches score below {cut}")
    unet.write_prob(p, comp, out, "URBIS-1 ensemble")
    return sm["threshold"]


def cmd_map(ap, a):
    bbox, grid, name = aoi(ap, a)
    out = Path(a.out or f"runs/{name}/slum_prob.tif")
    out.parent.mkdir(parents=True, exist_ok=True)
    comp = out.parent / "s2.tif"
    if comp.exists() and s2.Grid.from_raster(comp).bounds == grid.bounds:
        print(f"reusing {comp}")
    else:
        items = s2.search(grid, a.start, a.end, n=6)
        if not items:
            raise SystemExit("no clear Sentinel-2 scenes found for this AOI/date range")
        print(f"{len(items)} Sentinel-2 scenes")
        s2.save(comp, grid, s2.composite(grid, items))
    bld = None
    level = model.load(a.model)[1]["buildings"]
    if level:
        bld = out.parent / "bld.tif"
        if not bld.exists() or s2.Grid.from_raster(bld).bounds != grid.bounds:
            print(f"{buildings.fetch(grid, bld)} building footprints (Overture Maps)")
        elif level == 2 and not bld.with_name("shape.tif").exists():
            print(f"{buildings.fetch(grid, bld, only_shape=True)} building footprints for shape (Overture Maps)")
    th = predict(comp, bld, out, a.model, not a.no_patch_check)
    prof = None if a.map_only else profile.run(out, th, bbox)
    if a.living:
        from . import living
        print(f"wrote {living.write(out.parent, *living.score(out.parent)[:2])}  (URBIS-Living, 1 km cells)")
    html, st = report.build(out, th, title=a.place, bbox=bbox, profile=prof)
    print(f"wrote {out}  (value/250 = slum probability)")
    print(f"wrote {html}  slum area {st['slum_km2']} km2 ({st['slum_share_pct']} %), "
          f"{st['settlements']} settlements, ~{st.get('slum_pop', '?')} people")


def cmd_change(ap, a):
    from . import change
    bbox, _, name = aoi(ap, a)
    out = Path(a.out or f"runs/{name}/change")
    print(json.dumps(change.run(bbox, a.y0, a.y1, out), indent=2))
    print(f"wrote {out / 'change.tif'}")


def threshold_for_share(prob, share):
    v = prob[~np.isnan(prob)]
    return float(np.quantile(v, 1 - min(max(share, 1e-4), 0.999)))


def threshold_from_points(prob_path, points_path):
    """Best-F1 threshold on labelled points: GeoJSON Points with property slum = 1 / 0."""
    import rasterio
    from pyproj import Transformer
    from . import unet
    prob = unet.read_prob(prob_path)
    feats = json.loads(Path(points_path).read_text())["features"]
    with rasterio.open(prob_path) as f:
        tr = Transformer.from_crs("EPSG:4326", f.crs, always_xy=True)
        xy = [tr.transform(*ft["geometry"]["coordinates"][:2]) for ft in feats]
        rc = [f.index(x, y) for x, y in xy]
    p, y = [], []
    for (r, c), ft in zip(rc, feats):
        if 0 <= r < prob.shape[0] and 0 <= c < prob.shape[1] and not np.isnan(prob[r, c]):
            p.append(prob[r, c]); y.append(bool(int(ft["properties"]["slum"])))
    p, y = np.array(p), np.array(y)
    if len(p) < 20 or y.sum() < 3:
        raise SystemExit(f"need >= 20 labelled points inside the map with >= 3 slum points (got {len(p)}, {y.sum()})")
    ts = np.linspace(0.02, 0.98, 97)
    f1 = [2 * np.sum((p >= t) & y) / max(np.sum(p >= t) + y.sum(), 1) for t in ts]
    return float(ts[int(np.argmax(f1))]), len(p)


def cmd_calibrate(ap, a):
    from . import unet
    prob_path = Path(a.prob)
    if a.share is not None:
        th = threshold_for_share(unet.read_prob(prob_path), a.share)
        how = f"matches a slum share of {a.share:.1%} of the mapped area"
    elif a.points:
        th, n = threshold_from_points(prob_path, a.points)
        how = f"best F1 on {n} labelled points"
    else:
        ap.error("give --share or --points")
    print(f"calibrated threshold {th:.3f} ({how})")
    (prob_path.parent / "calibration.json").write_text(json.dumps(dict(threshold=th, method=how), indent=2))
    prof = None if a.map_only else profile.run(prob_path, th, None)
    html, st = report.build(prob_path, th, title=a.title, profile=prof)
    print(f"wrote {html}  slum area {st['slum_km2']} km2, ~{st.get('slum_pop', '?')} people")


ADAPT_W = 0.85     # adapted trees' share; scripts/adapt_eval.py: mean AP 0.66 -> 0.79, cities >= 0.9: 5 -> 14 of 31 (1.0: 0.80, 14)
MIN_SLUM_PX, MIN_REST_PX, MAX_PX = 50, 200, 300_000


def local_labels(prob_path, labels_path):
    """GeoJSON polygons with property slum = 1 (known settlement) or 0 (checked, not a settlement), burned onto the
    map grid: 1, 0, 255 = unlabelled. Slum wins where polygons overlap."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    feats = json.loads(Path(labels_path).read_text())["features"]
    with rasterio.open(prob_path) as f:
        shape, tr, crs = f.shape, f.transform, f.crs
    lab = np.full(shape, 255, np.uint8)
    for v in (0, 1):
        geoms = [transform_geom("EPSG:4326", crs, ft["geometry"]) for ft in feats
                 if int(ft["properties"]["slum"]) == v]
        if geoms:
            lab = np.where(rasterize(geoms, shape, transform=tr, fill=0, dtype="uint8") == 1, v, lab)
    return lab


def tune(comp, bld, lab, rounds, tree_path):
    """Continue training the shipped trees on lab (1 slum, 0 not, 255 unlabelled) -> (booster, labels used)."""
    import lightgbm as lgb
    from . import features
    base, meta = model.load(tree_path)
    sc = model.Scene(comp, bld if meta["buildings"] else None)
    rng = np.random.default_rng(0)
    n = int((lab != 255).sum())
    keep_p = min(1.0, MAX_PX / max(n, 1))
    X, Y = [], []
    for rd, crop, wr in features.tiles(sc.height, sc.width):
        y = lab[wr]
        k = (y != 255) & (rng.random(y.shape) < keep_p)
        if k.any():
            f = sc.features(rd, crop)
            k &= ~np.isnan(f).any(-1)
            X.append(f[k]); Y.append(y[k])
    sc.close()
    X, Y = np.concatenate(X), np.concatenate(Y)
    pos = Y.mean()
    return lgb.train(dict(model.PARAMS, scale_pos_weight=(1 - pos) / max(pos, 1e-6)), lgb.Dataset(X, Y), rounds,
                     init_model=base, keep_training_booster=True), Y


def cmd_adapt(ap, a):
    """Fine-tune the shipped trees on a city's own labelled areas, then redraw the map with them."""
    prob = Path(a.prob)
    d = prob.parent
    comp, bld = d / "s2.tif", d / "bld.tif"
    lab = local_labels(prob, a.labels)
    ns, nr = int((lab == 1).sum()), int((lab == 0).sum())
    if ns < MIN_SLUM_PX or nr < MIN_REST_PX:
        raise SystemExit(f"need >= {MIN_SLUM_PX} slum and >= {MIN_REST_PX} non-slum pixels (10 m) inside the map; "
                         f"got {ns} and {nr}")
    tree_path = M / "urbis1.txt"
    tuned, Y = tune(comp, bld, lab, a.rounds, tree_path)
    tuned.save_model(str(d / "urbis1_local.txt"))
    print(f"adapted on {len(Y)} labelled pixels ({int(Y.sum())} slum); trees saved to {d / 'urbis1_local.txt'}")
    th = predict(comp, bld, prob, str(tree_path), not a.no_patch_check, booster=tuned, tree_w=ADAPT_W)
    (d / "adaptation.json").write_text(json.dumps(dict(labels=str(a.labels), slum_px=ns, rest_px=nr, rounds=a.rounds,
                                                       tree_share=ADAPT_W), indent=2))
    prof = None if a.map_only else profile.run(prob, th, None)
    html, st = report.build(prob, th, title=a.title, profile=prof)
    print(f"wrote {html}  slum area {st['slum_km2']} km2, {st['settlements']} settlements, ~{st.get('slum_pop', '?')} people")


def main():
    import functools, builtins
    builtins.print = functools.partial(builtins.print, flush=True)  # show progress when piped to a log
    ap = argparse.ArgumentParser(prog="urbis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_aoi(p):
        p.add_argument("bbox", nargs="?", help="west,south,east,north in degrees (lon/lat)")
        p.add_argument("--place", help="geocode a place name instead of giving a bbox")
        p.add_argument("--bbox", dest="bbox_opt", help=argparse.SUPPRESS)
        return p

    m = with_aoi(sub.add_parser("map", help="slum map, people, risk, access and ranked settlements for a bbox"))
    m.add_argument("--out", help="output GeoTIFF (default runs/<name>/slum_prob.tif)")
    m.add_argument("--model", default=str(M / "urbis1.txt"))
    m.add_argument("--start", default="2023-01-01")
    m.add_argument("--end", default="2024-12-31")
    m.add_argument("--map-only", action="store_true", help="skip population, flood/heat and access layers")
    m.add_argument("--no-patch-check", action="store_true",
                   help="keep every candidate patch (higher recall, more false alarms); see prob_raw.tif")
    m.add_argument("--living", action="store_true",
                   help="also rank 1 km cells by relative deprivation (URBIS-Living; for cities with few slums)")
    c = with_aoi(sub.add_parser("change", help="settlement change between two years (imagery-only model)"))
    c.add_argument("--from", dest="y0", type=int, default=2019)
    c.add_argument("--to", dest="y1", type=int, default=2024)
    c.add_argument("--out", help="output folder (default runs/<name>/change)")
    k = sub.add_parser("calibrate", help="set a city-specific threshold from local knowledge and rebuild the report")
    k.add_argument("prob", help="slum_prob.tif from urbis map")
    k.add_argument("--share", type=float, help="known slum share of the mapped area, 0-1")
    k.add_argument("--points", help="GeoJSON points with property slum = 1 or 0")
    k.add_argument("--title")
    k.add_argument("--map-only", action="store_true")

    j = sub.add_parser("adapt", help="fine-tune the model on the city's own labelled areas and redraw the map")
    j.add_argument("prob", help="slum_prob.tif from urbis map")
    j.add_argument("--labels", required=True, help="GeoJSON polygons with property slum = 1 or 0")
    j.add_argument("--rounds", type=int, default=150)
    j.add_argument("--title")
    j.add_argument("--map-only", action="store_true")
    j.add_argument("--no-patch-check", action="store_true")

    # a bbox starting with a minus sign (western hemisphere) would otherwise be read as an option
    argv = [f"--bbox={t}" if re.fullmatch(r"-[\d.]+(,-?[\d.]+){3}", t) else t for t in sys.argv[1:]]
    a = ap.parse_args(argv)
    if hasattr(a, "bbox_opt"):
        a.bbox = a.bbox or a.bbox_opt
    dict(map=cmd_map, change=cmd_change, calibrate=cmd_calibrate, adapt=cmd_adapt)[a.cmd](ap, a)


if __name__ == "__main__":
    main()
