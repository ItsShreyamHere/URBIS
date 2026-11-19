"""URBIS settlement outlines + insights for a city reviewed with city_pipeline.py (2026-11-14).
  .venv/bin/python scripts/city_outlines.py <city> [model.pt]      (default model runs/outline/ho1/m34.pt; SIM=1 also grades vs official map)
For every group of confirmed slum boxes (8-connected "yes" answers): a 1.28 km Esri window around its central box, the click-guided
outline model (scripts/outline_train.py) draws the slum, keeping the parts connected to confirmed boxes. Overlapping windows are merged.
Writes runs/pipeline/<city>/settlements.geojson (boundary polygons, lon/lat) with per-settlement area_ha, buildings (Overture Maps
footprints inside the outline), buildings_per_ha, confirmed_boxes, lat/lon and a Google Maps link, and report.html (city summary).
Building counts come from footprint maps, which merge packed roofs: in an Accra check against community counts (SDI Know Your City,
9 settlements) they were 0.55x the community's structure count, so read them as a lower bound."""
import html, json, os, sys
from pathlib import Path
import duckdb, numpy as np, torch
from pyproj import Geod, Transformer
from scipy import ndimage as nd
from shapely import contains_xy
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely import transform as _tf
stransform = lambda f, g: _tf(g, lambda a: np.c_[f(a[:, 0], a[:, 1])])
from rasterio import features as rf
import rasterio
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("PRETRAINED", "0")   # inference: weights come from the checkpoint
import city_pipeline as cp, outline_data as od, outline_train as ot
CACHE = Path("runs/outline_city/cache"); CACHE.mkdir(parents=True, exist_ok=True)

def outlines(c, C, A, ck):
    m = None; t = float(os.environ.get("T", .6))
    H, W = C["H"], C["W"]; yes = np.zeros(H * W, bool); yes[[b for b, v in A.items() if v]] = True
    comp, n = nd.label(yes.reshape(H, W), np.ones((3, 3))); g = C["g"]; ym = np.kron(yes.reshape(H, W), np.ones((10, 10), np.uint8))
    to_ll = Transformer.from_crs(3857, 4326, always_xy=True); to_city = Transformer.from_crs(g.crs, 4326, always_xy=True); polys = []
    done = np.zeros((H, W), bool)                        # confirmed boxes already inside the central 800 m of a processed window
    for k in range(1, n + 1):
      while True:
        r, q = np.nonzero((comp == k) & ~done)
        if not len(r): break
        i = np.argmin((r - r.mean()) ** 2 + (q - q.mean()) ** 2); r0, q0 = r[i], q[i]; done[max(r0-4, 0):r0+5, max(q0-4, 0):q0+5] = True
        lon, lat = to_city.transform(*rasterio.transform.xy(g.transform, r0 * 10 + 5, q0 * 10 + 5)); b = od.window(lon, lat)
        sd = np.zeros_like(ym); sd[r0*10:r0*10+10, q0*10:q0*10+10] = 1
        seed = od.on_grid(sd, g, b, 0).astype(bool); conf = od.on_grid(ym, g, b, 0).astype(bool)
        cf = CACHE / f"{c}_{r0}_{q0}_{ck.get('tag', ck['enc'])}.npy"        # probability maps cached: threshold sweeps skip imagery + inference
        if cf.exists(): P = np.load(cf)
        else:
            im = od.render(b)
            if im is None: continue
            if m is None: ot.ENC = ck["enc"]; m = ot.Net().to(ot.DEV); m.load_state_dict(ck["sd"]); m.eval()   # only on a cache miss
            x = np.concatenate([(np.asarray(im, np.float32) / 255 - ot.MEAN) / ot.STD, np.kron(seed, np.ones((4, 4), np.float32))[..., None]], -1).transpose(2, 0, 1)
            with torch.no_grad():
                X = torch.from_numpy(np.ascontiguousarray(x))[None].to(ot.DEV)
                P = (sum(torch.rot90(torch.sigmoid(m(torch.rot90(X, j, (2, 3)))), -j, (1, 2)) for j in range(4)) / 4)[0].cpu().numpy()
            np.save(cf, P.astype(np.float16))
        P = P.astype(np.float32)
        R = P > t; cl, _ = nd.label(R, np.ones((3, 3))); ids = np.unique(cl[(seed | conf) & R]); R = np.isin(cl, ids[ids > 0]) if (ids > 0).any() else seed
        tr = rasterio.transform.from_bounds(*b, R.shape[1], R.shape[0])
        polys += [stransform(lambda x, y, z=None: to_ll.transform(x, y), shape(s).simplify(5 / np.cos(np.radians(lat)))) for s, v in rf.shapes(R.astype(np.uint8), mask=R, transform=tr)]
        print(f"  settlement group {k}/{n}: {int(R.sum())} px", flush=True)
    from shapely.geometry import Polygon                 # confirmed boxes are human-verified slum: always part of the map
    boxes = [Polygon(cp.corners(C, b)) for b, v in A.items() if v]
    if float(os.environ.get("BUF", 100)):                  # keep the outline only within BUF metres of a confirmed box
        near = unary_union(boxes).buffer(float(os.environ.get("BUF", 100)) / 111e3); polys = [p.buffer(0).intersection(near) for p in polys]
    polys += boxes
    d = 10 / 111e3                                       # smooth staircase pixel/box edges into boundary lines: close 10 m, simplify 4 m
    u = unary_union([p.buffer(0) for p in polys]).buffer(d, join_style="round").buffer(-d, join_style="round").simplify(4 / 111e3); return list(getattr(u, "geoms", [u])) if not u.is_empty else []

def buildings(c, P):
    """Overture building footprints (Google Open Buildings + Microsoft + OSM) around the outlines, cached per city as
    runs/pipeline/<city>/buildings.parquet (+ .json bounds). Returns, per outline, its footprints whose centre lies inside: [(polygon, source)]."""
    import shapely
    p = Path(f"runs/pipeline/{c}/buildings.parquet"); bj = p.with_suffix(".json"); x0, y0, x1, y1 = unary_union(P).bounds
    d = duckdb.connect(); d.sql("INSTALL spatial; LOAD spatial;")
    if not p.exists() or not bj.exists() or not (lambda q: q[0] <= x0 and q[1] <= y0 and q[2] >= x1 and q[3] >= y1)(json.loads(bj.read_text())):
        x0, y0, x1, y1 = x0 - .01, y0 - .01, x1 + .01, y1 + .01; d.sql("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';"); rel = os.environ.get("REL", "2026-10-21.0")
        d.sql(f"""COPY (select (bbox.xmin+bbox.xmax)/2 x, (bbox.ymin+bbox.ymax)/2 y, ST_AsWKB(geometry) wkb, sources[1].dataset src
            from read_parquet('s3://overturemaps-us-west-2/release/{rel}/theme=buildings/type=building/*', hive_partitioning=1)
            where bbox.xmin > {x0} and bbox.xmax < {x1} and bbox.ymin > {y0} and bbox.ymax < {y1}) TO '{p}' (FORMAT parquet)""")
        bj.write_text(json.dumps([x0, y0, x1, y1]))
    x, y, w, src = d.sql(f"select x, y, wkb, src from '{p}'").fetchnumpy().values(); out = []
    for s in P:
        k = np.flatnonzero(contains_xy(s, x, y)); out.append([(shapely.from_wkb(bytes(w[i])), src[i]) for i in k])
    return out

def report(c, F, out, grade):
    tot = lambda k: sum(f["properties"][k] for f in F); fs = sorted(F, key=lambda f: -f["properties"]["area_ha"])
    rows = "".join(f"<tr><td>{i+1}</td><td>{p['area_ha']:.1f}</td><td>{p['buildings']:,}</td><td>{p['buildings_per_ha']:.0f}</td><td>{p['confirmed_boxes']}</td>"
                   f"<td><a href=\"{html.escape(p['google_maps'])}\">{p['lat']:.5f}, {p['lon']:.5f}</a></td></tr>" for i, p in enumerate(f["properties"] for f in fs))
    big = [f["properties"]["area_ha"] for f in fs]; top = sum(big[:max(1, len(big) // 10)]) / max(sum(big), 1e-9)
    (out / "report.html").write_text(f"""<!doctype html><meta charset=utf-8><title>URBIS {html.escape(c)}</title>
<style>body{{font:15px/1.5 system-ui;margin:2rem auto;max-width:56rem;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}td,th{{padding:4px 8px;border-bottom:1px solid #ddd;text-align:right}}td:last-child,th:last-child{{text-align:left}}.k{{display:flex;gap:2rem;flex-wrap:wrap}}.k b{{font-size:1.6rem;display:block}}</style>
<h1>Slums of {html.escape(c.replace('_', ' ').title())}</h1>
<div class=k><div><b>{len(F)}</b>settlements</div><div><b>{tot('area_ha'):.0f} ha</b>slum land</div><div><b>{tot('buildings'):,}+</b>buildings (lower bound)</div>
<div><b>{100*top:.0f} %</b>of slum land in the largest 10 % of settlements</div></div>
<p>{grade}</p><p>Outlines: AI-drawn around boxes the city's officials confirmed (URBIS review). Building counts: Overture Maps footprints inside each outline;
footprint maps merge packed roofs and undercounted community counts by about half in an Accra check, so treat them as a minimum.
Files: settlements.kml (Google Earth), settlements.geojson (QGIS / GIS), slums.tif.</p>
<table><tr><th>#</th><th>Area (ha)</th><th>Buildings</th><th>per ha</th><th>Confirmed boxes</th><th>Location</th></tr>{rows}</table>""")

def kml(c, F, out):
    """settlements.kml for Google Earth: red boundary lines, figures in each popup."""
    pm = []
    for i, f in enumerate(F):
        p = f["properties"]; g = f["geometry"]; rings = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        geo = "".join("<Polygon><outerBoundaryIs><LinearRing><coordinates>" + " ".join(f"{x:.6f},{y:.6f},0" for x, y in r[0]) + "</coordinates></LinearRing></outerBoundaryIs>"
                      + "".join("<innerBoundaryIs><LinearRing><coordinates>" + " ".join(f"{x:.6f},{y:.6f},0" for x, y in h) + "</coordinates></LinearRing></innerBoundaryIs>" for h in r[1:]) + "</Polygon>" for r in rings)
        pm.append(f"<Placemark><name>Settlement {i+1}</name><description>{p['area_ha']} ha, at least {p['buildings']} buildings ({p['buildings_per_ha']}/ha), "
                  f"{p['confirmed_boxes']} confirmed boxes</description><styleUrl>#s</styleUrl><MultiGeometry>{geo}</MultiGeometry></Placemark>")
    (out / "settlements.kml").write_text('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f'<name>{html.escape(c)} slum settlements (URBIS)</name><Style id="s"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle><PolyStyle><color>330000ff</color></PolyStyle></Style>'
        + "".join(pm) + "</Document></kml>")

def main(c, model="runs/kaggle_outline/out1/mvit.pt"):
    C = cp.load(c); A = cp.answers(c); out = Path(os.environ.get("OUT", f"runs/pipeline/{c}")); out.mkdir(parents=True, exist_ok=True); ck = torch.load(model, map_location="cpu", weights_only=False)
    P = outlines(c, C, A, ck); G = Geod(ellps="WGS84"); P = sorted(P, key=lambda s: -abs(G.geometry_area_perimeter(s)[0])); hs = buildings(c, P)
    n = [len(h) for h in hs]; F = []
    for s, k in zip(P, n):
        a = abs(G.geometry_area_perimeter(s)[0]) / 1e4; pt = s.representative_point()
        F.append({"type": "Feature", "geometry": mapping(s), "properties": {"settlement": len(F) + 1, "area_ha": round(a, 2), "buildings": k, "buildings_per_ha": round(k / max(a, 1e-6), 1),
                  "confirmed_boxes": 0, "lat": round(pt.y, 6), "lon": round(pt.x, 6), "google_maps": f"https://www.google.com/maps/@{pt.y:.6f},{pt.x:.6f},400m/data=!3m1!1e3"}})
    lo, la = cp.lonlat(C, [b for b, v in A.items() if v])
    for f, s in zip(F, P): f["properties"]["confirmed_boxes"] = int(contains_xy(s.buffer(.0005), lo, la).sum())
    (out / "settlements.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": F})); grade = ""
    H = []                                               # every house: its footprint polygon, numbered north to south, west to east, within its settlement
    for f, h in zip(F, hs):
        h = sorted(h, key=lambda b: (-round(b[0].centroid.y, 4), b[0].centroid.x))
        H += [{"type": "Feature", "geometry": mapping(g), "properties": {"settlement": f["properties"]["settlement"], "house": i + 1,
               "area_m2": round(abs(G.geometry_area_perimeter(g)[0]), 1), "source": str(sr)}} for i, (g, sr) in enumerate(h)]
    (out / "houses.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": H}))
    if os.environ.get("SIM"):    # vs the official 10 m map, same measures as city_pipeline export
        g = C["g"]; L = g.read(1); tr = Transformer.from_crs(4326, g.crs, always_xy=True)
        M = rf.rasterize([stransform(lambda x, y, z=None: tr.transform(x, y), s) for s in P], out_shape=L.shape, transform=g.transform) > 0 if P else np.zeros(L.shape, bool)
        sl, kn = L == 1, L != 255; bt = Path(f"runs/pipeline/{c}/slums.tif"); box = rasterio.open(bt).read(1) > 0 if bt.exists() else None
        grade = (f"Check against the official map: outlines hold {100*M[sl].mean():.1f} % of mapped slum land and {100*sl[M & kn].mean():.1f} % of the outlined land is mapped slum"
                 + (f" (100 m box map: {100*box[sl].mean():.1f} % and {100*sl[box & kn].mean():.1f} %)." if box is not None else "."))
        print(grade)
    report(c, F, out, grade); kml(c, F, out); print(f"{c}: {len(F)} settlements, {sum(f['properties']['area_ha'] for f in F):.0f} ha, {sum(n):,} buildings -> {out}/settlements.geojson, report.html")

if __name__ == "__main__": main(*sys.argv[1:])
