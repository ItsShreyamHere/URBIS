"""Set up a brand-new city for the URBIS app from a bounding box (2026-11-14). No labels of the city are used.
  .venv/bin/python scripts/new_city.py <city_id> <lon0> <lat0> <lon1> <lat1>
Steps: 1 grid (UTM, 10 m pixels, 100 m review boxes) -> data/<id>/labels.tif (all 255 = unknown)
       2 building footprints (Overture: Google + Microsoft + OSM) -> runs/pipeline/<id>/buildings.parquet; built boxes = >= 1 building
       3 slum score of every built box with the round-3 two-view model (scripts/score_local.py views, on this machine)
         -> runs/app_cities/<id>.npz (sq, score, feat); city_pipeline.load() then treats the city like any scored city.
Progress: runs/app_cities/<id>.status.json {step, done, total, msg, error}."""
import json, math, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np, rasterio
from pyproj import Transformer
sys.path.insert(0, str(Path(__file__).parent))
import score_local as S

OUT = Path("runs/app_cities"); MAX_KM2 = 600

def status(cid, **k):
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / f"{cid}.status.json"
    s = json.loads(p.read_text()) if p.exists() else {}; s.update(k, t=time.time()); p.write_text(json.dumps(s))

def grid(cid, b):
    lon, lat = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    tr = Transformer.from_crs(4326, epsg, always_xy=True); xs, ys = tr.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
    x0, y0, x1, y1 = math.floor(min(xs) / 100) * 100, math.floor(min(ys) / 100) * 100, math.ceil(max(xs) / 100) * 100, math.ceil(max(ys) / 100) * 100
    assert (x1 - x0) * (y1 - y0) / 1e6 <= MAX_KM2, f"area {(x1 - x0) * (y1 - y0) / 1e6:.0f} km2 > {MAX_KM2} km2: draw a smaller box"
    W, H = int(x1 - x0) // 10, int(y1 - y0) // 10; d = Path(f"data/{cid}"); d.mkdir(parents=True, exist_ok=True)
    pr = dict(driver="GTiff", width=W, height=H, count=1, dtype="uint8", crs=f"EPSG:{epsg}", nodata=255, compress="deflate",
              transform=rasterio.transform.from_origin(x0, y1, 10, 10))
    with rasterio.open(d / "labels.tif", "w", **pr) as o: o.write(np.full((H, W), 255, np.uint8), 1)
    (d / "meta.json").write_text(json.dumps({"role": "app", "bbox": b, "epsg": epsg}))
    return rasterio.open(d / "labels.tif")

def buildings(cid, b, g):
    import duckdb
    p = Path(f"runs/pipeline/{cid}/buildings.parquet"); p.parent.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = b[0] - .01, b[1] - .01, b[2] + .01, b[3] + .01
    if not p.exists():
        d = duckdb.connect(); d.sql("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
        d.sql(f"""COPY (select (bbox.xmin+bbox.xmax)/2 x, (bbox.ymin+bbox.ymax)/2 y, ST_AsWKB(geometry) wkb, sources[1].dataset src
            from read_parquet('s3://overturemaps-us-west-2/release/2026-10-21.0/theme=buildings/type=building/*', hive_partitioning=1)
            where bbox.xmin > {x0} and bbox.xmax < {x1} and bbox.ymin > {y0} and bbox.ymax < {y1}) TO '{p}' (FORMAT parquet)""")
        p.with_suffix(".json").write_text(json.dumps([x0, y0, x1, y1]))
    import duckdb as dd
    x, y = dd.sql(f"select x, y from '{p}'").fetchnumpy().values()
    X, Y = Transformer.from_crs(4326, g.crs, always_xy=True).transform(x, y); r, q = rasterio.transform.rowcol(g.transform, X, Y)
    r, q = np.asarray(r) // 10, np.asarray(q) // 10; H, W = g.height // 10, g.width // 10; k = (r >= 0) & (r < H) & (q >= 0) & (q < W)
    return np.bincount(r[k] * W + q[k], minlength=H * W), len(x)

_fetch = S.tile

@lru_cache(maxsize=20000)
def tile_bytes(z, x, y):
    im = _fetch(z, x, y)
    if im is None: return None
    import io; b = io.BytesIO(); im.save(b, "JPEG", quality=92); return b.getvalue()

def _tile(z, x, y):                                       # neighbouring boxes share imagery tiles: cache them (compressed)
    import io; from PIL import Image; v = tile_bytes(z, x, y); return None if v is None else Image.open(io.BytesIO(v)).convert("RGB")

S.tile = _tile

def score(cid, g, sq):
    import torch
    W = g.width // 10; r, q = np.divmod(sq, W); X, Y = rasterio.transform.xy(g.transform, r * 10 + 5, q * 10 + 5)
    lon, lat = Transformer.from_crs(g.crs, 4326, always_xy=True).transform(np.asarray(X), np.asarray(Y))
    m, head, MEAN, STD = S.net(); dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"); m, head = m.to(dev), head.to(dev)
    tens = lambda im: torch.from_numpy(((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    sc, ft = [], []; B = 128
    with ThreadPoolExecutor(24) as ex:
        nxt = ex.submit(lambda: list(ex.map(lambda i: (S.view(lon[i], lat[i], 18, 160), S.view(lon[i], lat[i], 16, 640)), range(0, min(B, len(sq))))))
        for b in range(0, len(sq), B):
            ims = nxt.result()
            if b + B < len(sq): nxt = ex.submit(lambda s=b + B: list(ex.map(lambda i: (S.view(lon[i], lat[i], 18, 160), S.view(lon[i], lat[i], 16, 640)), range(s, min(s + B, len(sq))))))
            with torch.no_grad():
                for k in range(0, len(ims), 32):
                    cc = torch.stack([tens(a) for a, _ in ims[k:k + 32]]).to(dev); ww = torch.stack([tens(w) for _, w in ims[k:k + 32]]).to(dev)
                    f = torch.cat([m(cc), m(ww)], 1); sc.append(torch.sigmoid(head(f)[:, 0]).cpu().numpy()); ft.append(f.half().cpu().numpy())
            status(cid, step="score", done=min(b + B, len(sq)), total=len(sq), msg=f"Scoring built boxes with the slum model: {min(b + B, len(sq)):,} of {len(sq):,}")
    return np.concatenate(sc), np.concatenate(ft)

def main(cid, *b):
    b = [float(v) for v in b]
    try:
        status(cid, step="grid", done=0, total=1, msg="Preparing the 100 m grid", error=None, bbox=b); g = grid(cid, b)
        status(cid, step="buildings", done=0, total=1, msg="Downloading building footprints (Overture Maps); a few minutes")
        n, nb = buildings(cid, b, g); sq = np.flatnonzero(n >= 1)
        assert len(sq) >= 50, f"only {len(sq)} boxes with buildings in this area: draw a box over a built-up city"
        status(cid, step="score", done=0, total=len(sq), msg=f"{nb:,} buildings, {len(sq):,} built 100 m boxes; starting the slum model")
        s, f = score(cid, g, sq)
        np.savez(OUT / f"{cid}.npz", sq=sq, score=s, feat=f, count=n[sq])
        status(cid, step="ready", done=len(sq), total=len(sq), msg=f"Ready: {len(sq):,} built boxes scored, {nb:,} buildings")
    except Exception as e:
        status(cid, step="error", error=str(e), msg=f"Failed: {e}"); traceback.print_exc(); raise

if __name__ == "__main__": main(*sys.argv[1:])
