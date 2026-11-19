"""Click-guided outline data (2026-11-14): 1.28 km Esri windows (z16, 512 px = 2.5 m/px) centred near a confirmed slum box, with the
official 10 m slum mask resampled onto the same grid (128 px). Runs anywhere with rasterio + requests (built for the Windows box).
  python outline_data.py <data_dir> <out_dir> <cities.txt> <per_city> [held]
Writes <out>/<city>_<k>.jpg (RGB 512) and <out>/<city>_<k>.npz: y (128x128 uint8: 1 slum, 0 not, 255 unknown), seed (128x128 bool:
the confirmed 100 m box), sid (128x128 bool: the seeded settlement), meta (lon, lat, merc bounds). held=1: one window per settlement
(>= 0.1 ha, seed = its box with most slum); else up to per_city random settlements (>= 0.5 ha), seed = a random box with >= 20 % slum,
window centre shifted up to 300 m from the seed."""
import hashlib, io, json, math, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, rasterio, requests
from rasterio.warp import reproject, Resampling
from PIL import Image
from scipy import ndimage as nd

URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"; BLANK = "f27d9de7f80c13501f470595e327aa6d"
SPAN, PX, LP, R = 1280, 512, 128, 6378137.0
S = requests.Session(); S.headers["User-Agent"] = "urbis-research"

def tile(k):
    z, x, y = k
    for _ in range(4):
        try:
            b = S.get(URL.format(z=z, x=x, y=y), timeout=30); b.raise_for_status(); b = b.content
            return k, None if hashlib.md5(b).hexdigest() == BLANK else b
        except Exception: pass
    return k, None

def merc(lon, lat): return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

def window(lon, lat):
    """Mercator bounds of a SPAN-metre ground window (mercator stretch 1/cos(lat))."""
    mx, my = merc(lon, lat); h = SPAN / 2 / math.cos(math.radians(lat)); return mx - h, my - h, mx + h, my + h

def render(b, z=16):
    W = 2 * math.pi * R; n = 2 ** z * 256; px = lambda mx, my: ((mx + W / 2) / W * n, (W / 2 - my) / W * n)
    x0, y0 = px(b[0], b[3]); x1, y1 = px(b[2], b[1]); ks = [(z, tx, ty) for tx in range(int(x0) // 256, int(x1) // 256 + 1) for ty in range(int(y0) // 256, int(y1) // 256 + 1)]
    with ThreadPoolExecutor(12) as ex: got = dict(ex.map(tile, ks))
    if any(v is None for v in got.values()): return render(b, z - 1) if z > 15 else None
    ox, oy = int(x0) // 256 * 256, int(y0) // 256 * 256; mos = Image.new("RGB", ((int(x1) // 256 - int(x0) // 256 + 1) * 256, (int(y1) // 256 - int(y0) // 256 + 1) * 256))
    for (_, tx, ty), v in got.items(): mos.paste(Image.open(io.BytesIO(v)).convert("RGB"), (tx * 256 - ox, ty * 256 - oy))
    return mos.crop((x0 - ox, y0 - oy, x1 - ox, y1 - oy)).resize((PX, PX), Image.BILINEAR)

def on_grid(arr, src, b, nodata):
    """Nearest-neighbour sample of a raster on the window's LP x LP mercator grid (pixel centres)."""
    from pyproj import Transformer
    t = (np.arange(LP) + .5) / LP; mx = b[0] + t * (b[2] - b[0]); my = b[3] - t * (b[3] - b[1]); MX, MY = np.meshgrid(mx, my)
    X, Y = Transformer.from_crs(3857, src.crs, always_xy=True).transform(MX, MY); r, c = rasterio.transform.rowcol(src.transform, X, Y)
    r, c = np.asarray(r).reshape(LP, LP), np.asarray(c).reshape(LP, LP); ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.full((LP, LP), nodata, arr.dtype); out[ok] = arr[r[ok], c[ok]]; return out

def main(data, out, cities, per, held=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(0)
    for c in Path(cities).read_text().split():
        g = rasterio.open(f"{data}/{c}/labels.tif"); L = g.read(1); H, W = L.shape[0] // 10, L.shape[1] // 10
        B = L[:H*10, :W*10].reshape(H, 10, W, 10); px = (B == 1).sum((1, 3)); comp, nc = nd.label(L == 1, np.ones((3, 3))); sz = np.bincount(comp.ravel())
        ids = [k for k in range(1, nc + 1) if sz[k] >= (10 if held else 50)]
        if not held: ids = list(rng.permutation(ids)[:int(per)])
        from pyproj import Transformer
        tr = Transformer.from_crs(g.crs, 4326, always_xy=True); n = 0
        for k in ids:
            if (out / f"{c}_{k}.npz").exists(): n += 1; continue
            ys, xs = np.nonzero(comp[:H*10, :W*10] == k)
            if not len(ys): continue
            cnt = np.bincount((ys // 10) * W + xs // 10, minlength=H * W)
            if held: sb = int(cnt.argmax())
            else:
                ok = np.flatnonzero((cnt > 0) & (px.ravel() >= 20)); sb = int(rng.choice(ok)) if len(ok) else int(cnt.argmax())
            y0, x0 = divmod(sb, W); cx, cy = rasterio.transform.xy(g.transform, y0 * 10 + 5, x0 * 10 + 5)
            dx, dy = (0, 0) if held else rng.uniform(-300, 300, 2); lon, lat = tr.transform(cx + dx, cy + dy); b = window(lon, lat)
            im = render(b)
            if im is None: continue
            seedm = np.zeros(L.shape, np.uint8); seedm[y0*10:y0*10+10, x0*10:x0*10+10] = 1
            y = on_grid(L, g, b, 255); seed = on_grid(seedm, g, b, 0).astype(bool); sid = on_grid((comp == k).astype(np.uint8), g, b, 0).astype(bool)
            im.save(out / f"{c}_{k}.jpg", quality=90); np.savez_compressed(out / f"{c}_{k}.npz", y=y, seed=seed, sid=sid, meta=np.array([lon, lat, *b]))
            n += 1
        print(c, n, "windows", flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:5], held=len(sys.argv) > 5 and sys.argv[5] == "1")
