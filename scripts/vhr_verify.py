"""Automatic verifier: a CNN fine-tuned on Esri World Imagery (~0.3-0.6 m) judges each 100 m square the 10 m ensemble
flags. Goal: >= 80 % of slums found with >= 95 % of what is marked correct, no human.
Splits: CNN trained on 13 Brazilian training cities; fusion + threshold chosen on held-out cities (tune13.HELD);
validation cities scored once.
  python scripts/vhr_verify.py chips     # training chips (stored as jpg)
  python scripts/vhr_verify.py train
  python scripts/vhr_verify.py score     # held + validation candidates (streamed, scores only)
  python scripts/vhr_verify.py eval      -> urbis/weights/vhr_verify.json
"""
import hashlib, io, json, math, os, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from scipy import ndimage

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).parent)]
try:
    from review_eval import held, val, y_of  # noqa: E402  (training only; not shipped)
except ImportError:
    held = val = y_of = None
try:
    from tune13 import HELD  # noqa: E402  (training only; not shipped)
    from validation import cities  # noqa: E402
except ImportError:
    HELD, cities = [None] * 13, None

OUT = Path("runs/vhr_verify")
TRAIN = ["recife", "rio", "salvador", "sao_paulo", "sao_luis", "teresina", "natal", "maceio", "joao_pessoa",
         "goiania", "curitiba", "campinas", "vitoria"]
TUNE = HELD[:7]          # ponytail: 7 of 13 held cities keep download time sane; use all 13 if fusion looks noisy
SQ, CTX, PX, Z = 10, int(os.environ.get("CTX", 160)), int(os.environ.get("PX", 224)), 18   # square 10 px = 100 m, 160 m of imagery around it (CTX env: wider view), 224 px chip
CAND = 0.05              # candidate threshold on the 10 m ensemble
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
BLANK = "f27d9de7f80c13501f470595e327aa6d"
MB = [0.0]


_TL = threading.local()


def _session():   # one keep-alive connection pool per thread: no TLS handshake per tile
    if not hasattr(_TL, "s"):
        import requests
        _TL.s = requests.Session(); _TL.s.headers["User-Agent"] = "urbis-research"
    return _TL.s


def get(k):
    z, x, y = k
    for i in range(4):
        try:
            r = _session().get(URL.format(z=z, x=x, y=y), timeout=30); r.raise_for_status(); b = r.content
            MB[0] += len(b) / 1e6
            return k, None if hashlib.md5(b).hexdigest() == BLANK else b
        except Exception:
            if i == 3:
                return k, None


class Tiles:
    def __init__(self):
        self.c = {}

    def fetch(self, keys):
        keys = [k for k in set(keys) if k not in self.c]
        with ThreadPoolExecutor(int(os.environ.get("TILE_THREADS", 16))) as ex:
            for k, b in ex.map(get, keys):
                self.c[k] = b

    def img(self, k):
        if k not in self.c:
            self.fetch([k])
        b = self.c[k]
        return None if b is None else Image.open(io.BytesIO(b)).convert("RGB")


def gpx(lon, lat, z):
    n = 2 ** z * 256
    return (lon + 180) / 360 * n, (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n


def window(lon, lat, z):
    mpp = 156543.03 * math.cos(math.radians(lat)) / 2 ** z
    gx, gy = gpx(lon, lat, z)
    h = CTX / mpp / 2
    return int(gx - h), int(gy - h), int(gx + h), int(gy + h)


def keys_for(lon, lat, z=Z):
    x0, y0, x1, y1 = window(lon, lat, z)
    return [(z, tx, ty) for tx in range(x0 // 256, x1 // 256 + 1) for ty in range(y0 // 256, y1 // 256 + 1)]


def prefetch(T, pts):
    """Fetch every tile render() may need for these points, in parallel: the fallback zooms only where the finer zoom
    has a blank or failed tile (render() would otherwise fetch them one at a time)."""
    need = list(pts)
    for z in (Z, Z - 1, Z - 2):
        T.fetch([k for p in need for k in keys_for(p[0], p[1], z)])
        need = [p for p in need if any(T.c[k] is None for k in keys_for(p[0], p[1], z))]


def render(T, lon, lat):
    for z in (Z, Z - 1, Z - 2):
        x0, y0, x1, y1 = window(lon, lat, z)
        ims = {k: T.img(k) for k in keys_for(lon, lat, z)}
        if any(v is None for v in ims.values()):
            continue
        mos = Image.new("RGB", ((x1 // 256 - x0 // 256 + 1) * 256, (y1 // 256 - y0 // 256 + 1) * 256))
        for (_, tx, ty), im in ims.items():
            mos.paste(im, ((tx - x0 // 256) * 256, (ty - y0 // 256) * 256))
        ox, oy = x0 // 256 * 256, y0 // 256 * 256
        return mos.crop((x0 - ox, y0 - oy, x1 - ox, y1 - oy)).resize((PX, PX), Image.BILINEAR)
    return None


def grid(c):
    """Per 100 m square: slum share, valid share, centre lon/lat."""
    y = y_of(c)
    H, W = y.shape[0] // SQ * SQ, y.shape[1] // SQ * SQ
    b = lambda a: a[:H, :W].reshape(H // SQ, SQ, W // SQ, SQ).mean((1, 3))
    v = b(y != 255)
    fr = b(y == 1) / np.maximum(v, 1e-6)
    with rasterio.open(f"data/{c}/labels.tif") as f:
        Tr, crs = f.transform, f.crs
    r, cc = np.mgrid[:H // SQ, :W // SQ]
    X, Y = Tr * ((cc + .5) * SQ, (r + .5) * SQ)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(X, Y)
    return fr, v, lon, lat, (H, W)


def sample(c, per=2400, rng=None):
    """Training squares for one city: slum / mixed / formal-near-slum / formal-elsewhere strata."""
    fr, v, lon, lat, _ = grid(c)
    with rasterio.open(f"data/{c}/bld.tif") as f:
        bl = f.read(1).astype(np.float32)
    H, W = fr.shape[0] * SQ, fr.shape[1] * SQ
    bld = np.nan_to_num(bl[:H, :W]).reshape(fr.shape[0], SQ, fr.shape[1], SQ).mean((1, 3))
    ok = v > .5
    near = ndimage.binary_dilation(fr >= .5, iterations=10)          # within 1 km of a slum
    built = bld > np.quantile(bld[ok], .5)
    strata = [(ok & (fr >= .5), .45), (ok & (fr > .01) & (fr < .5), .15),
              (ok & (fr <= .01) & near & built, .25), (ok & (fr <= .01) & ~near & built, .15)]
    pick = []
    for m, s in strata:
        idx = np.flatnonzero(m)
        pick += list(rng.choice(idx, min(len(idx), int(per * s)), replace=False))
    return pick, fr, lon, lat


def chips(per=2400, seed=0, cities=None):
    rng = np.random.default_rng(seed)
    for c in cities or TRAIN:
        d = OUT / "chips" / c
        if (d / "meta.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        pick, fr, lon, lat = sample(c, per, rng)
        T = Tiles()
        T.fetch([k for i in pick for k in keys_for(lon.flat[i], lat.flat[i])])
        meta = []
        for j, i in enumerate(pick):
            im = render(T, lon.flat[i], lat.flat[i])
            if im is None:
                continue
            im.save(d / f"{j}.jpg", quality=85)
            meta.append(dict(f=f"{j}.jpg", sq=int(i), frac=float(fr.flat[i])))
        (d / "meta.json").write_text(json.dumps(meta))
        print(c, len(meta), "chips, MB so far", round(MB[0]), flush=True)


# ---------------------------------------------------------------- model
def net(arch="convnext_tiny.fb_in22k_ft_in1k"):
    import timm
    return timm.create_model(arch, pretrained=True, num_classes=1)


def tens(ims):
    import torch
    a = np.stack([np.asarray(i, np.float32) / 255 for i in ims])
    a = (a - [.485, .456, .406]) / [.229, .224, .225]
    return torch.from_numpy(a.transpose(0, 3, 1, 2).astype(np.float32))


def train(epochs=5, bs=48):
    import torch
    dev = "mps"
    items = [(OUT / "chips" / c / m["f"], m["frac"]) for c in TRAIN
             for m in json.loads((OUT / "chips" / c / "meta.json").read_text())]
    print(len(items), "chips", flush=True)
    m = net().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=.05)
    steps = epochs * (len(items) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=steps, pct_start=.1)
    rng = np.random.default_rng(0)
    for e in range(epochs):
        m.train(); order = rng.permutation(len(items)); tot = 0
        for b in range(len(items) // bs):
            ims, ys = [], []
            for i in order[b * bs:(b + 1) * bs]:
                im = Image.open(items[i][0]).convert("RGB")
                k = rng.integers(8)
                im = im.transpose(Image.FLIP_LEFT_RIGHT) if k & 4 else im
                im = im.rotate(90 * (k & 3))
                ims.append(im); ys.append(items[i][1])
            x = tens(ims).to(dev)
            x = x * float(rng.uniform(.8, 1.2)) + float(rng.uniform(-.2, .2))   # brightness / contrast jitter
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                m(x)[:, 0], torch.tensor(ys, dtype=torch.float32, device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item()
            if b % 100 == 0:
                print(e, b, round(tot / (b + 1), 4), flush=True)
        torch.save(m.state_dict(), OUT / "verifier.pt")


def score(names=None):
    import torch
    m = net(); m.load_state_dict(torch.load(OUT / "verifier.pt", map_location="cpu")); m = m.to("mps").eval()
    for c in names or TUNE + cities():
        f = OUT / "scores" / f"{c}.npz"
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        p = held(c) if c in HELD else val(c)
        fr, v, lon, lat, (H, W) = grid(c)
        pm = np.nan_to_num(p[:H, :W], nan=0).reshape(fr.shape[0], SQ, fr.shape[1], SQ)
        pmax, pmean = pm.max((1, 3)), pm.mean((1, 3))
        idx = np.flatnonzero(pmax >= CAND)
        T = Tiles()
        T.fetch([k for i in idx for k in keys_for(lon.flat[i], lat.flat[i])])
        s = np.full(fr.size, np.nan, np.float32)
        for b in range(0, len(idx), 64):
            ii = idx[b:b + 64]
            ims = [render(T, lon.flat[i], lat.flat[i]) for i in ii]
            okk = [k for k, im in enumerate(ims) if im is not None]
            if okk:
                with torch.no_grad():
                    x = tens([ims[k] for k in okk]).to("mps")
                    lg = sum(m(torch.rot90(x, r, (2, 3)))[:, 0] for r in range(4)) / 4   # 4-rotation TTA
                s[ii[okk]] = lg.cpu().numpy()
        np.savez_compressed(f, cnn=s.reshape(fr.shape), pmax=pmax, pmean=pmean, fr=fr, v=v)
        print(c, len(idx), "squares, MB so far", round(MB[0]), flush=True)
        del T


# ---------------------------------------------------------------- evaluation
def metrics(keep, c):
    """keep: accepted 100 m squares. -> square precision, pixel precision, slum-area recall, community recall."""
    y = y_of(c)
    H, W = keep.shape[0] * SQ, keep.shape[1] * SQ
    k = np.zeros(y.shape, bool); k[:H, :W] = np.kron(keep, np.ones((SQ, SQ), bool))
    slum, v = y == 1, y != 255
    d = np.load(OUT / "scores" / f"{c}.npz")
    fr, vv = d["fr"], d["v"]
    sq_prec = float((keep & (fr >= .5)).sum() / max(keep.sum(), 1))
    lab, n = ndimage.label(slum, np.ones((3, 3)))
    size = np.bincount(lab.ravel(), minlength=n + 1); cov = np.bincount(lab.ravel(), k.ravel(), n + 1)
    big = size >= 100; big[0] = False
    return dict(sq_prec=round(sq_prec, 3), px_prec=round(float((k & slum).sum() / max((k & v).sum(), 1)), 3),
                area_rec=round(float((k & slum).sum() / max(slum.sum(), 1)), 3),
                comm_rec=round(float((cov[big] / size[big] >= .5).mean()), 3), squares=int(keep.sum()))


def feats(d):
    lg = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    return np.stack([np.nan_to_num(d["cnn"], nan=-20), lg(d["pmax"]), lg(d["pmean"])], -1)


def evaluate():
    from sklearn.linear_model import LogisticRegression
    X, Y = [], []
    for c in TUNE:
        d = np.load(OUT / "scores" / f"{c}.npz"); m = ~np.isnan(d["cnn"]) & (d["v"] > .5)
        X.append(feats(d)[m]); Y.append(d["fr"][m] >= .5)
    lr = LogisticRegression(C=1.0).fit(np.concatenate(X), np.concatenate(Y))
    res = dict(coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(), tune={}, validation={})

    def run(c, th, use="fused"):
        d = np.load(OUT / "scores" / f"{c}.npz"); ok = ~np.isnan(d["cnn"])
        s = lr.predict_proba(feats(d).reshape(-1, 3))[:, 1].reshape(ok.shape) if use == "fused" else \
            1 / (1 + np.exp(-np.nan_to_num(d["cnn"], nan=-20)))
        return metrics(ok & (s >= th), c)

    THS = [.5, .6, .7, .8, .85, .9, .93, .95, .97, .98]
    for use in ("fused", "cnn"):
        for th in THS:
            r = [run(c, th, use) for c in TUNE]
            res["tune"][f"{use}_{th}"] = {k: round(float(np.mean([x[k] for x in r])), 3) for k in r[0]}
            print("tune", use, th, res["tune"][f"{use}_{th}"], flush=True)
    # pick: highest community recall whose held-out mean square precision >= 0.95
    ok = [k for k, g in res["tune"].items() if g["sq_prec"] >= .95]
    best = max(ok, key=lambda k: res["tune"][k]["comm_rec"]) if ok else max(res["tune"], key=lambda k: res["tune"][k]["sq_prec"])
    use, th = best.split("_")[0], float(best.split("_")[1])
    res["chosen"] = dict(use=use, th=th)
    print("chosen", use, th, flush=True)
    for c in cities():
        res["validation"][c] = run(c, th, use); print(c, res["validation"][c], flush=True)
    Path("urbis/weights/vhr_verify.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"chips": chips, "train": train, "score": score, "eval": evaluate}[sys.argv[1]]()
