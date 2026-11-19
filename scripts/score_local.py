"""Score a random sample of one new city's built squares on the Mac with the round-3 two-view model (no Kaggle).
  .venv/bin/python scripts/score_local.py hyderabad [n=2000]  -> runs/city_review/<city>/{pts.json, meta.npz, scores.npy}
Views and preprocessing copied from runs/kaggle_retrain3/kernel/kaggle_twoview.py (160 m z18 + 640 m z16, 224 px)."""
import hashlib, io, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, rasterio, requests, timm, torch
from PIL import Image

sys.path.insert(0, "scripts"); import vhr_verify as V  # noqa: E402

URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
BLANK = "f27d9de7f80c13501f470595e327aa6d"; S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)); S.headers["User-Agent"] = "urbis-research"
CK = ("runs/kaggle_retrain3/data/visible_verifier_sat_ibge.pt", "runs/kaggle_retrain3/out/twoview_e2.pt")


def tile(z, x, y):
    for _ in range(3):
        try:
            r = S.get(URL.format(z=z, x=x, y=y), timeout=20)
            if r.status_code == 200 and hashlib.md5(r.content).hexdigest() != BLANK: return Image.open(io.BytesIO(r.content)).convert("RGB")
            if r.status_code == 200: return None
        except Exception: time.sleep(1)
    return None


def view(lon, lat, z, metres):
    n = 2 ** z; px = (lon + 180) / 360 * n * 256; py = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n * 256
    half = metres / 2 / (156543.03392 * math.cos(math.radians(lat)) / n); x0, y0, x1, y1 = int(px - half), int(py - half), int(px + half), int(py + half)
    mos = Image.new("RGB", ((x1 // 256 - x0 // 256 + 1) * 256, (y1 // 256 - y0 // 256 + 1) * 256), (128, 128, 128))
    for tx in range(x0 // 256, x1 // 256 + 1):
        for ty in range(y0 // 256, y1 // 256 + 1):
            t = tile(z, tx, ty)
            if t is not None: mos.paste(t, ((tx - x0 // 256) * 256, (ty - y0 // 256) * 256))
    ox, oy = x0 // 256 * 256, y0 // 256 * 256
    return mos.crop((x0 - ox, y0 - oy, x1 - ox, y1 - oy)).resize((224, 224), Image.BICUBIC)


def net():
    m = timm.create_model("vit_large_patch16_dinov3.sat493m", pretrained=True, num_classes=0)
    ck = torch.load(CK[0], map_location="cpu"); m.load_state_dict({k: v.float() for k, v in ck["sd"].items() if not k.startswith("head.")}, strict=False)
    head = torch.nn.Linear(2 * m.num_features, 1)
    for k, v in torch.load(CK[1], map_location="cpu")["sd"].items():   # trained top blocks + head, names b.<...> / head.<...>
        if k.startswith("b."): m.get_parameter(k[2:]).data[:] = v.float()
        else: head.get_parameter(k[5:]).data[:] = v.float()
    return m.eval(), head.eval(), np.array(m.pretrained_cfg["mean"], np.float32), np.array(m.pretrained_cfg["std"], np.float32)


def main(c, n=2000):
    n = int(n); O = Path(f"runs/city_review/{c}"); O.mkdir(parents=True, exist_ok=True)
    fr, v, lon, lat, _ = V.grid(c)
    with rasterio.open(f"data/{c}/bld.tif") as f: bl = np.nan_to_num(f.read(1).astype(np.float32))
    H, W = fr.shape[0] * V.SQ, fr.shape[1] * V.SQ
    bld = bl[:H, :W].reshape(fr.shape[0], V.SQ, fr.shape[1], V.SQ).mean((1, 3))
    ok = v > .5; built = ok & (bld > np.quantile(bld[ok], .4))
    pick = np.random.default_rng(11).choice(np.flatnonzero(built), min(n, built.sum()), replace=False)
    pts = [[float(lon.flat[i]), float(lat.flat[i])] for i in pick]
    t0 = time.time()
    with ThreadPoolExecutor(24) as ex:
        ims = list(ex.map(lambda p: (view(p[0], p[1], 18, 160), view(p[0], p[1], 16, 640)), pts))
    print(c, len(ims), "squares fetched in", round(time.time() - t0), "s", flush=True)
    m, head, MEAN, STD = net(); dev = "mps" if torch.backends.mps.is_available() else "cpu"; m, head = m.to(dev), head.to(dev)
    tens = lambda im: torch.from_numpy(((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    sc = []
    with torch.no_grad():
        for b in range(0, len(ims), 32):
            cc = torch.stack([tens(a) for a, _ in ims[b:b + 32]]).to(dev); ww = torch.stack([tens(w) for _, w in ims[b:b + 32]]).to(dev)
            sc.append(torch.sigmoid(head(torch.cat([m(cc), m(ww)], 1))[:, 0]).cpu().numpy())
    sc = np.concatenate(sc); print("scored in", round(time.time() - t0), "s", flush=True)
    (O / "pts.json").write_text(json.dumps(pts)); np.save(O / "scores.npy", sc)
    np.savez(O / "meta.npz", sq=pick, frac=fr.flat[pick], bld=bld.flat[pick])
    y = fr.flat[pick] >= .5; top = sc >= np.quantile(sc, .95)
    print(f"map slum share {y.mean():.3f}; model top 5 %: {y[top].mean():.2f} map slum; AUC-ish rank check",
          f"{(sc[y][:, None] > sc[~y][None]).mean():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
