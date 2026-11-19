"""URBIS-1 Seg: a small U-Net that segments slums from Sentinel-2 + building rasters.

The tree model (model.py) sees each pixel through hand-made neighbourhood statistics; the U-Net learns
the spatial pattern itself (street grids, block shapes, edges against formal housing). The two make
different mistakes, so their average beats either (see scripts/evaluate.py).

Inputs per pixel, 22 channels:
  10 S2 bands scene-relative ((x - scene median) / scene IQR)  -> robust to sensor/season/haze shifts
  10 S2 bands absolute (reflectance x 3)                       -> keeps real brightness / greenness
  roof cover fraction, log1p(building count) / 2               -> Overture footprints (zeros if absent)
"""
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CH, PATCH = 22, 128
# LightGBM and PyTorch each ship an OpenMP runtime; once LightGBM has run in the process, torch's
# multi-threaded CPU copies deadlock on macOS. Our CPU-side torch work is tiny, so run it single-threaded.
torch.set_num_threads(1)
DEV = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data
def scene_norm(s2):
    """Per-band (median, IQR) over valid pixels of one scene, from a fixed 1% subsample."""
    v = s2.reshape(-1, s2.shape[-1])
    v = v[np.random.default_rng(0).random(len(v)) < 0.01]
    v = v[(v > 0).all(1)].astype(np.float32) / 10000
    q1, med, q3 = np.percentile(v, [25, 50, 75], axis=0)
    return med, np.maximum(q3 - q1, 1e-3)


def channels(s2, bld, norm):
    """uint16 s2 (..., 10) + uint8 bld (..., 2) or None -> float32 (..., 22), nodata -> 0, plus valid mask."""
    valid = (s2 > 0).all(-1)
    r = s2.astype(np.float32) / 10000
    med, iqr = norm
    rel = np.clip((r - med) / iqr, -5, 5)
    b = np.zeros(s2.shape[:-1] + (2,), np.float32)
    if bld is not None:
        b[..., 0] = bld[..., 0] / 250
        b[..., 1] = np.log1p(bld[..., 1].astype(np.float32)) / 2
    x = np.concatenate([rel, np.clip(r * 3, 0, 3), b], -1).astype(np.float32)
    x[~valid] = 0
    return x, valid


def source(d):
    """Which map-maker drew a city's labels: 0 = official outlines, 1 = VHR-GSIS (meta role "weak").
    The two draw the same settlements differently (IoU 0.47), so each gets its own output head."""
    m = Path(d) / "meta.json"
    return int(m.exists() and json.loads(m.read_text()).get("role") == "weak")


def read_city(d):
    d = Path(d)
    with rasterio.open(d / "s2.tif") as f:
        s2 = np.moveaxis(f.read(), 0, -1)
    bld = None
    if (d / "bld.tif").exists():
        with rasterio.open(d / "bld.tif") as f:
            bld = np.moveaxis(f.read(), 0, -1)
    lab = None
    if (d / "labels.tif").exists():
        with rasterio.open(d / "labels.tif") as f:
            lab = f.read(1)
    return s2, bld, lab


class Patches:
    """Random training patches; each label source equally likely, then every city in it equally likely;
    half the patches centred on a slum pixel."""

    def __init__(self, city_dirs, seed=0):
        self.c, self.src = [], [source(d) for d in city_dirs]
        self.groups = [[i for i, s in enumerate(self.src) if s == k] for k in sorted(set(self.src))]
        for d in city_dirs:
            s2, bld, lab = read_city(d)
            lab = lab.copy()
            lab[(s2 == 0).any(-1)] = 255
            pos = np.argwhere(lab == 1)
            self.c.append((s2, bld, lab, scene_norm(s2), pos[:: max(1, len(pos) // 200_000)]))
        self.rng = np.random.default_rng(seed)

    def one(self):
        r = self.rng
        g = self.groups[r.integers(len(self.groups))]
        ci = g[r.integers(len(g))]
        s2, bld, lab, norm, pos = self.c[ci]
        H, W = lab.shape
        for _ in range(20):
            if len(pos) and r.random() < 0.5:
                cy, cx = pos[r.integers(len(pos))] + r.integers(-PATCH // 3, PATCH // 3, 2)
            else:
                cy, cx = r.integers(0, H), r.integers(0, W)
            y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH)); x0 = int(np.clip(cx - PATCH // 2, 0, W - PATCH))
            y = lab[y0:y0 + PATCH, x0:x0 + PATCH]
            if (y != 255).mean() > 0.2:
                break
        sl = np.s_[y0:y0 + PATCH, x0:x0 + PATCH]
        s = s2[sl].astype(np.float32) * r.normal(1, 0.05, 10)          # per-band gain jitter: sensor/season shift
        x, _ = channels(np.clip(s, 1, 65535).astype(np.uint16) * (s2[sl] > 0), None if bld is None else bld[sl], norm)
        if bld is not None and r.random() < 0.1:
            x[..., 20:] = 0                                              # teach it to cope without footprints
        k = r.integers(4)
        x, y = np.rot90(x, k), np.rot90(y, k)
        if r.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        return np.ascontiguousarray(x), np.ascontiguousarray(y), self.src[ci]

    def batch(self, n):
        xs, ys, ss = zip(*(self.one() for _ in range(n)))
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2)
        return x, torch.from_numpy(np.stack(ys)), torch.tensor(ss)


# ---------------------------------------------------------------- model
def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=(32, 64, 128, 256, 256), heads=1):
        super().__init__()
        self.down = nn.ModuleList([_block(IN_CH, c[0])] + [_block(c[i], c[i + 1]) for i in range(len(c) - 1)])
        self.up = nn.ModuleList([_block(c[i + 1] + c[i], c[i]) for i in reversed(range(len(c) - 1))])
        self.head = nn.Conv2d(c[0], heads, 1)     # one output per label source; head 0 = official

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for u, s in zip(self.up, reversed(skips[:-1])):
            x = u(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.head(x)


def train(city_dirs, steps=4000, batch=16, lr=2e-3, log=print, seed=0):
    torch.manual_seed(seed)
    data = Patches(city_dirs, seed)
    net = UNet(heads=1 + max(data.src)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    run = 0.0
    for i in range(steps):
        x, y, src = data.batch(batch)
        x, y, src = x.to(DEV), y.to(DEV), src.to(DEV)
        m = (y != 255).float()
        logit = net(x)[torch.arange(len(src), device=DEV), src]   # each patch scored by its own source's head
        loss = (F.binary_cross_entropy_with_logits(logit, (y == 1).float(), reduction="none") * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if (i + 1) % 250 == 0:
            log(f"step {i + 1}/{steps} loss {run:.4f}")
    return net.eval()


# ---------------------------------------------------------------- inference
@torch.no_grad()
def prob_map(net, s2, bld, tile=512, margin=48):
    """Full-scene probability (float32, NaN where nodata), tiled with overlapping margins."""
    norm = scene_norm(s2)
    H, W = s2.shape[:2]
    out = np.full((H, W), np.nan, np.float32)
    step = tile - 2 * margin
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(0, y0 - margin), max(0, x0 - margin)
            yb, xb = min(H, y0 + step + margin), min(W, x0 + step + margin)
            x, valid = channels(s2[ya:yb, xa:xb], None if bld is None else bld[ya:yb, xa:xb], norm)
            h, w = x.shape[:2]
            ph, pw = (-h) % 16, (-w) % 16
            t = torch.from_numpy(np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect")).permute(2, 0, 1)[None]
            p = torch.sigmoid(net(t.to(DEV)))[0, 0, :h, :w].cpu().numpy()
            p[~valid] = np.nan
            cy, cx = y0 - ya, x0 - xa
            out[y0:min(H, y0 + step), x0:min(W, x0 + step)] = p[cy:cy + step, cx:cx + step][:min(H, y0 + step) - y0, :min(W, x0 + step) - x0]
    return out


def write_prob(p, like_path, out_path, model_name):
    with rasterio.open(like_path) as f:
        prof = dict(f.profile, count=1, dtype="uint8", nodata=255)
    a = np.where(np.isnan(p), 255, np.round(np.nan_to_num(p) * 250)).astype(np.uint8)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(a, 1)
        dst.update_tags(1, meaning="slum probability = value/250", model=model_name)


def read_prob(path):
    with rasterio.open(path) as f:
        a = f.read(1)
    return np.where(a == 255, np.nan, a / 250).astype(np.float32)


def predict(net, s2_path, out_path, bld_path=None):
    with rasterio.open(s2_path) as f:
        s2 = np.moveaxis(f.read(), 0, -1)
    bld = None
    if bld_path and Path(bld_path).exists():
        with rasterio.open(bld_path) as f:
            bld = np.moveaxis(f.read(), 0, -1)
    write_prob(prob_map(net, s2, bld), s2_path, out_path, "URBIS-1 Seg")


def save(net, path, meta):
    torch.save(net.state_dict(), path)
    heads = net.head.out_channels
    Path(path).with_suffix(".json").write_text(json.dumps(dict(meta, in_ch=IN_CH, heads=heads), indent=2))


def load(path):
    j = Path(path).with_suffix(".json")
    net = UNet(heads=json.loads(j.read_text()).get("heads", 1) if j.exists() else 1)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(DEV).eval()


if __name__ == "__main__":
    # self-check: shapes, and that tiled inference equals one-shot inference away from tile seams
    net = UNet().to(DEV).eval()
    rng = np.random.default_rng(0)
    s2 = rng.integers(200, 4000, (300, 260, 10)).astype(np.uint16)
    s2[:5, :5] = 0
    p = prob_map(net, s2, None, tile=160, margin=48)
    q = prob_map(net, s2, None, tile=1024, margin=0)
    assert p.shape == (300, 260) and np.isnan(p[:5, :5]).all() and not np.isnan(p[10:, 10:]).any()
    assert np.nanmax(np.abs(p - q)) < 0.05, np.nanmax(np.abs(p - q))
    print("ok", float(np.nanmax(np.abs(p - q))))
