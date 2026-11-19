"""Per-pixel features with multi-scale spatial context. Same code path for training and prediction.

Tiles are processed with a halo wider than the largest window, so tiled output == whole-scene output.
"""
import numpy as np
from scipy.ndimage import uniform_filter, sobel

from .s2 import BANDS

SCALES = (5, 11, 25, 51)          # windows in pixels: 50 m .. 510 m at 10 m
HALO = max(SCALES) // 2 + 2


def _nd(a, b):
    return (a - b) / (a + b + 1e-6)


def _mean(x, valid, w):
    """NaN-aware box mean (normalised convolution)."""
    num = uniform_filter(np.where(valid, x, 0), w, mode="reflect")
    den = uniform_filter(valid.astype(np.float32), w, mode="reflect")
    return num / np.maximum(den, 1e-6)


def base_layers(r):
    b = {k: r[..., i] for i, k in enumerate(BANDS)}
    return {
        "NDVI": _nd(b["B08"], b["B04"]),
        "NDBI": _nd(b["B11"], b["B08"]),
        "MNDWI": _nd(b["B03"], b["B11"]),
        "BSI": _nd(b["B11"] + b["B04"], b["B08"] + b["B02"]),
        "NDRE": _nd(b["B8A"], b["B05"]),
        "BRIGHT": (b["B02"] + b["B03"] + b["B04"]) / 3,
    }


CONTEXT = ("NDVI", "NDBI", "MNDWI", "BRIGHT", "B08")


def base_names(buildings=False):
    """buildings: 0/False none, 1/True roof cover and count (bld.tif), 2 also footprint shape (shape.tif),
    3 also built height (height.tif, scripts/heights.py), 4 roof cover and count + AlphaEarth embeddings
    (aef16.tif, scripts/embeddings.py; no shape or height), 5 roof cover and count + Sentinel-1 radar (s1.tif,
    scripts/radar.py)."""
    n = list(BANDS) + list(base_layers(np.zeros((1, 1, len(BANDS)), np.float32)))
    for k in CONTEXT:
        for w in SCALES:
            n += [f"{k}_mean{w}", f"{k}_std{w}"]
    n += [f"EDGE_mean{w}" for w in SCALES]
    if buildings:
        n += ["BLD_cover", "BLD_count"]
        for w in SCALES:
            n += [f"BLD_cover_mean{w}", f"BLD_cover_std{w}", f"BLD_count_mean{w}", f"BLD_size{w}"]
    if buildings in (2, 3):
        for w in SCALES:
            n += [f"BLD_align{w}", f"BLD_sizesd{w}"]
    if buildings == 3:
        n += ["HGT", "HGT_count"]
        for w in SCALES:
            n += [f"HGT_mean{w}", f"HGT_std{w}", f"HGT_per_bld{w}"]
    if buildings == 4:
        n += [f"AEF{i}" for i in range(AEF_K)] + [f"AEF{i}_mean{AEF_W}" for i in range(AEF_K)]
    if buildings == 5:
        n += [f"S1_{i}" for i in range(S1_K)] + [f"S1_{i}_mean{AEF_W}" for i in range(S1_K)]
    return n


def names(buildings=False):
    """Absolute features + the same features relative to their own scene (median / IQR).
    Relative features make the model ask 'denser and rougher than this city's norm?' instead of
    'this exact roof colour?', which is what lets a model trained in Mumbai work in Nairobi."""
    b = base_names(buildings)
    return b + [n + "_rel" for n in b]


def with_relative(f, stats):
    med, iqr = stats
    return np.concatenate([f, (f - med) / iqr], -1)


def scene_stats(tile_features, h, w, n_tiles=8, seed=0):
    """Robust per-scene (median, IQR) of base features from a fixed random set of tiles.
    tile_features(read_window) -> (h, w, F) base features for that window."""
    rng = np.random.default_rng(seed)
    ts = list(tiles(h, w, 512))
    vals = []
    for i in rng.choice(len(ts), min(n_tiles, len(ts)), replace=False):
        rd, crop, _ = ts[i]
        v = tile_features(rd)[crop]
        v = v.reshape(-1, v.shape[-1])
        v = v[~np.isnan(v).any(1)]
        if len(v):
            vals.append(v[rng.choice(len(v), min(20000, len(v)), replace=False)])
    if not vals:
        raise ValueError("scene has no clear pixels")
    v = np.concatenate(vals)
    q1, med, q3 = np.percentile(v, [25, 50, 75], 0)
    return med.astype(np.float32), np.maximum(q3 - q1, 1e-6).astype(np.float32)


def _shape(count, shp, valid, w):
    """Alignment (bias-corrected squared mean resultant of 4θ: ~1 planned grid, ~0 random) and spread of
    ln building size, over the buildings in a w x w window."""
    a = w * w
    n = _mean(count, valid, w) * a
    c, s_, la, la2 = (_mean(shp[..., i], valid, w) * a for i in range(4))
    align = np.where(n > 1.5, (c * c + s_ * s_ - n) / np.maximum(n * (n - 1), 1e-6), 0)
    m = la / np.maximum(n, 1e-6)
    sd = np.where(n > 1.5, np.sqrt(np.maximum(la2 / np.maximum(n, 1e-6) - m * m, 0)), 0)
    return np.clip(align, 0, 1), sd


AEF_K, AEF_W = 16, 11     # embedding components; window (110 m) for their local mean
S1_K = 5                  # Sentinel-1 bands (s1.tif, scripts/radar.py), level 5; same 110 m local mean


def compute(r, bld=None, shp=None, hgt=None, aef=None):
    """r: (H, W, 10) reflectance with NaN gaps; bld: optional (H, W, 2) roof cover fraction and
    building count (urbis.buildings); shp: optional (H, W, 4) footprint shape sums; hgt: optional (H, W, 2) built
    height (m over the cell) and buildings per cell (scripts/heights.py) -> (H, W, F) float32."""
    valid = ~np.isnan(r).any(-1)
    r = np.where(valid[..., None], r, 0).astype(np.float32)
    base = base_layers(r)
    out = [r[..., i] for i in range(r.shape[-1])] + list(base.values())
    src = dict(base, B08=r[..., BANDS.index("B08")])
    for k in CONTEXT:
        x = src[k]
        for w in SCALES:
            m = _mean(x, valid, w)
            m2 = _mean(x * x, valid, w)
            out += [m, np.sqrt(np.maximum(m2 - m * m, 0))]
    br = base["BRIGHT"]
    edge = np.hypot(sobel(br, 0), sobel(br, 1))
    out += [_mean(edge, valid, w) for w in SCALES]
    if bld is not None:
        cover, count = bld[..., 0], bld[..., 1]
        out += [cover, count]
        for w in SCALES:
            cm = _mean(cover, valid, w)
            nm = _mean(count, valid, w)
            out += [cm, np.sqrt(np.maximum(_mean(cover * cover, valid, w) - cm * cm, 0)), nm,
                    np.minimum(cm * 100 / np.maximum(nm, 1e-3), 5000)]   # mean m2 per building
        if shp is not None:
            for w in SCALES:
                out += list(_shape(count, shp, valid, w))
        if hgt is not None:
            h, hn = hgt[..., 0], hgt[..., 1]
            out += [h, hn]
            for w in SCALES:
                hm, nm = _mean(h, valid, w), _mean(hn, valid, w)
                out += [hm, np.sqrt(np.maximum(_mean(h * h, valid, w) - hm * hm, 0)),
                        np.minimum(hm / np.maximum(nm, 1e-3), 500)]   # built volume per building
        if aef is not None:
            k = aef.shape[-1]    # AlphaEarth (level 4) or Sentinel-1 (level 5): extra bands + their local means
            out += [aef[..., i] for i in range(k)] + [_mean(np.nan_to_num(aef[..., i]), valid, AEF_W) for i in range(k)]
    f = np.stack(out, -1).astype(np.float32)
    f[~valid] = np.nan
    return f


def tiles(h, w, size=1024):
    """Yield (read_window, crop_slices, write_window) with a halo for exact tiled computation."""
    for y in range(0, h, size):
        for x in range(0, w, size):
            y0, x0 = max(0, y - HALO), max(0, x - HALO)
            y1, x1 = min(h, y + size + HALO), min(w, x + size + HALO)
            ty, tx = min(size, h - y), min(size, w - x)
            yield (slice(y0, y1), slice(x0, x1)), (slice(y - y0, y - y0 + ty), slice(x - x0, x - x0 + tx)), \
                  (slice(y, y + ty), slice(x, x + tx))


if __name__ == "__main__":
    # self-check: tiled == whole-scene, names match feature count
    rng = np.random.default_rng(0)
    r = rng.uniform(0.01, 0.4, (300, 260, len(BANDS))).astype(np.float32)
    r[10:20, 30:40] = np.nan
    full = compute(r)
    assert full.shape[-1] == len(base_names()) and len(names()) == 2 * len(base_names())
    tiled = np.empty_like(full)
    for rd, crop, wr in tiles(*r.shape[:2], size=128):
        tiled[wr] = compute(r[rd])[crop]
    # reflect padding differs only at true image borders; interior must match exactly
    inner = (slice(HALO, -HALO), slice(HALO, -HALO))
    assert np.allclose(np.nan_to_num(full[inner]), np.nan_to_num(tiled[inner]), atol=1e-4)
    st = scene_stats(lambda rd: compute(r[rd]), *r.shape[:2], n_tiles=3)
    assert with_relative(full, st).shape[-1] == len(names())
    bld = np.stack([rng.uniform(0, 1, r.shape[:2]), rng.integers(0, 4, r.shape[:2])], -1).astype(np.float32)
    assert compute(r, bld).shape[-1] == len(base_names(True))
    # shape: 9 aligned buildings per pixel -> alignment 1; alternating 0/45 deg -> alignment ~0
    one = np.ones(r.shape[:2], np.float32)
    al = np.stack([one * 9, one * 0, one * 0, one * 0], -1)          # cos 4θ = 1 for all 9 buildings
    mx = np.stack([one * 9, one * 0, one * 0, one * 0], -1); mx[::2, :, 0] = -9
    b9 = np.stack([one * 0.5, one * 9], -1)
    fa, fm = compute(r, b9, al), compute(r, b9, mx)
    k = base_names(2).index("BLD_align11")
    assert len(base_names(2)) == fa.shape[-1] and fa[150, 100, k] > 0.95 and fm[150, 100, k] < 0.05, \
        (fa[150, 100, k], fm[150, 100, k])
    hg = compute(r, bld, np.stack([one * 9, one * 0, one * 0, one * 0], -1), np.stack([one * 6, one * 2], -1))
    k = base_names(3).index("HGT_per_bld5") if "HGT_per_bld5" in base_names(3) else -1
    assert hg.shape[-1] == len(base_names(3)) and abs(hg[150, 100, k] - 3) < 1e-3, "height features"
    ae = compute(r, bld, aef=np.stack([one * i for i in range(AEF_K)], -1))
    assert ae.shape[-1] == len(base_names(4)) and ae[150, 100, base_names(4).index("AEF5_mean11")] == 5, "aef features"
    print("ok", full.shape, len(names()))
