"""URBIS-1 second stage: judge each mapped patch as a whole.

The pixel models flag candidate patches (ensemble >= LO). Many confident patches are dense formal housing, so
a small tree model scores each patch from what a pixel cannot see: its size and compactness, how its
confidence is distributed, the footprints inside it (roof cover, building density, typical size, size
spread, alignment) and the same footprint measures in a ring around it, relative to the whole city.
"""
import numpy as np
from scipy import ndimage

LO = 0.4
MIN_PX = 50
RING = 5           # px (50 m) ring for context
CONFIRM = 0.95     # "high score" tier: 61 % real out of fold without the label-edge leak (97 % with it; docs/VALIDATION.md)
SCREEN = 0.05      # field screen: raw probability, imagery-only where footprints are missing (tiers.json)
GAP_PX = 11        # 110 m window with no footprint at all = footprint gap (Overture misses whole settlements)
EIGHT = np.ones((3, 3))
NAMES = ["log_area", "compact", "ens_mean", "ens_max", "ens_q90", "share_07", "trees_mean", "unet_mean",
         "cover", "dens", "lnsize", "size_sd", "align", "ring_cover", "ring_dens", "ring_lnsize",
         "cover_rel", "dens_rel", "lnsize_rel", "ens_rank", "city_share"] + [
    f"{w}_{b}" for b in ("ndvi", "ndbi", "bright", "swir") for w in ("in", "ring", "diff", "rel")] + ["bright_sd"]


def candidates(ens):
    m = np.nan_to_num(ens) >= LO
    lab, n = ndimage.label(m, EIGHT)
    size = ndimage.sum_labels(m, lab, np.arange(1, n + 1))
    keep = np.flatnonzero(size >= MIN_PX) + 1
    lut = np.zeros(n + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[lab], len(keep)


def spectral(s2):
    """(H, W, 4) NDVI, NDBI, visible brightness, SWIR-2 from (H, W, 10) reflectance (B02..B12 order)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        b, g, r, nir, sw1, sw2 = s2[..., 0], s2[..., 1], s2[..., 2], s2[..., 6], s2[..., 8], s2[..., 9]
        return np.stack([(nir - r) / (nir + r), (sw1 - nir) / (sw1 + nir), (b + g + r) / 3, sw2], -1)


TREE_W = 0.5     # trees' share of the ensemble. 0.7 with the source-flag trees lifted VHR-GSIS AP 0.62 -> 0.64 but cut
                 # confirmed settlements 97 % -> 93-94 % real at 0.5 or 0.7 (tiers_flag.json), so trees and weight stay


def ensemble(trees, unet, w=None):
    w = TREE_W if w is None else w
    return w * trees + (1 - w) * unet


def features(trees, unet, bld, shp, s2, lab=None):
    """trees, unet: (H, W) probabilities; bld: (H, W, 2) cover, count; shp: (H, W, 4) shape sums;
    s2: (H, W, 10) reflectance.
    -> lab (H, W) patch ids 1..n, X (n, len(NAMES))."""
    ens = ensemble(trees, unet)
    if lab is None:
        lab, n = candidates(ens)
    else:
        n = int(lab.max())
    if n == 0:
        return lab, np.zeros((0, len(NAMES)), np.float32)
    idx = np.arange(1, n + 1)
    S = lambda a, L=lab: ndimage.sum_labels(a, L, idx)
    e, t, u = np.nan_to_num(ens), np.nan_to_num(trees), np.nan_to_num(unet)
    area = S(np.ones_like(e))
    sl = ndimage.find_objects(lab)
    bbox = np.array([(s[0].stop - s[0].start) * (s[1].stop - s[1].start) for s in sl], np.float64)
    q90 = np.array([np.quantile(e[s][lab[s] == i + 1], 0.9) for i, s in enumerate(sl)])
    cover, count = bld[..., 0], bld[..., 1]
    ring = ndimage.grey_dilation(lab, size=2 * RING + 1) * (lab == 0)

    def fp(L):
        c, k = S(cover, L), np.maximum(S(count, L), 1e-9)
        a = S(np.ones_like(cover), L)
        mu = S(shp[..., 2], L) / k
        sd = np.sqrt(np.maximum(S(shp[..., 3], L) / k - mu * mu, 0))
        al = np.hypot(S(shp[..., 0], L), S(shp[..., 1], L)) / k
        return c / np.maximum(a, 1), k / np.maximum(a, 1), mu, sd, al

    pc, pd, pm, psd, pal = fp(lab)
    rc, rd, rm, _, _ = fp(ring)
    built = count > 0
    cm, dm = np.median(cover[built]) if built.any() else 0, np.median(count[built]) if built.any() else 0
    cnt = np.maximum(count.sum(), 1e-9)
    mm = shp[..., 2].sum() / cnt
    em = S(e) / area
    sp = spectral(s2)
    ok = np.isfinite(sp).all(-1)
    spz = np.where(ok[..., None], sp, 0)
    spec = []
    for j in range(4):
        pin = S(spz[..., j]) / np.maximum(S(ok), 1)
        pr = S(spz[..., j], ring) / np.maximum(S(ok, ring), 1)
        med = np.median(sp[..., j][ok & (count > 0)]) if (ok & (count > 0)).any() else 0
        spec += [pin, pr, pin - pr, pin - med]
    br = spz[..., 2]
    spec.append(np.sqrt(np.maximum(S(br * br) / np.maximum(S(ok), 1) - spec[8] ** 2, 0)))
    X = np.column_stack([np.log(area), area / bbox, em, ndimage.maximum(e, lab, idx), q90,
                         S(e >= 0.7) / area, S(t) / area, S(u) / area,
                         pc, pd, pm, psd, pal, rc, rd, rm,
                         pc - cm, pd - dm, pm - mm, (np.argsort(np.argsort(em)) + 0.5) / n,
                         np.full(n, (np.nan_to_num(ens) >= LO).mean())] + spec)
    return lab, X.astype(np.float32)


def screen(raw, s2only, count):
    """Field-screen probability: the raw ensemble, raised to the imagery-only probability wherever no footprint
    lies within 55 m (Overture gaps; 99 % of what Tshwane's screen missed). count: (H, W) footprints per pixel."""
    gap = ndimage.uniform_filter((count > 0).astype(np.float32), GAP_PX) == 0
    return np.where(gap & ~np.isnan(s2only), np.fmax(raw, s2only), raw)


def apply(prob, lab, score, cut):
    """Cap every pixel of a patch scored below `cut` at that score; other pixels are unchanged."""
    cap = np.concatenate([[np.inf], np.where(score < cut, score, np.inf)]).astype(np.float32)
    return np.minimum(prob, cap[lab])


def truth(lab, y, hit=0.25):
    """1 where >= hit of the patch lies in a labelled settlement (y == 1)."""
    n = int(lab.max())
    idx = np.arange(1, n + 1)
    return (ndimage.sum_labels(y == 1, lab, idx) / np.maximum(ndimage.sum_labels(lab > 0, lab, idx), 1) >= hit
            ).astype(np.int8)


if __name__ == "__main__":
    # self-check: two patches, a big one over dense small buildings and a small one; features are finite
    H = W = 120
    p = np.zeros((H, W), np.float32); p[10:40, 10:40] = 0.9; p[80:90, 80:90] = 0.5
    bld = np.zeros((H, W, 2), np.float32); bld[10:40, 10:40] = (0.9, 3)
    shp = np.zeros((H, W, 4), np.float32); shp[10:40, 10:40] = (3, 0, -1.5, 3.75)
    s2 = np.full((H, W, 10), 0.1, np.float32); s2[..., 6] = 0.3
    lab, X = features(p, p, bld, shp, s2)
    assert lab.max() == 2 and X.shape == (2, len(NAMES)) and np.isfinite(X).all(), X
    i = NAMES.index
    assert abs(X[0, i("align")] - 1) < 1e-6 and abs(X[0, i("size_sd")] - 1) < 1e-6 and X[0, i("dens")] == 3
    assert abs(X[0, i("in_ndvi")] - 0.5) < 1e-6 and abs(X[0, i("diff_ndvi")]) < 1e-6
    y = np.zeros((H, W), np.uint8); y[10:40, 10:40] = 1
    assert list(truth(lab, y)) == [1, 0]
    q = apply(p, lab, np.array([0.9, 0.1]), 0.3)
    assert q[20, 20] == np.float32(0.9) and abs(q[85, 85] - 0.1) < 1e-6 and q[0, 0] == 0
    cnt = np.zeros((H, W)); cnt[:60] = 1
    sc = screen(np.full((H, W), 0.1, np.float32), np.full((H, W), 0.8, np.float32), cnt)
    assert sc[10, 10] == np.float32(0.1) and sc[110, 110] == np.float32(0.8)
    print("ok")
