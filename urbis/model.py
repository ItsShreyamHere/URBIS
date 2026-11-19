"""Sampling, training, evaluation and tiled prediction for URBIS-1.

A city lives in data/<city>/ with s2.tif (from s2.save), labels.tif (1 = slum, 0 = not, 255 = unknown)
and optionally bld.tif and shape.tif (from buildings.fetch), all on the same grid.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import buildings as bld_mod, features, s2

BLOCK = 200          # 2 km spatial blocks at 10 m: the unit of CV splitting
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=8)


def _win(rd):
    ys, xs = rd
    return Window(xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start)


def _read(src, rd):
    a = np.moveaxis(src.read(window=_win(rd)), 0, -1).astype(np.float32)
    a[a == 0] = np.nan
    return a / 10000


class Scene:
    """Reflectance (+ optional building layers) for one grid, yielding the full feature vector per tile.
    Scene statistics for the relative features are computed once, from a fixed set of tiles."""

    def __init__(self, s2_path, bld_path=None, shape_path=None, height_path=None, aef_path=None):
        self.src = rasterio.open(s2_path)
        self.aef = rasterio.open(aef_path) if aef_path else None
        self.hgt = rasterio.open(height_path) if height_path else None
        self.bld = rasterio.open(bld_path) if bld_path else None
        self.shape_path = shape_path
        if self.bld is not None:
            assert (self.bld.height, self.bld.width) == (self.src.height, self.src.width), "bld grid differs"
        self.height, self.width, self.profile = self.src.height, self.src.width, self.src.profile
        self.stats = features.scene_stats(self.base, self.height, self.width)

    def base(self, rd):
        bld = None
        if self.bld is not None:
            a = np.moveaxis(self.bld.read(window=_win(rd)), 0, -1).astype(np.float32)
            bld = np.stack([a[..., 0] / 250, a[..., 1]], -1)
        shp = bld_mod.load_shape(self.shape_path, _win(rd)) if self.shape_path else None
        hgt = None
        if self.hgt is not None:
            a = np.moveaxis(self.hgt.read(window=_win(rd), boundless=True, fill_value=-99), 0, -1)
            hgt = np.where(a == -99, np.nan, a * [1, 400]).astype(np.float32)   # count -> buildings per 10 m cell
        aef = None
        if self.aef is not None:
            a = np.moveaxis(self.aef.read(window=_win(rd), boundless=True, fill_value=-128), 0, -1)
            aef = np.where(a == -128, np.nan, a / 32).astype(np.float32)
        return features.compute(_read(self.src, rd), bld, shp, hgt, aef)

    def features(self, rd, crop):
        return features.with_relative(self.base(rd)[crop], self.stats)

    def close(self):
        self.src.close()
        for f in (self.bld, self.hgt, self.aef):
            if f is not None:
                f.close()


def sample(city_dir, p_pos=0.25, p_neg=0.02, seed=0, buildings=False):
    """Random pixel sample. Returns X, y, w (inverse inclusion prob), block ids.
    buildings: 0 none, 1 bld.tif, 2 bld.tif + shape.tif, 3 also height.tif (features.base_names)."""
    d = Path(city_dir)
    rng = np.random.default_rng(seed)
    X, Y, W, B = [], [], [], []
    sc = Scene(d / "s2.tif", d / "bld.tif" if buildings else None, d / "shape.tif" if buildings in (2, 3) else None,
               d / "height.tif" if buildings == 3 else None, d / "aef16.tif" if buildings == 4 else d / "s1.tif" if buildings == 5 else None)
    with rasterio.open(d / "labels.tif") as lab:
        assert (sc.height, sc.width) == (lab.height, lab.width), "labels and s2 grids differ"
        for rd, crop, wr in features.tiles(sc.height, sc.width):
            y = lab.read(1, window=_win(wr))
            p = np.where(y == 1, p_pos, np.where(y == 0, p_neg, 0.0))
            keep = rng.random(y.shape) < p
            if not keep.any():
                continue
            f = sc.features(rd, crop)
            keep &= ~np.isnan(f).any(-1)
            yy, xx = np.nonzero(keep)
            X.append(f[yy, xx]); Y.append(y[yy, xx]); W.append(1 / p[yy, xx])
            B.append(((yy + wr[0].start) // BLOCK) * 10000 + (xx + wr[1].start) // BLOCK)
    sc.close()
    return np.concatenate(X), np.concatenate(Y).astype(np.uint8), np.concatenate(W), np.concatenate(B)


def fit(X, y, rounds=600):
    pos = y.mean()
    return lgb.train(dict(PARAMS, scale_pos_weight=(1 - pos) / max(pos, 1e-6)),
                     lgb.Dataset(X, y, free_raw_data=True), rounds)


def metrics(y, prob, w):
    """Weighted (population-level) metrics; threshold chosen for best F1."""
    pr, rc, th = precision_recall_curve(y, prob, sample_weight=w)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    i = int(np.nanargmax(f1[:-1]))
    p, r = pr[i], rc[i]
    return dict(ap=float(average_precision_score(y, prob, sample_weight=w)),
                auc=float(roc_auc_score(y, prob, sample_weight=w)),
                f1=float(f1[i]), precision=float(p), recall=float(r),
                iou=float(p * r / max(p + r - p * r, 1e-9)), threshold=float(th[i]))


def block_cv(X, y, w, b, k=5, seed=0):
    """Spatial k-fold: whole 2 km blocks go to one fold, so neighbours never leak across."""
    ub = np.unique(b)
    fold_of = dict(zip(ub, np.random.default_rng(seed).integers(0, k, len(ub))))
    fold = np.vectorize(fold_of.get)(b)
    prob = np.empty(len(y), np.float32)
    for f in range(k):
        te = fold == f
        prob[te] = fit(X[~te], y[~te]).predict(X[te])
    return metrics(y, prob, w), prob


def predict(booster, s2_path, out_path, bld_path=None, shape_path=None, height_path=None, aef_path=None):
    """Tiled probability map (uint8, 0..250 = p*250, 255 = nodata). shape_path only for buildings>=2 models,
    height_path for buildings=3."""
    sc = Scene(s2_path, bld_path, shape_path, height_path, aef_path)
    prof = dict(sc.profile, count=1, dtype="uint8", nodata=255)
    with rasterio.open(out_path, "w", **prof) as dst:
        for rd, crop, wr in features.tiles(sc.height, sc.width):
            f = sc.features(rd, crop)
            ok = ~np.isnan(f).any(-1)
            out = np.full(ok.shape, 255, np.uint8)
            if ok.any():
                out[ok] = np.round(booster.predict(f[ok]) * 250).astype(np.uint8)
            dst.write(out, 1, window=_win(wr))
        dst.update_tags(1, meaning="slum probability = value/250", model="URBIS-1")
    sc.close()


def save(booster, path, meta, buildings=False):
    booster.save_model(str(path))
    Path(path).with_suffix(".json").write_text(json.dumps(
        dict(meta, buildings=buildings, features=features.names(buildings)), indent=2))


class Official:
    """Trees trained with a last column saying who drew the label (1 = VHR-GSIS, 0 = official map).
    Reads every pixel as the official map-maker would (scripts/source_trees.py)."""
    def __init__(self, b):
        self.b = b

    def predict(self, f):
        return self.b.predict(np.c_[f, np.zeros(len(f), f.dtype)])


def load(path):
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    meta.setdefault("buildings", False)
    assert meta["features"] == features.names(meta["buildings"]), "model was trained with a different feature set"
    b = lgb.Booster(model_file=str(path))
    return (Official(b) if meta.get("source_flag") else b), meta
