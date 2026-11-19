"""URBIS-Living: relative neighbourhood deprivation for any city, including cities with no slums.

Per spatial unit (US census tracts in training; a regular 1 km grid anywhere else):
  imagery   mean of the 156 URBIS-1 pixel features (Sentinel-2 + Overture footprints)
  heat      Landsat hot-season surface temperature
  places    Overture POI density and category mix
Every column is ranked within the city, and the model predicts the within-city rank of CDC social
vulnerability. Output 0-1: 1 = most deprived unit in this city. It is relative: a city with no
deprived areas still has a top decile.
Validated leave-one-metro-out on 12 US metros (scripts/us_tracts.py). Outside the US it is a transfer.
"""
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
from pyproj import Transformer
from scipy.stats import rankdata

from . import features, model, risk, s2

W = Path(__file__).parent / "weights"
PLACES = "s3://overturemaps-us-west-2/release/2026-10-*/theme=places/type=place/*"


def imagery(s2_path, bld_path, ids):
    sc = model.Scene(s2_path, bld_path)
    n, F = ids.max() + 1, len(features.names(True))
    s, c = np.zeros((n, F)), np.zeros(n)
    for rd, crop, wr in features.tiles(sc.height, sc.width):
        f = sc.features(rd, crop)
        t = ids[wr[0], wr[1]]
        ok = (t > 0) & ~np.isnan(f).any(-1)
        np.add.at(s, t[ok], f[ok])
        c += np.bincount(t[ok], minlength=n)
    sc.close()
    return s[1:] / np.maximum(c[1:], 1)[:, None], c[1:]


def heat(grid, ids, lst=None):
    t = risk.lst(grid) if lst is None else lst
    ok = (ids > 0) & ~np.isnan(t)
    n = ids.max() + 1
    return (np.bincount(ids[ok], t[ok], n) / np.maximum(np.bincount(ids[ok], minlength=n), 1))[1:]


def places(grid, ids):
    """(categories, counts per unit x category) from Overture places with confidence >= 0.5."""
    w, s_, e, n_ = grid.lonlat_bbox()
    con = duckdb.connect()
    con.execute("LOAD spatial; LOAD httpfs; SET s3_region='us-west-2'; SET geometry_always_xy=true;")
    rows = con.execute(f"""SELECT ST_X(geometry), ST_Y(geometry), coalesce(basic_category, 'none')
        FROM read_parquet('{PLACES}', hive_partitioning=1)
        WHERE bbox.xmin < {e} AND bbox.xmax > {w} AND bbox.ymin < {n_} AND bbox.ymax > {s_} AND confidence >= 0.5""").fetchall()
    m = np.zeros((ids.max() + 1, 0), np.int32)
    if not rows:
        return np.array([], str), m[1:]
    lon, lat, cat = (np.array(v) for v in zip(*rows))
    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(lon.astype(float), lat.astype(float))
    col, row = ~grid.transform * (x, y)
    row, col = np.floor(row).astype(int), np.floor(col).astype(int)
    ok = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    t = ids[row[ok], col[ok]]
    cats, ci = np.unique(cat[ok], return_inverse=True)
    m = np.zeros((ids.max() + 1, len(cats)), np.int32)
    np.add.at(m, (t, ci), 1)
    return cats, m[1:]


def place_matrix(cats, counts, area_km2, vocab):
    idx = {c: i for i, c in enumerate(cats)}
    cnt = np.stack([counts[:, idx[c]] if c in idx else np.zeros(len(counts)) for c in vocab], 1)
    return np.column_stack([cnt.sum(1) / area_km2, cnt / np.maximum(cnt.sum(1, keepdims=True), 1)])


def rank(a):
    return (rankdata(a, axis=0) - 0.5) / len(a)


def design(imag, lst, poi):
    """Model input: every view ranked within the city, columns in training order."""
    return np.column_stack([rank(imag), rank(lst[:, None]), rank(poi)])


def cells(grid, size_px=100):
    """Regular grid of size_px x size_px cells (1 km at 10 m) as an ids raster, 1-based."""
    r, c = np.indices((grid.height, grid.width))
    nc = -(-grid.width // size_px)
    return (r // size_px) * nc + c // size_px + 1


def score(run_dir, size_px=100, target="rpl_themes", lst=None, log=print):
    """Deprivation rank per 1 km cell for a run folder with s2.tif + bld.tif. Returns (ids, score per id)."""
    run_dir = Path(run_dir)
    grid = s2.Grid.from_raster(run_dir / "s2.tif")
    ids = cells(grid, size_px)
    imag, npx = imagery(run_dir / "s2.tif", run_dir / "bld.tif", ids)
    t = heat(grid, ids, lst)
    log("imagery + heat aggregated")
    cats, counts = places(grid, ids)
    log(f"{counts.sum()} places")
    vocab = json.loads((W / "us_living_results.json").read_text())["poi_vocab"]
    poi = place_matrix(cats, counts, npx / 1e4 + 1e-6, vocab)
    built = imag[:, features.names(True).index("BLD_cover_mean25")] > 0.05
    ok = (npx > 0.5 * size_px ** 2) & built & ~np.isnan(t)
    out = np.full(len(npx), np.nan)
    bst = lgb.Booster(model_file=str(W / f"living_{target}.txt"))
    out[ok] = rank(bst.predict(design(imag[ok], t[ok], poi[ok])))
    return ids, out


def write(run_dir, ids, sc, size_px=100):
    """living.tif: one pixel per cell (1 km), float32 0-1, NaN where not scored."""
    import rasterio
    from rasterio.transform import Affine
    run_dir = Path(run_dir)
    grid = s2.Grid.from_raster(run_dir / "s2.tif")
    nr, nc = -(-grid.height // size_px), -(-grid.width // size_px)
    a = np.asarray(sc, np.float32).reshape(nr, nc)
    prof = dict(driver="GTiff", height=nr, width=nc, count=1, dtype="float32", crs=grid.crs, nodata=np.nan,
                transform=grid.transform * Affine.scale(size_px))
    with rasterio.open(run_dir / "living.tif", "w", **prof) as f:
        f.write(a, 1)
        f.update_tags(1, meaning="URBIS-Living relative deprivation rank within this AOI (1 = most deprived)")
    return run_dir / "living.tif"


if __name__ == "__main__":
    # self-check: cell ids tile the grid, and ranks are within (0, 1) and order-preserving
    from rasterio.transform import from_origin
    g = s2.Grid("EPSG:32643", from_origin(0, 0, 10, 10), 250, 130)
    ids = cells(g, 100)
    assert ids.max() == 3 * 2 and ids[0, 0] == 1 and ids[249, 129] == 6, ids.max()
    r = rank(np.array([3.0, 1.0, 2.0]))
    assert np.allclose(r, [5 / 6, 1 / 6, 0.5])
    print("ok")
