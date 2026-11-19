"""Open building footprints (Overture Maps: Google + Microsoft + OSM) rasterised onto a 10 m grid.

bld.tif band 1 = roof cover fraction x250 (rasterised at 2 m, averaged), band 2 = building centroids per pixel.
Slums show up as near-total roof cover made of many tiny buildings; formal housing has fewer, larger
footprints with gaps. This is the signal 10 m imagery cannot resolve on its own.

shape.tif (written beside bld.tif) holds per-pixel sums over the buildings whose centroid falls in it:
cos 4θ and sin 4θ of the footprint orientation (x100), ln(area/55 m²) (x100) and its square (x50), int16.
Summed over a neighbourhood they give how aligned buildings are (planned grids: aligned; self-built
camps: not) and how much their sizes vary, which density and cover alone cannot tell apart.
"""
from pathlib import Path

import duckdb
import numpy as np
import rasterio
import shapely
import shapely.affinity
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine

RELEASE = "s3://overturemaps-us-west-2/release/2026-10-*/theme=buildings/type=building/*"
SUPER = 5   # 2 m sub-pixels per 10 m pixel side


def query(bbox):
    w, s, e, n = bbox
    con = duckdb.connect()
    con.execute("LOAD spatial; LOAD httpfs; SET s3_region='us-west-2'; SET geometry_always_xy=true; SET http_timeout=300000; SET http_retries=10;")
    rows = con.execute(f"""SELECT ST_AsWKB(geometry) FROM read_parquet('{RELEASE}', hive_partitioning=1)
        WHERE bbox.xmin < {e} AND bbox.xmax > {w} AND bbox.ymin < {n} AND bbox.ymax > {s}""").fetchall()
    return shapely.from_wkb([bytes(r[0]) for r in rows])


SHAPE_SCALE = np.array([100, 100, 100, 50], np.float32)


def _project(geoms_4326, grid):
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    return shapely.transform(geoms_4326, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


def _pixel(g, grid):
    """Row, col of each footprint's centroid and a mask of those inside the grid."""
    c = shapely.get_coordinates(shapely.centroid(g))
    col, row = ~grid.transform * (c[:, 0], c[:, 1])
    row, col = np.floor(row).astype(int), np.floor(col).astype(int)
    return row, col, (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)


def shape_sums(geoms_4326, grid):
    """(H, W, 4) float32 per-pixel sums: cos 4θ, sin 4θ, ln(A/55), ln(A/55)^2 (see module doc)."""
    g = _project(geoms_4326, grid)
    g = g[(shapely.get_type_id(g) >= 3) & (shapely.area(g) > 1)]      # polygons only
    env = shapely.oriented_envelope(g)
    keep = shapely.get_type_id(env) == 3
    g, env = g[keep], env[keep]
    xy, idx = shapely.get_coordinates(env, return_index=True)
    first = np.searchsorted(idx, np.arange(len(env)))
    d = xy[first + 1] - xy[first]                     # any rectangle edge: 4θ is the same for all four
    t = 4 * np.arctan2(d[:, 1], d[:, 0])
    la = np.log(shapely.area(g) / 55)
    vals = np.stack([np.cos(t), np.sin(t), la, la * la], -1)
    row, col, ok = _pixel(g, grid)
    out = np.zeros((grid.height, grid.width, 4), np.float32)
    np.add.at(out, (row[ok], col[ok]), vals[ok])
    return out


def rasterise(geoms_4326, grid):
    g = _project(geoms_4326, grid)
    H, W = grid.height, grid.width
    fine = rasterize(((x, 1) for x in g), (H * SUPER, W * SUPER),
                     transform=grid.transform * Affine.scale(1 / SUPER), dtype="uint8")
    cover = fine.reshape(H, SUPER, W, SUPER).mean((1, 3))
    del fine
    row, col, ok = _pixel(g, grid)
    count = np.zeros((H, W), np.int32)
    np.add.at(count, (row[ok], col[ok]), 1)
    return cover.astype(np.float32), count


def fetch(grid, out, only_shape=False):
    """Write bld.tif to `out` and shape.tif beside it (only_shape: just shape.tif, for older runs)."""
    geoms = query(grid.lonlat_bbox())
    if not only_shape:
        cover, count = rasterise(geoms, grid)
        with rasterio.open(out, "w", **grid.profile(2, "uint8")) as f:
            f.write(np.round(cover * 250).astype(np.uint8), 1)
            f.write(np.clip(count, 0, 250).astype(np.uint8), 2)
            f.descriptions = ("roof_cover_x250", "building_count")
    sh = np.clip(np.round(shape_sums(geoms, grid) * SHAPE_SCALE), -32767, 32767).astype(np.int16)
    with rasterio.open(Path(out).with_name("shape.tif"), "w", **grid.profile(4, "int16")) as f:
        f.write(np.moveaxis(sh, -1, 0))
        f.descriptions = ("sum_cos4theta_x100", "sum_sin4theta_x100", "sum_lnarea55_x100", "sum_lnarea55_sq_x50")
    return len(geoms)


def load_shape(path, window=None):
    """(H, W, 4) float32 per-pixel sums from shape.tif."""
    with rasterio.open(path) as f:
        a = np.moveaxis(f.read(window=window), 0, -1).astype(np.float32)
    return a / SHAPE_SCALE


def load(path, window=None):
    """(H, W, 2) float32: cover fraction 0..1, building count."""
    with rasterio.open(path) as f:
        a = f.read(window=window).astype(np.float32)
    a[0] /= 250
    return np.moveaxis(a, 0, -1)


if __name__ == "__main__":
    # self-check: one 20x20 m square building on a 10 m grid -> 4 fully covered pixels, 1 centroid
    from .s2 import Grid
    from rasterio.transform import from_origin
    grid = Grid("EPSG:32643", from_origin(0, 100, 10, 10), 10, 10)
    tr = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    sq = shapely.transform(shapely.box(20, 60, 40, 80), lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    cover, count = rasterise(np.array([sq]), grid)
    assert abs(cover.sum() - 4) < 0.3 and count.sum() == 1, (cover.sum(), count.sum())
    # orientation: an axis-aligned square and the same square rotated 45 deg give cos 4θ = +1 and -1
    rot = shapely.transform(shapely.affinity.rotate(shapely.box(20, 60, 40, 80), 45),
                            lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    s1, s2_ = shape_sums(np.array([sq]), grid).sum((0, 1)), shape_sums(np.array([rot]), grid).sum((0, 1))
    assert abs(s1[0] - 1) < 0.05 and abs(s2_[0] + 1) < 0.05 and abs(s1[2] - np.log(400 / 55)) < 0.05, (s1, s2_)
    print("ok", round(float(cover.sum()), 2))
