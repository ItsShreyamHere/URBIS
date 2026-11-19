"""Rasterise slum polygons onto the Sentinel-2 grid of a bbox -> labels.tif (1 slum, 0 not, 255 unknown)."""
import numpy as np
import pyogrio
import rasterio
import shapely
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, binary_erosion

from .s2 import Grid

EDGE_PX = 2   # +-20 m around polygon boundaries is "unknown": hand-drawn edges vs 10 m pixels


def read(path, bbox, where=None):
    """Polygons intersecting bbox (lon/lat), reprojected to EPSG:4326."""
    m, _, g, _ = pyogrio.raw.read(path, bbox=bbox if _is_4326(path) else None, where=where)
    geoms = shapely.from_wkb(g)
    crs = m["crs"]
    if crs and crs not in ("EPSG:4326", "OGC:CRS84"):
        from pyproj import Transformer
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        geoms = shapely.transform(geoms, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    return geoms[shapely.intersects(geoms, shapely.box(*bbox))]


def _is_4326(path):
    return pyogrio.read_info(path).get("crs") in ("EPSG:4326", "OGC:CRS84")


def _burn(geoms, grid):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    geoms = shapely.transform(geoms, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    if not len(geoms):
        return np.zeros((grid.height, grid.width), bool)
    return rasterize(((g, 1) for g in geoms), (grid.height, grid.width), transform=grid.transform,
                     dtype="uint8").astype(bool)


def build(bbox, pos, neg=None, out=None, res=10.0, grid=None):
    """pos: slum polygons. neg: None = everything else is non-slum (wall-to-wall labels);
    otherwise only `neg` polygons are non-slum and the rest is unknown (reference samples)."""
    grid = grid or Grid.from_bbox(*bbox, res=res)
    p = _burn(pos, grid)
    edge = binary_dilation(p, iterations=EDGE_PX) & ~binary_erosion(p, iterations=EDGE_PX)
    lab = np.where(p, 1, 0).astype(np.uint8)
    if neg is not None:
        lab[~p & ~_burn(neg, grid)] = 255
    lab[edge] = 255
    if out:
        with rasterio.open(out, "w", **grid.profile(1, "uint8", nodata=255)) as f:
            f.write(lab, 1)
    return lab, grid


if __name__ == "__main__":
    # self-check: a square polygon -> 1 inside, 255 on its edge band, 0 outside
    bb = (72.80, 19.00, 72.82, 19.02)
    sq = np.array([shapely.box(72.805, 19.005, 72.815, 19.015)])
    lab, _ = build(bb, sq)
    h, w = lab.shape
    assert lab[h // 2, w // 2] == 1 and lab[2, 2] == 0 and (lab == 255).any()
    lab2, _ = build(bb, sq, neg=np.array([], dtype=object))
    assert lab2[2, 2] == 255, "with reference-only labels, outside should be unknown"
    print("ok", lab.shape, np.unique(lab, return_counts=True))
