"""URBIS-Access: straight-line distance (m) from every 10 m pixel to the nearest health facility and school,
from Overture Maps places (Meta + Microsoft + Foursquare POIs, open licence), and to the nearest drivable
road (Overture transportation segments): street access, the core of slum upgrading. A roof more than ROAD_M
from any drivable road is beyond fire-hose and stretcher reach.

ponytail: Euclidean distance, not walking time over the street network; 1 km straight-line ~ 1.3 km on foot.
Upgrade path: Overture transportation segments + a graph shortest path, if a partner needs travel times.
"""
import duckdb
import numpy as np
import shapely
import shapely.affinity
from affine import Affine
from rasterio.features import rasterize
from pyproj import Transformer
from scipy import ndimage

RELEASE = "s3://overturemaps-us-west-2/release/2026-10-*/theme=places/type=place/*"
ROADS = "s3://overturemaps-us-west-2/release/2026-10-*/theme=transportation/type=segment/*"
FOOT = ("footway", "pedestrian", "steps", "path", "cycleway", "bridleway", "track")   # not drivable
ROAD_M = 50
KINDS = {
    "health": ("hospital", "health_care", "medical_service", "outpatient_care_facility",
               "emergency_or_urgent_care_facility", "reproductive_perinatal_and_womens_care",
               "specialized_medical_facility", "clinic", "medical_center"),
    "school": ("elementary_school", "middle_school", "high_school", "preschool", "school"),
}


def places(bbox, pad=0.05):
    """{kind: (N, 2) lon/lat points}; bbox padded so facilities just outside the AOI still count."""
    w, s, e, n = bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad
    con = duckdb.connect()
    con.execute("LOAD spatial; LOAD httpfs; SET s3_region='us-west-2'; SET geometry_always_xy=true;")
    out = {}
    for k, cats in KINDS.items():
        rows = con.execute(f"""SELECT ST_X(geometry), ST_Y(geometry) FROM read_parquet('{RELEASE}', hive_partitioning=1)
            WHERE bbox.xmin < {e} AND bbox.xmax > {w} AND bbox.ymin < {n} AND bbox.ymax > {s}
            AND confidence >= 0.5 AND basic_category IN {tuple(cats)}""").fetchall()
        out[k] = np.array(rows, np.float64).reshape(-1, 2)
    return out


def roads(bbox, pad=0.01):
    """Drivable road centrelines as shapely LineStrings (lon/lat)."""
    w, s_, e, n = bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad
    con = duckdb.connect()
    con.execute("LOAD spatial; LOAD httpfs; SET s3_region='us-west-2'; SET geometry_always_xy=true;")
    rows = con.execute(f"""SELECT ST_AsWKB(geometry) FROM read_parquet('{ROADS}', hive_partitioning=1)
        WHERE bbox.xmin < {e} AND bbox.xmax > {w} AND bbox.ymin < {n} AND bbox.ymax > {s_}
        AND subtype = 'road' AND coalesce(class, '') NOT IN {FOOT}""").fetchall()
    return shapely.from_wkb([bytes(r[0]) for r in rows])


def road_distance(lines_lonlat, grid, pad_px=50):
    """Distance (m) from every pixel to the nearest drivable road centreline."""
    H, W, T = grid.height, grid.width, grid.transform
    if not len(lines_lonlat):
        return np.full((H, W), np.inf, np.float32)
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    geoms = shapely.transform(lines_lonlat, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    Tp = T * Affine.translation(-pad_px, -pad_px)
    on = rasterize(((g, 1) for g in geoms), out_shape=(H + 2 * pad_px, W + 2 * pad_px), transform=Tp,
                   all_touched=True, dtype=np.uint8)
    if not on.any():
        return np.full((H, W), np.inf, np.float32)
    d = ndimage.distance_transform_edt(on == 0, sampling=abs(T.a))
    return d[pad_px:pad_px + H, pad_px:pad_px + W].astype(np.float32)


def distance(points_lonlat, grid, pad_px=500):
    """Distance (m) to the nearest point, computed on a grid padded by pad_px so outside points count."""
    H, W, T = grid.height, grid.width, grid.transform
    if not len(points_lonlat):
        return np.full((H, W), np.inf, np.float32)
    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(*points_lonlat.T)
    col, row = ~T * (x, y)
    row, col = np.floor(row).astype(int) + pad_px, np.floor(col).astype(int) + pad_px
    ok = (row >= 0) & (row < H + 2 * pad_px) & (col >= 0) & (col < W + 2 * pad_px)
    free = np.ones((H + 2 * pad_px, W + 2 * pad_px), bool)
    free[row[ok], col[ok]] = False
    if free.all():
        return np.full((H, W), np.inf, np.float32)
    d = ndimage.distance_transform_edt(free, sampling=abs(T.a))
    return d[pad_px:pad_px + H, pad_px:pad_px + W].astype(np.float32)


def layers(grid, log=print):
    p = places(grid.lonlat_bbox())
    log("facilities: " + ", ".join(f"{len(v)} {k}" for k, v in p.items()))
    out = {f"dist_{k}": distance(v, grid) for k, v in p.items()}
    counts = {k: len(v) for k, v in p.items()}
    try:
        r = roads(grid.lonlat_bbox())
        out["dist_road"], counts["road_segments"] = road_distance(r, grid), len(r)
        log(f"roads: {len(r)} drivable segments")
    except Exception as e:  # supplementary: keep facilities if the road query fails
        log(f"warning: roads skipped ({type(e).__name__}: {e})")
    return out, counts


if __name__ == "__main__":
    # self-check: one facility at pixel (0, 0) of a 10 m grid -> pixel (3, 4) is 50 m away
    from rasterio.transform import from_origin
    from .s2 import Grid
    g = Grid("EPSG:32643", from_origin(500000, 2000000, 10, 10), 10, 10)
    lon, lat = Transformer.from_crs(g.crs, "EPSG:4326", always_xy=True).transform(500005, 1999995)
    d = distance(np.array([[lon, lat]]), g, pad_px=5)
    assert d[0, 0] == 0 and abs(d[3, 4] - 50) < 1e-3, (d[0, 0], d[3, 4])
    # a road along the top row of pixels: row 5 is 50 m from it, and a road outside the grid still counts
    x0, x1 = 500000, 500100
    line = shapely.LineString(np.column_stack(Transformer.from_crs(g.crs, "EPSG:4326", always_xy=True)
                                              .transform([x0, x1], [1999995, 1999995])))
    rd = road_distance(np.array([line]), g, pad_px=5)
    assert rd[0, 3] == 0 and abs(rd[5, 3] - 50) < 1e-3, (rd[0, 3], rd[5, 3])
    off = shapely.affinity.translate(line, yoff=30 / 111320)          # 30 m north, outside the grid
    assert abs(road_distance(np.array([off]), g, pad_px=5)[0, 3] - 30) <= 10
    print("ok")
