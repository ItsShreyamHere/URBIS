"""URBIS-Plan: where new clinics or schools would reach the most slum residents who have none nearby.

Demand: slum residents more than REACH_M (straight line) from the nearest existing facility, spread over
slum pixels by roof area (as in URBIS-Priorities). Sites are chosen greedily: each one goes where a
REACH_M circle covers the most still-unserved residents, which is within (1 - 1/e) of the best possible
set for this coverage objective. Candidate sites are 100 m cells; any cell may host a site, so check
land availability on the ground. Straight-line reach, not walking time.
"""
import numpy as np
import rasterio.warp
from scipy.signal import fftconvolve

REACH_M = 1000
CELL = 10            # 10 x 10 pixels = 100 m candidate cells at 10 m


def sites(people, dist, grid, k=10, reach_m=REACH_M):
    """people: (H, W) residents per pixel (slum only); dist: (H, W) metres to nearest facility.
    -> rows with rank, lon, lat, people newly reached, cumulative, share of unserved reached."""
    H, W = people.shape
    d = np.where(np.isfinite(dist) & (dist > reach_m), people, 0).astype(np.float64)
    hc, wc = -(-H // CELL), -(-W // CELL)
    d = np.pad(d, ((0, hc * CELL - H), (0, wc * CELL - W))).reshape(hc, CELL, wc, CELL).sum((1, 3))
    total = d.sum()
    res = abs(grid.transform.a) * CELL
    r = int(np.ceil(reach_m / res))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = ((yy * yy + xx * xx) * res * res <= reach_m * reach_m).astype(np.float64)
    rows, cum = [], 0.0
    for rank in range(1, k + 1):
        gain = fftconvolve(d, disk, mode="same")
        i, j = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[i, j] < 1:
            break
        y0, x0 = max(0, i - r), max(0, j - r)
        sub = disk[y0 - i + r:min(hc, i + r + 1) - i + r, x0 - j + r:min(wc, j + r + 1) - j + r]
        reached = float((d[y0:i + r + 1, x0:j + r + 1] * sub).sum())
        d[y0:i + r + 1, x0:j + r + 1] *= 1 - sub
        cum += reached
        x, y = grid.transform * ((j + 0.5) * CELL, (i + 0.5) * CELL)
        lon, lat = rasterio.warp.transform(grid.crs, "EPSG:4326", [x], [y])
        rows.append(dict(rank=rank, lon=round(lon[0], 5), lat=round(lat[0], 5), people_reached=int(round(reached)),
                         cumulative=int(round(cum)), share_of_unserved=round(float(cum / max(total, 1e-9)), 3)))
    return rows, int(round(total))


def people_raster(slum, cover, slum_pop):
    """Spread the city's slum population over slum pixels by roof area (the URBIS-Priorities rule)."""
    w = np.where(slum, cover if cover is not None else 1.0, 0).astype(np.float64)
    return w * (slum_pop / max(w.sum(), 1e-9))


if __name__ == "__main__":
    # self-check: two clusters of unserved people; the first site lands on the bigger one, the second
    # on the other; people already near a facility are never counted
    from rasterio.transform import from_origin
    from .s2 import Grid
    g = Grid("EPSG:32643", from_origin(500000, 2000000, 10, 10), 600, 600)
    p = np.zeros((600, 600)); p[100:120, 100:120] = 5; p[450:460, 450:460] = 4; p[300:310, 300:310] = 9
    dist = np.full(p.shape, 5000.0); dist[290:320, 290:320] = 100           # cluster 3 is already served
    rows, total = sites(p, dist, g, k=3)
    assert total == 2000 + 400 and [r["people_reached"] for r in rows] == [2000, 400], rows
    assert abs(rows[0]["lat"] - rows[1]["lat"]) > 0.02 and rows[-1]["share_of_unserved"] == 1.0
    print("ok", rows)
