"""Cloud-free Sentinel-2 L2A median composite for any grid, streamed from Earth Search (no account needed)."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely
import shapely.geometry
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from pystac_client import Client

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
import warnings; warnings.filterwarnings("ignore", module="pystac")

STAC = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
# order matters: features.py indexes by these names
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ASSET = {"B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1", "B06": "rededge2",
         "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B11": "swir16", "B12": "swir22"}
BAD_SCL = [0, 1, 3, 8, 9, 10]  # nodata, saturated, cloud shadow, cloud med/high, cirrus


class Grid:
    """Target pixel grid: crs + transform + shape."""
    def __init__(self, crs, transform, height, width):
        self.crs, self.transform, self.height, self.width = CRS.from_user_input(crs), transform, height, width

    @classmethod
    def from_bbox(cls, west, south, east, north, res=10.0):
        lon, lat = (west + east) / 2, (south + north) / 2
        epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
        l, b, r, t = transform_bounds("EPSG:4326", f"EPSG:{epsg}", west, south, east, north)
        w, h = int(np.ceil((r - l) / res)), int(np.ceil((t - b) / res))
        return cls(f"EPSG:{epsg}", from_origin(l, t, res, res), h, w)

    @classmethod
    def from_raster(cls, path):
        with rasterio.open(path) as s:
            return cls(s.crs, s.transform, s.height, s.width)

    @property
    def bounds(self):
        l, t = self.transform.c, self.transform.f
        return l, t + self.transform.e * self.height, l + self.transform.a * self.width, t

    def lonlat_bbox(self):
        return transform_bounds(self.crs, "EPSG:4326", *self.bounds)

    def profile(self, count, dtype, **kw):
        return dict(driver="GTiff", crs=self.crs, transform=self.transform, height=self.height,
                    width=self.width, count=count, dtype=dtype, compress="deflate", tiled=True, **kw)


def search(grid, start="2023-01-01", end="2024-12-31", max_cloud=20, n=8):
    """Least-cloudy scenes covering the grid. Mixes seasons on purpose (median is robust)."""
    items = Client.open(STAC).search(collections=[COLLECTION], bbox=grid.lonlat_bbox(),
                                     datetime=f"{start}/{end}",
                                     query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=400).item_collection()
    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])
    # ponytail: greedy by cloud cover until every point of a 20x20 lattice over the AOI is seen by n scenes.
    # Uses each scene's data footprint, so half-empty orbit-edge scenes no longer leave the AOI uncovered
    # (Tshwane, Vaal, Nakuru were ~2/3 empty with the old pick-n-per-MGRS-tile rule).
    w, s_, e, n_ = grid.lonlat_bbox()
    pts = shapely.points(*np.meshgrid(np.linspace(w, e, 20), np.linspace(s_, n_, 20)))
    seen, picked = np.zeros(pts.size, int), []
    for it in items:
        cov = shapely.contains(shapely.geometry.shape(it.geometry), pts).ravel()
        if (cov & (seen < n)).any():
            seen += cov
            picked.append(it)
        if (seen >= n).all():
            break
    return picked


def _read(href, grid, resampling):
    with rasterio.open(href) as src, WarpedVRT(src, crs=grid.crs, transform=grid.transform, height=grid.height,
                                               width=grid.width, resampling=resampling, nodata=0) as v:
        return v.read(1)


def _scale(item, asset):
    rb = item.assets[asset].extra_fields.get("raster:bands", [{}])[0]
    return rb.get("scale", 1e-4), rb.get("offset", 0.0)


def composite(grid, items, log=print, bands_at_once=3):
    """Median reflectance (H, W, len(BANDS)) float32, NaN where never clear.

    Reads are latency-bound (many small COG range requests), so run many in parallel:
    `bands_at_once` bands x all scenes. Memory peak ~ bands_at_once * scenes * H * W * 4 bytes.
    """
    with ThreadPoolExecutor(32) as ex:
        masks = list(ex.map(lambda it: _read(it.assets["scl"].href, grid, Resampling.nearest), items))
    clear = [~np.isin(m, BAD_SCL) for m in masks]
    del masks
    log(f"  clear-pixel coverage per scene: {[round(float(c.mean()), 2) for c in clear]}")
    use = [k for k, c in enumerate(clear) if c.any()]   # skip scenes with nothing clear over the AOI
    out = np.full((grid.height, grid.width, len(BANDS)), np.nan, np.float32)

    def load(b, k):
        it = items[k]
        dn = _read(it.assets[ASSET[b]].href, grid, Resampling.bilinear).astype(np.float32)
        s, o = _scale(it, ASSET[b])
        dn = dn * s + o
        dn[~clear[k] | (dn <= 0)] = np.nan
        return dn

    with ThreadPoolExecutor(min(48, bands_at_once * len(use))) as ex:
        for i in range(0, len(BANDS), bands_at_once):
            chunk = BANDS[i:i + bands_at_once]
            futs = {b: [ex.submit(load, b, k) for k in use] for b in chunk}
            for b in chunk:
                stack = np.stack([f.result() for f in futs[b]])
                out[..., BANDS.index(b)] = np.nanmedian(stack, axis=0)
                del stack, futs[b]
                log(f"  {b} done")
    return out


def save(path, grid, arr):
    """Store reflectance as uint16 (x10000) to keep disk use small; 0 = nodata."""
    q = np.nan_to_num(np.clip(arr * 10000, 1, 65535), nan=0).astype(np.uint16)
    with rasterio.open(path, "w", **grid.profile(q.shape[-1], "uint16", nodata=0)) as d:
        d.write(np.moveaxis(q, -1, 0))
        d.descriptions = tuple(BANDS)


def load(path):
    with rasterio.open(path) as s:
        a = np.moveaxis(s.read(), 0, -1).astype(np.float32)
        grid = Grid(s.crs, s.transform, s.height, s.width)
    a[a == 0] = np.nan
    return a / 10000, grid
