"""URBIS-Risk: flood and heat exposure on the same 10 m grid as the slum map.

Flood: height above nearest water (HAND-like proxy). Copernicus GLO-30 DEM minus the elevation of the
       closest open-water pixel (Sentinel-2 MNDWI > 0). < 5 m = low-lying, flood-exposed.
       ponytail: the DEM is a surface model (includes roofs) and straight-line nearest water ignores
       drainage paths; a true HAND with flow routing is the upgrade if a hydrology partner needs it.
Heat:  Landsat 8/9 Collection-2 surface temperature (Microsoft Planetary Computer), clear-sky median of
       the hottest season, as the anomaly vs the AOI median (deg C). > +2 = heat-exposed.
"""
import numpy as np
import pystac_client
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
LOW_M, HOT_C = 5.0, 2.0


def _warp(href, grid, resampling=Resampling.bilinear, nodata=None):
    with rasterio.open(href) as src, WarpedVRT(src, crs=grid.crs, transform=grid.transform, height=grid.height,
                                               width=grid.width, resampling=resampling,
                                               src_nodata=nodata if nodata is not None else src.nodata) as v:
        a = v.read(1, masked=True)
    return a.astype(np.float32).filled(np.nan)


def dem(grid):
    items = pystac_client.Client.open(EARTH_SEARCH).search(collections=["cop-dem-glo-30"],
                                                           bbox=grid.lonlat_bbox()).items()
    out = np.full((grid.height, grid.width), np.nan, np.float32)
    for it in items:
        a = _warp(it.assets["data"].href.replace("s3://copernicus-dem-30m/", "https://copernicus-dem-30m.s3.amazonaws.com/"), grid)
        out = np.where(np.isnan(out), a, out)
    return out


def height_above_water(elev, water):
    """Elevation minus elevation of the nearest water pixel (m). NaN if the AOI has no water."""
    if not water.any():
        return np.full(elev.shape, np.nan, np.float32)
    _, (iy, ix) = ndimage.distance_transform_edt(~water, return_indices=True)
    return (elev - elev[iy, ix]).astype(np.float32)


def lst(grid, years=("2023", "2024"), n=6):
    """Hot-season clear-sky median land-surface temperature (deg C), NaN where never clear."""
    lat = (grid.lonlat_bbox()[1] + grid.lonlat_bbox()[3]) / 2
    months = ("03-01", "09-30") if lat >= 0 else ("10-01", "03-31")
    items = []
    cat = pystac_client.Client.open(PC)
    for y in years:
        end_y = y if lat >= 0 else str(int(y) + 1)
        items += cat.search(collections=["landsat-c2-l2"], bbox=grid.lonlat_bbox(),
                            datetime=f"{y}-{months[0]}/{end_y}-{months[1]}",
                            query={"eo:cloud_cover": {"lt": 20}, "platform": {"in": ["landsat-8", "landsat-9"]}}).items()
    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:n]
    if not items:
        return np.full((grid.height, grid.width), np.nan, np.float32)
    tok = requests.get(f"{PC.rsplit('/stac', 1)[0]}/sas/v1/token/landsat-c2-l2", timeout=30).json()["token"]
    stack = []
    for it in items:
        st = _warp(f"{it.assets['lwir11'].href}?{tok}", grid, nodata=0)
        qa = _warp(f"{it.assets['qa_pixel'].href}?{tok}", grid, Resampling.nearest, nodata=1)
        bad = np.isnan(qa) | ((np.nan_to_num(qa).astype(np.uint16) & 0b11110) > 0)   # dilated cloud, cirrus, cloud, shadow
        stack.append(np.where(bad, np.nan, st * 0.00341802 + 149.0 - 273.15))
    return np.nanmedian(np.stack(stack), 0).astype(np.float32)


def layers(grid, s2_reflectance, log=print):
    """dict of float32 rasters: elev, hand (m above nearest water), lst (C), heat_anom (C vs AOI median)."""
    r = s2_reflectance
    g, sw = r[..., 1], r[..., 8]
    water = np.nan_to_num((g - sw) / (g + sw + 1e-6)) > 0
    water = ndimage.binary_opening(water, iterations=2)        # drop single-pixel shadows / dark roofs
    elev = dem(grid)
    log("DEM loaded")
    hand = height_above_water(elev, water)
    hand[water] = 0
    t = lst(grid)
    log("Landsat surface temperature loaded")
    return dict(elev=elev, hand=hand, lst=t, heat_anom=t - np.nanmedian(t))


if __name__ == "__main__":
    # self-check: a valley with a river at x=0 rising 1 m per pixel -> height above water == x
    elev = np.tile(np.arange(20, dtype=np.float32), (5, 1)) + 7
    water = np.zeros_like(elev, bool); water[:, 0] = True
    h = height_above_water(elev, water)
    assert np.allclose(h, np.arange(20)), h[0]
    print("ok")
