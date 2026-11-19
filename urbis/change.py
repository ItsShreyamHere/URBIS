"""URBIS-Change: where informal settlements appeared or disappeared between two years.

Both years are scored by the same imagery-only model (building footprints have no history), so the
difference reflects the land, not a change of inputs. Output change.tif (int8): (p_to - p_from) x 100.
  new     : p_to >= threshold and p_from < threshold - MARGIN   (emerging / densifying settlements)
  cleared : p_from >= threshold and p_to < threshold - MARGIN   (redevelopment, clearance, flooding)
MARGIN keeps pixels hovering at the threshold from counting as change.
Archive limit: Sentinel-2 L2A Collection-1 on Earth Search starts in 2019 here.
"""
from pathlib import Path

import numpy as np
import rasterio

from . import model, s2

MARGIN = 0.15
S2ONLY = Path(__file__).parent / "weights" / "no_mumbai" / "s2only_urbis1.txt"   # Mumbai is benchmark-only


def year_map(grid, year, out_dir, booster, log=print):
    comp, prob = out_dir / f"s2_{year}.tif", out_dir / f"prob_{year}.tif"
    if not comp.exists():
        items = s2.search(grid, f"{year}-01-01", f"{year}-12-31", n=6)
        if not items:
            raise SystemExit(f"no clear Sentinel-2 scenes in {year}")
        log(f"{year}: {len(items)} scenes")
        s2.save(comp, grid, s2.composite(grid, items, log=lambda m: None))
    if not prob.exists():
        model.predict(booster, comp, prob)
    with rasterio.open(prob) as f:
        a = f.read(1)
    return np.where(a == 255, np.nan, a / 250).astype(np.float32)


def run(bbox, y0, y1, out_dir, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = s2.Grid.from_bbox(*bbox)
    booster, meta = model.load(S2ONLY)
    th = meta["threshold"]
    p0, p1 = year_map(grid, y0, out_dir, booster, log), year_map(grid, y1, out_dir, booster, log)
    st = summary(p0, p1, th, abs(grid.transform.a * grid.transform.e) / 1e6)
    d = np.where(np.isnan(p0) | np.isnan(p1), -128, np.round((p1 - p0) * 100)).astype(np.int8)
    with rasterio.open(out_dir / "change.tif", "w", **grid.profile(1, "int8", nodata=-128)) as f:
        f.write(d, 1)
        f.update_tags(1, meaning=f"slum probability change {y0}->{y1} x100", threshold=th)
    return dict(st, from_year=y0, to_year=y1, threshold=round(th, 3))


def summary(p0, p1, th, px_km2):
    ok = ~np.isnan(p0) & ~np.isnan(p1)
    a0, a1 = ok & (p0 >= th), ok & (p1 >= th)
    new = ok & (p1 >= th) & (p0 < th - MARGIN)
    gone = ok & (p0 >= th) & (p1 < th - MARGIN)
    km = lambda m: round(float(m.sum() * px_km2), 2)
    return dict(slum_km2_from=km(a0), slum_km2_to=km(a1), new_km2=km(new), cleared_km2=km(gone))


if __name__ == "__main__":
    # self-check: one pixel appears, one disappears, one hovers at the threshold (not counted)
    p0 = np.array([[0.1, 0.9, 0.55]], np.float32)
    p1 = np.array([[0.9, 0.1, 0.62]], np.float32)
    s = summary(p0, p1, 0.6, 1.0)
    assert s == dict(slum_km2_from=1.0, slum_km2_to=2.0, new_km2=1.0, cleared_km2=1.0), s
    print("ok")
