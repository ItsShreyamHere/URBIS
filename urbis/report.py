"""Self-contained HTML map report: python -m urbis.report runs/<name>/slum_prob.tif [threshold] [W,S,E,N]"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .patches import SCREEN  # noqa: E402

MAX_PX = 2400


def _to_mercator(src, band_idx, resampling, dtype, nodata):
    """Reproject bands to EPSG:3857 (Leaflet's projection) so the overlay lines up exactly."""
    t, w, h = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
    f = max(1, int(np.ceil(max(w, h) / MAX_PX)))
    w, h = w // f, h // f
    t = t * t.scale(f, f)
    out = np.full((len(band_idx), h, w), nodata, dtype)
    for i, b in enumerate(band_idx):
        reproject(rasterio.band(src, b), out[i], dst_transform=t, dst_crs="EPSG:3857",
                  resampling=resampling, dst_nodata=nodata)
    l, b_, r, tp = rasterio.transform.array_bounds(h, w, t)
    wst, s, e, n = transform_bounds("EPSG:3857", "EPSG:4326", l, b_, r, tp)
    return out, [[s, wst], [n, e]]


def _png(rgba):
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stats(prob_path, threshold):
    """Area statistics in the raster's native (metric UTM) grid."""
    with rasterio.open(prob_path) as s:
        p = s.read(1)
        px_km2 = abs(s.transform.a * s.transform.e) / 1e6
    valid = p != 255
    hist = np.bincount(p[valid].ravel(), minlength=251)[:251]
    slum = valid & (p / 250 >= threshold)
    lab, n = ndimage.label(slum)
    sizes = np.bincount(lab.ravel())[1:] * px_km2 if n else np.array([0.0])
    return dict(valid_km2=round(float(valid.sum() * px_km2), 1),
                slum_km2=round(float(slum.sum() * px_km2), 2),
                slum_share_pct=round(100 * float(slum.sum()) / max(int(valid.sum()), 1), 2),
                settlements=int((sizes >= 0.005).sum()),         # >= 0.5 ha (URBIS definition)
                largest_km2=round(float(sizes.max()), 2),
                threshold=threshold), dict(hist=hist.tolist(), px_km2=px_km2)


def ranges(st, threshold):
    """Likely range of the true slum area and people at the shipped threshold, from how far mapped area
    strayed from labelled area in 16 cities the model never saw (interquartile, scripts/area_error.py)."""
    p = Path(__file__).parent / "weights" / "area_error.json"
    if not p.exists():
        return
    ae = json.loads(p.read_text())
    if abs(threshold - ae["threshold"]) > 1e-3:      # calibrated runs have their own, smaller error
        return
    lo, hi = 1 / ae["quantiles"]["q75"], 1 / ae["quantiles"]["q25"]
    sig = lambda x: float(f"{x:.2g}")
    st["slum_km2_range"] = [sig(st["slum_km2"] * lo), sig(st["slum_km2"] * hi)]
    if st.get("slum_pop") is not None:
        st["slum_pop_range"] = [int(sig(st["slum_pop"] * lo)), int(sig(st["slum_pop"] * hi))]


def build(prob_path, threshold=0.5, title=None, bbox=None, profile=None):
    """bbox (lon/lat) enables the WorldPop population estimate; skipped with a warning if offline.
    profile = (summary, settlement rows) from urbis.profile.run adds exposure cards and the priority table."""
    prob_path = Path(prob_path)
    st, live = stats(prob_path, threshold)
    rows = []
    if profile:
        st.update(profile[0]); rows = profile[1][:25]
    elif bbox:
        from . import population
        try:
            st.update(population.estimate(prob_path, threshold, bbox))
        except Exception as e:  # network/API failure must not lose the map
            print(f"warning: population estimate skipped ({e})")
    ranges(st, threshold)
    with rasterio.open(prob_path) as s:
        p, bounds = _to_mercator(s, [1], Resampling.nearest, "uint8", 255)
    g = p[0]                                   # raw uint8 probability codes, recoloured in the browser
    prob_png = _png(np.dstack([g, g, g, np.full_like(g, 255)]))
    layers, over_layers = {}, {}

    s2p = prob_path.parent / "s2.tif"
    if s2p.exists():
        with rasterio.open(s2p) as s:
            rgb, _ = _to_mercator(s, [3, 2, 1], Resampling.bilinear, "uint16", 0)
        rgb = np.moveaxis(rgb, 0, -1).astype(np.float32)
        ok = (rgb > 0).all(-1)
        lo, hi = np.percentile(rgb[ok], [2, 98]) if ok.any() else (0, 1)
        img = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8
        layers["Sentinel-2 true colour"] = _png(np.dstack([img, ok.astype(np.float32)]))

    lv = prob_path.parent / "living.tif"
    if lv.exists():
        with rasterio.open(lv) as s:
            a, lb = _to_mercator(s, [1], Resampling.nearest, "float32", np.nan)
        v = a[0]
        ok = np.isfinite(v)
        rgba = np.zeros(v.shape + (4,), np.float32)
        t = np.nan_to_num(v)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.2 + 0.75 * t, 0.35 + 0.25 * (1 - t), 0.85 - 0.6 * t
        rgba[..., 3] = ok * (0.15 + 0.6 * t)
        over_layers["Relative deprivation, 1 km (URBIS-Living)"] = [_png(rgba), lb]

    raw = next((prob_path.with_name(f) for f in ("prob_screen.tif", "prob_raw.tif") if prob_path.with_name(f).exists()),
               prob_path)
    with rasterio.open(raw) as src:
        a, sb = _to_mercator(src, [1], Resampling.nearest, "uint8", 255)
    scr = (a[0] != 255) & (a[0] >= round(SCREEN * 250))
    rgba = np.zeros(scr.shape + (4,), np.float32)
    rgba[scr] = (0.35, 0.75, 0.95, 0.45)
    over_layers["Field screen (survey here)"] = [_png(rgba), sb]

    title = title or prob_path.parent.name.replace("_", " ").title()
    html = TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(
        dict(bounds=bounds, layers=layers, over=over_layers, stats=st, prob=prob_png, rows=rows, **live)))
    out = prob_path.with_name("report.html")
    out.write_text(html)
    return out, st


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>URBIS-1 · __TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
:root{--bg:#f7f5f0;--fg:#1d1b18;--muted:#6b665d;--card:#fffdf8;--accent:#d94f3d;--line:#e4dfd4}
@media (prefers-color-scheme:dark){:root{--bg:#141311;--fg:#ece8df;--muted:#a19b8f;--card:#1e1c19;--line:#34302a}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:20px 20px 8px}h1{margin:0;font-size:22px;letter-spacing:-.01em}h1 b{color:var(--accent)}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;padding:12px 20px}
.k{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.k .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}.k .l{color:var(--muted);font-size:12px}
#map{height:68vh;margin:0 20px;border-radius:12px;border:1px solid var(--line)}
footer{color:var(--muted);font-size:12px;padding:12px 20px 24px}
.ctl{background:var(--card);color:var(--fg);padding:8px 10px;border-radius:8px;font-size:13px}
.ctl input{width:140px;vertical-align:middle}
.note{margin:8px 20px 0;padding:10px 14px;border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;background:var(--card);font-size:14px}
#prio{padding:14px 20px 0;overflow-x:auto}#prio h2{font-size:16px;margin:0 0 4px}#prio p{color:var(--muted);font-size:12px;margin:0 0 8px}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:500}
td:first-child,th:first-child{text-align:left}tbody tr{cursor:pointer}tbody tr:hover{background:var(--card)}
</style></head><body>
<header><h1><b>URBIS-1</b> · __TITLE__</h1>
<div class="sub">Slum and informal-settlement probability from Sentinel-2 (10 m). Model output, not an official census.</div></header>
<div class="note" id="note" hidden></div>
<section class="stats" id="stats"></section>
<div id="map" role="application" aria-label="Map of slum probability"></div>
<section id="prio"></section>
<footer>Slum area and share follow the threshold slider. The likely ranges come from how far mapped area strayed from labelled area in 16 cities the model never saw (middle half of cities); calibrating with a known slum share narrows them. Population and settlement counts use the model's default threshold; settlements counted at ≥ 0.5 ha. Contains modified Copernicus Sentinel data. Suggested sites (URBIS-Plan) are chosen greedily to reach the most slum residents now over 1 km (straight line) from a mapped facility; land and access must be checked on the ground. Population: WorldPop 2020 (100 m), a lower-bound estimate in dense settlements. Basemap © OpenStreetMap contributors.</footer>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const D=__DATA__, s=D.stats;
const fmt=(x,d=2)=>x.toLocaleString(undefined,{maximumFractionDigits:d});
const cards=[["slumkm","Slum area",fmt(s.slum_km2)+" km²"],["slumpct","Share of mapped area",s.slum_share_pct+" %"],
["n","Settlements ≥0.5 ha (default threshold)",s.settlements.toLocaleString()],["big","Largest settlement",s.largest_km2+" km²"],
["area","Area mapped",fmt(s.valid_km2,1)+" km²"],["th","Threshold",s.threshold.toFixed(2)]];
if(s.slum_pop!=null) cards.splice(2,0,["pop","People in slums (est., default threshold)",s.slum_pop.toLocaleString()]);
if(s.slum_pop_share_pct!=null) cards.splice(3,0,["sdg","Share of population · SDG 11.1.1",s.slum_pop_share_pct+" %"]);
if(s.slum_km2_range) cards.splice(1,0,["slumrange","Likely true slum area (see footnote)",fmt(s.slum_km2_range[0])+"–"+fmt(s.slum_km2_range[1])+" km²"]);
if(s.slum_pop_range) cards.splice(cards.findIndex(c=>c[0]=="pop")+1,0,["poprange","Likely people in slums (see footnote)",s.slum_pop_range[0].toLocaleString()+"–"+s.slum_pop_range[1].toLocaleString()]);
const vs=(a,b,u,d=0)=>a==null?null:[fmt(a,d)+u,`vs ${fmt(b,d)}${u} elsewhere`];
[["flood","Slum area low-lying (<5 m above water)",vs(s.flood_slum_pct,s.flood_rest_pct," %")],
 ["heat","Surface heat in slums vs city median",s.heat_slum_c==null?null:[(s.heat_slum_c>0?"+":"")+fmt(s.heat_slum_c,1)+" °C",`vs ${(s.heat_rest_c>0?"+":"")+fmt(s.heat_rest_c,1)} °C elsewhere`]],
 ["health","Median distance to health facility",vs(s.health_slum_m,s.health_rest_m," m")],
 ["school","Median distance to school",vs(s.school_slum_m,s.school_rest_m," m")],
 ["road","Roofs over 50 m from a drivable road",vs(s.no_road_slum_pct,s.no_road_rest_pct," %")]].forEach(([id,l,v])=>{if(v)cards.push([id,`${l} · ${v[1]}`,v[0]])});
const KIND={health:["clinics","#2bb3c0"],school:["schools","#9b7be0"]};
for(const [k,v] of Object.entries(s.plan||{})) if(v.sites.length){const last=v.sites[v.sites.length-1];
  cards.push([`plan_${k}`,`${v.sites.length} new ${KIND[k][0]} would reach ${last.cumulative.toLocaleString()} of ${v.unserved.toLocaleString()} slum residents now over 1 km from one`,fmt(100*last.share_of_unserved,0)+" %"]);}
if(s.confirmed_settlements!=null) cards.push(["confirmed",`High-score settlements (patch check ≥ 0.95; 61 % real in cities it never saw) · ${fmt(s.confirmed_ha,1)} ha`,s.confirmed_settlements]);
if(s.screen_pct!=null) cards.push(["screen","Field screen: share of the area to survey; in every one of 16 cities it found settlements holding at least 95 % of slum land",s.screen_pct+" %"]);
document.getElementById("stats").innerHTML=cards.map(([id,l,v])=>`<div class="k"><div class="v" id="v_${id}">${v}</div><div class="l">${l}</div></div>`).join("");
if(s.slum_share_pct<1){const n=document.getElementById("note");n.hidden=false;
  n.textContent="Very little slum-like area was found here. At this level, detections are usually dense formal housing (small rowhouses, bungalow colonies), not slums. For neighbourhood deprivation in cities like this, use the URBIS-Living layer (urbis map --living).";}
const total=D.hist.reduce((a,b)=>a+b,0);
function liveStats(t){let n=0;for(let v=Math.ceil(t*250);v<=250;v++)n+=D.hist[v];
  document.getElementById("v_slumkm").textContent=fmt(n*D.px_km2)+" km²";
  document.getElementById("v_slumpct").textContent=fmt(100*n/Math.max(total,1))+" %";
  document.getElementById("v_th").textContent=t.toFixed(2);}
const map=L.map("map",{zoomSnap:0.25}).fitBounds(D.bounds);
addEventListener("load",()=>{map.invalidateSize();map.fitBounds(D.bounds)});
const osm=L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19,attribution:"© OpenStreetMap"}).addTo(map);
const base={"OpenStreetMap":osm}, over={};
for(const [name,url] of Object.entries(D.layers)) base[name]=L.layerGroup([osm,L.imageOverlay(url,D.bounds,{opacity:.9})]);
// probability layer: decode the grey PNG once, recolour on threshold change
const cv=document.createElement("canvas"), cx=cv.getContext("2d"), img=new Image();
let src=null, slum=L.imageOverlay(D.prob,D.bounds,{opacity:0}).addTo(map);
over["Slum probability"]=slum;
for(const [name,[url,b]] of Object.entries(D.over)) over[name]=L.imageOverlay(url,b,{opacity:.8});
function draw(t){if(!src)return;const out=cx.createImageData(cv.width,cv.height),o=out.data,c=Math.ceil(t*250);
  for(let i=0;i<src.length;i+=4){const v=src[i];if(v!==255&&v>=c){const a=v/250;
    o[i]=237;o[i+1]=64+100*(1-a);o[i+2]=89;o[i+3]=90+140*a;}}
  cx.putImageData(out,0,0);slum.setUrl(cv.toDataURL());liveStats(t);}
img.onload=()=>{cv.width=img.width;cv.height=img.height;cx.drawImage(img,0,0);
  src=cx.getImageData(0,0,cv.width,cv.height).data;slum.setOpacity(1);draw(s.threshold);};
img.src=D.prob;
const lc=L.control.layers(base,over,{collapsed:false}).addTo(map);
const ctl=L.control({position:"bottomleft"});
ctl.onAdd=()=>{const d=L.DomUtil.create("div","ctl");
  d.innerHTML=`<label>Threshold <input id="th" type="range" min="0.05" max="0.95" step="0.01" value="${s.threshold}"></label><br>
  <label>Opacity <input id="op" type="range" min="0" max="1" step="0.05" value="1"></label>`;
  L.DomEvent.disableClickPropagation(d);
  d.querySelector("#th").oninput=e=>draw(+e.target.value);
  d.querySelector("#op").oninput=e=>slum.setOpacity(+e.target.value);return d};
ctl.addTo(map);
if(D.rows.length){
  const pts=L.layerGroup(D.rows.map(r=>L.circleMarker([r.lat,r.lon],{radius:4+Math.sqrt(r.people||r.area_ha*100)/25,color:"#1d1b18",weight:1,fillColor:"#f2c14e",fillOpacity:.85})
    .bindPopup(`<b>#${r.rank}</b> · ${r.area_ha} ha${r.people!=null?` · ~${r.people.toLocaleString()} people`:""}<br>flood-prone ${r.flood_pct??"–"} % · heat ${r.heat_c??"–"} °C<br>health ${r.health_m??"–"} m · school ${r.school_m??"–"} m · no road ${r.no_road_pct??"–"} %${r.confidence!=null?`<br>confidence ${r.confidence}${r.confirmed?" (confirmed)":""}`:""}`))).addTo(map);
  lc.addOverlay(pts,"Priority settlements");
  const f=v=>v==null?"–":v.toLocaleString();
  document.getElementById("prio").innerHTML=`<h2>Priority settlements</h2><p>Ranked by people × need. Need = mean of flood-prone share, heat anomaly (/4 °C), distance to health and to school (/2 km), share of roofs over 50 m from a drivable road, each capped at 1. Weights are simple on purpose. Confidence is the patch check's score; ✓ marks a score ≥ 0.95 (61 % real in cities the model never saw: a pointer, not a guarantee). For service planning only, never for eviction.</p>
  <table><thead><tr><th>#</th><th>Area (ha)</th><th>People (est.)</th><th>Flood-prone %</th><th>Heat °C</th><th>Health m</th><th>School m</th><th>No road %</th><th>Need</th><th>Confidence</th></tr></thead><tbody>${
  D.rows.map(r=>`<tr data-ll="${r.lat},${r.lon}"><td>${r.rank}</td><td>${r.area_ha}</td><td>${f(r.people)}</td><td>${f(r.flood_pct)}</td><td>${f(r.heat_c)}</td><td>${f(r.health_m)}</td><td>${f(r.school_m)}</td><td>${f(r.no_road_pct)}</td><td>${r.need??"–"}</td><td>${r.confidence==null?"–":r.confidence+(r.confirmed?" ✓":"")}</td></tr>`).join("")}</tbody></table>`;
  document.querySelectorAll("#prio tbody tr").forEach(tr=>tr.onclick=()=>{map.setView(tr.dataset.ll.split(",").map(Number),16);scrollTo({top:0,behavior:"smooth"})});
}
for(const [k,v] of Object.entries(s.plan||{})) if(v.sites.length)
  lc.addOverlay(L.layerGroup(v.sites.map(r=>L.circleMarker([r.lat,r.lon],{radius:8,color:"#fff",weight:2,fillColor:KIND[k][1],fillOpacity:1})
    .bindPopup(`<b>Suggested ${KIND[k][0].slice(0,-1)} site #${r.rank}</b><br>reaches ~${r.people_reached.toLocaleString()} more slum residents within 1 km (straight line)<br>check land and access on the ground`))).addTo(map),
    `Suggested new ${KIND[k][0]} (URBIS-Plan)`);
</script></body></html>"""


if __name__ == "__main__":
    bb = tuple(map(float, sys.argv[3].split(","))) if len(sys.argv) > 3 else None
    out, st = build(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.5, bbox=bb)
    print(out, st)
