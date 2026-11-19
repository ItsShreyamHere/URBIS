"""URBIS app: local web interface for a city's slum mapping (2026-11-14).
  .venv/bin/python scripts/urbis_app.py [port]      -> http://localhost:8765
Review (officials answer slum / not slum per 100 m box, in the adopted AI order), view (settlement outlines + every house polygon,
numbered), rebuild (AI outlines from the answers: scripts/city_outlines.py), export (GeoJSON, KML, GeoPackage, Shapefile, FlatGeobuf,
CSV, GeoTIFF, HTML report). Cities: those with whole-city scores (see city_pipeline.py). Standard library server; no data leaves the machine
except Esri imagery tiles in the browser and the Overture building download on rebuild."""
import csv, hashlib, io, json, os, re, secrets, subprocess, sys, threading, time, urllib.parse, urllib.request, zipfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np, rasterio
from pyproj import Transformer
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import city_pipeline as cp
from urbis.house_quality import selected_houses_path

ROOT = Path(__file__).resolve().parent.parent; APP = ROOT / "app"; PIPE = ROOT / "runs/pipeline"
CACHE, LOCK, JOBS = {}, threading.Lock(), {}
STORE = ROOT / "runs/app"; STORE.mkdir(parents=True, exist_ok=True); SESS = {}; SETUP = []   # SETUP: city ids waiting for / in set-up (one at a time)

def jload_nolock(n): p = STORE / n; return json.loads(p.read_text()) if p.exists() else {}
jload = jload_nolock
def jsave(n, v):
    with LOCK: (STORE / (n + ".tmp")).write_text(json.dumps(v, indent=1)); os.replace(STORE / (n + ".tmp"), STORE / n)

def pwhash(pw, salt): return hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1).hex()

def signup(u, pw):
    u = u.strip().lower(); U = jload("users.json")
    if not re.fullmatch(r"[a-z0-9._-]{3,32}", u): raise ValueError("Username: use 3–32 lowercase letters, digits, dots, dashes or underscores.")
    if len(pw) < 8: raise ValueError("Password: use at least 8 characters.")
    if u in U: raise ValueError("That username is taken. Try another.")
    salt = secrets.token_hex(16); U[u] = {"salt": salt, "hash": pwhash(pw, salt), "created": time.time()}; jsave("users.json", U); return u

def login(u, pw):
    u = u.strip().lower(); r = jload("users.json").get(u)
    if not r or not secrets.compare_digest(r["hash"], pwhash(pw, r["salt"])): raise ValueError("Wrong username or password.")
    t = secrets.token_urlsafe(32); SESS[t] = u; jsave("sessions.json", SESS); return t

def city(c):
    if c not in CACHE: CACHE[c] = cp.load(c)
    return CACHE[c]

def setup_state(c):
    p = ROOT / f"runs/app_cities/{c}.status.json"; s = json.loads(p.read_text()) if p.exists() else {"step": "queued", "msg": "Waiting to start"}
    if c in SETUP and c not in RUN: s = {"step": "queued", "msg": "Waiting for another city's set-up to finish" if len(RUN) >= 2 else "Starting"}
    return s

def my_cities(u):
    out = []
    for c, m in jload("cities.json").items():
        if m["owner"] != u: continue
        st = setup_state(c); d = dict(m, id=c, setup=st)
        if st.get("step") == "ready":
            A = cp.answers(c); S = fc(PIPE / c / "settlements.geojson")["features"]
            d.update(checked=len(A), yes=sum(A.values()), settlements=len(S), area_ha=round(sum(f["properties"]["area_ha"] for f in S), 1), buildings=sum(f["properties"]["buildings"] for f in S))
        out.append(d)
    return sorted(out, key=lambda d: -d["created"])

RUN = {}                                                     # city id -> running set-up process

def setup_worker(small_only):
    """Two workers: one takes any city in order, one only small ones (<= 30 km2), so a quick test never waits behind a big city."""
    while True:
        with LOCK:
            M = jload_nolock("cities.json")
            c = next((x for x in SETUP if x not in RUN and x in M and (not small_only or M[x]["km2"] <= 30)), None)
            if c: RUN[c] = subprocess.Popen([sys.executable, str(ROOT / "scripts/new_city.py"), c, *map(str, M[c]["bbox"])], cwd=ROOT,
                                            stdout=open(ROOT / f"runs/app_cities/{c}.log", "w"), stderr=subprocess.STDOUT)
        if c:
            RUN[c].wait(); RUN.pop(c, None)
            if c in SETUP: SETUP.remove(c)
        time.sleep(2)

def delete_city(c):
    """Remove a city: stop its set-up if running, drop it from the list, delete its files."""
    import shutil
    if c in RUN: RUN[c].kill()
    elif c in SETUP: SETUP.remove(c)
    C = jload("cities.json"); C.pop(c, None); jsave("cities.json", C); CACHE.pop(c, None)
    for d in (ROOT / "data" / c, PIPE / c): shutil.rmtree(d, ignore_errors=True)
    for f in (ROOT / "runs/app_cities").glob(f"{c}.*"): f.unlink(missing_ok=True)

def new_city(u, name, bbox):
    name = name.strip()[:80]
    try: b = [float(v) for v in bbox]; assert len(b) == 4
    except (TypeError, ValueError, AssertionError): raise ValueError("That area is not valid. Search a place or draw a box.")
    if not name: raise ValueError("Give the city a name.")
    if not (-180 <= b[0] < b[2] <= 180 and -85 <= b[1] < b[3] <= 85): raise ValueError("That area is not valid. Search a place or draw a box.")
    km2 = (b[2] - b[0]) * 111.32 * abs(__import__("math").cos(__import__("math").radians((b[1] + b[3]) / 2))) * (b[3] - b[1]) * 110.57
    if km2 > 600: raise ValueError(f"The box is {km2:.0f} km². Keep it under 600 km².")
    if km2 < 1: raise ValueError("The box is too small. Draw at least 1 km².")
    c = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] + "-" + secrets.token_hex(3)
    C = jload("cities.json"); C[c] = {"owner": u, "name": name, "bbox": b, "created": time.time(), "km2": round(km2, 1)}; jsave("cities.json", C)
    (ROOT / "runs/app_cities").mkdir(parents=True, exist_ok=True); SETUP.append(c); return c

def demos():
    """Pre-computed demo cities (scripts/make_demo.py): SIMULATED review, the official map answered in place of an officer."""
    out = []
    for f in sorted((ROOT / "runs/app_cities").glob("demo-in-*.status.json")):
        s = json.loads(f.read_text())
        if s.get("demo") and s.get("label") and s.get("base"): out.append(dict(id=f.name[:-len(".status.json")], name=s["label"], msg=s["msg"], bbox=s["bbox"]))
    return out

def try_demo(u, d):
    """Give the visitor a private copy of a demo city, so their own clicks never change the cached baseline."""
    import shutil
    if d not in {x["id"] for x in demos()}: raise ValueError("Unknown demo city.")
    c = d + "-" + secrets.token_hex(3); shutil.copytree(PIPE / d, PIPE / c); shutil.copytree(ROOT / "data" / d, ROOT / "data" / c)
    for ext in ("npz", "status.json"): shutil.copy(ROOT / f"runs/app_cities/{d}.{ext}", ROOT / f"runs/app_cities/{c}.{ext}")
    s = json.loads((ROOT / f"runs/app_cities/{c}.status.json").read_text()); b = s["bbox"]
    s["base"] = False; (ROOT / f"runs/app_cities/{c}.status.json").write_text(json.dumps(s))   # a visitor copy is not listed as a demo
    km2 = (b[2] - b[0]) * 111.32 * abs(__import__("math").cos(__import__("math").radians((b[1] + b[3]) / 2))) * (b[3] - b[1]) * 110.57
    C = jload("cities.json"); C[c] = {"owner": u, "name": s["label"] + " · demo", "bbox": b, "created": time.time(), "km2": round(km2, 1), "demo": True}; jsave("cities.json", C)
    return c

DEV = ROOT / "runs/devlabels"

def dev_next(c, at=None):
    """Dev labelling (2026-11-19): next box for the owner to judge by eye; learns from the owner's own answers, no fixed rules.
    Boxes are grouped into 40 visual types (k-means on the model's two-view features). Until the owner has >= 3 slum and >= 3 not-slum
    answers: one box per unseen type (explore). Then a per-city model on the owner's labels (features + base score) picks:
    50 % most likely slum, 30 % most uncertain, 20 % unseen type. 'unsure' answers are ignored for training. Official answer never shown."""
    from sklearn.cluster import KMeans; from sklearn.linear_model import LogisticRegression
    C = city(c); f = DEV / f"{c}.csv"; L = {int(r["box"]): r["answer"] for r in csv.DictReader(open(f))} if f.exists() else {}
    sq, rk, Xp = C["sq"], C["rk"], C["Xp"]; pos = {int(s): i for i, s in enumerate(sq)}
    if "types" not in C: C["types"] = KMeans(40, n_init=4, random_state=0).fit_predict(Xp)
    T = C["types"]; free = np.array([int(s) not in L for s in sq]); rng = np.random.default_rng(len(L)); u = rng.random()
    seen = {T[pos[b]] for b in L if b in pos}; unseen = free & ~np.isin(T, list(seen))
    k = np.array([pos[b] for b, a in L.items() if b in pos and a in ("yes", "no")], int); y = np.array([L[int(sq[j])] == "yes" for j in k], int)
    explore = lambda: (int(rng.choice(np.flatnonzero(unseen))) if unseen.any() else int(rng.choice(np.flatnonzero(free))), "new_type")
    ih = ROOT / f"runs/app_cities/{c}_india.npy"; yes_i = k[y == 1]
    if ih.exists():   # India slum head (scripts/india_head.py): show what it calls slum (85 %) or is unsure about (15 %), nothing else
        pi = np.load(ih); fr = np.flatnonzero(free)
        i, src = (int(fr[np.argmax(pi[fr])]), "india_top") if u < .85 else (int(fr[np.argmin(abs(pi[fr] - .5))]), "india_unsure")
    elif len(yes_i) and u < .85 and (y.sum() < 3 or u < .4):   # look like the slums you confirmed (feature similarity to your yes boxes)
        Z = Xp / np.linalg.norm(Xp, axis=1, keepdims=True); sim = (Z @ Z[yes_i].T).max(1); fr = np.flatnonzero(free); i, src = int(fr[np.argmax(sim[fr])]), "like_your_slums"
    elif y.sum() < 3 or (len(y) - y.sum()) < 3: i, src = explore()
    else:
        X = np.c_[Xp, np.log(rk / (1 - rk + 1e-6) + 1e-9)]; p = LogisticRegression(C=.3, max_iter=500, class_weight="balanced").fit(X[k], y).predict_proba(X)[:, 1]
        fr = np.flatnonzero(free)
        if u < .5: i, src = int(fr[np.argmax(p[fr])]), "likely_slum"
        elif u < .8: i, src = int(fr[np.argmin(abs(p[fr] - .5))]), "uncertain"
        else: i, src = explore()
    src += f" · type {T[i]}"
    sg = DEV / f"{c}_suggest.csv"   # places Claude picked by looking at the imagery: shown first, for the owner to judge
    for r in (csv.DictReader(open(sg)) if sg.exists() and not at else []):
        X0, Y0 = Transformer.from_crs(4326, C["g"].crs, always_xy=True).transform(float(r["lon"]), float(r["lat"])); r0, q0 = rasterio.transform.rowcol(C["g"].transform, X0, Y0)
        b0 = int(r0) // 10 * C["W"] + int(q0) // 10
        if b0 in pos and b0 not in L: at, src = f"{r['lat']},{r['lon']}", "claude_pick: " + r["why"]; break
    if at:   # the owner asked for a place (lat,lon pasted from Google Maps), or a Claude pick: the box under it
        la0, lo0 = [float(v) for v in at.split(",")]; X0, Y0 = Transformer.from_crs(4326, C["g"].crs, always_xy=True).transform(lo0, la0)
        r0, q0 = rasterio.transform.rowcol(C["g"].transform, X0, Y0); b0 = int(r0) // 10 * C["W"] + int(q0) // 10
        if b0 not in pos: raise ValueError("No built-up box at that place in this city window.")
        i = pos[b0]; src = src if src.startswith("claude_pick") else "your_place"
    b = int(sq[i]); lo, la = cp.lonlat(C, [b])
    ih = ROOT / f"runs/app_cities/{c}_india.npy"; pin = float(np.load(ih)[i]) if ih.exists() else None
    return dict(box=b, lon=float(lo[0]), lat=float(la[0]), corners=cp.corners(C, b), rank_pct=round(100 * (1 - float(rk[i])), 2), source=src, done=len(L), india_p=pin)

def dev_save(c, d):
    DEV.mkdir(parents=True, exist_ok=True); f = DEV / f"{c}.csv"; new = not f.exists()
    with LOCK, open(f, "a", newline="") as o:
        w = csv.writer(o); new and w.writerow(["box", "answer", "useful", "comment", "source", "rank_pct", "t"])
        w.writerow([int(d["box"]), d["answer"], d.get("useful", ""), str(d.get("comment", ""))[:2000], d.get("source", ""), d.get("rank_pct", ""), round(time.time())])

def geocode(q):
    r = urllib.request.urlopen(urllib.request.Request("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": q, "format": "jsonv2", "limit": 6}),
                               headers={"User-Agent": "URBIS slum mapping (local research app)"}), timeout=15)
    return [{"name": x["display_name"], "lat": float(x["lat"]), "lon": float(x["lon"]),
             "bbox": [float(x["boundingbox"][2]), float(x["boundingbox"][0]), float(x["boundingbox"][3]), float(x["boundingbox"][1])]} for x in json.load(r)]

def fc(p):
    return json.loads(p.read_text()) if p.exists() else {"type": "FeatureCollection", "features": []}

def houses(c):
    """Use learned polygons only after they pass the fixed two-city F1 gate."""
    return fc(selected_houses_path(PIPE / c))

def ai_houses_approved(c):
    return selected_houses_path(PIPE / c).name == "houses_ai.geojson"

def settlements(c):
    S = fc(PIPE / c / "settlements.geojson"); H = houses(c)["features"]
    if ai_houses_approved(c):
        n = {}
        for h in H: n[h["properties"]["settlement"]] = n.get(h["properties"]["settlement"], 0) + 1
        for f in S["features"]:
            p = f["properties"]; p["footprint_buildings"] = p["buildings"]; p["buildings"] = n.get(p["settlement"], 0)
            p["buildings_per_ha"] = round(p["buildings"] / max(p["area_ha"], 1e-6), 1); p["houses_source"] = "URBIS roof model"
    return S

def box_feature(C, b, **props):
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[list(x) for x in cp.corners(C, b)]]}, "properties": {"box": int(b), **props}}

def status(c):
    st = setup_state(c); m = jload("cities.json").get(c, {})
    if st.get("step") != "ready": return {"city": c, "name": m.get("name", c), "bbox": m.get("bbox"), "setup": st}
    C = city(c); A = cp.answers(c); s = {"city": c, "name": m.get("name", c), "bbox": m.get("bbox"), "setup": st, "checked": len(A), "yes": sum(A.values()), "built_boxes": C["NB"], "share_checked": len(A) / C["NB"]}
    if A:
        sc = cp.rank(C, A); P = cp.polygon(C, A); l, kp = cp.left(C, A, sc, P)
        s["left_share"] = None if l != l else l / max(l + kp, 1e-9); s["stop"] = bool(s["left_share"] is not None and s["left_share"] < cp.STOP_T)
    F = settlements(c)["features"]
    s.update(houses_source="URBIS roof model" if ai_houses_approved(c) else "footprint maps", settlements=len(F), area_ha=round(sum(f["properties"]["area_ha"] for f in F), 1), buildings=sum(f["properties"]["buildings"] for f in F),
             built=(PIPE / c / "settlements.geojson").exists(), job=JOBS.get(c))
    return s

def queue(c, n):
    C = city(c); A = cp.answers(c); sc = cp.rank(C, A); sq, W = C["sq"], C["W"]; blocked = np.zeros((C["H"], W), bool)
    for b in A: r, q = divmod(b, W); blocked[max(r-1, 0):r+2, max(q-1, 0):q+2] = True
    free = np.array([s not in A for s in sq]); r, q = np.divmod(sq, W); cand = np.flatnonzero(free & ~blocked[r, q])
    if not len(cand): cand = np.flatnonzero(free)
    pick = []
    for i in cand[np.argsort(-sc[cand])]:
        rr, qq = divmod(int(sq[i]), W)
        if blocked[rr, qq] and len(cand) != free.sum(): continue
        pick.append(int(sq[i])); blocked[max(rr-1, 0):rr+2, max(qq-1, 0):qq+2] = True
        if len(pick) >= n: break
    rk = dict(zip(C["sq"].tolist(), C["rk"].tolist()))
    return {"type": "FeatureCollection", "features": [box_feature(C, b, n=i + 1, rank=round(rk[b], 4)) for i, b in enumerate(pick)]}

def answer(c, box, yes):
    with LOCK:
        A = cp.answers(c); A[int(box)] = bool(yes); d = PIPE / c; d.mkdir(parents=True, exist_ok=True)
        with open(d / "answers.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["box", "answer"]); [w.writerow([b, "yes" if v else "no"]) for b, v in A.items()]
    # the map redraws itself: first after 10 answers, then every 25 (ponytail: in-memory counter; after a restart the next answer rebuilds once)
    if JOBS.get(c) != "running" and len(A) >= 10 and (not (d / "settlements.geojson").exists() or len(A) - BUILT_AT.get(c, 0) >= 25):
        BUILT_AT[c] = len(A); rebuild(c)

def guess(c, top=.05):
    """Model pseudo-labels before any review: the city's top-scored built boxes, with their rank (1.0 = most slum-like)."""
    C = city(c); k = np.flatnonzero(C["rk"] >= 1 - top)
    return {"type": "FeatureCollection", "features": [box_feature(C, C["sq"][i], rank=round(float(C["rk"][i]), 4)) for i in k]}

def reviewed(c):
    C = city(c); return {"type": "FeatureCollection", "features": [box_feature(C, b, answer="yes" if v else "no") for b, v in cp.answers(c).items()]}

BUILT_AT = {}

def rebuild(c):
    def run():
        JOBS[c] = "running"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/city_outlines.py"), c], cwd=ROOT, capture_output=True, text=True)
        JOBS[c] = "done" if r.returncode == 0 else "failed: " + (r.stderr.strip().split("\n") or [""])[-1][:200]
    if JOBS.get(c) != "running": threading.Thread(target=run, daemon=True).start()

def export(c, fmt):
    """-> (filename, content type, bytes). Vector formats carry two layers: settlements and houses."""
    d = PIPE / c; S, H = settlements(c)["features"], houses(c)["features"]
    if fmt == "geojson":
        return f"{c}_urbis.geojson", "application/geo+json", json.dumps({"type": "FeatureCollection", "name": f"{c} slum settlements and houses",
                "features": [dict(f, properties={"layer": "settlement", **f["properties"]}) for f in S] + [dict(f, properties={"layer": "house", **f["properties"]}) for f in H]}).encode()
    if fmt in ("kml", "report"):
        p = d / ("settlements.kml" if fmt == "kml" else "report.html"); return f"{c}_settlements.kml" if fmt == "kml" else f"{c}_report.html", "application/vnd.google-earth.kml+xml" if fmt == "kml" else "text/html", p.read_bytes()
    if fmt == "csv":
        o = io.StringIO(); w = csv.writer(o); w.writerow(["settlement", "area_ha", "buildings", "buildings_per_ha", "confirmed_boxes", "lat", "lon", "google_maps"])
        for f in S: p = f["properties"]; w.writerow([p["settlement"], p["area_ha"], p["buildings"], p["buildings_per_ha"], p["confirmed_boxes"], p["lat"], p["lon"], p["google_maps"]])
        return f"{c}_settlements.csv", "text/csv", o.getvalue().encode()
    if fmt == "houses_csv":
        from shapely.geometry import shape
        o = io.StringIO(); w = csv.writer(o); w.writerow(["settlement", "house", "area_m2", "source", "lat", "lon", "wkt"])
        for f in H:
            g = shape(f["geometry"]); p = f["properties"]; w.writerow([p["settlement"], p["house"], p["area_m2"], p["source"], round(g.centroid.y, 7), round(g.centroid.x, 7), g.wkt])
        return f"{c}_houses.csv", "text/csv", o.getvalue().encode()
    if fmt == "tif":
        import rasterio
        from rasterio import features as rf
        from pyproj import Transformer
        from shapely.geometry import shape
        from shapely import transform as tf
        g = city(c)["g"]; tr = Transformer.from_crs(4326, g.crs, always_xy=True)
        M = rf.rasterize([(tf(shape(f["geometry"]), lambda a: np.c_[tr.transform(a[:, 0], a[:, 1])]), f["properties"]["settlement"]) for f in S],
                         out_shape=(g.height, g.width), transform=g.transform, dtype="uint16") if S else np.zeros((g.height, g.width), np.uint16)
        pr = g.profile; pr.update(dtype="uint16", count=1, nodata=0, compress="deflate"); b = io.BytesIO()
        with rasterio.MemoryFile() as mf:
            with mf.open(**pr) as o: o.write(M, 1); o.update_tags(description="URBIS slum settlements: pixel value = settlement number, 0 = none (10 m)")
            b.write(mf.read())
        return f"{c}_settlements_10m.tif", "image/tiff", b.getvalue()
    import pyogrio, shapely, tempfile
    from shapely.geometry import shape
    drv = {"gpkg": ("GPKG", ".gpkg"), "shp": ("ESRI Shapefile", ".shp"), "fgb": ("FlatGeobuf", ".fgb")}[fmt]
    with tempfile.TemporaryDirectory() as t:
        files = []
        for name, fs in (("settlements", S), ("houses", H)):
            if not fs: continue
            keys = list(fs[0]["properties"]); geom = shapely.to_wkb([shape(f["geometry"]) for f in fs])
            field = [np.array([f["properties"][k] for f in fs], dtype=object if isinstance(fs[0]["properties"][k], str) else None) for k in keys]
            path = Path(t) / (f"{c}{drv[1]}" if fmt == "gpkg" else f"{c}_{name}{drv[1]}")
            pyogrio.raw.write(path, geom, field, fields=keys, layer=name, driver=drv[0], crs="EPSG:4326", geometry_type="MultiPolygon" if fmt == "shp" else "Unknown",
                              append=fmt == "gpkg" and path.exists(), promote_to_multi=True)
            files.append(path)
        if fmt == "gpkg": return f"{c}_urbis.gpkg", "application/geopackage+sqlite3", files[0].read_bytes()
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
            for f in Path(t).iterdir(): z.write(f, f.name)
        return f"{c}_urbis_{fmt}.zip", "application/zip", b.getvalue()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json", name=None, cookie=None):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        if name: self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        if cookie is not None: self.send_header("Set-Cookie", f"urbis={cookie}; Path=/; HttpOnly; SameSite=Strict" + ("" if cookie else "; Max-Age=0"))
        self.end_headers(); self.wfile.write(b)
    def user(self):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "urbis" and v in SESS and SESS[v] in jload("users.json"): return SESS[v]   # a deleted account loses its sessions
    def route(self, method):
        u = urlparse(self.path); q = parse_qs(u.query); parts = [p for p in u.path.split("/") if p]
        try:
            if not parts or parts[0] != "api":
                f = (APP / ("index.html" if not parts else "/".join(parts))).resolve()
                if APP not in f.parents and f != APP / "index.html" or not f.is_file(): return self.send(404, {"error": "not found"})
                return self.send(200, f.read_bytes(), {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".jpg": "image/jpeg", ".png": "image/png"}.get(f.suffix, "application/octet-stream"))
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}") if method == "POST" else {}
            if parts[1:] == ["signup"]: signup(body.get("user", ""), body.get("password", ""))
            if parts[1] in ("signup", "login"): return self.send(200, {"ok": True}, cookie=login(body.get("user", ""), body.get("password", "")))
            if parts[1:] == ["logout"]:
                for part in (self.headers.get("Cookie") or "").split(";"):
                    k, _, v = part.strip().partition("="); SESS.pop(v, None) if k == "urbis" else None
                jsave("sessions.json", SESS)
                return self.send(200, {"ok": True}, cookie="")
            me = self.user()
            if not me: return self.send(401, {"error": "sign in first"})
            if parts[1:] == ["me"]: return self.send(200, {"user": me})
            if parts[1:] == ["demos"]: return self.send(200, demos())
            if parts[1] == "demo" and method == "POST" and len(parts) == 3: return self.send(200, {"id": try_demo(me, parts[2])})
            if parts[1:] == ["geocode"]: return self.send(200, geocode(q.get("q", [""])[0]))
            if parts[1:] == ["cities"]:
                if method == "POST": return self.send(200, {"id": new_city(me, body.get("name", ""), body.get("bbox", []))})
                return self.send(200, my_cities(me))
            c = parts[2] if len(parts) > 2 and parts[1] == "city" else None
            if jload("cities.json").get(c, {}).get("owner") != me: return self.send(404, {"error": "unknown city"})
            what = parts[3] if len(parts) > 3 else "status"
            if method == "POST" and what == "delete": delete_city(c); return self.send(200, {"ok": True})
            if what != "status" and setup_state(c).get("step") != "ready": return self.send(409, {"error": "this city is still being set up"})
            if method == "GET":
                if what == "status": return self.send(200, status(c))
                if what == "settlements": return self.send(200, settlements(c))
                if what == "houses": return self.send(200, houses(c))
                if what == "queue": return self.send(200, queue(c, int(q.get("n", ["40"])[0])))
                if what == "reviewed": return self.send(200, reviewed(c))
                if what == "guess": return self.send(200, guess(c))
                if what == "devnext": return self.send(200, dev_next(c, q.get("at", [None])[0]))
                if what == "export": n, t, b = export(c, parts[4]); return self.send(200, b, t, n)
            else:
                if what == "answer": answer(c, body["box"], body["answer"] == "yes"); return self.send(200, {"ok": True})
                if what == "rebuild": rebuild(c); return self.send(200, {"ok": True})
                if what == "devlabel": dev_save(c, body); return self.send(200, {"ok": True})
            self.send(404, {"error": "not found"})
        except ValueError as e: self.send(400, {"error": str(e)})
        except FileNotFoundError as e: self.send(404, {"error": f"not built yet: {Path(e.filename).name if e.filename else e}"})
        except Exception as e: self.send(500, {"error": f"{type(e).__name__}: {e}"})
    def do_GET(self): self.route("GET")
    def do_POST(self): self.route("POST")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    SESS.update(jload("sessions.json")); C0 = jload("cities.json"); SETUP += [c for c in C0 if setup_state(c).get("step") not in ("ready", "error")]   # resume unfinished set-ups
    for small in (False, True): threading.Thread(target=setup_worker, args=(small,), daemon=True).start()
    print(f"URBIS app on http://localhost:{port}", flush=True); ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
