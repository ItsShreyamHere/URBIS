"""URBIS city onboarding: AI-ranked review by city officials -> slum map (2026-11-14).
Loop:  queue  -> officials answer yes/no per box in Google Maps / Earth -> answers.csv -> queue again ... -> export.
  .venv/bin/python scripts/city_pipeline.py queue  <city> [--n 200]   next boxes to check: runs/pipeline/<city>/batch_NN.{csv,kml}
  .venv/bin/python scripts/city_pipeline.py status <city>              answers so far, estimated slum left, stop advice
  .venv/bin/python scripts/city_pipeline.py export <city>              slum map: runs/pipeline/<city>/slums.{geojson,tif}
answers.csv columns: box,answer   (answer = yes / no; a "yes" means slum anywhere in the box, also a partly-slum box)
Ranking = the adopted review order (scripts/active_city.py METHODS=bayes): prior = general model rank; correction = logistic on the
city's image features learned from its own answers, weight chosen by cross-validation on those answers; spread: a box next to a
checked box waits until the first pass is done. Stop when the calibrated estimate of slum boxes left is under STOP_T of known
(0.03 = certified 99 % of settlements >= 3 ha in 23/23 held-out cities, scripts/estimate_stop.py). Needs the city's whole-city
scores + features (runs/kaggle_dense3all, dense3f2/3xf2): currently the 23 held-out cities.
SIM=1 queue/status answer from the official map (for demos/tests)."""
import argparse, csv, json, os
from pathlib import Path
import numpy as np, rasterio
from rasterio import features as rf
from pyproj import Transformer
from scipy import ndimage as nd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

STOP_T = .03
I = np.load("runs/kaggle_dense3all/index.npz"); D = np.load("runs/kaggle_dense3all/out/vb_scores_init.npy").astype(float)

def load(c):
    if Path(f"runs/app_cities/{c}.npz").exists():         # a city set up in the app (scripts/new_city.py): its own scores and features
        z = np.load(f"runs/app_cities/{c}.npz"); sq = z["sq"]; Xp = PCA(32, random_state=0).fit_transform(z["feat"].astype(np.float32)); Xp /= Xp.std(0) + 1e-9
        g = rasterio.open(f"data/{c}/labels.tif")
        return dict(sq=sq, rk=rankdata(z["score"]) / len(sq), Xp=Xp, NB=len(sq), W=g.width // 10, H=g.height // 10, g=g)
    j = np.flatnonzero(I["city"] == c); assert len(j), f"{c}: no whole-city scores"
    f = np.load(f"runs/context/{c}.npz"); X0 = f["X"]; nm = list(f["names"]); nh = X0[:, nm.index("non-housing land use share")] > .5
    j = j[~nh[I["sq"][j]]]; sq = I["sq"][j]; built = ((np.nan_to_num(X0[:, nm.index("count")]) >= 1) | (np.nan_to_num(X0[:, nm.index("cover")]) > 0)) & ~nh
    F = np.concatenate([np.load(f"runs/kaggle_dense{t}/out/vb_feats_init.npy", mmap_mode="r") for t in ("3f2", "3xf2")])
    Xp = PCA(32, random_state=0).fit_transform(np.asarray(F[j], np.float32)); Xp /= Xp.std(0) + 1e-9
    g = rasterio.open(f"data/{c}/labels.tif"); W = g.width // 10
    return dict(sq=sq, rk=rankdata(D[j]) / len(j), Xp=Xp, NB=int(built.sum()), W=W, H=g.height // 10, g=g)

def answers(c):
    p = Path(f"runs/pipeline/{c}/answers.csv")
    return {int(r["box"]): r["answer"].strip().lower() == "yes" for r in csv.DictReader(open(p))} if p.exists() else {}

def rank(C, A):
    """Current priority of every box given answers A {box: bool} (the adopted bayes order with cross-validated weight)."""
    sq, rk, Xp = C["sq"], C["rk"], C["Xp"]; pri = np.log(rk / (1 - rk + 1e-6) + 1e-9); pos = {s: i for i, s in enumerate(sq)}
    k = np.array([pos[b] for b in A if b in pos], int); y = np.array([A[b] for b in A if b in pos], int)
    if y.sum() < 5 or (len(y) - y.sum()) < 5: return pri
    d = LogisticRegression(C=.1, max_iter=300, class_weight="balanced").fit(Xp[k], y).decision_function(Xp)
    fo = np.arange(len(k)) % 3; dcv = np.zeros(len(k))
    for q in range(3):
        t = fo != q
        if 0 < y[t].sum() < t.sum(): dcv[~t] = LogisticRegression(C=.1, max_iter=300, class_weight="balanced").fit(Xp[k[t]], y[t]).decision_function(Xp[k[~t]])
    ab = LogisticRegression(C=1., max_iter=300, class_weight="balanced").fit(np.c_[pri[k], dcv], y).coef_[0]
    lam = float(np.clip(ab[1] / max(ab[0], 1e-3), 0, 3)) * min(1., y.sum() / 300)
    return pri + lam * d

def left(C, A, sc, P):
    """Calibrated expected slum boxes among unchecked boxes outside the current slum polygons, and boxes inside them."""
    pos = {s: i for i, s in enumerate(C["sq"])}; k = np.array([pos[b] for b in A if b in pos], int); y = np.array([A[b] for b in A if b in pos], int)
    if not (0 < y.sum() < len(y)): return np.nan, int(y.sum())
    p = LogisticRegression(C=1., max_iter=200).fit(sc[k, None], y).predict_proba(sc[:, None])[:, 1]; un = ~P[C["sq"]]; un[k] = False
    return float(p[un].sum()), int(P.sum())

def sim_answer(C, boxes):
    L = C["g"].read(1); H, W = C["H"], C["W"]; B = L[:H*10, :W*10].reshape(H, 10, W, 10); px = (B == 1).sum((1, 3)).ravel()
    return {int(b): bool(px[b] >= 20) for b in boxes}

def lonlat(C, boxes):
    r, q = np.divmod(np.asarray(boxes), C["W"]); x, y = rasterio.transform.xy(C["g"].transform, r * 10 + 5, q * 10 + 5)
    return Transformer.from_crs(C["g"].crs, 4326, always_xy=True).transform(np.asarray(x), np.asarray(y))

def corners(C, b):
    r, q = divmod(int(b), C["W"]); t = C["g"].transform; tr = Transformer.from_crs(C["g"].crs, 4326, always_xy=True)
    pts = [rasterio.transform.xy(t, r * 10 + dy, q * 10 + dx, offset="ul") for dy, dx in ((0, 0), (0, 10), (10, 10), (10, 0), (0, 0))]
    return [tr.transform(x, y) for x, y in pts]

def queue(c, n, C=None):
    C = C or load(c); A = answers(c); out = Path(f"runs/pipeline/{c}"); out.mkdir(parents=True, exist_ok=True)
    sc = rank(C, A); sq, W = C["sq"], C["W"]; blocked = np.zeros((C["H"], C["W"]), bool)
    for b in A: r, q = divmod(b, W); blocked[max(r-1, 0):r+2, max(q-1, 0):q+2] = True
    free = np.array([s not in A for s in sq]); r, q = np.divmod(sq, W); cand = np.flatnonzero(free & ~blocked[r, q])
    if not len(cand): cand = np.flatnonzero(free)
    pick = []
    for i in cand[np.argsort(-sc[cand])]:                       # spread inside the batch too
        rr, qq = divmod(int(sq[i]), W)
        if blocked[rr, qq] and len(cand) != free.sum(): continue
        pick.append(int(sq[i])); blocked[max(rr-1, 0):rr+2, max(qq-1, 0):qq+2] = True
        if len(pick) >= n: break
    k = len(list(out.glob("batch_*.csv"))); lo, la = lonlat(C, pick)
    with open(out / f"batch_{k:02d}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["box", "lat", "lon", "google_maps", "answer"])
        for b, x, y in zip(pick, lo, la): w.writerow([b, f"{y:.6f}", f"{x:.6f}", f"https://www.google.com/maps/@{y:.6f},{x:.6f},120m/data=!3m1!1e3", ""])
    pm = "".join(f"<Placemark><name>{b}</name><Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>"
                 f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{' '.join(f'{x},{y},0' for x, y in corners(C, b))}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>" for b in pick)
    (out / f"batch_{k:02d}.kml").write_text(f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>{c} batch {k}</name>{pm}</Document></kml>')
    print(f"{c}: batch {k} with {len(pick)} boxes -> {out}/batch_{k:02d}.csv/.kml  (checked so far {len(A)} = {100*len(A)/C['NB']:.1f} % of built boxes)")
    if os.environ.get("SIM"):
        A.update(sim_answer(C, pick))
        with open(out / "answers.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["box", "answer"]); [w.writerow([b, "yes" if v else "no"]) for b, v in A.items()]
    return C, A

def status(c, C=None, A=None):
    C = C or load(c); A = answers(c) if A is None else A; sc = rank(C, A); P = polygon(C, A); l, kp = left(C, A, sc, P)
    share = l / max(l + kp, 1e-9) if l == l else np.nan; done = share < STOP_T
    print(f"{c}: checked {len(A)} boxes ({100*len(A)/C['NB']:.1f} % of built), boxes in slum polygons {kp}, expected slum boxes left {l:.0f} -> left share {share:.3f} "
          f"{'-> STOP: map is complete (99 % of settlements >= 3 ha, certified across 23 test cities)' if done else '-> keep checking'}")
    return done

def polygon(C, A):
    """Slum boxes: confirmed boxes plus unchecked 8-connected neighbours in the city's top 10 % (LOCO-chosen q, scripts/polygons100.py)."""
    H, W = C["H"], C["W"]; yes = np.zeros(H * W, bool); yes[[b for b, v in A.items() if v]] = True
    S = np.zeros(H * W); S[C["sq"]] = C["rk"]; no = np.zeros(H * W, bool); no[[b for b, v in A.items() if not v]] = True
    cl, _ = nd.label(((S >= .9) & ~no | yes).reshape(H, W), np.ones((3, 3))); keep = np.unique(cl.ravel()[yes]); return np.isin(cl.ravel(), keep[keep > 0])

def export(c, C=None):
    C = C or load(c); A = answers(c); H, W = C["H"], C["W"]; yes = polygon(C, A)
    m = np.kron(yes.reshape(H, W), np.ones((10, 10), np.uint8)).astype(np.uint8); out = Path(f"runs/pipeline/{c}")
    pr = C["g"].profile; pr.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    full = np.zeros((C["g"].height, C["g"].width), np.uint8); full[:H*10, :W*10] = m
    with rasterio.open(out / "slums.tif", "w", **pr) as o: o.write(full, 1)
    tr = Transformer.from_crs(C["g"].crs, 4326, always_xy=True)
    fs = [{"type": "Feature", "properties": {"id": i}, "geometry": {"type": "Polygon", "coordinates": [[list(tr.transform(x, y)) for x, y in ring] for ring in geom["coordinates"]]}}
          for i, (geom, v) in enumerate(rf.shapes(full, mask=full > 0, transform=C["g"].transform)) if v]
    (out / "slums.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": fs}))
    print(f"{c}: {len(fs)} slum polygons, {int(yes.sum())} boxes -> {out}/slums.geojson, slums.tif")
    if os.environ.get("SIM"):   # demo grade against the official map (10 m pixels)
        L = C["g"].read(1); kn = L != 255; sl = L == 1; comp, nc = nd.label(sl, np.ones((3, 3))); sz = np.bincount(comp.ravel())
        hit = np.unique(comp[(full > 0) & sl]); big = np.flatnonzero(sz >= 300)[1:] if nc else []
        print(f"   vs official map: slum land inside the map {100*(full[sl] > 0).mean():.1f} %, map that is slum {100*sl[(full > 0) & kn].mean():.1f} %, "
              f"settlements >= 3 ha touched {np.isin(big, hit).sum()}/{len(big)}")

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["queue", "status", "export", "demo"]); a.add_argument("city"); a.add_argument("--n", type=int, default=200)
    x = a.parse_args()
    if x.cmd == "queue": queue(x.city, x.n)
    elif x.cmd == "status": status(x.city)
    elif x.cmd == "export": export(x.city)
    else:   # demo: SIM answers until the stop rule fires or 10 % checked, then export
        os.environ["SIM"] = "1"
        C = None
        while True:
            C, A = queue(x.city, x.n, C)
            if status(x.city, C, A) or len(A) >= .1 * C["NB"]: break
        export(x.city, C)
