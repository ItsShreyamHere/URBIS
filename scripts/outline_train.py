"""Click-guided settlement outline model (2026-11-14). Input: 512 px RGB (1.28 km, 2.5 m/px) + the confirmed box as a 4th channel.
Output: 128 x 128 slum logits (10 m). U-Net on a timm encoder (ImageNet weights), BCE + Dice on known pixels. Data from outline_data.py.
  python outline_train.py <train_dir> <out.pt> [epochs]          trains; val = every 10th training city (by name), prints val IoU of the seeded settlement
  python outline_train.py predict <model.pt> <dir> <out.npz>      probability maps for every window in <dir> (keys = file stems)"""
import glob, os, sys, time
from pathlib import Path
import numpy as np, timm, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy import ndimage as nd
DEV = "cuda" if torch.cuda.is_available() else "cpu"; ENC = os.environ.get("ENC", "resnet34")
MEAN, STD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)

class VitNet(nn.Module):
    """ENC=vit_*: DINOv3 sat ViT; features of 4 blocks (stride 16) fused, upsampled x4 to 128 x 128. FREEZE first blocks (default 16)."""
    def __init__(s):
        super().__init__(); s.e = timm.create_model(ENC, pretrained=os.environ.get("PRETRAINED", "1") == "1", in_chans=4, num_classes=0, dynamic_img_size=True); n = len(s.e.blocks)
        s.idx = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]; fz = int(os.environ.get("FREEZE", 16))
        for p in s.e.patch_embed.parameters(): p.requires_grad = True
        for b in s.e.blocks[:fz]:
            for p in b.parameters(): p.requires_grad = False
        global MEAN, STD; MEAN, STD = np.array(s.e.pretrained_cfg["mean"], np.float32), np.array(s.e.pretrained_cfg["std"], np.float32)
        if os.environ.get("INIT"):   # slum-tuned last blocks of the shipped two-view ranker (keys b.blocks.N...)
            sd = torch.load(os.environ["INIT"], map_location="cpu", weights_only=False)["sd"]
            print("init from ranker:", s.e.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("b.")}, strict=False).unexpected_keys[:3], flush=True)
        d = s.e.num_features; s.fuse = nn.Sequential(nn.Conv2d(4 * d, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        s.up = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Upsample(scale_factor=2, mode="bilinear"),
                             nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(64, 1, 1))
    def forward(s, x):
        fs = s.e.forward_intermediates(x, indices=s.idx, output_fmt="NCHW", intermediates_only=True)
        p = s.up(s.fuse(torch.cat(fs, 1))); return F.interpolate(p, size=(x.shape[-2] // 4, x.shape[-1] // 4), mode="bilinear", align_corners=False)[:, 0]

class Net(nn.Module):
    def __new__(cls, *a, **k): return VitNet() if ENC.startswith("vit") else super().__new__(cls)
    def __init__(s):
        super().__init__(); s.e = timm.create_model(ENC, pretrained=True, features_only=True, in_chans=4, out_indices=(1, 2, 3, 4)); ch = s.e.feature_info.channels()
        s.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in ch]); s.head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 1, 1))
    def forward(s, x):
        fs = s.e(x); p = s.lat[-1](fs[-1])
        for f, l in zip(fs[-2::-1], s.lat[-2::-1]): p = F.interpolate(p, size=f.shape[-2:], mode="bilinear", align_corners=False) + l(f)
        return s.head(p)[:, 0]                                  # stride 4 -> 128 x 128

def load(stem):
    im = (np.asarray(Image.open(stem + ".jpg"), np.float32) / 255 - MEAN) / STD; d = np.load(stem + ".npz")
    seed = np.kron(d["seed"], np.ones((4, 4), np.float32))[..., None]; return np.concatenate([im, seed], -1).transpose(2, 0, 1), d["y"], d["sid"]

def batches(stems, bs, aug):
    idx = np.random.permutation(len(stems)) if aug else np.arange(len(stems))
    for b in range(0, len(idx), bs):
        X, Y = [], []
        for i in idx[b:b + bs]:
            x, y, _ = load(stems[i])
            if aug:
                k = np.random.randint(4); x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
                if np.random.rand() < .5: x, y = x[:, :, ::-1], y[:, ::-1]
                x = x.copy(); x[:3] = x[:3] * np.random.uniform(.8, 1.2) + np.random.uniform(-.2, .2)
            X.append(np.ascontiguousarray(x)); Y.append(np.ascontiguousarray(y))
        yield torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))

def loss_fn(lg, y):
    k = y != 255; t = (y == 1).float(); w = k.float()
    bce = (F.binary_cross_entropy_with_logits(lg, t, reduction="none") * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(lg) * w; dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * w).sum() + 1); return bce + dice

def seeded_iou(P, y, sid, seed, t):
    A = P > t; cl, _ = nd.label(A, np.ones((3, 3))); ids = np.unique(cl[seed & A]); A = np.isin(cl, ids[ids > 0]); kn = y != 255
    return (A & sid & kn).sum() / max(((A | sid) & kn).sum(), 1), sid.sum()

@torch.no_grad()
def predict(m, stems, bs=16):
    m.eval(); out = []
    for X, _ in batches(stems, bs, False):
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            X = X.to(DEV); p = sum(torch.rot90(torch.sigmoid(m(torch.rot90(X, r, (2, 3))).float()), -r, (1, 2)) for r in range(4)) / 4
        out.append(p.cpu().numpy())
    return np.concatenate(out)

def train(d, out, ep=12):
    stems = sorted(f[:-4] for f in glob.glob(f"{d}/*.npz")); city = lambda s: Path(s).name.rsplit("_", 1)[0]; cs = sorted({city(s) for s in stems})
    vc = set(cs[::10]); tr = [s for s in stems if city(s) not in vc]; va = [s for s in stems if city(s) in vc]
    print(f"train {len(tr)} windows / {len(cs) - len(vc)} cities, val {len(va)} / {len(vc)}", flush=True)
    m = Net().to(DEV); BS = int(os.environ.get("BS", 8)); enc = [p for p in m.e.parameters() if p.requires_grad]; eid = {id(p) for p in enc}
    opt = torch.optim.AdamW([{"params": enc, "lr": 3e-4 * float(os.environ.get("ENCLR", 1))}, {"params": [p for p in m.parameters() if p.requires_grad and id(p) not in eid]}], 3e-4, weight_decay=1e-4); n = int(ep) * (len(tr) // BS + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in opt.param_groups], total_steps=n, pct_start=.1); sc = torch.amp.GradScaler(enabled=DEV == "cuda"); best = -1
    T0 = time.time()
    for e in range(int(ep)):
        if (time.time() - T0) / 60 > float(os.environ.get("MAXMIN", 1e9)): print("time budget reached", flush=True); break
        m.train(); t0 = time.time(); L = 0
        for i, (X, Y) in enumerate(batches(tr, BS, True)):
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"): lo = loss_fn(m(X.to(DEV)).float(), Y.to(DEV))
            opt.zero_grad(); sc.scale(lo).backward(); sc.step(opt); sc.update(); sch.step(); L += lo.item()
        P = predict(m, va); r = {t: [] for t in (.3, .4, .5, .6)}
        for p, s in zip(P, va):
            _, y, sid = load(s); seed = np.load(s + ".npz")["seed"]
            for t in r: r[t].append(seeded_iou(p, y, sid, seed, t))
        v = {t: np.average([a for a, _ in x], weights=[w for _, w in x]) for t, x in r.items()}; tb = max(v, key=v.get)
        print(f"epoch {e} loss {L / (i + 1):.3f} val IoU {', '.join(f'{t}:{x:.3f}' for t, x in v.items())} ({time.time() - t0:.0f}s)", flush=True)
        if v[tb] > best: best = v[tb]; torch.save({"sd": m.state_dict(), "t": float(tb), "enc": ENC, "val": float(best)}, out)
    print("best val IoU", round(best, 3), "saved", out, flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "predict":
        ck = torch.load(sys.argv[2], map_location=DEV, weights_only=False); ENC = ck["enc"]; m = Net().to(DEV); m.load_state_dict(ck["sd"])
        stems = sorted(f[:-4] for f in glob.glob(f"{sys.argv[3]}/*.npz")); P = predict(m, stems)
        np.savez_compressed(sys.argv[4], P=P.astype(np.float16), stems=np.array([Path(s).name for s in stems]), t=ck["t"]); print("predicted", len(stems), flush=True)
    else: train(*sys.argv[1:])
