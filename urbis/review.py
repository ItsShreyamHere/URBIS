"""Flag-then-review: the way to a map that finds ~95 % of slums and marks almost nothing else.

No pixel model gets there alone on a new city (docs/VALIDATION.md). So the model flags generously (ensemble >= TH)
and a reviewer decides. To keep that cheap the reviewer works top-down: each flagged patch first, then only the
mixed ones are split into 200 m, 100 m, ... squares. A unit is accepted or rejected in one click; whatever is
still mixed at a level is split again; at the finest level the reviewer's majority call stands.
"""
import numpy as np
from scipy import ndimage

TH = 0.03                  # flag threshold (scripts/review_eval.py picks it on held-out cities, never validation)
LEVELS = (None, 20, 10, 5, 2)  # None = whole patch, then square sizes in 10 m px (200 m .. 20 m)
PURE = 0.95                # accept in one click when at least this share is slum
NONE = 0.01                # reject in one click only when at most this share is slum (a reviewer sees "no slum here")


def squares(mask, t):
    h, w = mask.shape
    ids = (np.arange(h)[:, None] // t) * ((w + t - 1) // t) + np.arange(w)[None] // t
    return np.where(mask, ids + 1, 0)


def perfect(fr, has, level, last):
    """A reviewer who is always right: -> (accept, reject) per unit."""
    return has & (fr >= (0.5 if last else PURE)), has & (fr <= (0.5 if last else NONE))


def simulate(prob, y, th=TH, levels=LEVELS, reviewer=perfect):
    """Review (labels y: 1 slum, 0 not, 255 ignored). reviewer(fr, has, level, last) -> (accept, reject) per unit, where fr is
    the true slum share; a noisy reviewer may answer wrongly. -> clicks, recall, precision, recall of the flags alone."""
    v, slum = y != 255, y == 1
    flag = np.nan_to_num(prob, nan=-np.inf) >= th   # edge band (255) stays in the units; it only has no truth
    keep, todo, clicks = np.zeros_like(flag), flag.copy(), 0
    patch = ndimage.label(flag, np.ones((3, 3)))[0]
    for i, t in enumerate(levels):
        ids = patch * todo if t is None else squares(todo, t)
        n = int(ids.max()) + 1
        real, tot = np.bincount(ids.ravel(), slum.ravel(), n), np.bincount(ids.ravel(), v.ravel(), n)
        fr, has = real / np.maximum(tot, 1), np.bincount(ids.ravel(), None, n) > 0
        has[0] = False
        clicks += int(has.sum())
        last = i == len(levels) - 1
        acc, rej = reviewer(fr, has, i, last)
        keep |= acc[ids] & todo
        todo &= ~(acc | rej)[ids]
    s = max(int(slum[v].sum()), 1)
    return clicks, (keep & slum).sum() / s, (keep & slum).sum() / max(int((keep & v).sum()), 1), (flag & slum).sum() / s


if __name__ == "__main__":
    # self-check: one pure slum block and one pure non-slum block -> 2 clicks, perfect map
    p = np.zeros((60, 60), np.float32); p[5:25, 5:25] = 0.9; p[40:50, 40:50] = 0.5
    y = np.zeros((60, 60), np.uint8); y[5:25, 5:25] = 1
    c, r, pr, fr = simulate(p, y)
    assert (c, r, pr, fr) == (2, 1.0, 1.0, 1.0), (c, r, pr, fr)
    print("ok")
