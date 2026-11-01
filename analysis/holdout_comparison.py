"""Holdout comparison of drive specifications on the FIC meals.

Each model is fitted by least squares to the cumulative bite curve of each of the
first 17 meals (25 random starts, lower bound 1e-6 on positive parameters); the
mean parameters then predict the cumulative curves of the remaining 4 meals
(entry_17 to entry_20). Scores: R^2 on the cumulative curve, R^2 on bites counted
in consecutive 60-s windows (increments), and mean absolute error in bites.
A negative increment R^2 means the model predicts within-meal variation worse
than the holdout meals' own mean count per window.

This reproduces the comparison of Kieckens & Izgarshev (2026), including its
random-number stream (the shared-gain model of that paper is fitted but not part
of this paper's argument). As a check that does not depend on one split, the same
scores are also computed leave-one-meal-out over all 21 meals.

Writes analysis/out/holdout.csv.
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
xl = pd.ExcelFile(os.path.join(ROOT, "data", "fic_bite_timelines.xlsx"))
data = [xl.parse(s) for s in xl.sheet_names]
LB, INF = 1e-6, np.inf

MODELS = {
    "constant rate": (lambda t, p: p[0] * t, [0.05], [LB], [INF]),
    "straight line": (lambda t, p: p[0] + p[1] * t, [0, 0.05], [-INF, LB], [INF, INF]),
    "saturating exponential": (lambda t, p: p[0] / p[1] * (1 - np.exp(-p[1] * t)), [0.05, 1e-3], [LB, LB], [INF, INF]),
    "shared-gain (other paper)": (lambda t, p: np.exp(np.clip(p[0] / p[1] * (1 - np.exp(-p[1] * t)), 0, 50)) - 1,
                                  [0.02, 5e-3], [LB, LB], [INF, INF]),
    "Kissileff quadratic": (lambda t, p: p[0] * t - p[1] * t ** 2, [0.05, 1e-5], [LB, 0], [INF, INF]),
    "Thomas logistic": (lambda t, p: p[0] / (1 + np.exp(-np.clip(p[1] * (t - p[2]), -50, 50))),
                        [80, 0.01, 300], [LB, LB, LB], [INF, INF, INF]),
}


def fit(f, t, y, x0, lo, hi, rng):
    best = None
    for _ in range(25):
        s = np.clip(np.array(x0) * 10 ** rng.uniform(-1, 1, len(x0)), np.array(lo) * 1.001, np.array(hi) * 0.999)
        try:
            r = least_squares(lambda p: f(t, p) - y, s, bounds=(lo, hi), max_nfev=5000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    return best.x


def score(f, pm, hold):
    yh, yo, ih, io = [], [], [], []
    for k in hold:
        d = data[k]; t = d.timestamp.values; y = d.cumulative_bites.values.astype(float)
        p = f(t, pm); yh.append(p); yo.append(y)
        idx = np.arange(0, len(t), 600)          # 0.1-s grid -> 60-s windows
        ih.append(np.diff(p[idx])); io.append(np.diff(y[idx]))
    yh, yo, ih, io = map(np.concatenate, (yh, yo, ih, io))
    r2 = lambda o, h: 1 - ((h - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    return r2(yo, yh), r2(io, ih), np.abs(yh - yo).mean()


rng = np.random.default_rng(0)
rows = []
for name, (f, x0, lo, hi) in MODELS.items():
    P = np.array([fit(f, data[k].timestamp.values, data[k].cumulative_bites.values.astype(float), x0, lo, hi, rng)
                  for k in range(17)])
    cum, inc, mae = score(f, P.mean(0), range(17, 21))
    rows.append(dict(model=name, split="17 fit / 4 held out", n_par=len(x0), cum_R2=cum, incr_R2=inc, MAE=mae))

# leave-one-meal-out over all 21 meals: fit every meal once, then predict each from the other 20
rng = np.random.default_rng(1)
for name, (f, x0, lo, hi) in MODELS.items():
    P = np.array([fit(f, data[k].timestamp.values, data[k].cumulative_bites.values.astype(float), x0, lo, hi, rng)
                  for k in range(21)])
    sc = [score(f, np.delete(P, k, 0).mean(0), [k]) for k in range(21)]
    # pooled scores across the 21 held-out meals
    yh, yo, ih, io = [], [], [], []
    for k in range(21):
        d = data[k]; t = d.timestamp.values; y = d.cumulative_bites.values.astype(float)
        p = f(t, np.delete(P, k, 0).mean(0)); yh.append(p); yo.append(y)
        idx = np.arange(0, len(t), 600); ih.append(np.diff(p[idx])); io.append(np.diff(y[idx]))
    yh, yo, ih, io = map(np.concatenate, (yh, yo, ih, io))
    r2 = lambda o, h: 1 - ((h - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    rows.append(dict(model=name, split="leave one meal out (21)", n_par=len(x0), cum_R2=r2(yo, yh),
                     incr_R2=r2(io, ih), MAE=np.abs(yh - yo).mean()))

R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "holdout.csv"), index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
