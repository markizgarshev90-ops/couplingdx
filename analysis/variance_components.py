"""Split between-meal variance of eating rate into stable and occasion parts.

Input: CSV with columns  subject, n_bites, duration_s  (one row per solo meal).
Method: per-meal rate r = N/T with counting variance phi*N/T^2.
  occasion variance  sigma_o^2 = pooled within-subject variance of r
                                  - mean counting variance (subjects with >= 2 meals)
  between-meal       sigma_b^2 = variance of r across meals - mean counting variance
  stable             sigma_p^2 = sigma_b^2 - sigma_o^2
Negative estimates are reported as 0. A bootstrap over SUBJECTS gives intervals.
Usage: python variance_components.py meals.csv [--phi 0.7] [--boot 5000]
"""
import argparse, csv, random, statistics as st
from collections import defaultdict


def components(rows, phi):
    r = [n / t for _, n, t in rows]
    cv = [phi * n / t ** 2 for _, n, t in rows]
    sb2 = st.variance(r) - st.fmean(cv)
    by = defaultdict(list)
    for (s, n, t), ri, ci in zip(rows, r, cv):
        by[s].append((ri, ci))
    ss = df = 0.0; cw = []
    for vals in by.values():
        if len(vals) >= 2:
            rs = [v[0] for v in vals]; m = st.fmean(rs)
            ss += sum((x - m) ** 2 for x in rs); df += len(rs) - 1
            cw += [v[1] for v in vals]
    so2 = ss / df - st.fmean(cw) if df else float("nan")
    sp2 = sb2 - so2
    return {"sigma_b": max(sb2, 0) ** .5, "sigma_o": max(so2, 0) ** .5,
            "sigma_p": max(sp2, 0) ** .5, "occasion_share": min(max(so2, 0) / sb2, 1.0) if sb2 > 0 else float("nan"),
            "subjects": len(by), "meals": len(rows), "within_df": int(df)}


def bootstrap(rows, phi, B, seed=1):
    rng = random.Random(seed)
    by = defaultdict(list)
    for row in rows:
        by[row[0]].append(row)
    subs = list(by)
    out = []
    for b in range(B):
        samp = []
        for k, s in enumerate(rng.choices(subs, k=len(subs))):
            samp += [(f"{s}#{k}", n, t) for _, n, t in by[s]]  # resampled subjects are distinct
        try:
            out.append(components(samp, phi))
        except st.StatisticsError:
            pass
    def ci(key):
        v = sorted(o[key] for o in out if o[key] == o[key])
        return v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
    return {k: ci(k) for k in ("sigma_b", "sigma_o", "sigma_p", "occasion_share")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--phi", type=float, default=0.7)
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    rows = [(d["subject"], float(d["n_bites"]), float(d["duration_s"])) for d in csv.DictReader(open(a.csv))]
    est = components(rows, a.phi); cis = bootstrap(rows, a.phi, a.boot)
    for k in ("sigma_b", "sigma_p", "sigma_o", "occasion_share"):
        print(f"{k:15s} {est[k]:.4f}   95% CI [{cis[k][0]:.4f}, {cis[k][1]:.4f}]")
    print(f"{est['subjects']} subjects, {est['meals']} meals, {est['within_df']} within-subject df")
