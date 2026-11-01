"""Plain (HC0) vs leverage-corrected (HC3) sandwich standard errors in small samples.

Null rejection rate of the two-group test (theta = 0 in both groups) at 20 and 30
dyads per group, model as assumed (slow drive, normal rates, phi = 0.7, 12-min joint
and solo sessions, m = 4, random pairing), 6,000 replicates.
Run: python hc_comparison.py   (numpy, scipy)
"""
import numpy as np
from scipy.stats import norm
from robustness_sim import counts, draw_people, MU, TS

Z = norm.ppf(0.975)


def group(reps, n, rng, theta=0.0, T=740.0, m=4, sp=0.013):
    bi = draw_people((reps, n), sp, rng, False); bj = draw_people((reps, n), sp, rng, False)
    si = counts(bi[..., None] * np.ones(m) * TS, rng) / TS
    sj = counts(bj[..., None] * np.ones(m) * TS, rng) / TS
    hh = 1.0 if theta == 0 else (1 - np.exp(-theta)) / theta
    ki = counts(T * ((1 + hh) / 2 * bi + (1 - hh) / 2 * bj), rng)
    kj = counts(T * ((1 + hh) / 2 * bj + (1 - hh) / 2 * bi), rng)
    bhi, bhj = si.mean(-1), sj.mean(-1)
    y = (bhi + bhj) * (kj - ki) / np.maximum(ki + kj, 1); x = bhj - bhi
    v = (si.var(-1, ddof=1) + sj.var(-1, ddof=1)) / m
    w = x * x - v; den = w.sum(1); est = (y * x).sum(1) / den
    psi = y * x - est[:, None] * w
    se0 = np.sqrt((psi ** 2).sum(1)) / np.abs(den)
    se3 = np.sqrt(((psi / (1 - w / den[:, None])) ** 2).sum(1)) / np.abs(den)
    return est, se0, se3


if __name__ == "__main__":
    rng = np.random.default_rng(20261007)
    R = 6000
    print(f"{'dyads per group':>16s} {'HC0 size':>9s} {'HC3 size':>9s}   (MC SE about 0.003)")
    for n in (20, 30, 50, 100):
        a0, a_s0, a_s3 = group(R, n, rng); b0, b_s0, b_s3 = group(R, n, rng)
        r0 = np.mean(np.abs(a0 - b0) / np.hypot(a_s0, b_s0) > Z)
        r3 = np.mean(np.abs(a0 - b0) / np.hypot(a_s3, b_s3) > Z)
        print(f"{n:16d} {r0:9.3f} {r3:9.3f}")
