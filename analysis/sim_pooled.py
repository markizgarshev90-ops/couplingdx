"""Pooled (population-level) estimation of the contraction factor h.

For each dyad d, with joint-meal counts k_i, k_j and m solo meals per person:
    y_d = (k_j - k_i)/(k_j + k_i) * S_hat_d        (~ h * Gamma_d, A and T cancel)
    x_d = Gamma_hat_d = beta_hat_j - beta_hat_i    (signed; labels are arbitrary)
    v_d = (s_i^2 + s_j^2)/m                        (replicate-based error variance of x_d)

  MoM  h = sum(y x) / sum(x^2 - v)                 errors-in-variables, latent baselines
  IV   h = sum(y x1) / sum(x2 x1)                  split-half baselines as instruments
  OLS  h = sum(y x) / sum(x^2)                     naive: attenuated toward 0 (= spurious coupling)

Two closeness groups (theta_low, theta_high); the test is a z-test on the difference
of the group estimates with sandwich standard errors.  Writes analysis/out/pooled_*.csv.
"""
import os, json, itertools
import numpy as np, pandas as pd
from simlib import h, counts, joint_meal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
F = json.load(open(os.path.join(OUT, "fic_numbers.json")))
PHI = F["PHI"]; MU = F["rate_mean"]; SDB = F["sd_b"]
TS = 740.0
Z = 1.959964
rng = np.random.default_rng(5)


def draw_group(n, R, theta, T, m, sigma_p, sigma_o, pairing, phi=PHI, Ts=TS):
    """Return y, x, v, x1, x2 arrays of shape (R, n)."""
    if pairing == "random":
        bi = MU + sigma_p * rng.standard_normal((R, n))
        bj = MU + sigma_p * rng.standard_normal((R, n))
    else:   # opposite terciles on ONE independent screening meal per candidate
        pool = MU + sigma_p * rng.standard_normal((R, 3 * n))
        scr_rate = pool + sigma_o * rng.standard_normal(pool.shape)
        scr = counts(Ts * np.maximum(scr_rate, 1e-6), phi, rng) / Ts
        order = np.argsort(scr + 1e-9 * rng.standard_normal(scr.shape), axis=1)
        srt = np.take_along_axis(pool, order, axis=1)
        lo, hi = srt[:, :n], srt[:, 2 * n:]
        perm = rng.permuted(np.tile(np.arange(n), (R, 1)), axis=1)
        hi = np.take_along_axis(hi, perm, axis=1)
        flip = rng.random((R, n)) < 0.5                    # arbitrary labels
        bi = np.where(flip, lo, hi); bj = np.where(flip, hi, lo)
    # joint occasion
    oi = bi + sigma_o * rng.standard_normal((R, n)); oj = bj + sigma_o * rng.standard_normal((R, n))
    ki, kj = joint_meal(np.maximum(oi, 1e-6), np.maximum(oj, 1e-6), T, theta, phi, rng)
    # solo meals
    def solo(b):
        r = b[..., None] + sigma_o * rng.standard_normal(b.shape + (m,))
        return counts(Ts * np.maximum(r, 1e-6), phi, rng) / Ts
    ri, rj = solo(bi), solo(bj)
    bhi, bhj = ri.mean(-1), rj.mean(-1)
    x = bhj - bhi; Sh = bhj + bhi
    v = (ri.var(-1, ddof=1) + rj.var(-1, ddof=1)) / m
    half = m // 2
    x1 = rj[..., :half].mean(-1) - ri[..., :half].mean(-1)
    x2 = rj[..., half:].mean(-1) - ri[..., half:].mean(-1)
    y = (kj - ki) / np.maximum(kj + ki, 1) * Sh
    return y, x, v, x1, x2


def estimators(y, x, v, x1, x2):
    n = y.shape[1]
    out = {}
    den = (x * x - v).sum(1); hm = (y * x).sum(1) / den
    psi = y * x - hm[:, None] * (x * x - v)
    out["mom"] = (hm, np.sqrt((psi ** 2).sum(1)) / np.abs(den))
    den = (x2 * x1).sum(1); hi_ = (y * x1).sum(1) / den
    psi = (y - hi_[:, None] * x2) * x1
    out["iv"] = (hi_, np.sqrt((psi ** 2).sum(1)) / np.abs(den))
    den = (x * x).sum(1); ho = (y * x).sum(1) / den
    psi = (y - ho[:, None] * x) * x
    out["ols"] = (ho, np.sqrt((psi ** 2).sum(1)) / den)
    return out


def analytic_var(n, theta, T, m, sigma_p, sigma_o, pairing, phi=PHI, Ts=TS, A=1.0):
    """Large-n variance of the MoM estimator (Supplementary S4)."""
    hh = h(theta); S = 2 * MU
    sig_e2 = 2 * hh ** 2 * sigma_o ** 2 + phi * S / (A * T)
    sig_u2 = (2 * sigma_o ** 2 + phi * S / Ts) / m
    if pairing == "random":
        EG2 = 2 * sigma_p ** 2
    else:
        R_ = sigma_p ** 2 / (sigma_p ** 2 + sigma_o ** 2 + phi * MU / Ts)
        EG2 = sigma_p ** 2 * (2 + 3.32 * R_)
    vpsi = sig_e2 * (EG2 + sig_u2) + hh ** 2 * EG2 * sig_u2 + hh ** 2 * sig_u2 ** 2 * (2 + 1 / max(m - 1, 1))
    return vpsi / (n * EG2 ** 2)


if __name__ == "__main__":
    from math import erf, sqrt
    rows = []
    for pairing, share, T, m, n in itertools.product(
            ("random", "screened"), (0.0, 0.5), (740.0, 1500.0, 3000.0), (2, 4, 8), (50, 100, 200, 400)):
        for thetas in ((0.0, 1.0), (0.0, 0.0)):
            R = 1000 if n >= 200 else 2000
            sigma_p = np.sqrt(1 - share) * SDB; sigma_o = np.sqrt(share) * SDB
            g = [estimators(*draw_group(n, R, th, T, m, sigma_p, sigma_o, pairing)) for th in thetas]
            rec = dict(n=n, T=T, m=m, occ_share=share, pairing=pairing, theta_low=thetas[0], theta_high=thetas[1])
            for k in ("mom", "iv", "ols"):
                (h0, s0), (h1, s1) = g[0][k], g[1][k]
                zst = (h0 - h1) / np.sqrt(s0 ** 2 + s1 ** 2)
                rec[f"{k}_reject"] = float(np.mean(np.abs(zst) > Z))
                rec[f"{k}_h_low_median"] = float(np.median(h0)); rec[f"{k}_h_high_median"] = float(np.median(h1))
                rec[f"{k}_sd_low"] = float(np.std(h0)); rec[f"{k}_se_low_mean"] = float(np.mean(s0))
                rec[f"{k}_cover_low"] = float(np.mean(np.abs(h0 - h(thetas[0])) <= Z * s0))
            va = analytic_var(n, thetas[0], T, m, sigma_p, sigma_o, pairing) + \
                analytic_var(n, thetas[1], T, m, sigma_p, sigma_o, pairing)
            dh = h(thetas[0]) - h(thetas[1])
            rec["analytic_sd_low"] = float(np.sqrt(analytic_var(n, thetas[0], T, m, sigma_p, sigma_o, pairing)))
            rec["analytic_power"] = float(0.5 * (1 + erf((abs(dh) / sqrt(va) - Z) / sqrt(2)))) if dh else 0.05
            rows.append(rec)
        print(pairing, share, T, m, n, f"power={rows[-2]['mom_reject']:.3f} (an {rows[-2]['analytic_power']:.3f})"
              f" size={rows[-1]['mom_reject']:.3f} ols_h0={rows[-1]['ols_h_low_median']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "pooled_grid.csv"), index=False)
