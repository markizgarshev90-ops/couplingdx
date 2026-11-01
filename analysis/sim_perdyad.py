"""Per-dyad ratio estimator: null behaviour, truncation, filtering, and recovery.

Writes analysis/out/perdyad_*.csv and perdyad_numbers.json.
"""
import os, json
import numpy as np, pandas as pd
from simlib import h, inv_h, counts, joint_meal, solo_meals

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
F = json.load(open(os.path.join(OUT, "fic_numbers.json")))
PHI = F["PHI"]; RATE = F["rate_mean"]; S = 2 * RATE; SDB = F["sd_b"]
TS = 740.0          # natural (FIC median) solo meal length, s
R = 6000
rng = np.random.default_rng(11)
Z = 1.959964


def one_cell(Gam, T, m, theta, phi=PHI, sigma_o=0.0, Ts=None, R=R, Gam_vec=None):
    """Simulate R dyads with true contrast Gam (or a vector), return per-replicate arrays."""
    Ts = T if Ts is None else Ts
    G = np.full(R, Gam) if Gam_vec is None else Gam_vec
    bi_p = S / 2 - G / 2; bj_p = S / 2 + G / 2
    bi = bi_p + sigma_o * rng.standard_normal(R); bj = bj_p + sigma_o * rng.standard_normal(R)
    ki, kj = joint_meal(bi, bj, T, theta, phi, rng)
    ri = solo_meals(bi_p, m, Ts, phi, sigma_o, rng); rj = solo_meals(bj_p, m, Ts, phi, sigma_o, rng)
    bhi, bhj = ri.mean(-1), rj.mean(-1)
    Gh, Sh = bhj - bhi, bhj + bhi
    D, Sig = kj - ki, kj + ki
    a = D / np.maximum(Sig, 1) * Sh                     # ~ h * Gamma
    with np.errstate(divide="ignore", invalid="ignore"):
        hh = a / Gh
    # plug-in variances (phi assumed known; occasion variance ignored, as a
    # per-dyad analyst would have to)
    va = phi * Sh ** 2 / np.maximum(Sig, 1)
    vb = phi * Sh / (m * Ts)
    nu_hat = np.abs(Gh) / np.sqrt(vb)
    return dict(hh=hh, a=a, b=Gh, va=va, vb=vb, nu_hat=nu_hat, G=G)


def summarise(r, theta):
    ht = h(theta)
    hh = np.where(np.isfinite(r["hh"]), r["hh"], -np.inf)  # Gh==0 -> undefined, count as failure
    th = inv_h(hh)                                         # +inf where hh <= 0
    q25, q50, q75 = np.percentile(th, [25, 50, 75])
    # old procedure: keep 0 < hh < 1 only
    keep = (hh > 0) & (hh < 1)
    th_keep = inv_h(hh[keep])
    # coverage: exact Fieller set membership, and delta-method interval
    a, b, va, vb = r["a"], r["b"], r["va"], r["vb"]
    fiel = (a - ht * b) ** 2 <= Z ** 2 * (va + ht ** 2 * vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_delta = np.abs(hh) * np.sqrt(va / a ** 2 + vb / b ** 2)
    delta = np.abs(hh - ht) <= Z * sd_delta
    return dict(theta=theta, p_noninvertible=float(np.mean(hh <= 0)),
                median_theta=float(q50), bias=float(q50 - theta),
                iqr=float(q75 - q25) if np.isfinite(q75) else np.inf,
                q25=float(q25), q75=float(q75),
                old_estimability=float(keep.mean()),
                old_median_theta=float(np.median(th_keep)) if keep.any() else np.nan,
                cover_fieller=float(fiel.mean()), cover_delta=float(np.nanmean(delta)))


rows = []
# ---- A. recovery grid: Gamma x T x m, theta in {0, 1}, sigma_o = 0, Ts = T ----
for Gam in (0.0124, 0.025, 0.0375, 0.05):
    for T in (740.0, 1500.0, 3000.0):
        z0 = Gam * T / np.sqrt(PHI * S * T)
        for m in (1, 2, 4, 8, 16, 64):
            nu = Gam / np.sqrt(PHI * S / (m * T))
            for theta in (0.0, 1.0):
                r = one_cell(Gam, T, m, theta)
                rows.append(dict(Gamma=Gam, T=T, m=m, z0=z0, nu=nu, **summarise(r, theta)))
A = pd.DataFrame(rows); A.to_csv(os.path.join(OUT, "perdyad_grid.csv"), index=False)

# ---- B. FIC-like null: median true contrast, natural meal length, m = 8 ----
fic = dict(Gam=F["true_contrast_median_random"], T=740.0, m=8)
null_fic = summarise(one_cell(fic["Gam"], fic["T"], fic["m"], 0.0), 0.0)
alt_fic = summarise(one_cell(fic["Gam"], fic["T"], fic["m"], 1.0), 1.0)

# ---- C. selecting dyads on their own noisy baselines (old protocol filter), theta = 0 ----
def filter_rows(r, label):
    out = []
    for thr in (0, 2, 3):
        sel = r["nu_hat"] >= thr
        hh = r["hh"][sel]; hh = np.where(np.isfinite(hh), hh, -1.0)
        out.append(dict(case=label, threshold=thr, frac_kept=float(sel.mean()),
                        median_h=float(np.median(hh)), median_theta=float(np.median(inv_h(hh))),
                        median_true_over_est_contrast=float(np.median(r["G"][sel] / np.maximum(np.abs(r["b"][sel]), 1e-12)))))
    return out
filt = filter_rows(one_cell(F["true_contrast_median_random"], 740.0, 8, 0.0, R=200000), "fixed contrast (FIC median)")
Gpop = np.abs(np.sqrt(2) * SDB * rng.standard_normal(200000))
filt += filter_rows(one_cell(None, 740.0, 8, 0.0, R=200000, Gam_vec=Gpop), "random pairing population")
FILT = pd.DataFrame(filt); FILT.to_csv(os.path.join(OUT, "perdyad_filter.csv"), index=False)

# ---- D. occasion variance: z0 saturates ----
occ = []
for share in (0.0, 0.25, 0.5):
    so = np.sqrt(share) * SDB
    for T in (740.0, 3000.0, 12000.0):
        r = one_cell(0.05, T, 8, 1.0, sigma_o=so, Ts=740.0)
        s = summarise(r, 1.0)
        occ.append(dict(share=share, sigma_o=so, T=T, **s))
OCC = pd.DataFrame(occ); OCC.to_csv(os.path.join(OUT, "perdyad_occasion.csv"), index=False)

json.dump(dict(fic_null=null_fic, fic_alt=alt_fic, fic_cell=fic), open(os.path.join(OUT, "perdyad_numbers.json"), "w"), indent=1)

pd.set_option("display.width", 200)
print(A[A.theta == 1][["Gamma", "T", "m", "z0", "nu", "p_noninvertible", "median_theta", "iqr",
                        "old_estimability", "cover_fieller", "cover_delta"]].round(3).to_string())
print("\nnull rows:")
print(A[A.theta == 0][["Gamma", "T", "m", "z0", "median_theta", "iqr", "old_estimability",
                        "old_median_theta", "cover_fieller", "cover_delta"]].round(3).to_string())
print("\nFIC null:", null_fic); print("FIC alt:", alt_fic)
print(FILT.round(3)); print(OCC.round(3))
