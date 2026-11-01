"""Quantities estimated from the FIC bite timelines.

Outputs analysis/out/fic_numbers.json and analysis/out/pseudo_dyads.csv.

* two-parameter Poisson ML fit of lambda(t)=beta*exp(-l t) per meal
  (used only to show that l is not identified at meal length)
* slow-drive drive estimate: the mean bite rate N/T, with sampling SE sqrt(phi*N)/T
* dispersion phi: Fano factor of window counts and squared CV of inter-bite intervals
* variance components: between-meal SD of the rate net of sampling variance
  (method of moments), with a bootstrap interval over meals
* pseudo-dyads: all 210 pairs of meals
"""
import os, json, itertools
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out"); os.makedirs(OUT, exist_ok=True)
B = pd.read_csv(os.path.join(ROOT, "data", "fic_bite_times.csv"))
M = pd.read_csv(os.path.join(ROOT, "data", "fic_meals.csv"))
rng = np.random.default_rng(2026)
times = [np.sort(B.t[B.meal == i].to_numpy()) for i in range(21)]
T = M["T"].to_numpy(); N = M["N"].to_numpy().astype(float)

# ---------- 1. two-parameter Poisson ML (identifiability of l) ----------
def poisson_mle(tn, T_):
    n = len(tn); s = tn.sum()
    def negll(u):
        l = np.exp(u); Mi = -np.expm1(-l * T_) / l; b = n / Mi
        return -(n * np.log(b) - l * s - b * Mi)
    r = minimize_scalar(negll, bounds=(np.log(1e-7), np.log(1.0)), method="bounded",
                        options=dict(xatol=1e-10))
    l = float(np.exp(r.x)); return n / (-np.expm1(-l * T_) / l), l
fits = [poisson_mle(times[i], T[i]) for i in range(21)]
l_hat = np.array([f[1] for f in fits])
n_l_bound = int((l_hat < 2e-7).sum())
lT = l_hat * T

# ---------- 2. dispersion ----------
def fano(tn, T_, w):
    nw = int(T_ // w)
    c = np.histogram(tn, bins=np.arange(0, nw * w + 1e-9, w))[0]
    return c.var(ddof=1) / c.mean(), nw
fano_tab = {}
for w in (30, 60, 120, 180):
    vals = np.array([fano(times[i], T[i], w) for i in range(21)])
    fano_tab[w] = float(np.average(vals[:, 0], weights=vals[:, 1] - 1))
cv2 = np.array([(np.diff(t).std(ddof=1) / np.diff(t).mean()) ** 2 for t in times])
phi_cv2 = float(np.average(cv2, weights=N - 1))
def phi_point(idx):
    vals = np.array([fano(times[i], T[i], 120) for i in idx])
    return np.average(vals[:, 0], weights=vals[:, 1] - 1)
phi_boot = np.array([phi_point(rng.integers(0, 21, 21)) for _ in range(4000)])
PHI = 0.70   # central value carried into the design calculations (see paper)

# ---------- 3. variance components of the mean rate ----------
rate = N / T
def comps(idx, phi):
    r = rate[idx]; samp = phi * r / T[idx]      # sampling variance of N/T
    tot = r.var(ddof=1); exc = tot - samp.mean()
    return tot, samp.mean(), exc
tot, samp1, exc1 = comps(np.arange(21), 1.0)
_, samp_phi, exc_phi = comps(np.arange(21), PHI)
boot = np.array([comps(rng.integers(0, 21, 21), PHI)[2] for _ in range(8000)])
sd_b = float(np.sqrt(max(exc_phi, 0)))
sd_b_ci = [float(np.sqrt(max(q, 0))) for q in np.percentile(boot, [2.5, 97.5])]
rel1 = exc_phi / tot                              # single-meal reliability

# ---------- 4. pseudo-dyads ----------
rows = []
for a, b in itertools.combinations(range(21), 2):
    G = abs(rate[b] - rate[a]); S = rate[a] + rate[b]; Tm = .5 * (T[a] + T[b])
    se_pair = np.sqrt(PHI * rate[a] / T[a] + PHI * rate[b] / T[b])
    sigD = np.sqrt(PHI * S * Tm)
    rows.append(dict(a=a, b=b, Gamma_hat=G, S=S, Tm=Tm, gap=G * Tm,
                     nu_hat=G / se_pair, z0_hat=G * Tm / sigD))
P = pd.DataFrame(rows)
# drop pairs of two meals from the same person when the participant link is available
# (data/fic_meals_subjects.csv, written by fic_subjects.py from the original FIC pickle)
SUBJ = os.path.join(ROOT, "data", "fic_meals_subjects.csv")
if os.path.exists(SUBJ):
    sid = pd.read_csv(SUBJ).set_index("meal").subject
    P["same_person"] = [bool(sid[a_] == sid[b_]) for a_, b_ in zip(P.a, P.b)]
else:
    P["same_person"] = False
P.to_csv(os.path.join(OUT, "pseudo_dyads.csv"), index=False)
n_pairs_all = len(P); n_same = int(P.same_person.sum())
P = P[~P.same_person]

nums = dict(
    n_meals=21, N_mean=float(N.mean()), T_mean=float(T.mean()), T_median=float(np.median(T)),
    rate_mean=float(rate.mean()), rate_pooled=float(N.sum() / T.sum()),
    l_at_bound=n_l_bound, lT_max=float(lT.max()), lT_median=float(np.median(lT)),
    fano=fano_tab, phi_cv2=phi_cv2, phi_fano120_ci=[float(x) for x in np.percentile(phi_boot, [2.5, 97.5])],
    PHI=PHI,
    sd_rate_between_meals=float(np.sqrt(tot)), se_rate_phi1=float(np.sqrt(samp1)),
    se_rate_phi=float(np.sqrt(samp_phi)),
    ratio_between_within_phi1=float(np.sqrt(tot / samp1)),
    ratio_between_within_phi=float(np.sqrt(tot / samp_phi)),
    sd_b=sd_b, sd_b_ci=sd_b_ci, sd_b_phi1=float(np.sqrt(max(exc1, 0))),
    reliability_single_meal=float(rel1),
    true_contrast_median_random=float(0.9539 * sd_b),
    pd_pairs_all=n_pairs_all, pd_pairs_same_person=n_same, pd_pairs_used=len(P),
    pd_z0_iqr=[float(x) for x in P.z0_hat.quantile([.25, .75])],
    pd_Gamma_hat_median=float(P.Gamma_hat.median()), pd_Gamma_hat_iqr=[float(x) for x in P.Gamma_hat.quantile([.25, .75])],
    pd_Gamma_hat_max=float(P.Gamma_hat.max()),
    pd_gap_median=float(P.gap.median()), pd_gap_iqr=[float(x) for x in P.gap.quantile([.25, .75])], pd_gap_max=float(P.gap.max()),
    pd_nu_median=float(P.nu_hat.median()), pd_nu_iqr=[float(x) for x in P.nu_hat.quantile([.25, .75])], pd_nu_max=float(P.nu_hat.max()),
    pd_frac_nu2=float((P.nu_hat >= 2).mean()), pd_frac_nu3=float((P.nu_hat >= 3).mean()),
    pd_z0_median=float(P.z0_hat.median()), pd_z0_max=float(P.z0_hat.max()),
    sigmaD_poisson=float(np.sqrt(2 * N.mean())), sigmaD_phi=float(np.sqrt(PHI * 2 * N.mean())),
)
json.dump(nums, open(os.path.join(OUT, "fic_numbers.json"), "w"), indent=1)
for k, v in nums.items(): print(f"{k:32s} {v}")
