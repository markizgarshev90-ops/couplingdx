"""Per-meal fits of the decaying drive lambda(t) = beta * exp(-l t), all 21 FIC meals.

* Maximum likelihood (non-homogeneous Poisson process), l searched on
  [1e-7, 1] s^-1 with beta profiled out; standard error of beta from the observed
  information of the two-parameter fit (analytic second derivatives).
* Least squares on the cumulative curve, k(t) = (beta/l)(1 - exp(-l t)), lower
  bound 1e-6 s^-1 on both parameters, 25 random starts (as in Kieckens &
  Izgarshev, 2026, which used 17 of the 21 meals).

Writes analysis/out/drive_fits.csv and analysis/out/drive_numbers.json.
"""
import os, json
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, curve_fit
from scipy.stats import t as tdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out"); os.makedirs(OUT, exist_ok=True)
B = pd.read_csv(os.path.join(ROOT, "data", "fic_bite_times.csv"))
M = pd.read_csv(os.path.join(ROOT, "data", "fic_meals.csv"))
L_MIN_ML, L_MAX_ML = 1e-7, 1.0
L_MIN_LS = 1e-6


def Mfun(l, T):
    """int_0^T exp(-l t) dt and its first two derivatives in l."""
    if l * T < 1e-4:   # series, avoids cancellation
        return (T - l * T**2 / 2 + l**2 * T**3 / 6,
                -T**2 / 2 + l * T**3 / 3 - l**2 * T**4 / 8,
                T**3 / 3 - l * T**4 / 4)
    e = np.exp(-l * T)
    M0 = (1 - e) / l
    M1 = (T * e) / l - (1 - e) / l**2
    M2 = -(T**2 * e) / l - 2 * T * e / l**2 + 2 * (1 - e) / l**3
    return M0, M1, M2


def ml_fit(tn, T):
    n = len(tn); s = tn.sum()
    def negll(u):
        l = np.exp(u); M0 = Mfun(l, T)[0]; b = n / M0
        return -(n * np.log(b) - l * s - b * M0)
    r = minimize_scalar(negll, bounds=(np.log(L_MIN_ML), np.log(L_MAX_ML)),
                        method="bounded", options=dict(xatol=1e-10))
    l = float(np.exp(r.x)); M0, M1, M2 = Mfun(l, T); b = n / M0
    info = np.array([[n / b**2, M1], [M1, b * M2]])      # observed information
    se_b = float(np.sqrt(np.linalg.inv(info)[0, 0]))
    return b, l, se_b


def k_model(t, b, l):
    return (b / l) * (-np.expm1(-l * t))


def ls_fit(t, y, rng, starts=25):
    best, best_rmse = None, np.inf
    for _ in range(starts):
        p0 = [10 ** rng.uniform(-4, 1), 10 ** rng.uniform(-6, 0)]
        try:
            p, _ = curve_fit(k_model, t, y, p0=p0, bounds=([L_MIN_LS, L_MIN_LS], [np.inf, np.inf]),
                             maxfev=20000)
        except RuntimeError:
            continue
        rmse = np.sqrt(np.mean((k_model(t, *p) - y) ** 2))
        if rmse < best_rmse:
            best, best_rmse = p, rmse
    return best


xl = pd.ExcelFile(os.path.join(ROOT, "data", "fic_bite_timelines.xlsx"))
rng = np.random.default_rng(0)
rows = []
for i in range(21):
    tn = np.sort(B.t[B.meal == i].to_numpy()); T = float(M["T"][i]); N = len(tn)
    b, l, se_b = ml_fit(tn, T)
    d = xl.parse(f"entry_{i}")
    bl, ll = ls_fit(d.timestamp.values, d.cumulative_bites.values.astype(float), rng)
    rows.append(dict(meal=i, T=T, N=N, rate=N / T, se_rate_phi07=np.sqrt(0.7 * N) / T,
                     beta_ml=b, l_ml=l, se_beta_ml=se_b, beta_ls=bl, l_ls=ll))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "drive_fits.csv"), index=False)


def mean_ci(x):
    n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); q = tdist.ppf(.975, n - 1)
    return float(m), float(max(m - q * se, 0)), float(m + q * se)


P = pd.read_csv(os.path.join(OUT, "pseudo_dyads.csv"))
out = dict(
    ml_l_lower_bound=L_MIN_ML, ml_l_at_bound=int((R.l_ml < 2 * L_MIN_ML).sum()),
    ml_l_mean_ci=mean_ci(R.l_ml), ml_l_median=float(R.l_ml.median()),
    ml_se_beta_mean=float(R.se_beta_ml.mean()), ml_se_beta_rms=float(np.sqrt((R.se_beta_ml ** 2).mean())),
    ml_se_beta_median=float(R.se_beta_ml.median()),
    ratio_sd_rate_over_ml_se=float(R.rate.std(ddof=1) / R.se_beta_ml.mean()),
    ratio_sd_betaml_over_ml_se=float(R.beta_ml.std(ddof=1) / R.se_beta_ml.mean()),
    ls_l_lower_bound=L_MIN_LS, ls_l_at_bound=int((R.l_ls < 2 * L_MIN_LS).sum()),
    ls_l_mean_ci=mean_ci(R.l_ls), ls_l_mean_ci_first17=mean_ci(R.l_ls[:17]),
    ls_l_at_bound_first17=int((R.l_ls[:17] < 2 * L_MIN_LS).sum()),
    pd_z0_iqr=[float(x) for x in P.z0_hat.quantile([.25, .75])],
    pd_z0_median=float(P.z0_hat.median()), pd_z0_max=float(P.z0_hat.max()),
)
json.dump(out, open(os.path.join(OUT, "drive_numbers.json"), "w"), indent=1)
pd.set_option("display.width", 200)
print(R.round(6).to_string(index=False))
for k, v in out.items():
    print(f"{k:28s} {v}")
