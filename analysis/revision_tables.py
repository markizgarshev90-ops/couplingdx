"""LaTeX rows for Supplementary Tables S2-S5 from analysis/out/revision_*.csv."""
import os, pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
from decimal import Decimal, ROUND_HALF_UP
_r = lambda v, q: str(Decimal(repr(round(float(v), 9))).quantize(Decimal(q), rounding=ROUND_HALF_UP))
f3 = lambda v: _r(v, "0.001")
f2 = lambda v: _r(v, "0.01")
A, B, C, D, E = (pd.read_csv(os.path.join(OUT, f"revision_{p}.csv")) for p in "ABCDE")
lines = {}

# S2: facilitation differing between groups; partner-effect test
b = B[B.test == "group comparison"]
r = []
for d in (-0.05, 0.05):
    x0 = b[(b.delta_close == d) & (b.theta_close == 0)].iloc[0]; x1 = b[(b.delta_close == d) & (b.theta_close == 1)].iloc[0]
    r.append(f"Close group only, $\\delta={d:+.2f}$ & {f3(x0.pooled_reject)} & {f3(x1.pooled_reject)} & "
             f"{f3(x0.partner_diff_reject)} & {f3(x1.partner_diff_reject)} & {f3(x0.partner_close_reject)} & {f3(x1.partner_close_reject)}\\\\")
lines["S2a"] = "\n".join(r)
o = B[B.test == "partner effect, one group"]
r = []
for pairing in ("random", "screened"):
    for d in (-0.05, 0.0, 0.05):
        x = o[(o.pairing == pairing) & (o.delta == d) & (o.theta == 0)].iloc[0]
        r.append(f"{pairing.capitalize()} pairing, $\\delta={d:+.2f}$, $\\theta=0$ & {f3(x.partner_median)} & --- & {f3(x.partner_reject)}\\\\")
for th in (0.5, 1.0):
    x = o[(o.theta == th)].iloc[0]
    r.append(f"Random pairing, $\\delta=0$, $\\theta={th:g}$ & {f3(x.partner_median)} & {f3(x.partner_target)} & {f3(x.partner_reject)}\\\\")
lines["S2b"] = "\n".join(r)

# S3: existing pairs
c = C
r = []
for rho in (0.3, 0.6):
    x0 = c[(c.case == "homogeneous coupling") & (c.rho == rho) & (c.theta_close == 0)].iloc[0]
    x1 = c[(c.case == "homogeneous coupling") & (c.rho == rho) & (c.theta_close == 1)].iloc[0]
    r.append(f"Constant coupling, $\\rho={rho:g}$ & {f3(x0.reject)} & {f3(x1.reject)} & {f3(x1.formula)} & {f3(x1.median_close)}\\\\")
for rho in (0.3, 0.6):
    x = c[(c.case == "coupling depends on contrast, same rule") & (c.rho == rho)].iloc[0]
    r.append(f"Coupling falls with contrast, $\\rho={rho:g}$ & {f3(x.reject)} & --- & --- & {f3(x.median_close)} ({f3(x.median_strangers)})\\\\")
for rho in (0.3, 0.6):
    for d in (0.0, 0.05):
        x = c[(c.case == "partner test, existing pairs") & (c.rho == rho) & (c.delta == d)].iloc[0]
        r.append(f"Partner-effect test, $\\rho={rho:g}$, $\\delta={d:.2f}$ & {f3(x.reject)} & --- & --- & {f3(x.median_partner)}\\\\")
lines["S3"] = "\n".join(r)

# S4: rank test
r = []
for sc in ("all stable", "half occasional"):
    for n in (50, 100, 200, 400):
        x0 = A[(A.scenario == sc) & (A.n == n) & (A.theta_high == 0)].iloc[0]; x1 = A[(A.scenario == sc) & (A.n == n) & (A.theta_high == 1)].iloc[0]
        lab = (sc.capitalize() if n == 50 else "")
        r.append(f"{lab} & {n} & {f3(x0.rank_reject)} & {f3(x1.rank_reject)} & {f3(x0.pooled_reject)} & {f3(x1.pooled_reject)} & {f2(x0.median_h_low)}\\\\")
lines["S4"] = "\n".join(r)

# S5: time course
r = []
for sc in ("all stable", "half occasional"):
    for n in (50, 100, 200, 400):
        x0 = E[(E.scenario == sc) & (E.n == n) & (E.theta_high == 0)].iloc[0]; x1 = E[(E.scenario == sc) & (E.n == n) & (E.theta_high == 1)].iloc[0]
        lab = (sc.capitalize() if n == 50 else "")
        r.append(f"{lab} & {n} & {f3(x0.tc_median_low)} & {f3(x1.tc_median_high)} & {f3(x0.tc_reject)} & {f3(x1.tc_reject)} & {f3(x0.pooled_reject)} & {f3(x1.pooled_reject)}\\\\")
lines["S5a"] = "\n".join(r)
names = {"dispersion of half-meal counts 0.9, assumed 0.7": "Half-meal dispersion 0.9, assumed 0.7",
         "shared second-half profile, SD 0.1 (log), mean 1.04": "Shared second-half change, mean $+4\\%$, SD 10\\%",
         "faster eaters slow 5% more per SD in second half": "Faster eaters slow 5\\% more per SD",
         "decaying drive, l up to 2.4e-4": "Decaying drive, $l_{\\max}=2.4\\times10^{-4}$",
         "decaying drive, l up to 5e-4": "Decaying drive, $l_{\\max}=5\\times10^{-4}$"}
r = []
for k, lab in names.items():
    x0 = E[(E.scenario == k) & (E.theta_high == 0)].iloc[0]; x1 = E[(E.scenario == k) & (E.theta_high == 1)].iloc[0]
    r.append(f"{lab} & {f3(x0.tc_median_low)} & {f3(x0.tc_reject)} & {f3(x1.tc_reject)} & {f3(x0.pooled_median_low)} & {f3(x0.pooled_reject)} & {f3(x1.pooled_reject)}\\\\")
lines["S5b"] = "\n".join(r)

# S6: pilot
r = []
for sc in ("half occasional", "all stable"):
    for k in (12, 24, 48, 96):
        for m in (2, 4):
            x = D[(D.scenario == sc) & (D.people == k) & (D.meals_each == m)].iloc[0]
            lab = (f"{sc.capitalize()} ({f2(x.true_share)})" if (k == 12 and m == 2) else "")
            r.append(f"{lab} & {k} & {m} & {f2(x.share_q025)}--{f2(x.share_q975)} & {f3(x.sigma_p_q025)}--{f3(x.sigma_p_q975)}\\\\")
lines["S6"] = "\n".join(r)
if __name__ == "__main__":
    for k, v in lines.items():
        print(f"%% ---- {k}\n{v}")
