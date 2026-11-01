"""Design requirements for the worked cases (Table 3), computed with couplingdx,
and the agreement between couplingdx's analytic pooled power and simulation."""
import os, sys, json, math
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "couplingdx"))
import couplingdx as c

OUT = os.path.join(ROOT, "analysis", "out")
F = json.load(open(os.path.join(OUT, "fic_numbers.json")))
# reported (rounded) values, so that the paper, the package examples and this table agree
MU, SB, PHI = round(F["rate_mean"], 4), round(F["sd_b"], 4), F["PHI"]; S = 2 * MU; TS = 740.0
res = {}

# ---------------- per-dyad: eating ----------------
G_rand = c.expected_contrast(SB)
G_scr = c.expected_contrast(SB, "screened", mean_rate=MU, dispersion=PHI, screen_duration=TS)
G_true = c.expected_contrast(SB, "screened")
rows = []
for lab, G in (("random", G_rand), ("screened (one 12-min meal)", G_scr), ("screened (true rates)", G_true)):
    rows.append(dict(case="eating", pairing=lab, contrast=G,
                     T_unlimited_min=c.required_duration(G, S, PHI) / 60,
                     T_m8_min=c.required_duration(G, S, PHI, 8, TS) / 60,
                     T_m8_occ_half_min=c.required_duration(G * math.sqrt(.5), S, PHI, 8, TS,
                                                            occasion_sd=SB * math.sqrt(.5)) / 60))
# ---------------- per-dyad: speech ----------------
PHI_SP = (6.9 ** 2 + 12.7 ** 2) / 12.7     # compound Poisson, syllables per phrase
LAM = 2 * 5.12
for lab, G in (("random", c.expected_contrast(0.59)), ("screened (true rates)", c.expected_contrast(0.59, "screened"))):
    Tu = c.required_duration(G, LAM, PHI_SP)
    Teq = Tu * (1 + c.h(1) ** 2)            # baseline as long as the conversation
    rows.append(dict(case="speech", pairing=lab, contrast=G, T_unlimited_min=Tu / 60,
                     T_equal_baseline_min=Teq / 60))
PD = pd.DataFrame(rows); PD.to_csv(os.path.join(OUT, "design_perdyad.csv"), index=False)
print(PD.round(4).to_string()); res["phi_speech"] = PHI_SP

# ---------------- pooled: eating ----------------
scen = [("all stable", SB, 0.0), ("half occasion", SB / math.sqrt(2), SB / math.sqrt(2)),
        ("half the stable SD", SB / 2, 0.0)]
prow = []
for lab, sp, so in scen:
    for pairing in ("random", "screened"):
        for T in (740.0, 3000.0):
            for m in (4, 8):
                d = dict(duration=T, n_baseline=m, baseline_session=TS, mean_rate=MU,
                         between_sd=sp, occasion_sd=so, dispersion=PHI, pairing=pairing)
                prow.append(dict(scenario=lab, sigma_p=sp, sigma_o=so, pairing=pairing, T=T, m=m,
                                 n80=c.pooled_n(0.8, **d), n90=c.pooled_n(0.9, **d)))
PP = pd.DataFrame(prow); PP.to_csv(os.path.join(OUT, "design_pooled.csv"), index=False)
print(PP.to_string())

# ---------------- analytic vs simulated power ----------------
G = pd.read_csv(os.path.join(OUT, "pooled_grid.csv"))
G = G[G.theta_high == 1].copy()
an = []
SBu, MUu = F["sd_b"], F["rate_mean"]
for _, r in G.iterrows():
    sp = math.sqrt(1 - r.occ_share) * SBu; so = math.sqrt(r.occ_share) * SBu
    an.append(c.pooled_power(int(r.n), 0, 1, duration=r["T"], n_baseline=int(r.m), baseline_session=TS,
                             mean_rate=MUu, between_sd=sp, occasion_sd=so, dispersion=PHI, pairing=r.pairing))
G["pkg_power"] = an   # NB: uses the unrounded values the simulation used (below)
G["diff"] = G.mom_reject - G.pkg_power
for mmin in (2, 4):
    sub = G[G.m >= mmin]
    print(f"m>={mmin}: max |sim - analytic| = {sub['diff'].abs().max():.3f}, mean |diff| = {sub['diff'].abs().mean():.3f}")
    res[f"maxdiff_m{mmin}"] = float(sub["diff"].abs().max()); res[f"meandiff_m{mmin}"] = float(sub["diff"].abs().mean())
N = pd.read_csv(os.path.join(OUT, "pooled_grid.csv")); N = N[N.theta_high == 0]
res["size_m4"] = [float(N[N.m >= 4].mom_reject.min()), float(N[N.m >= 4].mom_reject.max()), float(N[N.m >= 4].mom_reject.mean())]
res["size_all"] = [float(N.mom_reject.min()), float(N.mom_reject.max())]
res["ols_null_h"] = [float(N.ols_h_low_median.min()), float(N.ols_h_low_median.max())]
res["mom_null_h"] = [float(N.mom_h_low_median.min()), float(N.mom_h_low_median.max())]
A = pd.read_csv(os.path.join(OUT, "pooled_grid.csv")); A = A[A.theta_high == 1]
res["mom_cover_m4"] = [float(A[A.m >= 4].mom_cover_low.min()), float(A[A.m >= 4].mom_cover_low.max())]
res["iv_vs_mom_power_ratio_median"] = float((A.iv_reject / A.mom_reject.clip(lower=1e-3))[A.mom_reject.between(0.2, 0.95)].median())
json.dump(res, open(os.path.join(OUT, "design_numbers.json"), "w"), indent=1)
print(res)
