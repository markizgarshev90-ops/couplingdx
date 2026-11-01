"""Simulations added in revision (reviewer points 3-7).

  A  group-level rank test on untruncated per-dyad estimates vs the pooled estimator
  B  facilitation that differs between groups; unconstrained APIM partner-effect test
  C  closeness groups formed from existing pairs (correlated partner baselines)
  D  pilot study size needed to split between-meal variance into stable and occasion parts
  E  what the within-meal time course adds: a pooled half-meal estimator

Model: slow-drive limit of main-text Eq. (1) with constant rates inside each half of
a session, so that a second-half drive multiplier w (common or person-specific) can be
added. Within each half, D' = G - 2 kappa D, with G the facilitated rate difference, so
    D(T/2) = G1 (T/2) h(theta/2),   D(T) = D(T/2) exp(-theta/2) + G2 (T/2) h(theta/2).
Counts in each half are drawn independently with variance phi * mean (discretised
normal), as in the paper's other simulations.  Run:  python revision_sims.py [A B C D E]
Writes analysis/out/revision_*.csv and revision_numbers.json.
"""
import os, sys, json, math
import numpy as np, pandas as pd
from scipy.stats import norm, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(ROOT, "couplingdx"))
import couplingdx as cdx
sys.path.insert(0, HERE)
from variance_components import components

MU, SB, PHI, TS = 0.0867, 0.0130, 0.7, 740.0
Z = norm.ppf(0.975)
SP_REF = 0.013          # standardisation of person rates in facilitation / profile gradients


def h(t):
    t = np.asarray(t, float); out = np.ones_like(t); nz = np.abs(t) > 1e-12
    out[nz] = -np.expm1(-t[nz]) / t[nz]
    return out


def counts(mean, rng, phi=PHI):
    mean = np.maximum(mean, 0)
    return np.maximum(np.rint(mean + np.sqrt(phi * mean + 1e-12) * rng.standard_normal(mean.shape)), 0)


def baselines(R, n, rng, sp, pairing="random", rho=0.0, so=0.0):
    if pairing == "random":
        bi = MU + sp * rng.standard_normal((R, n)); bj = MU + sp * rng.standard_normal((R, n))
    elif pairing == "existing":            # pre-existing pairs, partner rates correlated rho
        z1 = rng.standard_normal((R, n)); z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.standard_normal((R, n))
        bi, bj = MU + sp * z1, MU + sp * z2
    elif pairing == "screened":            # opposite terciles of one independent 12-min meal
        pool = MU + sp * rng.standard_normal((R, 3 * n))
        scr = counts(TS * np.maximum(pool + so * rng.standard_normal(pool.shape), 1e-6), rng) / TS
        order = np.argsort(scr, axis=1)
        lo = np.take_along_axis(pool, order[:, :n], 1); hi = np.take_along_axis(pool, order[:, 2 * n:], 1)
        hi = np.take_along_axis(hi, rng.permuted(np.tile(np.arange(n), (R, 1)), axis=1), 1)
        bi, bj = lo, hi
    else:
        raise ValueError(pairing)
    flip = rng.random((R, n)) < 0.5        # arbitrary labels
    return np.where(flip, bj, bi), np.where(flip, bi, bj)


def simulate(R, n, theta, rng, sp=SB, so=0.0, m=4, T=740.0, Ts=TS, pairing="random", rho=0.0,
             facil=0.0, A0=1.0, theta_fn=None, w_shared_sd=0.0, w_shared_mean=1.0, w_diff=0.0,
             lmax=0.0, phi=PHI):
    """One group of n dyads, R replicates. Returns dict of (R, n) arrays."""
    bi, bj = baselines(R, n, rng, sp, pairing, rho, so)
    th = np.full((R, n), float(theta)) if theta_fn is None else theta_fn(np.abs(bj - bi), rng)
    Ai = A0 * (1 + facil * (bi - MU) / SP_REF); Aj = A0 * (1 + facil * (bj - MU) / SP_REF)
    # person-specific second-half multiplier: rate gradient and/or decaying drive
    wi = 1 + w_diff * (bi - MU) / SP_REF; wj = 1 + w_diff * (bj - MU) / SP_REF
    if lmax > 0:
        li = rng.uniform(0, lmax, (R, n)); lj = rng.uniform(0, lmax, (R, n))
        di, dj = np.exp(-li * T / 2), np.exp(-lj * T / 2)    # second-half drive relative to first
    else:
        li = lj = np.zeros((R, n)); di = dj = np.ones((R, n))
    ws = np.exp(np.log(w_shared_mean) + w_shared_sd * rng.standard_normal((R, n))) if w_shared_sd > 0 \
        else np.full((R, n), w_shared_mean)
    # joint session
    ri = np.maximum(bi + so * rng.standard_normal((R, n)), 0); rj = np.maximum(bj + so * rng.standard_normal((R, n)), 0)
    if lmax > 0:   # first-half drive is the mean of the decaying drive over the first half
        f1i = np.where(li * T > 1e-9, (1 - np.exp(-li * T / 2)) / np.maximum(li * T / 2, 1e-12), 1.0)
        f1j = np.where(lj * T > 1e-9, (1 - np.exp(-lj * T / 2)) / np.maximum(lj * T / 2, 1e-12), 1.0)
    else:
        f1i = f1j = 1.0
    ei1, ej1 = Ai * ri * f1i, Aj * rj * f1j                       # first-half drives
    ei2, ej2 = ei1 * wi * di * ws, ej1 * wj * dj * ws               # second-half drives
    hh = h(th / 2)
    D1 = (ej1 - ei1) * (T / 2) * hh
    DT = D1 * np.exp(-th / 2) + (ej2 - ei2) * (T / 2) * hh
    S1 = (ei1 + ej1) * T / 2; S2 = (ei2 + ej2) * T / 2; D2 = DT - D1
    ki1, kj1 = counts((S1 - D1) / 2, rng, phi), counts((S1 + D1) / 2, rng, phi)
    ki2, kj2 = counts((S2 - D2) / 2, rng, phi), counts((S2 + D2) / 2, rng, phi)
    # solo sessions (same person-specific profile and decay; no facilitation, no shared w)
    def solo(b, w, d, f1):
        r = np.maximum(b[..., None] + so * rng.standard_normal(b.shape + (m,)), 0)
        mean_rate = r * (f1 * (1 + w * d) / 2)[..., None]
        return counts(mean_rate * Ts, rng, phi) / Ts
    si, sj = solo(bi, wi, di, f1i), solo(bj, wj, dj, f1j)
    return dict(si=si, sj=sj, ki1=ki1, kj1=kj1, ki2=ki2, kj2=kj2, bi=bi, bj=bj, th=th, T=T)


# ------------------------------- estimators -------------------------------
def pooled_h(d, hc3=True):
    si, sj = d["si"], d["sj"]; m = si.shape[-1]
    ki = d["ki1"] + d["ki2"]; kj = d["kj1"] + d["kj2"]
    bhi, bhj = si.mean(-1), sj.mean(-1)
    y = (bhi + bhj) * (kj - ki) / np.maximum(ki + kj, 1); x = bhj - bhi
    v = (si.var(-1, ddof=1) + sj.var(-1, ddof=1)) / m
    w = x * x - v; den = w.sum(1); hh = (y * x).sum(1) / den
    psi = y * x - hh[:, None] * w
    if hc3:
        psi = psi / (1 - w / den[:, None])
    return hh, np.sqrt((psi ** 2).sum(1)) / np.abs(den)


def perdyad_h(d):
    si, sj = d["si"], d["sj"]
    ki = d["ki1"] + d["ki2"]; kj = d["kj1"] + d["kj2"]
    bhi, bhj = si.mean(-1), sj.mean(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hh = (kj - ki) / np.maximum(ki + kj, 1) * (bhi + bhj) / (bhj - bhi)
    return np.where(np.isfinite(hh), hh, 0.0)        # Gamma_hat = 0: undefined; counted as 0 (rare)


def _apim_from_sums(N, Sy, So, Sp, Soo, Spp, Sop, Soy, Spy, Vo, Vp):
    Moo = Soo - So * So / N - Vo; Mpp = Spp - Sp * Sp / N - Vp; Mop = Sop - So * Sp / N
    Coy = Soy - So * Sy / N; Cpy = Spy - Sp * Sy / N
    det = Moo * Mpp - Mop ** 2
    return (Mpp * Coy - Mop * Cpy) / det, (Moo * Cpy - Mop * Coy) / det


def partner_effect(d):
    """Unconstrained APIM, k/T = c + a*own + p*partner, errors-in-variables with the
    replicate solo sessions (method of moments), both members of each dyad as rows.
    Standard error of p by the delete-one-dyad jackknife. Returns (a, p, se_p)."""
    si, sj = d["si"], d["sj"]; m = si.shape[-1]; T = d["T"]
    ki = d["ki1"] + d["ki2"]; kj = d["kj1"] + d["kj2"]
    bi, bj = si.mean(-1), sj.mean(-1); vi, vj = si.var(-1, ddof=1) / m, sj.var(-1, ddof=1) / m
    yi, yj = ki / T, kj / T
    # per-dyad contributions (two rows: i with partner j, j with partner i)
    c = dict(Sy=yi + yj, So=bi + bj, Sp=bj + bi, Soo=bi * bi + bj * bj, Spp=bj * bj + bi * bi,
             Sop=2 * bi * bj, Soy=bi * yi + bj * yj, Spy=bj * yi + bi * yj, Vo=vi + vj, Vp=vj + vi)
    n = si.shape[1]
    tot = {k: v.sum(1) for k, v in c.items()}
    a, p = _apim_from_sums(2 * n, **tot)
    loo = {k: tot[k][:, None] - c[k] for k in c}
    _, pj = _apim_from_sums(2 * (n - 1), **loo)
    se = np.sqrt((n - 1) / n * ((pj - pj.mean(1, keepdims=True)) ** 2).sum(1))
    return a, p, se


def timecourse_rho(d, phi_assumed=PHI, hc3=True):
    """Pooled half-meal estimator of exp(-theta/2): sum D1 D2 / sum(D1^2 - phi N1)."""
    D1 = d["kj1"] - d["ki1"]; D2 = d["kj2"] - d["ki2"]; N1 = d["ki1"] + d["kj1"]
    w = D1 * D1 - phi_assumed * N1; den = w.sum(1); rho = (D1 * D2).sum(1) / den
    psi = D1 * D2 - rho[:, None] * w
    if hc3:
        psi = psi / (1 - w / den[:, None])
    return rho, np.sqrt((psi ** 2).sum(1)) / np.abs(den)


def ztest(e0, s0, e1, s1):
    return np.abs((e1 - e0) / np.sqrt(s0 ** 2 + s1 ** 2)) > Z


def mcse(p, R):
    return math.sqrt(max(p * (1 - p), 1e-12) / R)


# ------------------------------- A: rank test -------------------------------
def part_A(R=2000):
    rng = np.random.default_rng(301)
    rows = []
    for label, kw in (("all stable", dict(sp=SB)), ("half occasional", dict(sp=SB / math.sqrt(2), so=SB / math.sqrt(2)))):
        for n in (50, 100, 200, 400):
            for th1 in (0.0, 1.0):
                g0 = simulate(R, n, 0.0, rng, **kw); g1 = simulate(R, n, th1, rng, **kw)
                h0, h1 = perdyad_h(g0), perdyad_h(g1)
                pmw = mannwhitneyu(h0, h1, axis=1, alternative="two-sided").pvalue
                med = np.median(h0, 1) - np.median(h1, 1)
                p0, s0 = pooled_h(g0); p1, s1 = pooled_h(g1)
                rows.append(dict(scenario=label, n=n, theta_high=th1, R=R,
                                 rank_reject=float(np.mean(pmw < 0.05)),
                                 pooled_reject=float(np.mean(ztest(p0, s0, p1, s1))),
                                 median_h_low=float(np.median(np.median(h0, 1))),
                                 median_h_high=float(np.median(np.median(h1, 1))),
                                 median_diff_sd=float(np.std(med))))
                print("A", rows[-1], flush=True)
    return rows


# ------------------------- B: facilitation differs by group -------------------------
def part_B(R=4000, n=100):
    rng = np.random.default_rng(302)
    rows = []
    for d_close in (-0.05, 0.05):
        for th1 in (0.0, 1.0):
            g0 = simulate(R, n, 0.0, rng, facil=0.0, A0=1.1)          # strangers
            g1 = simulate(R, n, th1, rng, facil=d_close, A0=1.1)      # close
            p0, s0 = pooled_h(g0); p1, s1 = pooled_h(g1)
            _, b0, sb0 = partner_effect(g0); _, b1, sb1 = partner_effect(g1)
            rows.append(dict(test="group comparison", delta_strangers=0.0, delta_close=d_close, theta_close=th1, n=n, R=R,
                             pooled_reject=float(np.mean(ztest(p0, s0, p1, s1))),
                             pooled_median_close=float(np.median(p1)), pooled_median_strangers=float(np.median(p0)),
                             partner_diff_reject=float(np.mean(ztest(b0, sb0, b1, sb1))),
                             partner_close_reject=float(np.mean(np.abs(b1 / sb1) > Z)),
                             partner_median_close=float(np.median(b1))))
            print("B", rows[-1], flush=True)
    # partner test within one group: size under facilitation gradients and pairing schemes
    for pairing in ("random", "screened"):
        for dl in (-0.05, 0.0, 0.05):
            g = simulate(R, n, 0.0, rng, facil=dl, A0=1.1, pairing=pairing)
            _, b, sb = partner_effect(g)
            rows.append(dict(test="partner effect, one group", pairing=pairing, delta=dl, theta=0.0, n=n, R=R,
                             partner_reject=float(np.mean(np.abs(b / sb) > Z)), partner_median=float(np.median(b))))
            print("B", rows[-1], flush=True)
    for th in (0.5, 1.0):
        g = simulate(R, n, th, rng, A0=1.1)
        _, b, sb = partner_effect(g)
        rows.append(dict(test="partner effect, one group", pairing="random", delta=0.0, theta=th, n=n, R=R,
                         partner_reject=float(np.mean(np.abs(b / sb) > Z)), partner_median=float(np.median(b)),
                         partner_target=float(1.1 * (1 - float(h(th))) / 2)))
        print("B", rows[-1], flush=True)
    return rows


# ------------------------- C: existing pairs, correlated baselines -------------------------
def part_C(R=4000, n=100):
    rng = np.random.default_rng(303)
    rows = []
    for rho in (0.3, 0.6):
        for th1 in (0.0, 1.0):
            g0 = simulate(R, n, 0.0, rng)                                    # strangers, random pairing
            g1 = simulate(R, n, th1, rng, pairing="existing", rho=rho)       # close, existing pairs
            p0, s0 = pooled_h(g0); p1, s1 = pooled_h(g1)
            d = dict(duration=740, n_baseline=4, baseline_session=740, mean_rate=MU, between_sd=SB, dispersion=PHI)
            v = cdx.pooled_variance(n, 0.0, **d) + cdx.pooled_variance(n, th1, **dict(d, pairing="existing", partner_correlation=rho))
            dh = 1 - float(h(th1))
            form = (norm.cdf(dh / math.sqrt(v) - Z) + norm.cdf(-dh / math.sqrt(v) - Z)) if th1 else 0.05
            rows.append(dict(case="homogeneous coupling", rho=rho, theta_close=th1, n=n, R=R,
                             reject=float(np.mean(ztest(p0, s0, p1, s1))), formula=float(form),
                             median_close=float(np.median(p1)), target_close=float(h(th1))))
            print("C", rows[-1], flush=True)
    # heterogeneous coupling that depends on the pair's contrast, same rule in both groups
    def theta_fn(absG, rng_):          # mean 1 over random pairs; similar pairs couple more
        return 2.0 * np.exp(-absG / (SB * math.sqrt(2) * 0.7978845608 * 1.0))  # scale = E|Gamma| for random pairs
    for rho in (0.3, 0.6):
        g0 = simulate(R, n, None, rng, theta_fn=theta_fn)
        g1 = simulate(R, n, None, rng, pairing="existing", rho=rho, theta_fn=theta_fn)
        p0, s0 = pooled_h(g0); p1, s1 = pooled_h(g1)
        rows.append(dict(case="coupling depends on contrast, same rule", rho=rho, n=n, R=R,
                         reject=float(np.mean(ztest(p0, s0, p1, s1))),
                         median_strangers=float(np.median(p0)), median_close=float(np.median(p1)),
                         mean_theta_strangers=float(g0["th"].mean()), mean_theta_close=float(g1["th"].mean())))
        print("C", rows[-1], flush=True)
    # partner-effect test in existing pairs (theta = 0), with and without a facilitation gradient
    for rho in (0.3, 0.6):
        for dl in (0.0, 0.05):
            g = simulate(R, n, 0.0, rng, pairing="existing", rho=rho, facil=dl, A0=1.1)
            _, b, sb = partner_effect(g)
            rows.append(dict(case="partner test, existing pairs", rho=rho, delta=dl, n=n, R=R,
                             reject=float(np.mean(np.abs(b / sb) > Z)), median_partner=float(np.median(b))))
            print("C", rows[-1], flush=True)
    return rows


# ------------------------------- D: pilot size -------------------------------
def part_D(R=2000):
    rng = np.random.default_rng(304)
    rows = []
    for label, sp, so in (("half occasional", SB / math.sqrt(2), SB / math.sqrt(2)), ("all stable", SB, 0.0)):
        for k in (12, 24, 48, 96):
            for m in (2, 4):
                share, sps = [], []
                for _ in range(R):
                    b = MU + sp * rng.standard_normal(k)
                    r = np.maximum(b[:, None] + so * rng.standard_normal((k, m)), 0)
                    N = counts(r * TS, rng)
                    rws = [(str(p), float(N[p, j]), TS) for p in range(k) for j in range(m)]
                    c = components(rws, PHI)
                    share.append(c["occasion_share"]); sps.append(c["sigma_p"])
                share = np.array(share); sps = np.array(sps)
                rows.append(dict(scenario=label, people=k, meals_each=m, true_share=so ** 2 / (sp ** 2 + so ** 2),
                                 share_q025=float(np.nanpercentile(share, 2.5)), share_median=float(np.nanmedian(share)),
                                 share_q975=float(np.nanpercentile(share, 97.5)),
                                 sigma_p_q025=float(np.percentile(sps, 2.5)), sigma_p_q975=float(np.percentile(sps, 97.5))))
                print("D", rows[-1], flush=True)
    return rows


# ------------------------------- E: time course -------------------------------
def part_E(R=2000):
    rng = np.random.default_rng(305)
    rows = []
    base = [("all stable", dict(sp=SB)), ("half occasional", dict(sp=SB / math.sqrt(2), so=SB / math.sqrt(2)))]
    for label, kw in base:
        for n in (50, 100, 200, 400):
            for th1 in (0.0, 1.0):
                g0 = simulate(R, n, 0.0, rng, **kw); g1 = simulate(R, n, th1, rng, **kw)
                r0, q0 = timecourse_rho(g0); r1, q1 = timecourse_rho(g1)
                p0, s0 = pooled_h(g0); p1, s1 = pooled_h(g1)
                rows.append(dict(scenario=label, n=n, theta_high=th1, R=R,
                                 tc_reject=float(np.mean(ztest(r0, q0, r1, q1))),
                                 pooled_reject=float(np.mean(ztest(p0, s0, p1, s1))),
                                 tc_median_low=float(np.median(r0)), tc_median_high=float(np.median(r1)),
                                 tc_target_high=float(math.exp(-th1 / 2)),
                                 tc_cover_low=float(np.mean(np.abs(r0 - 1) <= Z * q0))))
                print("E", rows[-1], flush=True)
    # departures, n = 100, theta 0 vs 1 (power) and 0 vs 0 (size), both estimators
    deps = [("dispersion of half-meal counts 0.9, assumed 0.7", dict(phi=0.9)),
            ("shared second-half profile, SD 0.1 (log), mean 1.04", dict(w_shared_sd=0.1, w_shared_mean=1.04)),
            ("faster eaters slow 5% more per SD in second half", dict(w_diff=-0.05)),
            ("decaying drive, l up to 2.4e-4", dict(lmax=2.4e-4)),
            ("decaying drive, l up to 5e-4", dict(lmax=5e-4))]
    for label, kw in deps:
        for th1 in (0.0, 1.0):
            g0 = simulate(R, 100, 0.0, rng, **kw); g1 = simulate(R, 100, th1, rng, **kw)
            r0, q0 = timecourse_rho(g0); r1, q1 = timecourse_rho(g1)
            p0, s0 = pooled_h(g0); p1, s1 = pooled_h(g1)
            rows.append(dict(scenario=label, n=100, theta_high=th1, R=R,
                             tc_reject=float(np.mean(ztest(r0, q0, r1, q1))),
                             pooled_reject=float(np.mean(ztest(p0, s0, p1, s1))),
                             tc_median_low=float(np.median(r0)), tc_median_high=float(np.median(r1)),
                             tc_target_high=float(math.exp(-th1 / 2)),
                             tc_cover_low=float(np.mean(np.abs(r0 - 1) <= Z * q0)),
                             pooled_median_low=float(np.median(p0)), pooled_cover_low=float(np.mean(np.abs(p0 - 1) <= Z * s0))))
            print("E", rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    parts = sys.argv[1:] or list("ABCDE")
    fns = dict(A=part_A, B=part_B, C=part_C, D=part_D, E=part_E)
    for p in parts:
        rows = fns[p]()
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"revision_{p}.csv"), index=False)
