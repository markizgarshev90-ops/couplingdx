"""Robustness of the pooled estimator to departures from its own model.

Reproduces the paper's pooled simulation design (slow-drive APIM, normal rates,
dispersion phi = 0.7 counts) and adds five departures:
  R1 small samples, R2 skewed rates, R3 decaying drive (full Eq. 1),
  R4 heterogeneous coupling, R5 differential facilitation.
Two groups of n dyads, theta = 0 vs theta = 1 unless stated; two-sided z-test
with sandwich SEs. Run: python robustness_sim.py  (numpy, scipy)
R3 at the decay bound refitted on all 21 FIC meals: python robustness_sim.py --r3-fic21
"""
import numpy as np
from scipy.stats import norm

MU, PHI, TS = 0.0867, 0.7, 740.0
ZT = norm.ppf(2 / 3); KT = norm.pdf(ZT) * 3; VT = 1 + ZT * KT - KT ** 2


def h(t):
    t = np.asarray(t, float)
    out = np.ones_like(t)
    nz = np.abs(t) > 1e-12
    out[nz] = (1 - np.exp(-t[nz])) / t[nz]
    return out


def gap_kernel(l, c, T):
    """(e^{-lT} - e^{-cT})/(c - l), with the l -> c and l -> 0, c -> 0 limits."""
    l = np.broadcast_to(l, np.broadcast(l, c).shape).astype(float)
    c = np.broadcast_to(c, l.shape).astype(float)
    out = np.empty_like(l)
    close = np.abs(c - l) * T < 1e-8
    out[close] = T * np.exp(-l[close] * T)
    o = ~close
    out[o] = (np.exp(-l[o] * T) - np.exp(-c[o] * T)) / (c[o] - l[o])
    return out


def cum_drive(l, T):
    l = np.asarray(l, float)
    out = np.full_like(l, T)
    nz = l * T > 1e-10
    out[nz] = (1 - np.exp(-l[nz] * T)) / l[nz]
    return out


def counts(mean, rng):
    mean = np.maximum(mean, 0)
    return np.maximum(np.rint(rng.normal(mean, np.sqrt(PHI * mean + 1e-12))), 0)


def draw_people(shape, sp, rng, skew):
    if skew:  # lognormal with mean MU and SD sp
        s2 = np.log(1 + (sp / MU) ** 2)
        return rng.lognormal(np.log(MU) - s2 / 2, np.sqrt(s2), shape)
    return rng.normal(MU, sp, shape)


def occasion(shape, so, rng, skew):
    if so == 0:
        return np.zeros(shape)
    if skew:  # centred gamma, shape 2 (skewness 1.41), SD so
        k = 2.0
        return (rng.gamma(k, so / np.sqrt(k), shape) - so * np.sqrt(k))
    return rng.normal(0, so, shape)


def simulate_group(reps, n, theta, rng, T=740, m=4, sp=0.013, so=0.0,
                   pairing="random", skew=False, decay=None, theta_dist=None,
                   facil=None, Ts=TS):
    """Return (h_hat, se, denom_ok, target_h) arrays over reps."""
    if pairing == "random":
        bi = draw_people((reps, n), sp, rng, skew)
        bj = draw_people((reps, n), sp, rng, skew)
    else:  # opposite terciles of one independent screening meal
        pool = draw_people((reps, 3 * n), sp, rng, skew)
        scr = counts((pool + occasion(pool.shape, so, rng, skew)) * TS, rng) / TS  # 12-min screening meal
        order = np.argsort(scr, axis=1)
        low = np.take_along_axis(pool, order[:, :n], 1)
        high = np.take_along_axis(pool, order[:, 2 * n:], 1)
        perm = rng.permuted(np.tile(np.arange(n), (reps, 1)), axis=1)
        bi, bj = low, np.take_along_axis(high, perm, 1)
        flip = rng.random((reps, n)) < 0.5  # random labelling
        bi, bj = np.where(flip, bj, bi), np.where(flip, bi, bj)
    if decay is None:
        li = lj = np.zeros((reps, n))
    else:
        li = rng.uniform(0, decay, (reps, n)); lj = rng.uniform(0, decay, (reps, n))
    # solo sessions: per-session rate = count / TS (what an analyst would compute)
    def solo(b, l):
        r = b[..., None] + occasion((reps, n, m), so, rng, skew)
        c = counts(np.maximum(r, 0) * cum_drive(l, Ts)[..., None], rng)
        return c / Ts
    si, sj = solo(bi, li), solo(bj, lj)
    # joint session (exact solution of main-text Eq. 1 with facilitation A_i, A_j)
    th = np.full((reps, n), float(theta)) if theta_dist is None else theta_dist((reps, n), rng)
    kappa2 = th / T  # 2*kappa
    ri = np.maximum(bi + occasion((reps, n), so, rng, skew), 0)
    rj = np.maximum(bj + occasion((reps, n), so, rng, skew), 0)
    if facil is None:
        Ai = Aj = np.ones((reps, n))
    else:  # A = 1.1 * (1 + delta * standardized baseline)
        Ai = 1.1 * (1 + facil * (bi - MU) / sp); Aj = 1.1 * (1 + facil * (bj - MU) / sp)
    Sig = Ai * ri * cum_drive(li, T) + Aj * rj * cum_drive(lj, T)
    D = Aj * rj * gap_kernel(lj, kappa2, T) - Ai * ri * gap_kernel(li, kappa2, T)
    ki, kj = counts((Sig - D) / 2, rng), counts((Sig + D) / 2, rng)
    # estimator
    bhi, bhj = si.mean(-1), sj.mean(-1)
    y = (bhi + bhj) * (kj - ki) / np.maximum(ki + kj, 1)
    x = bhj - bhi
    v = (si.var(-1, ddof=1) + sj.var(-1, ddof=1)) / m
    den = (x ** 2 - v).sum(1)
    hh = (y * x).sum(1) / den
    w = x ** 2 - v
    psi = y * x - hh[:, None] * w
    lev = w / den[:, None]  # HC3 leverage correction (small-sample)
    se = np.sqrt(((psi / (1 - lev)) ** 2).sum(1)) / np.abs(den)
    ok = den > 0.1 * (x ** 2).sum(1)
    target = np.mean(h(th)) if theta_dist is None else np.mean(h(th) * x ** 2) / np.mean(x ** 2)
    return hh, se, ok, target


def formula_power(n, T=740, m=4, sp=0.013, so=0.0, pairing="random", th1=1.0):
    S = 2 * MU
    su2 = (PHI * S / TS + 2 * so ** 2) / m
    if pairing == "random":
        G2 = 2 * sp ** 2
    else:
        R = sp ** 2 / (sp ** 2 + so ** 2 + PHI * MU / TS)
        G2 = 2 * sp ** 2 * (1 - (1 - VT) * R) + 4 * KT ** 2 * R * sp ** 2
    def V(hh):
        se2 = PHI * S / T + 2 * hh ** 2 * so ** 2
        return (se2 * (G2 + su2) + hh ** 2 * su2 * G2 + hh ** 2 * su2 ** 2 * (2 + 1 / (m - 1))) / G2 ** 2
    d = 1 - float(h(th1))
    se = np.sqrt((V(1.0) + V(float(h(th1)))) / n)
    return norm.cdf(d / se - 1.96) + norm.cdf(-d / se - 1.96)


def run(label, reps, n, rng, th0=0.0, th1=1.0, **kw):
    kw0 = {k: v for k, v in kw.items() if k != "theta_dist"}
    h0, s0, ok0, t0 = simulate_group(reps, n, th0, rng, **kw0)
    h1, s1, ok1, t1 = simulate_group(reps, n, th1, rng, **kw)
    okb = ok0 & ok1
    z = (h1 - h0) / np.sqrt(s0 ** 2 + s1 ** 2)
    rej = (np.abs(z) > 1.96) & okb
    cover1 = (np.abs(h1 - t1) <= 1.96 * s1)[ok1].mean()
    cover0 = (np.abs(h0 - t0) <= 1.96 * s0)[ok0].mean()
    fkw = {k: kw[k] for k in ("T", "m", "sp", "so", "pairing") if k in kw}
    fp = formula_power(n, th1=th1, **fkw) if th1 != th0 else 0.05
    return dict(label=label, n=n, reps=reps,
                med_h0=np.median(h0[ok0]), target0=t0,
                med_h1=np.median(h1[ok1]), target1=t1,
                cover0=cover0, cover1=cover1,
                reject=rej.mean(), mcse=np.sqrt(rej.mean() * (1 - rej.mean()) / reps),
                formula=fp, unusable=1 - okb.mean())


if __name__ == "__main__" and "--r3-fic21" not in __import__("sys").argv:
    rng = np.random.default_rng(20261006)
    R = 4000
    rows = []
    # baseline: reproduce the paper's setting
    rows.append(run("Baseline (model as assumed)", R, 100, rng))
    rows.append(run("Baseline, size (theta 0 vs 0)", R, 100, rng, th1=0.0))
    # R1 small samples
    for n in (20, 30):
        rows.append(run(f"R1 small n", R, n, rng))
        rows.append(run(f"R1 small n, size", R, n, rng, th1=0.0))
    rows.append(run("R1 small n, screened", R, 20, rng, pairing="screened"))
    rows.append(run("R1 small n, screened, size", R, 20, rng, pairing="screened", th1=0.0))
    # R2 skewed rates (lognormal people, gamma occasions), half occasional
    sp = so = 0.013 / np.sqrt(2)
    rows.append(run("R2 normal, half occasional", R, 300, rng, sp=sp, so=so))
    rows.append(run("R2 skewed, half occasional", R, 300, rng, sp=sp, so=so, skew=True))
    rows.append(run("R2 skewed, half occasional, size", R, 300, rng, sp=sp, so=so, skew=True, th1=0.0))
    rows.append(run("R2 skewed, all stable", R, 100, rng, skew=True))
    # R3 decaying drive (full Eq. 1); l ~ U(0, lmax) per person
    rows.append(run("R3 decay, l up to 1.3e-4 (17-meal FIC upper)", R, 100, rng, decay=1.3e-4))
    rows.append(run("R3 decay, l up to 1.3e-4, size", R, 100, rng, decay=1.3e-4, th1=0.0))
    rows.append(run("R3 decay, l up to 5e-4, T=3000 (stress)", R, 100, rng, decay=5e-4, T=3000))
    rows.append(run("R3 decay stress, size", R, 100, rng, decay=5e-4, T=3000, th1=0.0))
    rows.append(run("R3 decay up to 5e-4, T=Ts=740", R, 100, rng, decay=5e-4))
    rows.append(run("R3 decay up to 5e-4, T=Ts=3000", R, 100, rng, decay=5e-4, T=3000, Ts=3000))
    rows.append(run("R3 no decay, T=3000, Ts=740", R, 100, rng, T=3000))
    # R4 heterogeneous coupling: theta_d ~ Gamma(2, 0.5), mean 1
    gam = lambda shape, r: r.gamma(2.0, 0.5, shape)
    rows.append(run("R4 heterogeneous theta (Gamma, mean 1)", R, 100, rng, theta_dist=gam))
    # R5 differential facilitation, common to both groups
    for d in (-0.05, 0.05):
        rows.append(run(f"R5 facilitation delta={d:+.2f}", R, 100, rng, facil=d))
        rows.append(run(f"R5 facilitation delta={d:+.2f}, size", R, 100, rng, facil=d, th1=0.0))
    import json
    out = [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()} for r in rows]
    json.dump(out, open("robustness_results.json", "w"), indent=1)
    hdr = f"{'scenario':44s} {'n':>4s} {'medh0':>6s} {'tgt0':>5s} {'medh1':>6s} {'tgt1':>5s} {'cov0':>5s} {'cov1':>5s} {'rej':>6s} {'mcse':>5s} {'form':>5s} {'unus':>5s}"
    print(hdr)
    for r in out:
        print(f"{r['label']:44s} {r['n']:4d} {r['med_h0']:6.3f} {r['target0']:5.3f} {r['med_h1']:6.3f} {r['target1']:5.3f} "
              f"{r['cover0']:5.3f} {r['cover1']:5.3f} {r['reject']:6.3f} {r['mcse']:5.3f} {r['formula']:5.3f} {r['unusable']:5.3f}")


def r3_fic21():
    """R3 at the upper end of the 95% interval for the mean decay rate refitted on all
    21 FIC meals (maximum likelihood, 2.4e-4 /s; fic_drive_fits.py). Separate seed, so
    the rows above are unchanged."""
    rng2 = np.random.default_rng(20261011)
    rows2 = [run("R3 decay, l up to 2.4e-4 (FIC 21-meal upper)", 4000, 100, rng2, decay=2.4e-4),
             run("R3 decay, l up to 2.4e-4, size", 4000, 100, rng2, decay=2.4e-4, th1=0.0)]
    for r in rows2:
        print(f"{r['label']:44s} {r['n']:4d} {r['med_h0']:6.3f} {r['target0']:5.3f} {r['med_h1']:6.3f} {r['target1']:5.3f} "
              f"{r['cover0']:5.3f} {r['cover1']:5.3f} {r['reject']:6.3f} {r['mcse']:5.3f} {r['formula']:5.3f} {r['unusable']:5.3f}")
    return rows2


if __name__ == "__main__" and "--r3-fic21" in __import__("sys").argv:
    r3_fic21()
