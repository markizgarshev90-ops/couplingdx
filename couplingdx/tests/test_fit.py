import math
import random
import pytest
import couplingdx as c


def simulate(n, theta, seed, m=4, T=740.0, mu=0.0867, sp=0.013, phi=0.7):
    """Slow-drive dyads: normal per-session rates, counts with variance phi * mean."""
    rng = random.Random(seed)
    hh = c.h(theta)
    out = []
    for _ in range(n):
        bi, bj = rng.gauss(mu, sp), rng.gauss(mu, sp)
        si = [rng.gauss(bi, math.sqrt(phi * bi / T)) for _ in range(m)]
        sj = [rng.gauss(bj, math.sqrt(phi * bj / T)) for _ in range(m)]
        mi = T * ((1 + hh) / 2 * bi + (1 - hh) / 2 * bj)
        mj = T * ((1 + hh) / 2 * bj + (1 - hh) / 2 * bi)
        out.append((si, sj, rng.gauss(mi, math.sqrt(phi * mi)), rng.gauss(mj, math.sqrt(phi * mj))))
    return out


def test_recovers_h():
    f = c.fit_pooled(simulate(400, 1.0, seed=3))
    assert abs(f.h - c.h(1.0)) < 3 * f.se
    lo, hi = f.ci()
    assert lo < f.h < hi and f.n == 400
    assert abs(c.h(f.theta) - f.h) < 1e-9


def test_label_invariance():
    d = simulate(60, 1.0, seed=4)
    swapped = [(sj, si, kj, ki) if k % 2 else (si, sj, ki, kj) for k, (si, sj, ki, kj) in enumerate(d)]
    a, b = c.fit_pooled(d), c.fit_pooled(swapped)
    assert abs(a.h - b.h) < 1e-12 and abs(a.se - b.se) < 1e-12


def test_matches_formula_by_hand():
    d = simulate(30, 0.5, seed=5)
    t = [c.dyad_terms(*x) for x in d]
    den = sum(x * x - v for _, x, v in t)
    h = sum(y * x for y, x, _ in t) / den
    se0 = math.sqrt(sum((y * x - h * (x * x - v)) ** 2 for y, x, v in t)) / den
    f0 = c.fit_pooled(d, se_type="HC0"); f3 = c.fit_pooled(d)
    assert abs(f0.h - h) < 1e-12 and abs(f0.se - se0) < 1e-12
    assert f3.se > f0.se                     # leverage correction widens the interval


def test_refuses_without_contrast():
    # identical baselines in every dyad: no true contrast, denominator not positive
    rng = random.Random(6)
    d = [([0.08 + rng.gauss(0, .01) for _ in range(4)], [0.08 + rng.gauss(0, .01) for _ in range(4)], 60, 61)
         for _ in range(40)]
    with pytest.raises(ValueError):
        c.fit_pooled(d)
    with pytest.raises(ValueError):
        c.dyad_terms([0.08], [0.09, 0.1], 60, 61)          # one solo session
    with pytest.raises(ValueError):
        c.fit_pooled(d, se_type="HC1")


def test_compare_groups():
    a = c.fit_pooled(simulate(200, 0.0, seed=7)); b = c.fit_pooled(simulate(200, 1.0, seed=8))
    r = c.compare_groups(a, b)
    assert r["diff"] > 0 and 0 <= r["p"] <= 1
    assert abs(r["se"] - math.hypot(a.se, b.se)) < 1e-15
    r2 = c.compare_groups(b, a)
    assert abs(r2["z"] + r["z"]) < 1e-12 and abs(r2["p"] - r["p"]) < 1e-12


def simulate_facil(n, theta, seed, delta=0.0, A0=1.1, m=4, T=740.0, mu=0.0867, sp=0.013, phi=0.7):
    """As `simulate`, with facilitation A = A0 (1 + delta * standardised own rate)."""
    rng = random.Random(seed)
    hh = c.h(theta)
    out = []
    for _ in range(n):
        bi, bj = rng.gauss(mu, sp), rng.gauss(mu, sp)
        si = [rng.gauss(bi, math.sqrt(phi * bi / T)) for _ in range(m)]
        sj = [rng.gauss(bj, math.sqrt(phi * bj / T)) for _ in range(m)]
        ei, ej = A0 * (1 + delta * (bi - mu) / sp) * bi, A0 * (1 + delta * (bj - mu) / sp) * bj
        mi = T * ((1 + hh) / 2 * ei + (1 - hh) / 2 * ej)
        mj = T * ((1 + hh) / 2 * ej + (1 - hh) / 2 * ei)
        out.append((si, sj, rng.gauss(mi, math.sqrt(phi * mi)), rng.gauss(mj, math.sqrt(phi * mj))))
    return out


def test_partner_effect():
    f = c.fit_partner_effect(simulate_facil(600, 1.0, seed=11, delta=0.0), 740.0)
    target = 1.1 * (1 - c.h(1.0)) / 2
    assert abs(f.partner - target) < 3 * f.se and f.p < 0.001
    # no coupling, strong facilitation gradient: partner effect stays near zero
    g = c.fit_partner_effect(simulate_facil(600, 0.0, seed=12, delta=0.05), 740.0)
    assert abs(g.partner) < 3 * g.se
    # durations per dyad and label invariance
    d = simulate_facil(50, 1.0, seed=13)
    a = c.fit_partner_effect(d, [740.0] * 50)
    sw = [(sj, si, kj, ki) for si, sj, ki, kj in d]
    b = c.fit_partner_effect(sw, 740.0)
    assert abs(a.partner - b.partner) < 1e-12 and abs(a.se - b.se) < 1e-12
    with pytest.raises(ValueError):
        c.fit_partner_effect(d[:2], 740.0)
