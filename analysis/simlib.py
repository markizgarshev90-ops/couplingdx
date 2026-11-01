"""Shared simulation machinery for the per-dyad and pooled coupling estimators.

Generative model (slow-drive limit of either coupling, main text Eq. 4):
  person p has a stable mean bite rate beta_p; on any one occasion the rate is
  beta_p + eps, eps ~ N(0, sigma_o^2) (occasion variance; 0 in the base case).
  In a joint meal of length T with contraction factor h = h(theta), the expected
  counts are
      E k_i = T [ (1+h)/2 b_i + (1-h)/2 b_j ],   E k_j = T [ (1+h)/2 b_j + (1-h)/2 b_i ]
  with b the occasion rates.  A solo meal of length Ts has E k = Ts * b.
  Counts are drawn with variance phi * mean (discretised normal; phi < 1 is the
  under-dispersion measured in FIC).
"""
import numpy as np

H_EPS = 1e-9


def h(theta):
    """Contraction factor, extended smoothly to theta < 0 (h > 1 there)."""
    t = np.asarray(theta, float)
    out = np.ones_like(t)
    nz = np.abs(t) > 1e-8
    out[nz] = -np.expm1(-t[nz]) / t[nz]
    out[~nz] = 1 - t[~nz] / 2
    return out if out.ndim else float(out)


def inv_h(v, lo=-60.0, hi=1e4, iters=80):
    """Vectorised inverse of h on (0, inf). Returns +inf where v <= 0."""
    v = np.atleast_1d(np.asarray(v, float))
    out = np.full(v.shape, np.inf)
    ok = v > 0
    a = np.full(ok.sum(), lo); b = np.full(ok.sum(), hi); vv = v[ok]
    for _ in range(iters * 2):
        m = 0.5 * (a + b)
        hm = h(m)
        big = hm > vv          # h decreasing: h(m) > v means root is to the right
        a = np.where(big, m, a); b = np.where(big, b, m)
    out[ok] = 0.5 * (a + b)
    return out


def counts(mean, phi, rng):
    mean = np.maximum(mean, 0.0)
    if phi == 1.0:
        return rng.poisson(mean).astype(float)
    x = np.rint(mean + np.sqrt(phi * mean) * rng.standard_normal(mean.shape))
    return np.maximum(x, 0.0)


def joint_meal(bi, bj, T, theta, phi, rng):
    hh = h(theta)
    mi = T * ((1 + hh) / 2 * bi + (1 - hh) / 2 * bj)
    mj = T * ((1 + hh) / 2 * bj + (1 - hh) / 2 * bi)
    return counts(mi, phi, rng), counts(mj, phi, rng)


def solo_meals(beta, m, Ts, phi, sigma_o, rng):
    """Returns per-meal rates, shape beta.shape + (m,)."""
    b = beta[..., None] + sigma_o * rng.standard_normal(beta.shape + (m,))
    return counts(Ts * np.maximum(b, 1e-6), phi, rng) / Ts


def fieller(a, b, va, vb, z=1.959964):
    """95% Fieller set for a/b with independent a, b. Returns (lo, hi, bounded)."""
    A = b * b - z * z * vb
    Bq = -2 * a * b
    C = a * a - z * z * va
    disc = Bq * Bq - 4 * A * C
    bounded = (A > 0) & (disc >= 0)
    sq = np.sqrt(np.maximum(disc, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = (-Bq - sq) / (2 * A); hi = (-Bq + sq) / (2 * A)
    return np.where(bounded, lo, -np.inf), np.where(bounded, hi, np.inf), bounded
