"""Fitting the pooled errors-in-variables estimator of the contraction factor h.

Implements main-text Eq. (10),

    h_pool = sum_d y_d x_d / sum_d (x_d^2 - v_d),

with its M-estimation sandwich standard error, and the two-group comparison.
Pure Python, standard library only.

Input per dyad
--------------
solo_i, solo_j   : sequences of per-session solo *rates* (events / session length)
                   for the two members, at least 2 sessions each (4+ recommended).
joint_i, joint_j : event counts of the two members in the joint session.

Labelling of i and j within a dyad is arbitrary; the estimator is invariant to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist, fmean, variance
from typing import Sequence

from .core import inv_h

__all__ = ["PooledFit", "dyad_terms", "fit_pooled", "compare_groups",
           "PartnerFit", "fit_partner_effect"]


@dataclass
class PooledFit:
    h: float            # estimated contraction factor h (1 = no coupling)
    se: float           # sandwich standard error (HC3 or HC0)
    n: int              # number of dyads used
    denom: float        # sum(x^2 - v); must be clearly positive
    theta: float        # h inverted to theta over the whole real line (inf if h <= 0)

    def ci(self, level: float = 0.95) -> tuple:
        z = NormalDist().inv_cdf(0.5 + level / 2)
        return self.h - z * self.se, self.h + z * self.se

    def __str__(self) -> str:
        lo, hi = self.ci()
        return (f"h_pool = {self.h:.3f} (SE {self.se:.3f}; 95% CI {lo:.3f} to {hi:.3f}), "
                f"theta = {self.theta:.2f}, n = {self.n} dyads")


def dyad_terms(solo_i: Sequence[float], solo_j: Sequence[float],
               joint_i: float, joint_j: float) -> tuple:
    """Return (y_d, x_d, v_d) for one dyad."""
    m_i, m_j = len(solo_i), len(solo_j)
    if m_i < 2 or m_j < 2:
        raise ValueError("each person needs at least 2 solo sessions (4+ recommended)")
    b_i, b_j = fmean(solo_i), fmean(solo_j)
    total = joint_i + joint_j
    if total <= 0:
        raise ValueError("joint session has no events")
    y = (b_i + b_j) * (joint_j - joint_i) / total
    x = b_j - b_i
    v = variance(solo_i) / m_i + variance(solo_j) / m_j
    return y, x, v


def fit_pooled(dyads: Sequence[tuple], min_denom_ratio: float = 0.1,
               se_type: str = "HC3") -> PooledFit:
    """Fit h_pool to a list of dyads (solo_i, solo_j, joint_i, joint_j).

    Raises ValueError if the corrected denominator sum(x^2 - v) is not clearly
    positive (below ``min_denom_ratio`` times sum(x^2)): the sample then carries
    too little true contrast relative to baseline noise for the estimator to be
    usable.

    se_type: "HC3" (default) divides each dyad's score by (1 - leverage), with
    leverage (x_d^2 - v_d) / sum(x^2 - v); this kept the two-group z-test at its
    nominal size down to 20 dyads per group in simulation. "HC0" is the plain
    sandwich, adequate from about 50 dyads per group.
    """
    if se_type not in ("HC0", "HC3"):
        raise ValueError("se_type must be 'HC0' or 'HC3'")
    terms = [dyad_terms(*d) for d in dyads]
    if len(terms) < 2:
        raise ValueError("need at least 2 dyads")
    sxx = sum(x * x for _, x, _ in terms)
    denom = sum(x * x - v for _, x, v in terms)
    if denom <= min_denom_ratio * sxx:
        raise ValueError(
            f"corrected denominator {denom:.3g} is not clearly positive "
            f"(sum x^2 = {sxx:.3g}); add dyads, solo sessions, or screen for contrast")
    h = sum(y * x for y, x, _ in terms) / denom
    psi2 = 0.0
    for y, x, v in terms:
        w = x * x - v
        psi = y * x - h * w
        if se_type == "HC3":
            psi /= 1 - w / denom
        psi2 += psi * psi
    se = math.sqrt(psi2) / denom
    return PooledFit(h=h, se=se, n=len(terms), denom=denom, theta=inv_h(h))


def compare_groups(fit_a: PooledFit, fit_b: PooledFit) -> dict:
    """Two-sided z-test of h_a = h_b for independent groups.
    Returns diff (h_a - h_b), its se, z and the p-value."""
    diff = fit_a.h - fit_b.h
    se = math.hypot(fit_a.se, fit_b.se)
    z = diff / se
    p = 2 * (1 - NormalDist().cdf(abs(z)))
    return {"diff": diff, "se": se, "z": z, "p": p}


# --------------------------------------------------------------------------
# unconstrained actor-partner model: a test of kappa = 0 that does not need
# facilitation to be common to the two diners
# --------------------------------------------------------------------------
@dataclass
class PartnerFit:
    actor: float        # coefficient on the person's own baseline (A (1+h)/2 under the model)
    partner: float      # coefficient on the partner's baseline (A (1-h)/2; 0 if no coupling)
    se: float           # delete-one-dyad jackknife standard error of `partner`
    n: int              # number of dyads
    z: float
    p: float            # two-sided p-value for partner = 0

    def __str__(self) -> str:
        return (f"partner effect = {self.partner:.3f} (SE {self.se:.3f}; z = {self.z:.2f}, "
                f"p = {self.p:.3g}), actor effect = {self.actor:.3f}, n = {self.n} dyads")


def _apim(rows_sums, N):
    Sy, So, Sp, Soo, Spp, Sop, Soy, Spy, Vo, Vp = rows_sums
    Moo = Soo - So * So / N - Vo
    Mpp = Spp - Sp * Sp / N - Vp
    Mop = Sop - So * Sp / N
    Coy = Soy - So * Sy / N
    Cpy = Spy - Sp * Sy / N
    det = Moo * Mpp - Mop * Mop
    if det <= 0 or Moo <= 0 or Mpp <= 0:
        raise ValueError("corrected moment matrix is not positive definite: "
                         "too little spread in baselines relative to their noise")
    return (Mpp * Coy - Mop * Cpy) / det, (Moo * Cpy - Mop * Coy) / det


def fit_partner_effect(dyads: Sequence[tuple], duration) -> PartnerFit:
    """Unconstrained actor-partner model fitted across dyads.

    Each person's joint-session rate k / duration is regressed on their own and their
    partner's baseline (mean solo rate), with an intercept, correcting both
    regressors for baseline noise with the replicate solo sessions (method of
    moments); both members of a dyad enter as rows. Under the slow-drive model the
    partner coefficient is A (1-h)/2: it is zero when there is no coupling, whatever
    the form of each person's facilitation, as long as facilitation depends only on
    that person's own rate. Its size depends on A, so it is not an estimate of h.

    dyads    : (solo_i, solo_j, joint_i, joint_j) as for `fit_pooled`
    duration : joint-session length, one number or one per dyad (same unit as rates)
    """
    dyads = list(dyads)
    n = len(dyads)
    if n < 3:
        raise ValueError("need at least 3 dyads")
    Ts = list(duration) if hasattr(duration, "__len__") else [duration] * n
    if len(Ts) != n or min(Ts) <= 0:
        raise ValueError("duration must be positive, one value or one per dyad")
    contrib = []
    for (si, sj, ki, kj), T in zip(dyads, Ts):
        if len(si) < 2 or len(sj) < 2:
            raise ValueError("each person needs at least 2 solo sessions (4+ recommended)")
        bi, bj = fmean(si), fmean(sj)
        vi, vj = variance(si) / len(si), variance(sj) / len(sj)
        yi, yj = ki / T, kj / T
        contrib.append((yi + yj, bi + bj, bj + bi, bi * bi + bj * bj, bj * bj + bi * bi,
                        2 * bi * bj, bi * yi + bj * yj, bj * yi + bi * yj, vi + vj, vj + vi))
    tot = [sum(c[k] for c in contrib) for k in range(10)]
    actor, partner = _apim(tot, 2 * n)
    loo = [_apim([tot[k] - c[k] for k in range(10)], 2 * (n - 1))[1] for c in contrib]
    mean_loo = fmean(loo)
    se = math.sqrt((n - 1) / n * sum((x - mean_loo) ** 2 for x in loo))
    z = partner / se
    return PartnerFit(actor=actor, partner=partner, se=se, n=n, z=z,
                      p=2 * (1 - NormalDist().cdf(abs(z))))
