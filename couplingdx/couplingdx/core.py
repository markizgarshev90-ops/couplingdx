"""Design diagnostics for two-person coupling studies.

A coupling design observes two individuals together and compares the joint
observation against baselines measured separately for each.  This module answers,
before data are collected, two questions:

1. Can the coupling be estimated *dyad by dyad*?  The per-dyad estimator

       h_hat = (k_j - k_i)/(k_j + k_i) * S_hat / Gamma_hat

   divides by a difference of two noisy baselines.  Its precision is governed by
   one combined signal-to-noise ratio

       zeta = [ 1/z0^2 + h^2/nu^2 ]^(-1/2),

   where z0 is the expected joint-session gap in units of its sampling SD and nu
   is the baseline contrast in units of its standard error.  Simulation shows
   that the interquartile range of theta_hat is at most theta (at theta = 1)
   once zeta >= ZETA_STAR = 5.2.

2. How many dyads does a *pooled* estimator need?  Regressing the joint-session
   asymmetry on the baseline contrast across dyads, with replicate baselines
   correcting for their measurement error, never divides by a single dyad's
   contrast.  Its variance is available in closed form (`pooled_variance`).

The contraction factor is h(theta) = (1 - exp(-theta))/theta, theta = 2*kappa*T,
defined for all real theta (h > 1 for theta < 0, i.e. divergence).

All rates are per unit time in the unit the caller chooses (bites/s in the paper);
`dispersion` is the variance-to-mean ratio of the event counts (about 0.7 for
bites in the FIC data, so bites are *less* variable than Poisson).
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field

__all__ = [
    "h", "inv_h", "ceiling", "counting_sd", "sign_error_probability",
    "per_dyad_precision", "required_duration", "per_dyad_check", "PerDyadReport",
    "expected_contrast", "expected_squared_contrast", "pooled_variance",
    "pooled_power", "pooled_n", "ZETA_STAR", "Z_975",
]

#: Combined per-dyad signal-to-noise ratio at which the interquartile range of
#: theta_hat falls to the true theta (theta = 1); from simulation, see paper.
ZETA_STAR = 5.2
Z_975 = 1.959963984540054
_K_TERCILE = 1.0909      # E[Z | Z > z_(2/3)] for a standard normal
_V_TERCILE = 0.2798      # Var[Z | Z > z_(2/3)]
_MEDIAN_ABS_DIFF = 0.9539  # median |X - Y| / sd for X, Y iid normal


# --------------------------------------------------------------------------
# the observable
# --------------------------------------------------------------------------
def h(theta: float) -> float:
    """Contraction factor h(theta) = (1 - e^-theta)/theta, h(0) = 1.

    Strictly decreasing on the whole real line; h > 1 for theta < 0.
    """
    if abs(theta) < 1e-8:
        return 1.0 - theta / 2.0
    return -math.expm1(-theta) / theta


def _dh(theta: float) -> float:
    if abs(theta) < 1e-6:
        return -0.5 + theta / 3.0
    return ((1.0 + theta) * math.exp(-theta) - 1.0) / theta ** 2


def inv_h(value: float) -> float:
    """Inverse of h.  Any positive value is invertible (values above 1 give
    theta < 0); a non-positive value returns +inf (gap reversed relative to the
    baseline contrast).  Never truncate at 1: doing so biases toward coupling."""
    if value <= 0.0:
        return math.inf
    lo, hi = -60.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if h(mid) > value:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ceiling(scale_points: int = 7, baseline_offset: float = 0.0) -> float:
    """Largest spread in the asymptotic contraction factor across an ordinal scale.

    Under kappa = c * (baseline_offset + s) for s = 1..scale_points the
    effective ratio is q = (offset + max)/(offset + 1) and the attainable spread
    is (sqrt(q) - 1)/(sqrt(q) + 1), before any noise.
    """
    if scale_points < 2:
        raise ValueError("scale_points must be at least 2")
    if baseline_offset < 0:
        raise ValueError("baseline_offset must be non-negative")
    q = (baseline_offset + scale_points) / (baseline_offset + 1.0)
    return (math.sqrt(q) - 1.0) / (math.sqrt(q) + 1.0)


def counting_sd(n_events_total: float, dispersion: float = 1.0) -> float:
    """Sampling SD of a difference of two independent counts totalling
    n_events_total, sqrt(dispersion * N).  A benchmark, not a floor: regular
    (under-dispersed) processes fall below the Poisson value, and coupling that
    acts on realised events lowers it further."""
    if n_events_total < 0 or dispersion <= 0:
        raise ValueError("n_events_total must be >= 0 and dispersion > 0")
    return math.sqrt(dispersion * n_events_total)


def sign_error_probability(nu: float) -> float:
    """P(the estimated baseline contrast has the wrong sign), normal approx.

    This, not the existence of moments, is what nu ~ 2-3 controls: the ratio
    estimator has no finite moments at any nu."""
    return 0.5 * math.erfc(nu / math.sqrt(2.0))


# --------------------------------------------------------------------------
# per-dyad estimation
# --------------------------------------------------------------------------
def per_dyad_precision(contrast: float, rate_total: float, duration: float,
                       n_baseline: int, baseline_session: float, dispersion: float = 1.0,
                       occasion_sd: float = 0.0, facilitation: float = 1.0,
                       theta_ref: float = 1.0) -> dict:
    """Signal-to-noise ratios of the per-dyad estimator.

    contrast          |difference| between the two people's mean rates (Gamma)
    rate_total        sum of the two mean rates (S)
    duration          joint-session length T
    n_baseline        solo sessions per person (m)
    baseline_session  length of each solo session
    occasion_sd       SD of a person's rate across occasions (0 if perfectly stable)

    Returns
      z0    uncoupled joint-session gap / its sampling SD
      nu    baseline contrast / its standard error
      zeta  combined ratio [ s_D^2/(A Gamma T)^2 + h^2/nu^2 ]^(-1/2) at theta_ref,
            which is 1/SD(h_hat) relative to h's scale (Supplementary S2)
      sign_error  P(baseline contrast estimated with the wrong sign)
    """
    if min(contrast, rate_total, duration, baseline_session) <= 0 or n_baseline < 1:
        raise ValueError("contrast, rate_total, durations and n_baseline must be positive")
    A, T, phi = facilitation, duration, dispersion
    hr = h(theta_ref)
    gap_var_0 = phi * A * rate_total * T + 2 * (A * occasion_sd * T) ** 2
    gap_var = phi * A * rate_total * T + 2 * (A * hr * occasion_sd * T) ** 2
    base_var = (phi * rate_total / baseline_session + 2 * occasion_sd ** 2) / n_baseline
    z0 = A * contrast * T / math.sqrt(gap_var_0)
    nu = contrast / math.sqrt(base_var)
    zeta = 1.0 / math.sqrt(gap_var / (A * contrast * T) ** 2 + hr ** 2 / nu ** 2)
    return dict(z0=z0, nu=nu, zeta=zeta, sign_error=sign_error_probability(nu))


def required_duration(contrast: float, rate_total: float, dispersion: float = 1.0,
                      n_baseline: int = None, baseline_session: float = None,
                      facilitation: float = 1.0, occasion_sd: float = 0.0,
                      zeta_star: float = ZETA_STAR, theta_ref: float = 1.0) -> float:
    """Joint-session length at which a single dyad reaches zeta_star.

    With unlimited baselines (n_baseline=None) and no occasion variance:

        T_req = zeta_star^2 * dispersion * rate_total / (facilitation * contrast^2)

    Finite baselines add h^2/nu^2 to the error budget; if that alone exceeds
    1/zeta_star^2 no duration suffices and inf is returned.  Occasion variance
    adds 2 h^2 occasion_sd^2 / contrast^2, which does not shrink with T: if it
    exceeds what is left of the budget, inf is returned.
    """
    if contrast <= 0 or rate_total <= 0:
        raise ValueError("contrast and rate_total must be positive")
    hr = h(theta_ref); phi, A = dispersion, facilitation
    budget = 1.0 / zeta_star ** 2
    if n_baseline is not None:
        base_var = (phi * rate_total / baseline_session + 2 * occasion_sd ** 2) / n_baseline
        budget -= hr ** 2 * base_var / contrast ** 2
    budget -= 2 * hr ** 2 * occasion_sd ** 2 / contrast ** 2
    if budget <= 0:
        return math.inf
    return phi * rate_total / (A * contrast ** 2 * budget)


@dataclass
class PerDyadReport:
    z0: float
    nu: float
    zeta: float
    zeta_star: float
    sign_error: float
    feasible: bool
    required_duration: float
    notes: list = field(default_factory=list)

    def __str__(self) -> str:
        L = ["Per-dyad coupling estimate", "=" * 40,
             f"  gap signal-to-noise      z0   = {self.z0:7.2f}",
             f"  baseline contrast ratio  nu   = {self.nu:7.2f}   "
             f"(P sign error = {self.sign_error:.3f})",
             f"  combined                 zeta = {self.zeta:7.2f}   (need >= {self.zeta_star:g})",
             ""]
        L.append("  FEASIBLE per dyad." if self.feasible else
                 "  NOT FEASIBLE per dyad: theta_hat would be less precise than theta itself.")
        rd = self.required_duration
        L.append("  required joint-session length: " +
                 ("none suffices" if not math.isfinite(rd) else f"{rd:,.0f}"))
        if self.notes:
            L += ["", "Notes:"]
            for n in self.notes:
                L += textwrap.wrap(n, width=72, initial_indent="  - ", subsequent_indent="    ")
        return "\n".join(L)


def per_dyad_check(contrast: float, rate_total: float, duration: float,
                   n_baseline: int, baseline_session: float, dispersion: float = 1.0,
                   occasion_sd: float = 0.0, facilitation: float = 1.0,
                   zeta_star: float = ZETA_STAR, theta_ref: float = 1.0) -> PerDyadReport:
    """Can theta be estimated for one dyad?  See `per_dyad_precision`."""
    p = per_dyad_precision(contrast, rate_total, duration, n_baseline, baseline_session,
                           dispersion, occasion_sd, facilitation, theta_ref)
    T_req = required_duration(contrast, rate_total, dispersion, n_baseline, baseline_session,
                              facilitation, occasion_sd, zeta_star, theta_ref)
    notes = []
    if p["nu"] < h(theta_ref) * zeta_star:
        notes.append("baseline precision alone rules out the target: more or longer "
                     "baseline sessions are needed before joint-session length matters.")
    if occasion_sd > 0:
        notes.append("occasion-to-occasion variability adds gap noise that does not shrink "
                     "with session length.")
    notes.append("never discard estimates with h_hat >= 1, and never select dyads on "
                 "the same baselines used in the estimate: both bias toward coupling.")
    return PerDyadReport(z0=p["z0"], nu=p["nu"], zeta=p["zeta"], zeta_star=zeta_star,
                         sign_error=p["sign_error"], feasible=p["zeta"] >= zeta_star,
                         required_duration=T_req, notes=notes)


# --------------------------------------------------------------------------
# pairing schemes
# --------------------------------------------------------------------------
def _screen_reliability(between_sd, occasion_sd, mean_rate, dispersion, screen_duration):
    noise = occasion_sd ** 2 + dispersion * mean_rate / screen_duration
    return between_sd ** 2 / (between_sd ** 2 + noise)


def expected_contrast(between_sd: float, pairing: str = "random", *, occasion_sd: float = 0.0,
                      mean_rate: float = None, dispersion: float = 1.0,
                      screen_duration: float = None, partner_correlation: float = 0.0) -> float:
    """Typical |contrast| of a dyad: the median for random pairing, the mean for
    pairing opposite terciles of one screening session (attenuated by that
    session's reliability; pass screen_duration=math.inf to screen on true rates),
    and the median for pre-existing pairs ("existing") whose stable rates have
    correlation partner_correlation."""
    if pairing == "random":
        return _MEDIAN_ABS_DIFF * between_sd
    if pairing == "existing":
        return _MEDIAN_ABS_DIFF * between_sd * math.sqrt(1.0 - _check_rho(partner_correlation))
    if pairing == "screened":
        R = 1.0 if not screen_duration or math.isinf(screen_duration) else \
            _screen_reliability(between_sd, occasion_sd, mean_rate, dispersion, screen_duration)
        return 2 * _K_TERCILE * math.sqrt(R) * between_sd
    raise ValueError("pairing must be 'random', 'screened' or 'existing'")


def _check_rho(rho):
    if not -1.0 < rho < 1.0:
        raise ValueError("partner_correlation must lie strictly between -1 and 1")
    return rho


def expected_squared_contrast(between_sd: float, pairing: str = "random", *,
                              occasion_sd: float = 0.0, mean_rate: float = None,
                              dispersion: float = 1.0, screen_duration: float = None,
                              partner_correlation: float = 0.0) -> float:
    """E[Gamma^2] across dyads; this is what pooled precision depends on.
    For pre-existing pairs ("existing", e.g. friends or couples), whose stable
    rates may be correlated, E[Gamma^2] = 2 between_sd^2 (1 - partner_correlation)."""
    if pairing == "random":
        return 2 * between_sd ** 2
    if pairing == "existing":
        return 2 * between_sd ** 2 * (1.0 - _check_rho(partner_correlation))
    if pairing == "screened":
        R = 1.0 if not screen_duration or math.isinf(screen_duration) else \
            _screen_reliability(between_sd, occasion_sd, mean_rate, dispersion, screen_duration)
        return between_sd ** 2 * (2 - 2 * (1 - _V_TERCILE) * R + 4 * _K_TERCILE ** 2 * R)
    raise ValueError("pairing must be 'random', 'screened' or 'existing'")


# --------------------------------------------------------------------------
# pooled estimation
# --------------------------------------------------------------------------
def pooled_variance(n_dyads: int, theta: float, duration: float, n_baseline: int,
                    baseline_session: float, mean_rate: float, between_sd: float,
                    occasion_sd: float = 0.0, dispersion: float = 1.0,
                    facilitation: float = 1.0, pairing: str = "random",
                    screen_duration: float = None, partner_correlation: float = 0.0) -> float:
    """Large-sample variance of the pooled (errors-in-variables) estimate of h.

    Estimator: h = sum(y x) / sum(x^2 - v) over dyads, y the joint-session
    asymmetry times the baseline sum, x the signed baseline contrast, v its
    replicate-based error variance.  Requires n_baseline >= 2 (>= 4 advised).
    pairing: "random", "screened" (opposite terciles of one independent screening
    session) or "existing" (pre-existing pairs whose stable rates have correlation
    partner_correlation; this lowers the mean squared contrast and so the power).
    """
    if n_baseline < 2:
        raise ValueError("the pooled estimator needs at least 2 baseline sessions per person")
    hh = h(theta); S = 2 * mean_rate; phi = dispersion
    sig_e2 = 2 * hh ** 2 * occasion_sd ** 2 + phi * S / (facilitation * duration)
    sig_u2 = (2 * occasion_sd ** 2 + phi * S / baseline_session) / n_baseline
    EG2 = expected_squared_contrast(between_sd, pairing, occasion_sd=occasion_sd,
                                    mean_rate=mean_rate, dispersion=dispersion,
                                    screen_duration=screen_duration or baseline_session,
                                    partner_correlation=partner_correlation)
    vpsi = (sig_e2 * (EG2 + sig_u2) + hh ** 2 * EG2 * sig_u2
            + hh ** 2 * sig_u2 ** 2 * (2 + 1.0 / (n_baseline - 1)))
    return vpsi / (n_dyads * EG2 ** 2)


def _Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def pooled_power(n_per_group: int, theta_low: float, theta_high: float,
                 high_design: dict = None, **design) -> float:
    """Power of a two-sided 5% z-test comparing h between two groups of dyads
    (e.g. low vs high closeness).  `design` is passed to `pooled_variance`;
    `high_design` overrides entries for the theta_high group only, e.g.
    dict(pairing="existing", partner_correlation=0.4) when the close group consists
    of pre-existing pairs and the comparison group is randomly paired."""
    v = (pooled_variance(n_per_group, theta_low, **design)
         + pooled_variance(n_per_group, theta_high, **{**design, **(high_design or {})}))
    d = abs(h(theta_low) - h(theta_high)) / math.sqrt(v)
    return _Phi(d - Z_975) + _Phi(-d - Z_975)


def pooled_n(power: float = 0.8, theta_low: float = 0.0, theta_high: float = 1.0,
             high_design: dict = None, **design) -> int:
    """Smallest number of dyads per group reaching `power` (see `pooled_power`)."""
    v1 = (pooled_variance(1, theta_low, **design)
          + pooled_variance(1, theta_high, **{**design, **(high_design or {})}))
    dh = abs(h(theta_low) - h(theta_high))
    zb = -_inv_Phi(1 - power)
    n = max(1, math.ceil((Z_975 + zb) ** 2 * v1 / dh ** 2))
    while n > 1 and pooled_power(n - 1, theta_low, theta_high, high_design, **design) >= power:
        n -= 1
    while pooled_power(n, theta_low, theta_high, high_design, **design) < power:
        n += 1
    return n


def _inv_Phi(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
