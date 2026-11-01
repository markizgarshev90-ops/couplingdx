# couplingdx

Design diagnostics and a pooled estimator for two-person coupling studies
(mimicry, synchrony, convergence), from *Dividing by a difference of noisy
baselines: why per-dyad coupling estimates fail, and how to size a pooled
alternative*.

No dependencies. Python 3.8+.

```bash
pip install "git+https://github.com/markizgarshev90-ops/couplingdx#subdirectory=couplingdx"
# or, from this folder:
pip install .
python -m pytest tests        # optional
```

## What it answers

A coupling design observes two people together and compares them with baselines
measured separately for each. Two questions decide whether it can work:

1. **Can coupling be estimated for a single dyad?** The per-dyad estimate divides
   by the difference of two noisy baselines. Its precision is one combined
   signal-to-noise ratio, `zeta`; simulation shows the interquartile range of
   `theta_hat` is about `5.2 / zeta`, so a dyad is estimable only when
   `zeta >= 5.2` (`ZETA_STAR`).
2. **How many dyads does a pooled estimate need?** Regressing the joint-session
   asymmetry on the baseline contrast across dyads, correcting for baseline noise
   with replicate solo sessions, never divides by one dyad's contrast. Its power
   has a closed form, checked against simulation to within 0.03.

Once data are in, `fit_pooled` fits that estimator and `compare_groups` tests
the difference between two groups. `fit_partner_effect` tests for coupling without
needing facilitation to be the same for both diners or across groups.

## Tutorial (the eating example from the paper)

Inputs: mean bite rate 0.0867/s per person, between-meal SD of rates 0.0130/s,
dispersion 0.7 (bites are more regular than Poisson), 12-minute (740 s) meals.

```python
import couplingdx as c

G = c.expected_contrast(0.0130)                 # median contrast, random pairs: 0.0124
print(c.per_dyad_check(contrast=G, rate_total=2*0.0867, duration=740,
                       n_baseline=8, baseline_session=740, dispersion=0.7))
```

```
Per-dyad coupling estimate
========================================
  gap signal-to-noise      z0   =    0.97
  baseline contrast ratio  nu   =    2.74   (P sign error = 0.003)
  combined                 zeta =    0.94   (need >= 5.2)

  NOT FEASIBLE per dyad: theta_hat would be less precise than theta itself.
  required joint-session length: none suffices

Notes:
  - baseline precision alone rules out the target: more or longer
    baseline sessions are needed before joint-session length matters.
  - never discard estimates with h_hat >= 1, and never select dyads on
    the same baselines used in the estimate: both bias toward coupling.
```

How long would a meal have to be?

```python
c.required_duration(G, 2*0.0867, 0.7) / 3600            # 5.93 h, unlimited baselines
Gs = c.expected_contrast(0.0130, "screened", mean_rate=0.0867,
                         dispersion=0.7, screen_duration=740)
c.required_duration(Gs, 2*0.0867, 0.7, n_baseline=8, baseline_session=740) / 60   # 171 min
```

How many dyads per group would a pooled comparison of two closeness groups
(theta = 0 vs 1) need for 80% power?

```python
d = dict(duration=740, n_baseline=4, baseline_session=740,
         mean_rate=0.0867, between_sd=0.0130, dispersion=0.7)
c.pooled_n(0.8, **d)                                                       # 76
c.pooled_n(0.8, **dict(d, between_sd=0.0130/2**0.5, occasion_sd=0.0130/2**0.5))  # 376
c.pooled_n(0.8, **dict(d, pairing="screened"))                             # 34
# close group made of existing pairs whose solo rates correlate 0.3; strangers random
c.pooled_n(0.8, **d, high_design=dict(pairing="existing", partner_correlation=0.3))  # 94
```

## Fitting the pooled estimator

Each dyad is a tuple of the two members' per-session solo rates (events per unit
time, at least 2 sessions each, 4 or more recommended) and their joint-session
counts. Fit each group separately, then compare:

```python
dyad = ([0.081, 0.090, 0.077, 0.085],   # solo rates, person i (bites/s)
        [0.102, 0.095, 0.110, 0.099],   # solo rates, person j
        66, 73)                          # joint-session counts, i and j
fit_close = c.fit_pooled(dyads_close)            # list of such tuples
fit_strangers = c.fit_pooled(dyads_strangers)
print(fit_close)       # h_pool = ... (SE ...; 95% CI ... to ...), theta = ..., n = ... dyads
c.compare_groups(fit_close, fit_strangers)       # {'diff', 'se', 'z', 'p'}
```

`fit_pooled` uses the leverage-corrected (HC3) sandwich standard error by default
(`se_type="HC0"` gives the plain sandwich, adequate from about 50 dyads per
group), and raises `ValueError` if the corrected denominator `sum(x^2 - v)` is not
clearly positive, which means the sample carries too little true contrast. The
labelling of i and j within a dyad does not matter. Interpret the absolute level
of `h_pool` only relative to a comparison group: differential facilitation can
move it in either direction, and if facilitation differs *between* groups it
biases the comparison too.

## The partner-effect test

Dropping the constraint that the actor and partner effects sum to one gives a test
of no coupling that does not rely on facilitation being common to the two diners:

```python
print(c.fit_partner_effect(dyads_close, duration=740))
# partner effect = ... (SE ...; z = ..., p = ...), actor effect = ..., n = ... dyads
```

Each person's joint-session rate is regressed on their own and their partner's
baseline, correcting both for baseline noise with the replicate solo sessions; the
standard error is a delete-one-dyad jackknife. Under no coupling the partner effect
is zero whenever a person's facilitation depends only on their own rate (with
random pairing, or with existing pairs whose rates are jointly normal). Its size is
`A (1 - h) / 2`, so it detects coupling but does not estimate `h`.

Command line:

```bash
couplingdx per-dyad --contrast 0.0124 --rate-total 0.1734 --duration 740 \
    --n-baseline 8 --baseline-session 740 --dispersion 0.7
couplingdx pooled --mean-rate 0.0867 --between-sd 0.013 --duration 740 \
    --n-baseline 4 --baseline-session 740 --dispersion 0.7
```

## Functions

| Function | Returns |
|---|---|
| `h`, `inv_h` | contraction factor and its inverse, on the whole real line |
| `ceiling(scale_points, baseline_offset)` | maximum spread across an ordinal closeness scale |
| `counting_sd(n, dispersion)` | sampling SD of a difference of two counts (a benchmark, not a floor) |
| `sign_error_probability(nu)` | P(baseline contrast estimated with the wrong sign) |
| `per_dyad_precision`, `per_dyad_check` | `z0`, `nu`, `zeta`, verdict, required session length |
| `required_duration` | joint-session length for `zeta >= zeta_star` (inf if none suffices) |
| `expected_contrast`, `expected_squared_contrast` | contrasts under random, tercile-screened or existing pairs |
| `pooled_variance`, `pooled_power`, `pooled_n` | pooled estimator precision, power, dyads per group (`high_design` lets the two groups differ) |
| `fit_pooled`, `PooledFit` | the pooled estimate of h, its standard error, interval and theta |
| `fit_partner_effect`, `PartnerFit` | the partner-effect test of no coupling |
| `dyad_terms` | one dyad's `(y, x, v)` |
| `compare_groups` | two-group z-test on h |

## Practical rules the package encodes

- Never discard estimates with `h_hat >= 1`; invert `h` over the whole real line.
- Never select dyads on the same baselines that enter the estimate. Screen, if at
  all, on a separate session.
- Use at least 4 solo sessions per person for the pooled estimator, of the same
  length as the joint session.
- Occasion-to-occasion variability in a person's rate caps per-dyad precision at
  any session length; report it.
- Use the leverage-corrected standard error (the default); below about 50 dyads
  per group, also check the large-sample power formula by simulation.
- Report the partner-effect test alongside the comparison of h.
- Measure the stable and occasion parts of between-session variance first: about
  50 people with four solo sessions each pin the occasion share to within about 0.2.

## Changes

**0.3** adds the estimators: `fit_pooled`, `compare_groups`, `dyad_terms`,
`PooledFit`, and the partner-effect test `fit_partner_effect`; sizing for
pre-existing pairs (`pairing="existing"`, `partner_correlation`) and for groups
with different designs (`high_design`). Report notes are wrapped to 72 characters.

**0.2** replaced `design_check` with `per_dyad_check` (combined criterion
`zeta`), added the dispersion and occasion-variance terms, and added the pooled
sizing functions. 0.1 treated `sqrt(N)` as an irreducible floor, used a threshold
of `nu >= 12` (an artefact of a fixed number of baseline sessions), and had no
pooled estimator.

## License

MIT.
