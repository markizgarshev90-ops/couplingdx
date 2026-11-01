import math
import pytest
import couplingdx as c

MU, SB, PHI = 0.0867, 0.01304, 0.7
S = 2 * MU


def test_h_identities():
    assert c.h(0) == 1.0
    assert abs(c.h(1) - (1 - math.exp(-1))) < 1e-12
    assert c.h(-1) > 1 > c.h(1)                       # extended to divergence
    for t in (-2.0, -0.3, 0.0, 0.7, 5.0):
        assert abs(c.inv_h(c.h(t)) - t) < 1e-8
    assert c.inv_h(0.0) == math.inf and c.inv_h(-0.2) == math.inf


def test_ceiling():
    assert abs(c.ceiling(7) - 0.4514) < 1e-3
    assert abs(c.ceiling(7, 1) - 1 / 3) < 1e-9
    assert c.ceiling(7, 5) < c.ceiling(7, 2) < c.ceiling(7, 0)


def test_counting_and_sign():
    assert abs(c.counting_sd(126.9, 1.0) - 11.265) < 1e-2
    assert c.counting_sd(126.9, 0.7) < c.counting_sd(126.9, 1.0)
    assert abs(c.sign_error_probability(2) - 0.02275) < 1e-4


def test_zeta_combines_and_matches_requirement():
    # with the session at exactly the required length, zeta equals zeta_star
    G = 0.03
    T = c.required_duration(G, S, PHI, 8, 740)
    p = c.per_dyad_precision(G, S, T, 8, 740, PHI)
    assert abs(p["zeta"] - c.ZETA_STAR) < 1e-6
    # baselines alone can rule a design out
    assert c.required_duration(0.0124, S, PHI, 8, 740) == math.inf
    # occasion variance can rule it out at any length
    assert c.required_duration(0.05, S, PHI, 64, 740, occasion_sd=0.02) == math.inf


def test_worked_case_eating_per_dyad():
    G = c.expected_contrast(SB)
    hours = c.required_duration(G, S, PHI) / 3600
    assert 5.5 < hours < 6.3                           # paper: about 6 h, random pairing


def test_worked_case_speech_per_dyad():
    G = c.expected_contrast(0.59)
    hours = c.required_duration(G, 2 * 5.12, 16.4) / 3600
    assert 3.7 < hours < 4.3                           # paper: about 4 h, random pairing
    Gt = c.expected_contrast(0.59, "screened")
    assert 40 < c.required_duration(Gt, 2 * 5.12, 16.4) / 60 < 52


def test_pooled_sizes():
    d = dict(duration=740, n_baseline=4, baseline_session=740, mean_rate=MU,
             between_sd=SB, dispersion=PHI)
    n = c.pooled_n(0.8, **d)
    assert 60 <= n <= 90                               # validated by simulation in the paper
    assert c.pooled_power(n, 0, 1, **d) >= 0.8 > c.pooled_power(n - 1, 0, 1, **d)
    # half the stable variance moved to occasion variance needs many more dyads
    d2 = dict(d, between_sd=SB / math.sqrt(2), occasion_sd=SB / math.sqrt(2))
    assert c.pooled_n(0.8, **d2) > 3 * n
    # screening on one independent session reduces the requirement
    assert c.pooled_n(0.8, **dict(d, pairing="screened")) < n
    with pytest.raises(ValueError):
        c.pooled_variance(100, 0, 740, 1, 740, MU, SB)


def test_cli_runs(capsys):
    from couplingdx.__main__ import main
    main(["per-dyad", "--contrast", "0.0124", "--rate-total", "0.173", "--duration", "740",
          "--n-baseline", "8", "--baseline-session", "740", "--dispersion", "0.7"])
    main(["pooled", "--mean-rate", "0.0867", "--between-sd", "0.013", "--duration", "740",
          "--n-baseline", "4", "--baseline-session", "740", "--dispersion", "0.7"])
    out = capsys.readouterr().out
    assert "NOT FEASIBLE" in out and "dyads per group" in out


def test_existing_pairs():
    d = dict(duration=740, n_baseline=4, baseline_session=740, mean_rate=MU,
             between_sd=SB, dispersion=PHI)
    assert c.expected_squared_contrast(SB, "existing", partner_correlation=0.0) == c.expected_squared_contrast(SB)
    assert abs(c.expected_squared_contrast(SB, "existing", partner_correlation=0.5) - SB ** 2) < 1e-15
    n0 = c.pooled_n(0.8, **d)
    n5 = c.pooled_n(0.8, **dict(d, pairing="existing", partner_correlation=0.5))
    assert n5 > n0                     # more similar partners, less contrast, more dyads
    with pytest.raises(ValueError):
        c.expected_squared_contrast(SB, "existing", partner_correlation=1.0)
