"""Command-line interface.

  python -m couplingdx per-dyad --contrast 0.0124 --rate-total 0.173 --duration 740 \
        --n-baseline 8 --baseline-session 740 --dispersion 0.7
  python -m couplingdx pooled --mean-rate 0.0867 --between-sd 0.013 --duration 740 \
        --n-baseline 4 --baseline-session 740 --dispersion 0.7
"""
import argparse, math

from .core import per_dyad_check, pooled_n, pooled_power, ZETA_STAR


def main(argv=None):
    p = argparse.ArgumentParser(prog="couplingdx",
                                description="Design diagnostics for two-person coupling studies.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("per-dyad", help="can theta be estimated for a single dyad?")
    a.add_argument("--contrast", type=float, required=True, help="|difference| of the two mean rates")
    a.add_argument("--rate-total", type=float, required=True, help="sum of the two mean rates")
    a.add_argument("--duration", type=float, required=True, help="joint-session length")
    a.add_argument("--n-baseline", type=int, required=True, help="solo sessions per person")
    a.add_argument("--baseline-session", type=float, required=True, help="length of one solo session")
    a.add_argument("--dispersion", type=float, default=1.0)
    a.add_argument("--occasion-sd", type=float, default=0.0)
    a.add_argument("--facilitation", type=float, default=1.0)
    a.add_argument("--zeta-star", type=float, default=ZETA_STAR)

    b = sub.add_parser("pooled", help="dyads per group for a pooled two-group comparison")
    b.add_argument("--mean-rate", type=float, required=True, help="population mean rate per person")
    b.add_argument("--between-sd", type=float, required=True, help="SD of stable person-level rates")
    b.add_argument("--duration", type=float, required=True)
    b.add_argument("--n-baseline", type=int, required=True)
    b.add_argument("--baseline-session", type=float, required=True)
    b.add_argument("--dispersion", type=float, default=1.0)
    b.add_argument("--occasion-sd", type=float, default=0.0)
    b.add_argument("--facilitation", type=float, default=1.0)
    b.add_argument("--pairing", choices=["random", "screened", "existing"], default="random")
    b.add_argument("--partner-correlation", type=float, default=0.0,
                   help="correlation of partners' stable rates (pairing=existing)")
    b.add_argument("--theta-low", type=float, default=0.0)
    b.add_argument("--theta-high", type=float, default=1.0)
    b.add_argument("--power", type=float, default=0.8)

    x = p.parse_args(argv)
    if x.cmd == "per-dyad":
        print(per_dyad_check(x.contrast, x.rate_total, x.duration, x.n_baseline,
                             x.baseline_session, x.dispersion, x.occasion_sd,
                             x.facilitation, x.zeta_star))
    else:
        d = dict(duration=x.duration, n_baseline=x.n_baseline, baseline_session=x.baseline_session,
                 mean_rate=x.mean_rate, between_sd=x.between_sd, occasion_sd=x.occasion_sd,
                 dispersion=x.dispersion, facilitation=x.facilitation, pairing=x.pairing,
                 partner_correlation=x.partner_correlation)
        n = pooled_n(x.power, x.theta_low, x.theta_high, **d)
        print(f"Pooled estimator, theta {x.theta_low:g} vs {x.theta_high:g}, two-sided 5% test")
        print(f"  dyads per group for power {x.power:.2f}: {n}")
        for m in (50, 100, 200, 400):
            print(f"  power at {m:4d} per group: {pooled_power(m, x.theta_low, x.theta_high, **d):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
