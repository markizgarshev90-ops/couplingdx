"""couplingdx: design diagnostics and the pooled estimator for two-person coupling studies."""
from .core import (h, inv_h, ceiling, counting_sd, sign_error_probability,
                   per_dyad_precision, required_duration, per_dyad_check, PerDyadReport,
                   expected_contrast, expected_squared_contrast, pooled_variance,
                   pooled_power, pooled_n, ZETA_STAR, Z_975)
from .fit import (PooledFit, dyad_terms, fit_pooled, compare_groups,
                  PartnerFit, fit_partner_effect)

__version__ = "0.3.0"
__all__ = ["h", "inv_h", "ceiling", "counting_sd", "sign_error_probability",
           "per_dyad_precision", "required_duration", "per_dyad_check", "PerDyadReport",
           "expected_contrast", "expected_squared_contrast", "pooled_variance",
           "pooled_power", "pooled_n", "ZETA_STAR", "Z_975",
           "PooledFit", "dyad_terms", "fit_pooled", "compare_groups",
           "PartnerFit", "fit_partner_effect"]
