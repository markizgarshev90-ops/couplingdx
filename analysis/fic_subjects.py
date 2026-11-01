"""Link the 21 FIC meals to participants and split between-meal variance.

Needs the original FIC pickle (Kyritsis et al., 2021; Zenodo record 4421861),
which carries a subject_id (1-12) for each meal, aligned by index with bite_gt.
The derived workbook was built from bite_gt in pickle order (entry_k = meal k),
so subject_id[k] belongs to entry_k. This script checks that alignment against
the workbook (bite count and meal length per meal) before using it.

Usage:  python fic_subjects.py /path/to/FIC.pkl [--boot 5000]
Writes: data/fic_meals_subjects.csv (subject, session, n_bites, duration_s)
        analysis/out/variance_components.json
"""
import os, sys, json, pickle, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from variance_components import components, bootstrap

ap = argparse.ArgumentParser(); ap.add_argument("pkl"); ap.add_argument("--phi", type=float, default=0.7)
ap.add_argument("--boot", type=int, default=5000); a = ap.parse_args()
class _NumpyOnly(pickle.Unpickler):
    """Load only numpy arrays: a pickle can run code, so refuse any other class."""
    ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy._core.multiarray", "_reconstruct")}
    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


with open(a.pkl, "rb") as fh:
    fic = _NumpyOnly(fh).load()
W = pd.read_csv(os.path.join(ROOT, "data", "fic_meals.csv"))          # from the workbook
rows = []
for k in range(len(fic["bite_gt"])):
    bites = np.asarray(fic["bite_gt"][k], float)
    T = np.ceil(bites[:, 1].max() * 10 - 1e-9) / 10     # workbook: recording start to end of last bite, 0.1-s grid
    starts = bites[:, 0]
    rows.append(dict(meal=k, subject=int(fic["subject_id"][k]), session=int(fic["session_id"][k]),
                     n_bites=int((starts <= T).sum()), duration_s=T))
S = pd.DataFrame(rows)
ok = (len(S) == len(W)) and (S.n_bites.values == W.N.values).all() and np.allclose(S.duration_s, W["T"], atol=0.11)
print(S.to_string(index=False))
if not ok:
    print("\nALIGNMENT CHECK FAILED: pickle meals do not match the workbook sheets one to one.")
    print(pd.DataFrame(dict(pkl_N=S.n_bites, wb_N=W.N, pkl_T=S.duration_s, wb_T=W["T"])).to_string())
    sys.exit(1)
print("\nalignment check passed: bite counts and meal lengths match the workbook for all", len(S), "meals")
S.to_csv(os.path.join(ROOT, "data", "fic_meals_subjects.csv"), index=False)

rws = [(str(r.subject), float(r.n_bites), float(r.duration_s)) for r in S.itertuples()]
est = components(rws, a.phi); ci = bootstrap(rws, a.phi, a.boot)
out = dict(phi=a.phi, boot=a.boot, **{k: est[k] for k in ("subjects", "meals", "within_df")},
           meals_per_subject=S.groupby("subject").size().value_counts().sort_index().to_dict(),
           **{k: dict(est=est[k], ci95=list(ci[k])) for k in ("sigma_b", "sigma_p", "sigma_o", "occasion_share")})
json.dump(out, open(os.path.join(ROOT, "analysis", "out", "variance_components.json"), "w"), indent=1, default=int)
# pseudo-dyads: drop pairs of two meals from the same person
P = pd.read_csv(os.path.join(ROOT, "analysis", "out", "pseudo_dyads.csv"))
subj = dict(zip(S.meal, S.subject))
P["same_person"] = [subj[a_] == subj[b_] for a_, b_ in zip(P.a, P.b)]
Q = P[~P.same_person]
q = lambda x: [float(v) for v in x.quantile([.25, .5, .75])]
pdx = dict(pairs_total=len(P), pairs_same_person=int(P.same_person.sum()), pairs_kept=len(Q),
           Gamma_hat_q=q(Q.Gamma_hat), Gamma_hat_max=float(Q.Gamma_hat.max()),
           gap_q=q(Q.gap), gap_max=float(Q.gap.max()),
           nu_q=q(Q.nu_hat), nu_max=float(Q.nu_hat.max()),
           frac_nu2=float((Q.nu_hat >= 2).mean()), frac_nu3=float((Q.nu_hat >= 3).mean()),
           z0_q=q(Q.z0_hat), z0_max=float(Q.z0_hat.max()),
           same_person_median_Gamma_hat=float(P[P.same_person].Gamma_hat.median()))
out["pseudo_dyads_different_people"] = pdx
json.dump(out, open(os.path.join(ROOT, "analysis", "out", "variance_components.json"), "w"), indent=1, default=int)
print(f"\n{est['subjects']} subjects, {est['meals']} meals, {est['within_df']} within-subject df "
      f"(phi = {a.phi}, {a.boot} bootstrap resamples of subjects)")
for k in ("sigma_b", "sigma_p", "sigma_o", "occasion_share"):
    print(f"  {k:15s} {est[k]:.4f}   95% CI [{ci[k][0]:.4f}, {ci[k][1]:.4f}]")
print("\npseudo-dyads without same-person pairs:")
for k, v in pdx.items():
    print(f"  {k:30s} {v}")
