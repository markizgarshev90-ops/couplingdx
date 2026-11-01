# Dividing by a difference of noisy baselines

Code, derived data and software for *Dividing by a difference of noisy baselines:
why per-dyad estimates of a coupling parameter fail, and how to size a pooled
alternative* (Izgarshev & Kieckens).

```
couplingdx/   the package: design diagnostics and the estimators (no dependencies)
analysis/     scripts reproducing every number, table and figure; outputs in analysis/out/
data/         derived FIC bite timelines, extracted bite times, meal-to-participant table
figures/      Figures 1-4
paper/        supplement.tex (LaTeX source of the Supplementary Material)
```

## The package

```bash
pip install "git+https://github.com/markizgarshev90-ops/couplingdx#subdirectory=couplingdx"
```

See `couplingdx/README.md` for a tutorial.

## Reproducing the paper

Requires `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`. Run from `analysis/`:

```bash
python extract_bites.py          # workbook -> data/fic_bite_times.csv, data/fic_meals.csv
python fic_subjects.py FIC.pkl   # optional, needs the original FIC pickle (see below)
python fic_analysis.py           # dispersion, variance components, pseudo-dyads (Table 2)
python fic_drive_fits.py         # per-meal drive fits on all 21 meals, SE of beta, z0 IQR
python holdout_comparison.py     # holdout comparison of drive specifications
python sim_perdyad.py            # per-dyad simulations (Figure 3)
python sim_pooled.py             # pooled power and size grid (Figure 4), about 10 min
python design_tables.py          # Table 3, and the formula against simulation
python robustness_sim.py         # Supplementary Table S1
python robustness_sim.py --r3-fic21   # R3 row at the 21-meal decay bound
python hc_comparison.py          # plain vs leverage-corrected standard errors (S7, R1)
python pooled_fit.py             # worked example of the estimator (S6)
python revision_sims.py          # S8-S11: facilitation by group, partner-effect test,
                                 #   existing pairs, rank test, time course, pilot size
python revision_tables.py        # rows of Supplementary Tables S2-S6
python make_figures.py           # figures/fig1-4.pdf
```

`fic_subjects.py` links meals to participants. It needs the original FIC pickle
(Zenodo record 4421861), which records a subject identifier for each meal. It loads
only numpy arrays from the pickle, checks that its meals match the workbook one to
one (bite counts and lengths), writes `data/fic_meals_subjects.csv`, splits the
between-meal variance with `variance_components.py`, and summarises the
pseudo-dyads without same-person pairs. Its output is committed, so the other steps
do not need the pickle.

## Data

The derived timelines and the meal-to-participant table are built from the Food
Intake Cycle dataset (Kyritsis, Diou & Delopoulos, 2019; Kyritsis, Tatli, Diou &
Delopoulos, 2017). A meal's length is the time from the start of its recording to
the end of its last bite. The underlying FIC recordings are the property of their
originators and are available from them under their own terms.

## License

MIT for the code. The FIC data are under their originators' terms.
