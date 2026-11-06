<p align="center">
  <img src="assets/logo.svg" alt="couplingdx logo" width="560">
</p>

<p align="center">
  <a href="LICENSE.txt"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8%2B-blue.svg">
  <img alt="Dependencies: none" src="https://img.shields.io/badge/dependencies-none-brightgreen.svg">
  <img alt="Version 0.3.0" src="https://img.shields.io/badge/version-0.3.0-informational.svg">
</p>

<p align="center">
  <a href="#what-the-tool-does">What it does</a> &middot;
  <a href="#how-researchers-can-use-it">For researchers</a> &middot;
  <a href="#install">Install</a> &middot;
  <a href="#reproducing-the-paper">Reproduce the paper</a> &middot;
  <a href="#citing">Cite</a>
</p>

# Dividing by a difference of noisy baselines

Code, derived data and software for *Dividing by a difference of noisy baselines:
why per-dyad estimates of a coupling parameter fail, and how to size a pooled
alternative* (Izgarshev & Kieckens).

## Install

```bash
pip install "git+https://github.com/markizgarshev90-ops/couplingdx#subdirectory=couplingdx"
```

## What the tool does

`couplingdx` answers two questions that decide whether a study of mimicry,
synchrony or convergence between two people can work at all.

1. **Can coupling be estimated for a single pair?** A per-pair estimate divides by
   the difference between two noisy solo baselines. When that difference is small
   compared with the noise, the estimate is less precise than the quantity it
   targets. The package gives a single number, `zeta`, and a feasibility verdict
   (`zeta >= 5.2`).
2. **How many pairs does a pooled estimate need?** Pooling across pairs never
   divides by one pair's difference. The package computes the number of pairs per
   group for a chosen power, and fits the pooled estimator once data are in.

It has no dependencies, a command-line interface, and a Python API.

```bash
python -m couplingdx per-dyad --contrast 0.0124 --rate-total 0.173 --duration 740 \
      --n-baseline 8 --baseline-session 740 --dispersion 0.7
python -m couplingdx pooled --mean-rate 0.0867 --between-sd 0.013 --duration 740 \
      --n-baseline 4 --baseline-session 740 --dispersion 0.7
```

## Repository layout


```
couplingdx/   the package: design diagnostics and the estimators (no dependencies)
analysis/     scripts reproducing every number, table and figure; outputs in analysis/out/
data/         derived FIC bite timelines, extracted bite times, meal-to-participant table
figures/      Figures 1-4 as printed in the manuscript
paper/        supplement.tex (LaTeX source of the Supplementary Material)
```


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
python make_figures.py           # figures/fig1-4.pdf (same data; earlier panel styling)
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
