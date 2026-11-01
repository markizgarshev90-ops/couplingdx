"""Worked example of the pooled estimator (Supplementary Section S6).

The estimator itself lives in the package (couplingdx.fit_pooled); this script runs
the example quoted in the supplement: 300 simulated dyads at theta = 1.
Run: python pooled_fit.py
"""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "couplingdx"))
import couplingdx as c

rng = random.Random(1)
dyads = []
for _ in range(300):
    bi, bj = rng.gauss(0.0867, 0.013), rng.gauss(0.0867, 0.013)
    si = [rng.gauss(bi, 0.009) for _ in range(4)]
    sj = [rng.gauss(bj, 0.009) for _ in range(4)]
    h = 0.632  # theta = 1
    ki = 740 * ((1 + h) / 2 * bi + (1 - h) / 2 * bj) + rng.gauss(0, 7)
    kj = 740 * ((1 + h) / 2 * bj + (1 - h) / 2 * bi) + rng.gauss(0, 7)
    dyads.append((si, sj, ki, kj))
print(c.fit_pooled(dyads))
