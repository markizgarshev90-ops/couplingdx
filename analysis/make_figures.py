"""Figures 1-4 of the paper.

Run after fic_analysis.py, sim_perdyad.py and sim_pooled.py:
    python make_figures.py          -> ../figures/fig1.pdf ... fig4.pdf
"""
import os, sys, json, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq, minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "couplingdx"))
import couplingdx as c

OUT = os.path.join(ROOT, "analysis", "out")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
F = json.load(open(os.path.join(OUT, "fic_numbers.json")))

# ---- style: Computer Modern to match the LaTeX text; recessive axes ----
INK, INK2, MUTED, BAND = "#1a1a19", "#52514e", "#8a8984", "#e6e5e1"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": True,
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.labelcolor": INK, "text.color": INK, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})
W = 6.5                                    # full text width of the manuscript, inches
hv = np.vectorize(c.h)


def panel(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left", family="DejaVu Sans")


# ======================= Figure 1: observable and ceiling =======================
def spread_h(theta, q):
    return c.h(theta) - c.h(q * theta)


def h_optimum(q):
    r = minimize_scalar(lambda t: -spread_h(t, q), bounds=(1e-4, 20), method="bounded",
                        options=dict(xatol=1e-10))
    return r.x, -r.fun


q7 = 7.0
th_star, S_star = h_optimum(q7)
half = S_star / 2
band = (brentq(lambda t: spread_h(t, q7) - half, 1e-4, th_star),
        brentq(lambda t: spread_h(t, q7) - half, th_star, 50))

fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.32))
a = ax[0]
t = np.linspace(-1, 8, 900)
a.axvspan(*band, color=BAND, lw=0, zorder=0)
a.axvline(0, color=MUTED, lw=0.7, ls=":", zorder=1)
a.axhline(1, color=MUTED, lw=0.7, ls=":", zorder=1)
a.plot(t, hv(t), color=INK, lw=1.8, zorder=2)
lo_pt, hi_pt = (th_star, c.h(th_star)), (q7 * th_star, c.h(q7 * th_star))
a.plot(*lo_pt, "o", ms=6.5, color=C2, mec="white", mew=1.2, zorder=4)
a.plot(*hi_pt, "s", ms=6.5, color=C2, mec="white", mew=1.2, zorder=4)
a.annotate("", xy=(hi_pt[0], hi_pt[1]), xytext=(hi_pt[0], lo_pt[1]),
           arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9, shrinkA=0, shrinkB=4))
a.plot([lo_pt[0], hi_pt[0]], [lo_pt[1], lo_pt[1]], color=INK2, lw=0.6, ls=(0, (2, 2)))
a.text(hi_pt[0] + 0.22, 0.5 * (lo_pt[1] + hi_pt[1]), f"spread {S_star:.2f}", fontsize=8, va="center")
a.text(lo_pt[0] + 0.2, lo_pt[1] + 0.05, "bottom of scale", fontsize=7.5, color=INK2)
a.text(hi_pt[0] + 0.3, hi_pt[1] + 0.06, "top of scale", fontsize=7.5, color=INK2, ha="left")
a.annotate(r"divergence ($h>1$)", xy=(-0.55, c.h(-0.55)), xytext=(0.35, 1.52), fontsize=7.5, color=INK2,
           va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=2))
a.text(0.5 * (band[0] + band[1]), 0.05, "at least half the\nmaximal spread", fontsize=7, color=INK2,
       ha="center", va="bottom")
a.set_xlim(-1, 8); a.set_ylim(0, 1.7)
a.set_xlabel(r"$\theta=2\kappa T$"); a.set_ylabel(r"contraction factor $h(\theta)$")
panel(a, "A")

b = ax[1]
s0 = np.linspace(0, 8, 161); q = (s0 + 7) / (s0 + 1)
b.plot(s0, (np.sqrt(q) - 1) / (np.sqrt(q) + 1), color=C1, lw=1.8)
Sh = np.array([h_optimum(qq)[1] for qq in q])
b.plot(s0, Sh, color=C2, lw=1.8, ls=(0, (5, 2)))
for x in (0, 1, 2, 5):
    qq = (x + 7) / (x + 1)
    b.plot([x], [(math.sqrt(qq) - 1) / (math.sqrt(qq) + 1)], "o", ms=5.5, color=C1, mec="white", mew=1)
    b.plot([x], [h_optimum(qq)[1]], "s", ms=5.5, color=C2, mec="white", mew=1)
b.text(8, Sh[-1] + 0.025, r"slow-drive $h$", fontsize=7.5, ha="right", va="bottom")
b.text(8, (math.sqrt(q[-1]) - 1) / (math.sqrt(q[-1]) + 1) - 0.025, r"asymptotic $\rho$", fontsize=7.5,
       ha="right", va="top")
b.set_xlim(-0.2, 8.2); b.set_ylim(0, 0.6)
b.set_xlabel(r"baseline coupling $s_0$ (scale points)")
b.set_ylabel(r"maximum attainable spread $\Sigma^{*}$")
b.legend(handles=[Line2D([], [], color=C1, marker="o", mec="white", label=r"asymptotic $\rho$"),
                  Line2D([], [], color=C2, ls=(0, (5, 2)), marker="s", mec="white", label=r"slow-drive $h$")],
         loc="upper right")
panel(b, "B")
fig.savefig(os.path.join(FIG, "fig1.pdf")); plt.close(fig)
print(f"fig1: theta*={th_star:.3f}, spread={S_star:.3f}, half-spread band=[{band[0]:.3f}, {band[1]:.3f}]")

# ================= Figure 2: dispersion and pseudo-dyads (FIC) =================
B = pd.read_csv(os.path.join(ROOT, "data", "fic_bite_times.csv"))
M = pd.read_csv(os.path.join(ROOT, "data", "fic_meals.csv"))
P = pd.read_csv(os.path.join(OUT, "pseudo_dyads.csv"))
P = P[~P.same_person]                       # pairs of two meals from different people
times = [np.sort(B.t[B.meal == i].to_numpy()) for i in range(21)]
T = M["T"].to_numpy(); N = M["N"].to_numpy().astype(float)
PHI = F["PHI"]

fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.32))
a = ax[0]
ws = [20, 30, 45, 60, 90, 120, 180]
pooled = []
jit = np.random.default_rng(3)
for w in ws:
    vals, wts = [], []
    for i in range(21):
        nw = int(T[i] // w)
        cnt = np.histogram(times[i], bins=np.arange(0, nw * w + 1e-9, w))[0]
        vals.append(cnt.var(ddof=1) / cnt.mean()); wts.append(nw - 1)
    pooled.append(np.average(vals, weights=wts))
    a.scatter(w * np.exp(0.07 * jit.standard_normal(21)), vals, s=9, color=MUTED, alpha=0.55, lw=0, zorder=2)
a.axhline(1, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
a.axhline(F["phi_cv2"], color=C2, lw=1.2, ls=(0, (1, 1.5)), zorder=1)
a.plot(ws, pooled, "o-", color=C1, ms=5.5, mec="white", mew=1, zorder=3)
a.set_xscale("log"); a.set_xticks(ws); a.set_xticklabels([str(w) for w in ws])
a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
a.set_xlim(16, 225); a.set_ylim(0, 2.85)
a.set_xlabel("counting window (s)"); a.set_ylabel("variance / mean of bite counts")
a.legend(handles=[Line2D([], [], color=C1, marker="o", mec="white", label="pooled across meals"),
                  Line2D([], [], color=MUTED, marker="o", ls="none", ms=4, label="one meal"),
                  Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2)), label="Poisson"),
                  Line2D([], [], color=C2, lw=1.2, ls=(0, (1, 1.5)), label=r"CV$^2$ of intervals")],
         loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.8, fontsize=7)
panel(a, "A")

# reference: meals that differ only by sampling noise (common rate, same lengths, phi)
rng = np.random.default_rng(17)
mu = N.sum() / T.sum()
iu = np.triu_indices(21, 1)
keep = ~pd.read_csv(os.path.join(OUT, "pseudo_dyads.csv")).same_person.to_numpy()   # same pair order
ref = []
for _ in range(4000):
    Nk = np.maximum(np.rint(mu * T + np.sqrt(PHI * mu * T) * rng.standard_normal(21)), 0)
    r = Nk / T
    G = np.abs(r[:, None] - r[None, :])[iu]; S = (r[:, None] + r[None, :])[iu]
    Tm = 0.5 * (T[:, None] + T[None, :])[iu]
    ref.append((G * Tm / np.sqrt(PHI * S * Tm))[keep])
ref = np.concatenate(ref)
b = ax[1]
bins = np.arange(0, 6.01, 0.25)
b.hist(P.z0_hat, bins=bins, color=C1, edgecolor="white", lw=0.8, zorder=2)
dens, _ = np.histogram(ref, bins=bins, density=True)
b.step(np.r_[bins[:-1], bins[-1]], np.r_[dens, dens[-1]] * len(P) * 0.25, where="post",
       color=INK, lw=1.1, ls=(0, (4, 2)), zorder=3)
b.axvline(c.ZETA_STAR, color=C2, lw=1.6, zorder=3)
b.text(c.ZETA_STAR - 0.12, 20, "per-dyad requirement\n" r"$\zeta^{*}=5.2$", fontsize=7.5,
       ha="right", va="center")
b.set_xlim(0, 6); b.set_ylim(0, 44)
b.set_xlabel(r"gap signal-to-noise ratio $\hat z_0$"); b.set_ylabel("pseudo-dyads")
b.legend(handles=[matplotlib.patches.Patch(color=C1, label=f"{len(P)} FIC pseudo-dyads"),
                  Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 2)), label="sampling noise only")],
         loc="upper right", bbox_to_anchor=(0.84, 1.0))
panel(b, "B")
fig.savefig(os.path.join(FIG, "fig2.pdf")); plt.close(fig)
print(f"fig2: pooled Fano {dict(zip(ws, np.round(pooled, 3)))}; reference median z0 {np.median(ref):.2f}, "
      f"observed median {P.z0_hat.median():.2f}, max {P.z0_hat.max():.2f}")

# ===================== Figure 3: per-dyad estimator (simulation) =====================
A = pd.read_csv(os.path.join(OUT, "perdyad_grid.csv"))
hr = c.h(1.0)
A["zeta"] = 1 / np.sqrt(1 / A.z0 ** 2 + hr ** 2 / A.nu ** 2)
fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw=dict(wspace=0.32))
a = ax[0]
a1 = A[(A.theta == 1) & np.isfinite(A.iqr)]
zz = np.linspace(0.9, 12, 300)
a.plot(zz, c.ZETA_STAR / zz, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
a.axhline(1, color=MUTED, lw=0.7, ls=":"); a.axvline(c.ZETA_STAR, color=MUTED, lw=0.7, ls=":")
groups = [(a1.m <= 2, C2, "s", r"$m=1$ or $2$"), (a1.m.isin([4, 8]), C1, "o", r"$m=4$ or $8$"),
          (a1.m >= 16, C3, "^", r"$m=16$ or $64$")]
for sel, col, mk, lab in groups:
    a.scatter(a1.zeta[sel], a1.iqr[sel], s=24, color=col, marker=mk, edgecolor="white", lw=0.6, zorder=3, label=lab)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlim(0.9, 12); a.set_ylim(0.35, 20)
a.set_xticks([1, 2, 5, 10]); a.set_xticklabels(["1", "2", "5", "10"])
a.set_yticks([0.5, 1, 2, 5, 10]); a.set_yticklabels(["0.5", "1", "2", "5", "10"])
a.minorticks_off()
a.text(1.22, 2.0, r"$5.2/\zeta$", fontsize=8, ha="left", va="center")
a.text(1.0, 1.08, r"IQR $=\theta$", fontsize=7.5, color=INK2)
a.set_xlabel(r"combined signal-to-noise ratio $\zeta$")
a.set_ylabel(r"IQR of $\hat\theta$ (true $\theta=1$)")
a.legend(loc="upper right", handletextpad=0.3)
panel(a, "A")

b = ax[1]
s = A[A.m == 8].sort_values("z0")
null, alt = s[s.theta == 0], s[s.theta == 1]
b.axhline(0, color=MUTED, lw=0.7)
b.axhline(1, color=MUTED, lw=0.7, ls=":")
b.plot(alt.z0, alt.old_median_theta, "^-", color=C3, ms=5.5, mec="white", mew=0.8,
       label=r"$\theta=1$, keep $0<\hat h<1$")
b.plot(null.z0, null.old_median_theta, "s-", color=C2, ms=5.5, mec="white", mew=0.8,
       label=r"$\kappa=0$, keep $0<\hat h<1$")
b.plot(null.z0, null.median_theta, "o-", color=C1, ms=5.5, mec="white", mew=0.8,
       label=r"$\kappa=0$, all estimates")
b.set_xscale("log"); b.set_xticks([1, 2, 4, 8]); b.set_xticklabels(["1", "2", "4", "8"]); b.minorticks_off()
b.set_ylim(-0.25, 1.9)
b.set_xlabel(r"gap signal-to-noise ratio $z_0$ ($m=8$)"); b.set_ylabel(r"median $\hat\theta$")
b.legend(loc="upper right")
panel(b, "B")
fig.savefig(os.path.join(FIG, "fig3.pdf")); plt.close(fig)
print(f"fig3: {len(a1)} cells with bounded IQR plotted of {(A.theta == 1).sum()}")

# ========================= Figure 4: pooled estimator power =========================
G = pd.read_csv(os.path.join(OUT, "pooled_grid.csv"))
g = G[(G.theta_high == 1) & (G["T"] == 740) & (G.m == 4)]
MU, SB = F["rate_mean"], F["sd_b"]          # the values the simulation used
fig, a = plt.subplots(figsize=(4.3, 3.0))
nn = np.arange(10, 451)
handles = []
for pairing, col, mk in (("random", C1, "o"), ("screened", C2, "s")):
    for share, ls in ((0.0, "-"), (0.5, (0, (5, 2)))):
        sp, so = math.sqrt(1 - share) * SB, math.sqrt(share) * SB
        d = dict(duration=740, n_baseline=4, baseline_session=740, mean_rate=MU, between_sd=sp,
                 occasion_sd=so, dispersion=PHI, pairing=pairing)
        a.plot(nn, [c.pooled_power(int(n), 0, 1, **d) for n in nn], color=col, ls=ls, lw=1.6, zorder=2)
        sel = g[(g.pairing == pairing) & (g.occ_share == share)]
        a.plot(sel.n, sel.mom_reject, mk, color=col, ms=6, mec="white", mew=1, zorder=3)
        handles.append(Line2D([], [], color=col, ls=ls, marker=mk, ms=5, mec="white",
                              label=f"{pairing}, " + ("all stable" if share == 0 else "half occasional")))
a.axhline(0.8, color=MUTED, lw=0.8, ls=":", zorder=1)
a.set_xlim(0, 450); a.set_ylim(0, 1.02)
a.set_xticks([0, 50, 100, 200, 300, 400])
a.set_xlabel("dyads per group"); a.set_ylabel(r"power, $\theta=0$ vs $\theta=1$")
a.legend(handles=handles, loc="lower right", title="pairing, variance components", title_fontsize=7.5)
fig.savefig(os.path.join(FIG, "fig4.pdf")); plt.close(fig)
d = (g.mom_reject - g.analytic_power).abs()
print(f"fig4: {len(g)} simulated points, max |sim - formula| = {d.max():.3f}")
