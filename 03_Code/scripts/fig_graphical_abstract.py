"""Graphical abstract for Swarm and Evolutionary Computation (531 x 1328 px or proportionally more; 5 x 13 cm).

usage: python fig_graphical_abstract.py <tables_dir> <figures_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd  # noqa: E402

import plotstyle as ps  # noqa: E402

LABEL = {"DE>NM": "DE→NM", "LSHADE>CMAES": "L-SHADE→CMA-ES"}
T, OUT = sys.argv[1], sys.argv[2]
facts = json.load(open(os.path.join(T, "facts.json")))
summ = pd.read_csv(os.path.join(T, "hybrid_summary.csv"), index_col=0).sort_values("NHG")
by_dim = facts["pooled_by_dim"]
p = facts["pooled"]

ps.apply()
plt = ps.plt
plt.rcParams["savefig.bbox"] = "standard"  # keep the exact 531:1328 aspect ratio
CM = 1 / 2.54
fig = plt.figure(figsize=(13 * CM, 13 * CM * 531 / 1328))
fs = 6.5

# ---------------------------------------------------------------- (1) the question and the set-up
ax = fig.add_axes([0.025, 0.12, 0.27, 0.66])
ax.axis("off")
ax.text(0, 1.27, "Does hybridization pay off?", fontsize=9, fontweight="bold", transform=ax.transAxes, va="top")
setup = ("9 hybrids (all 4 Talbi classes)\nvs. their own best parent\n\n"
         f"CEC 2017 · D = 10, 30, 50 · 51 runs\n{int(p['N'])} comparisons, equal budgets\n\n"
         "+ TSP case study (ACS, ILS)")
ax.text(0, 0.88, setup, fontsize=fs, va="top", transform=ax.transAxes, linespacing=1.35, color=ps.INK)

# ---------------------------------------------------------------- (2) pooled outcome: not by default
ax = fig.add_axes([0.32, 0.12, 0.20, 0.66])
shares = [("Win", p["HAR"], ps.SERIES[0], None), ("Tie", p["T"] / p["N"], "#c3c2bd", None),
          ("Loss", p["HLR"], ps.SERIES[1], "////")]
left = 0.0
for name, s, col, hatch in shares:
    ax.barh(0, s, left=left, height=0.55, color=col, hatch=hatch, edgecolor="white", linewidth=1.2)
    ax.text(left + s / 2, -0.52, f"{name}\n{100 * s:.0f}%", ha="center", va="top", fontsize=fs, color=ps.INK)
    left += s
ax.set_xlim(0, 1)
ax.set_ylim(-1.75, 0.45)
ax.axis("off")
ax.text(0, 1.08, "Not by default", transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
nhg_d = " → ".join(f"{by_dim[str(d)]['NHG']:+.2f}".replace("-", "−") for d in (10, 30, 50))
ax.text(0, 0.24, f"Net gain NHG = {p['NHG']:+.2f}".replace("-", "−"), transform=ax.transAxes, fontsize=fs,
        fontweight="bold", color=ps.INK, va="top")
ax.text(0, 0.04, f"by D = 10 → 30 → 50:\n{nhg_d}", transform=ax.transAxes, fontsize=fs - 0.5, va="top",
        color=ps.INK2, linespacing=1.3)

# ---------------------------------------------------------------- (3) per design: repair a weakness
ax = fig.add_axes([0.695, 0.17, 0.285, 0.60])
y = range(len(summ))
cols = [ps.DIVERGING_POS if v > 0 else ps.DIVERGING_NEG for v in summ.NHG]
ax.barh(list(y), summ.NHG, color=cols, height=0.62, edgecolor="white", linewidth=0.8)
ax.set_yticks(list(y))
ax.set_yticklabels([LABEL.get(h, h) for h in summ.index], fontsize=5.6)
ax.axvline(0, color=ps.INK2, lw=0.6)
ax.set_xlim(-0.75, 0.75)
ax.set_xticks([-0.5, 0, 0.5])
ax.set_xticklabels(["−0.5", "0", "+0.5"], fontsize=5.6)
ax.tick_params(length=0, pad=1.5)
ax.grid(False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("NHG vs. best parent", fontsize=fs, labelpad=1)
for h in ("DE>NM", "MA-DE", "GA-PSO"):
    v = summ.NHG[h]
    inside = v < 0  # negative labels sit inside the bar so that they do not collide with the axis labels
    ax.text(v + 0.03 if v > 0 else v + 0.03, list(summ.index).index(h), f"{v:+.2f}".replace("-", "−"),
            va="center", ha="left", fontsize=5.6, color="white" if inside else ps.INK,
            fontweight="bold" if inside else "normal")
fig.text(0.565, 0.935, "Pays off only when a component\nrepairs a weakness of the other", fontsize=7.5,
         fontweight="bold", va="top", color=ps.INK)

fig.savefig(os.path.join(OUT, "graphical_abstract.png"), dpi=600, facecolor="white")
fig.savefig(os.path.join(OUT, "graphical_abstract.pdf"), facecolor="white")
print("saved graphical abstract")
