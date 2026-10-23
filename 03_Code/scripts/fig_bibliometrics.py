"""Fig. biblio (growth and hybrid share) and Fig. pairs (co-occurrence heatmap) from OpenAlex counts."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import plotstyle as ps  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
ps.apply()
plt = ps.plt

years = np.arange(2000, 2026)
allc = np.array([d["all_by_year"].get(str(y), 0) for y in years])
hyb = np.array([d["hybrid_by_year"].get(str(y), 0) for y in years])
share = 100 * hyb / allc

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(years, allc / 1000, color=ps.SERIES[0], marker="o", ms=3, label="All metaheuristic titles")
ax[0].plot(years, hyb / 1000, color=ps.SERIES[1], marker="s", ms=3, label="Titles containing \"hybrid\"")
ax[0].set_ylabel("Publications per year (thousands)")
ax[0].set_xlabel("Year")
ax[0].legend(loc="upper left")
ax[0].set_title("(a) Publication volume", loc="left")
ax[1].plot(years, share, color=ps.SERIES[1], marker="s", ms=3)
ax[1].set_ylabel("Hybrid share of titles (%)")
ax[1].set_xlabel("Year")
ax[1].set_ylim(0, max(share) * 1.25)
for y, off, ha in ((2000, (4, -14), "left"), (2025, (-4, 8), "right")):
    i = y - 2000
    ax[1].annotate(f"{share[i]:.1f}%", (y, share[i]), textcoords="offset points",
                   xytext=off, ha=ha, color=ps.INK, fontsize=8)
ax[1].set_title("(b) Share of hybrid titles", loc="left")
fig.tight_layout(w_pad=2.5)
ps.save(fig, os.path.join(out, "fig_biblio.png"))

keys = ["GA", "PSO", "DE", "ACO", "SA", "TS", "LS", "NM", "ABC", "GWO", "WOA", "CS", "FA", "HS"]
M = np.full((len(keys), len(keys)), np.nan)
for k, v in d["pairs"].items():
    a, b = k.split("|")
    i, j = keys.index(a), keys.index(b)
    M[max(i, j), min(i, j)] = v
M = M[1:, :-1]  # drop the empty first row and last column of the lower triangle
rows, cols = keys[1:], keys[:-1]
cmap = LinearSegmentedColormap.from_list("blues", ps.BLUES)
fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.grid(False)
im = ax.imshow(np.log10(np.where(np.isnan(M), np.nan, M + 1)), cmap=cmap)
for i in range(len(rows)):
    for j in range(len(cols)):
        if not np.isnan(M[i, j]):
            lv = np.log10(M[i, j] + 1) / np.nanmax(np.log10(M + 1))
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=6.5,
                    color="white" if lv > 0.6 else ps.INK)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(rows)))
ax.set_xticklabels(cols, rotation=90)
ax.set_yticklabels(rows)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("log10(publications + 1)")
ps.save(fig, os.path.join(out, "fig_pairs.png"))

order = sorted(d["pairs"].items(), key=lambda kv: -kv[1])
summary = dict(date=d["query_date"], all_2000=int(allc[0]), all_2025=int(allc[-1]), h_2000=int(hyb[0]),
               h_2025=int(hyb[-1]), share_2000=round(share[0], 1), share_2025=round(share[-1], 1),
               growth=round(hyb[-1] / hyb[0], 1), total_h=int(hyb.sum()), top_pairs=order[:6])
json.dump(summary, open(os.path.join(os.path.dirname(src), "summary.json"), "w"), indent=1)
print(summary)
