"""Tables and figures for the TSP case study.

usage: python analyze_tsp.py <tsp_main.csv> <tables_dir> <figures_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plotstyle as ps  # noqa: E402
import stats as S  # noqa: E402
from core import CHECKPOINTS  # noqa: E402
from tsp import OPTIMA, TSP_ALGS, TSP_HYBRIDS  # noqa: E402

LABEL = {"ACS+2opt": "ACS+2-opt", "ACS>ILS": "ACS→ILS"}


def lab(a):
    return LABEL.get(a, a)


def main(path, tdir, fdir):
    ps.apply()
    plt = ps.plt
    d = pd.read_csv(path)
    insts = [i for i in OPTIMA if i in d.instance.unique()]
    E = {k: g.sort_values("run").rpd.to_numpy() for k, g in d.groupby(["alg", "instance"])}

    summ = d.groupby(["instance", "alg"]).rpd.agg(["mean", "std", "median", "min"]).reset_index()
    summ["hits"] = d.assign(h=d.rpd <= 1e-9).groupby(["instance", "alg"]).h.sum().to_numpy()
    summ.to_csv(os.path.join(tdir, "tsp_summary.csv"), index=False)

    rows = []
    for h, (cls, parents) in TSP_HYBRIDS.items():
        for inst in insts:
            cand = sorted((np.median(E[(p, inst)]), np.mean(E[(p, inst)]), p) for p in parents)
            bp = cand[0][2]
            xh, xp = E[(h, inst)], E[(bp, inst)]
            rows.append(dict(hybrid=h, cls=cls, instance=inst, n=int(d[d.instance == inst].n.iloc[0]),
                             best_parent=bp, mean_h=xh.mean(), mean_p=xp.mean(), p=S.rank_sum_p(xh, xp),
                             A12=S.a12(xh, xp)))
    r = pd.DataFrame(rows)
    r["p_adj"] = r.groupby("hybrid").p.transform(S.holm)
    r["outcome"] = [S.outcome(p, a) for p, a in zip(r.p_adj, r.A12)]
    r.to_csv(os.path.join(tdir, "tsp_hybrid_vs_best_parent.csv"), index=False)
    facts = {h: dict(W=int((g.outcome == "win").sum()), T=int((g.outcome == "tie").sum()),
                     L=int((g.outcome == "loss").sum())) for h, g in r.groupby("hybrid")}
    json.dump(facts, open(os.path.join(tdir, "tsp_facts.json"), "w"), indent=1)

    # Figure: mean RPD per instance (log scale), one marker per algorithm
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    algs = list(TSP_ALGS)
    x = np.arange(len(insts))
    for k, a in enumerate(algs):
        m = [summ[(summ.instance == i) & (summ.alg == a)]["mean"].iloc[0] for i in insts]
        ax.plot(x + (k - 1.5) * 0.08, m, color=ps.SERIES[k], marker=ps.MARKERS[k], ms=5,
                lw=2.0 if a in TSP_HYBRIDS else 1.4, ls="-" if a in TSP_HYBRIDS else "--", label=lab(a))
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
    ax.set_yticks([0, 0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0", "0.01", "0.1", "1", "10"])
    ax.set_ylim(-0.002, 12)
    ax.set_xticks(x)
    ax.set_xticklabels(insts)
    ax.set_ylabel("Mean RPD (%), symmetric log scale")
    ax.set_xlabel("TSPLIB instance (ordered by size)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ps.save(fig, os.path.join(fdir, "fig_tsp_rpd.png"))

    # Figure: convergence on the two largest instances
    big = insts[-2:]
    fig, axes = plt.subplots(1, len(big), figsize=(6.6, 2.9))
    for ax, inst in zip(np.atleast_1d(axes), big):
        for k, a in enumerate(algs):
            g = d[(d.alg == a) & (d.instance == inst)]
            med = g[[f"cp{c}" for c in CHECKPOINTS]].median().to_numpy()
            ax.plot(np.array(CHECKPOINTS) * 100, med, color=ps.SERIES[k], marker=ps.MARKERS[k], ms=3,
                    lw=2.0 if a in TSP_HYBRIDS else 1.4, ls="-" if a in TSP_HYBRIDS else "--", label=lab(a))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_title(inst, loc="left")
        ax.set_xlabel("CPU-time budget (%)")
    np.atleast_1d(axes)[0].set_ylabel("Median RPD (%)")
    h_, l_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    ps.save(fig, os.path.join(fdir, "fig_tsp_convergence.png"))
    print(r[["hybrid", "instance", "best_parent", "mean_h", "mean_p", "A12", "p_adj", "outcome"]].round(4))
    print(facts)


if __name__ == "__main__":
    main(*sys.argv[1:4])
