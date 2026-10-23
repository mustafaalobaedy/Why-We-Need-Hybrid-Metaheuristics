"""Tables and figures for the main CEC2017 experiment.

usage: python analyze.py <results.csv> <tables_dir> <figures_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import plotstyle as ps  # noqa: E402
import stats as S  # noqa: E402
from algorithms import ALGORITHMS  # noqa: E402
from core import CHECKPOINTS  # noqa: E402

LABEL = {"DE>NM": "DE→NM", "LSHADE>CMAES": "L-SHADE→CMA-ES"}
CATS = ["Unimodal", "Multimodal", "Hybrid", "Composition"]
CLASSES = ["LRH", "LTH", "HRH", "HTH"]
HYBRIDS = [a for a, v in ALGORITHMS.items() if v[1] == "hybrid"]
SINGLES = [a for a, v in ALGORITHMS.items() if v[1] == "single"]


def lab(a):
    return LABEL.get(a, a)


def wtl(df):
    n = len(df)
    w, l_ = (df.outcome == "win").sum(), (df.outcome == "loss").sum()
    return pd.Series(dict(W=w, T=n - w - l_, L=l_, N=n, HAR=w / n, HLR=l_ / n, NHG=(w - l_) / n,
                          A12_median=df.A12.median()))


def complete(d, runs):
    c = d.groupby(["alg", "fid", "dim"]).size()
    ok = c[c == runs].reset_index()[["alg", "fid", "dim"]]
    return d.merge(ok)


def main(path, tdir, fdir, runs=51):
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    ps.apply()
    plt = ps.plt
    d = complete(S.load(path), runs)
    dims = sorted(d.dim.unique())
    facts = {}

    # ---------------------------------------------------------- RQ1: hybrid vs best parent
    hp = S.hybrid_vs_parents(d)
    hp.to_csv(os.path.join(tdir, "hybrid_vs_best_parent_detail.csv"), index=False)
    per_h = hp.groupby("hybrid").apply(wtl).reindex(HYBRIDS).dropna()
    per_h["class"] = [ALGORITHMS[h][2] for h in per_h.index]
    per_h = per_h.sort_values("NHG", ascending=False)
    per_h.to_csv(os.path.join(tdir, "hybrid_summary.csv"))
    pooled = wtl(hp)
    facts["pooled"] = pooled.round(4).to_dict()
    facts["per_hybrid"] = per_h.round(4).reset_index().to_dict(orient="records")

    # ---------------------------------------------------------- RQ2: category and dimension
    by_cat = hp.groupby(["hybrid", "category"]).apply(wtl).NHG.unstack().reindex(index=per_h.index, columns=CATS)
    by_dim = hp.groupby(["hybrid", "dim"]).apply(wtl).NHG.unstack().reindex(index=per_h.index)
    by_cat.to_csv(os.path.join(tdir, "nhg_by_category.csv"))
    by_dim.to_csv(os.path.join(tdir, "nhg_by_dim.csv"))
    facts["pooled_by_cat"] = hp.groupby("category").apply(wtl).round(4).to_dict(orient="index")
    facts["pooled_by_dim"] = hp.groupby("dim").apply(wtl).round(4).to_dict(orient="index")
    facts["pooled_by_class"] = hp.groupby("cls").apply(wtl).round(4).to_dict(orient="index")

    # ---------------------------------------------------------- RQ2: budget (checkpoints)
    rows = []
    for c in CHECKPOINTS:
        col = f"cp{c}"
        r = S.hybrid_vs_parents(d, col=col)
        for key, g in [("All hybrids", r)] + [(k, g) for k, g in r.groupby("cls")]:
            s = wtl(g)
            rows.append(dict(budget=c, group=key, HAR=s.HAR, HLR=s.HLR, NHG=s.NHG))
        for h, g in r.groupby("hybrid"):
            s = wtl(g)
            rows.append(dict(budget=c, group=h, HAR=s.HAR, HLR=s.HLR, NHG=s.NHG))
    bud = pd.DataFrame(rows)
    bud.to_csv(os.path.join(tdir, "nhg_by_budget.csv"), index=False)

    # ---------------------------------------------------------- Friedman ranks
    fr_rows, ph_rows = [], []
    for dim in dims + [None]:
        ranks, stat, p, n = S.friedman(d, dim)
        ph, best = S.friedman_posthoc_holm(ranks, n)
        tag = "All" if dim is None else f"D={dim}"
        for a, r in ranks.items():
            fr_rows.append(dict(setting=tag, alg=a, rank=r))
        ph["setting"], ph["control"], ph["chi2"], ph["p_friedman"] = tag, best, stat, p
        ph_rows.append(ph.reset_index().rename(columns={"index": "alg"}))
    fr = pd.DataFrame(fr_rows).pivot(index="alg", columns="setting", values="rank")
    fr = fr[[f"D={x}" for x in dims] + ["All"]].sort_values("All")
    fr.to_csv(os.path.join(tdir, "friedman_ranks.csv"))
    pd.concat(ph_rows).to_csv(os.path.join(tdir, "friedman_posthoc_holm.csv"), index=False)
    facts["friedman_all"] = fr["All"].round(2).to_dict()

    # ---------------------------------------------------------- hybrids vs the strongest single
    vs = []
    for h in HYBRIDS:
        for ref in ("L-SHADE", "CMA-ES"):
            r = S.pairwise(d, h, ref)
            s = wtl(r)
            vs.append(dict(hybrid=h, reference=ref, W=s["W"], T=s["T"], L=s["L"], NHG=s["NHG"]))
    vs = pd.DataFrame(vs)
    vs.to_csv(os.path.join(tdir, "hybrid_vs_state_of_art.csv"), index=False)

    # ---------------------------------------------------------- complementarity of the parents
    E = d.groupby(["alg", "fid", "dim"]).error.median()
    comp = []
    for h in HYBRIDS:
        parents = ALGORITHMS[h][3]
        best = []
        for (fid, dim), _ in d.groupby(["fid", "dim"]):
            meds = {p_: E[(p_, fid, dim)] for p_ in parents}
            best.append(min(meds, key=meds.get))
        share = pd.Series(best).value_counts(normalize=True)
        comp.append(dict(hybrid=h, complementarity=1 - share.max(), NHG=per_h.loc[h, "NHG"],
                         dominant_parent=share.idxmax()))
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(tdir, "parent_complementarity.csv"), index=False)
    facts["complementarity_corr"] = float(np.corrcoef(comp.complementarity, comp.NHG)[0, 1])

    # ---------------------------------------------------------- cost: run time and parameters
    t = d.groupby("alg").time.mean()
    cost = pd.DataFrame([dict(hybrid=h, time_s=t[h], parent_time_s=np.mean([t[p_] for p_ in ALGORITHMS[h][3]]))
                         for h in HYBRIDS])
    cost["time_ratio"] = cost.time_s / cost.parent_time_s
    cost.to_csv(os.path.join(tdir, "runtime.csv"), index=False)

    # ---------------------------------------------------------- mean/std tables (supplementary)
    ms = d.groupby(["dim", "fid", "alg"]).error.agg(["mean", "std", "median"]).reset_index()
    ms.to_csv(os.path.join(tdir, "mean_std_all.csv"), index=False)

    json.dump(facts, open(os.path.join(tdir, "facts.json"), "w"), indent=1, default=float)

    # ========================================================== figures
    # W/T/L bars
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    y = np.arange(len(per_h))[::-1]
    W, T, L = (per_h[c] / per_h["N"] * 100 for c in ("W", "T", "L"))
    ax.barh(y, W, color=ps.DIVERGING_POS, height=0.62, label="Win", edgecolor="white", linewidth=1)
    ax.barh(y, T, left=W, color="#c9c8c3", height=0.62, label="Tie", edgecolor="white", linewidth=1)
    ax.barh(y, L, left=W + T, color=ps.DIVERGING_NEG, height=0.62, label="Loss", edgecolor="white", linewidth=1,
            hatch="////")
    for yi, (w_, t_, l_, nhg) in zip(y, zip(W, T, L, per_h.NHG)):
        if w_ >= 7:
            ax.text(w_ / 2, yi, f"{w_:.0f}", ha="center", va="center", color="white", fontsize=7.5)
        if l_ >= 7:
            ax.text(w_ + t_ + l_ / 2, yi, f"{l_:.0f}", ha="center", va="center", color=ps.INK, fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax.text(102, yi, f"{nhg:+.2f}", ha="left", va="center", color=ps.INK, fontsize=8)
    ax.text(102, len(per_h) - 0.35, "NHG", ha="left", va="bottom", fontsize=8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab(h)} ({per_h.loc[h, 'class']})" for h in per_h.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Share of the {int(per_h['N'].iloc[0])} function–dimension cases (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    ps.save(fig, os.path.join(fdir, "fig_wtl.png"))

    # NHG heatmaps (category and dimension)
    cmap = LinearSegmentedColormap.from_list("div", [ps.DIVERGING_NEG, ps.DIVERGING_MID, ps.DIVERGING_POS])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw=dict(width_ratios=[4, len(dims)]))
    for ax, M, xl in [(axes[0], by_cat, CATS), (axes[1], by_dim, [f"D = {x}" for x in by_dim.columns])]:
        im = ax.imshow(M.to_numpy(float), cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.iat[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if abs(v) > 0.6 else ps.INK)
        ax.set_xticks(range(len(xl)))
        ax.set_xticklabels(xl, rotation=30, ha="right")
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels([lab(h) for h in M.index] if ax is axes[0] else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_title("(a) By function group", loc="left")
    axes[1].set_title("(b) By dimension", loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Net hybridization gain (NHG)")
    ps.save(fig, os.path.join(fdir, "fig_nhg_heatmap.png"))

    # NHG versus budget, by hybridization class
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    groups = ["All hybrids"] + [c for c in CLASSES if c in bud.group.unique()]
    colors = [ps.INK] + ps.SERIES[:4]
    for k, (g, col) in enumerate(zip(groups, colors)):
        s = bud[bud.group == g].sort_values("budget")
        ax.plot(s.budget * 100, s.NHG, color=col, marker=(["o"] + ps.MARKERS)[k], ms=4,
                lw=2.4 if g == "All hybrids" else 1.6, ls="-" if g == "All hybrids" else "--", label=g)
    ax.axhline(0, color=ps.INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Consumed budget (% of MaxFEs, log scale)")
    ax.set_ylabel("Net hybridization gain (NHG)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ps.save(fig, os.path.join(fdir, "fig_nhg_budget.png"))

    # Friedman ranks
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2, 3.4), sharey=False)
    for ax, dim in zip(np.atleast_1d(axes), dims):
        s = fr[f"D={dim}"].sort_values(ascending=False)
        cols = [ps.SERIES[1] if ALGORITHMS[a][1] == "hybrid" else ps.SERIES[0] for a in s.index]
        ax.barh(range(len(s)), s.to_numpy(), color=cols, height=0.65, edgecolor="white", linewidth=1)
        for i, v in enumerate(s.to_numpy()):
            ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=6.5, color=ps.INK)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([lab(a) for a in s.index], fontsize=7)
        ax.set_title(f"D = {dim}", loc="left")
        ax.set_xlim(0, len(ALGORITHMS) + 1.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Average Friedman rank")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ps.SERIES[0], label="Single algorithm"),
                        Patch(color=ps.SERIES[1], label="Hybrid")], ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.2)
    ps.save(fig, os.path.join(fdir, "fig_friedman.png"))

    # Complementarity scatter
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for k, cls in enumerate(CLASSES):
        s = comp[[ALGORITHMS[h][2] == cls for h in comp.hybrid]]
        ax.scatter(s.complementarity, s.NHG, s=44, color=ps.SERIES[k], marker=ps.MARKERS[k], label=cls,
                   edgecolor="white", linewidth=1, zorder=3)
    from adjustText import adjust_text
    pts = comp.assign(cx=comp.complementarity.round(3), cy=comp.NHG.round(3))
    texts = [ax.text(cx, cy, "\n".join(lab(h) for h in g.hybrid), fontsize=7, color=ps.INK)
             for (cx, cy), g in pts.groupby(["cx", "cy"])]
    adjust_text(texts, ax=ax, expand=(1.3, 1.6), arrowprops=dict(arrowstyle="-", color=ps.INK2, lw=0.6))
    ax.axhline(0, color=ps.INK2, lw=0.8)
    ax.set_xlabel("Parent complementarity C")
    ax.set_ylabel("Net hybridization gain (NHG)")
    ax.legend(title="Class", loc="best")
    ps.save(fig, os.path.join(fdir, "fig_complementarity.png"))

    # Convergence of representative cases
    conv_d = 30 if 30 in dims else dims[-1]
    cases = [("MA-DE", 5), ("LSHADE>CMAES", 13), ("Island", 22), ("DE>NM", 1)]
    cases = [c for c in cases if c[1] in d.fid.unique()]
    if cases:
        fig, axes = plt.subplots(1, len(cases), figsize=(7.4, 3.1))
        for ax, (h, fid) in zip(np.atleast_1d(axes), cases):
            algs = [h] + list(ALGORITHMS[h][3])
            for k, a in enumerate(algs):
                g = d[(d.alg == a) & (d.fid == fid) & (d.dim == conv_d)]
                if g.empty:
                    continue
                med = g[S.CP_COLS].median().to_numpy()
                j = SINGLES.index(a) if a in SINGLES else None
                ax.plot(np.array(CHECKPOINTS) * 100, np.maximum(med, 1e-8),
                        color=ps.INK if j is None else ps.SERIES[j], marker="o" if j is None else ps.MARKERS[j + 1],
                        ms=3, lw=2.2 if j is None else 1.4, ls="-" if j is None else "--", label=lab(a))
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(["1", "10", "100"])
            ax.set_title(f"F{fid}, D = {conv_d}", loc="left")
            ax.set_xlabel("Budget (% of MaxFEs)")
            ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
                      handlelength=2.2, columnspacing=1.0)
        np.atleast_1d(axes)[0].set_ylabel("Median error")
        fig.tight_layout(w_pad=0.8)
        ps.save(fig, os.path.join(fdir, "fig_convergence.png"))
    print(json.dumps(facts["pooled"], indent=1))
    print(per_h.round(3))


if __name__ == "__main__":
    main(*sys.argv[1:4], runs=int(sys.argv[4]) if len(sys.argv) > 4 else 51)
