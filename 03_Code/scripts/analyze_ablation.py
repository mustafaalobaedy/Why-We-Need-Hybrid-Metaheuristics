"""Budget-split ablation: NHG of four hybrids against their best parent for rho in {0.05, 0.1, 0.2, 0.4}.

The rho = 0.2 runs and the parent runs are taken from the main experiment.

usage: python analyze_ablation.py <main.csv> <ablation.csv> <tables_dir> <figures_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd  # noqa: E402

import plotstyle as ps  # noqa: E402
import stats as S  # noqa: E402
from algorithms import ALGORITHMS  # noqa: E402

LABEL = {"DE>NM": "DE→NM", "LSHADE>CMAES": "L-SHADE→CMA-ES"}
BASES = ["MA-DE", "MA-LSHADE", "DE>NM", "LSHADE>CMAES"]
RHO_MAIN = 0.2


def complete(d, runs):
    c = d.groupby(["alg", "fid", "dim"]).size()
    ok = c[c == runs].reset_index()[["alg", "fid", "dim"]]
    return d.merge(ok)


def main(main_path, abl_path, tdir, fdir, runs=51):
    os.makedirs(tdir, exist_ok=True)
    ps.apply()
    plt = ps.plt
    abl = complete(S.load(abl_path), runs)
    dims = sorted(abl.dim.unique())
    m = S.load(main_path)
    m = m[m.dim.isin(dims)]
    abl[["base", "rho"]] = abl.alg.str.split("@", expand=True)
    abl["rho"] = abl.rho.astype(float)

    rows, detail = [], []
    for base in BASES:
        parents = list(ALGORITHMS[base][3])
        variants = [(RHO_MAIN, m[m.alg == base])] + [(r, g.assign(alg=base)) for r, g in
                                                     abl[abl.base == base].groupby("rho")]
        for rho, hyb in variants:
            # Restrict every variant to the same (fid, dim) cases so that the Holm families are equal.
            cases = abl[abl.base == base][["fid", "dim"]].drop_duplicates()
            sub = pd.concat([hyb, m[m.alg.isin(parents)]]).merge(cases)
            r = S.hybrid_vs_parents(sub)
            r = r[r.hybrid == base].assign(rho=rho)
            detail.append(r)
            for key, g in [("All", r)] + [(f"D={dd}", gg) for dd, gg in r.groupby("dim")]:
                n = len(g)
                w, l_ = (g.outcome == "win").sum(), (g.outcome == "loss").sum()
                rows.append(dict(hybrid=base, rho=rho, scope=key, W=w, T=n - w - l_, L=l_, N=n,
                                 HAR=w / n, HLR=l_ / n, NHG=(w - l_) / n))
    res = pd.DataFrame(rows).sort_values(["hybrid", "scope", "rho"])
    res.to_csv(os.path.join(tdir, "ablation_rho.csv"), index=False)
    pd.concat(detail).to_csv(os.path.join(tdir, "ablation_rho_detail.csv"), index=False)

    # NHG versus rho, one panel per dimension
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2, 3.0), sharey=True)
    for ax, dim in zip(axes if len(dims) > 1 else [axes], dims):
        for k, base in enumerate(BASES):
            s = res[(res.hybrid == base) & (res.scope == f"D={dim}")].sort_values("rho")
            # small horizontal dodge (log axis) and white marker edges so that coinciding points stay visible
            ax.plot(s.rho * 1.035 ** (k - 1.5), s.NHG, color=ps.SERIES[k], marker=ps.MARKERS[k], ms=5.5, lw=1.6,
                    mec="white", mew=0.9, label=LABEL.get(base, base))
        ax.axhline(0, color=ps.INK2, lw=0.8)
        ax.axvline(RHO_MAIN, color=ps.INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.2, 0.4])
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
        ax.minorticks_off()
        ax.set_xlabel(r"Budget share of the second component $\rho$")
        ax.set_title(f"D = {dim}", loc="left")
    (axes[0] if len(dims) > 1 else axes).set_ylabel("Net hybridization gain (NHG)")
    (axes[-1] if len(dims) > 1 else axes).legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ps.save(fig, os.path.join(fdir, "fig_ablation_rho.png"))

    facts = {f"{h}|{r}": g.set_index("scope")[["W", "T", "L", "NHG"]].round(4).to_dict(orient="index")
             for (h, r), g in res.groupby(["hybrid", "rho"])}
    json.dump(facts, open(os.path.join(tdir, "ablation_facts.json"), "w"), indent=1)
    print(res[res.scope == "All"].pivot(index="hybrid", columns="rho", values="NHG").round(3))


if __name__ == "__main__":
    main(*sys.argv[1:5])
