"""Statistical comparison utilities (minimization: lower error is better)."""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata

import cec17
from algorithms import ALGORITHMS
from core import CHECKPOINTS, ERR_TOL

CP_COLS = [f"cp{c}" for c in CHECKPOINTS]
ALPHA = 0.05


def load(path):
    d = pd.read_csv(path)
    for c in ["error"] + CP_COLS:
        d.loc[d[c] < ERR_TOL, c] = 0.0
    d["category"] = d.fid.map(cec17.category)
    return d


def a12(x, y):
    """Vargha-Delaney A: probability that a run of x yields a lower error than a run of y."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def rank_sum_p(x, y):
    if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
        return 1.0
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def outcome(p, a, alpha=ALPHA):
    if p < alpha:
        return "win" if a > 0.5 else "loss"
    return "tie"


def errors(d, col="error"):
    """dict (alg, fid, dim) -> sorted-by-run array of the given error column."""
    return {k: g.sort_values("run")[col].to_numpy() for k, g in d.groupby(["alg", "fid", "dim"])}


def hybrid_vs_parents(d, col="error", correction="holm"):
    """Compare every hybrid with the best of its parents (lowest median) on every (fid, dim)."""
    E = errors(d, col)
    rows = []
    for h, (_, fam, cls, parents) in ALGORITHMS.items():
        if fam != "hybrid":
            continue
        for (alg, fid, dim), xh in E.items():
            if alg != h:
                continue
            cand = [(np.median(E[(pa, fid, dim)]), np.mean(E[(pa, fid, dim)]), pa) for pa in parents
                    if (pa, fid, dim) in E]
            if not cand:
                continue
            bp = min(cand)[2]
            xp = E[(bp, fid, dim)]
            rows.append(dict(hybrid=h, cls=cls, fid=fid, dim=dim, category=cec17.category(fid),
                             best_parent=bp, med_h=np.median(xh), med_p=np.median(xp),
                             mean_h=xh.mean(), mean_p=xp.mean(),
                             p=rank_sum_p(xh, xp), A12=a12(xh, xp)))
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    if correction == "holm":
        r["p_adj"] = r.groupby("hybrid").p.transform(holm)
    else:
        r["p_adj"] = r.p
    r["outcome"] = [outcome(p, a) for p, a in zip(r.p_adj, r.A12)]
    return r


def pairwise(d, a, b, col="error"):
    """Win/tie/loss of algorithm a against algorithm b over all (fid, dim)."""
    E = errors(d, col)
    rows = []
    for (alg, fid, dim), xa in E.items():
        if alg != a or (b, fid, dim) not in E:
            continue
        xb = E[(b, fid, dim)]
        rows.append(dict(a=a, b=b, fid=fid, dim=dim, p=rank_sum_p(xa, xb), A12=a12(xa, xb)))
    r = pd.DataFrame(rows)
    r["p_adj"] = holm(r.p)
    r["outcome"] = [outcome(p, x) for p, x in zip(r.p_adj, r.A12)]
    return r


def friedman(d, dim=None, col="error"):
    """Average Friedman ranks of all algorithms over functions (mean error per function)."""
    sub = d if dim is None else d[d.dim == dim]
    m = sub.groupby(["fid", "dim", "alg"])[col].mean().unstack("alg")
    ranks = m.apply(lambda row: pd.Series(rankdata(row.to_numpy()), index=row.index), axis=1)
    stat, p = friedmanchisquare(*[m[c].to_numpy() for c in m.columns])
    return ranks.mean().sort_values(), stat, p, len(m)


def friedman_posthoc_holm(avg_ranks, n_problems):
    """Holm post-hoc (control = best-ranked algorithm), z-test on Friedman average ranks."""
    from scipy.stats import norm
    k = len(avg_ranks)
    se = np.sqrt(k * (k + 1) / (6.0 * n_problems))
    best = avg_ranks.idxmin()
    others = avg_ranks.drop(best)
    z = (others - avg_ranks[best]) / se
    p = pd.Series(2 * (1 - norm.cdf(np.abs(z))), index=others.index)
    return pd.DataFrame(dict(rank=others, z=z, p=p, p_holm=holm(p.to_numpy()))), best
