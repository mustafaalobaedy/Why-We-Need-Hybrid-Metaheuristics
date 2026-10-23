"""Single (parent) metaheuristics and their hybrids.

Every algorithm is a callable ``run(problem, rng)`` that keeps optimizing until the
problem raises BudgetExhausted. Population-based components expose ``X``/``F`` and a
``step()`` method so that hybrids can be assembled from the same building blocks.
"""
import warnings

import cma
import numpy as np
from scipy.optimize import minimize

from core import BudgetExhausted

warnings.filterwarnings("ignore")

RHO = 0.2          # share of the budget given to the second (local/relay) component
LS_CALL = 50       # function evaluations per local-search call, multiplied by D
TERMINAL = -1.0    # L-SHADE terminal value for M_CR


# ----------------------------------------------------------------------------- helpers
def distinct_indices(n, k, rng):
    """k mutually distinct indices per row i, all different from i."""
    idx = np.empty((n, k), dtype=int)
    ar = np.arange(n)
    for j in range(k):
        col = rng.integers(0, n, n)
        while True:
            bad = col == ar
            for jj in range(j):
                bad |= col == idx[:, jj]
            if not bad.any():
                break
            col[bad] = rng.integers(0, n, bad.sum())
        idx[:, j] = col
    return idx


def midpoint_repair(V, X, lb, ub):
    V = np.where(V < lb, (lb + X) / 2.0, V)
    return np.where(V > ub, (ub + X) / 2.0, V)


def binomial_crossover(V, X, CR, rng):
    n, d = X.shape
    mask = rng.random((n, d)) < np.reshape(CR, (-1, 1))
    mask[np.arange(n), rng.integers(0, d, n)] = True
    return np.where(mask, V, X)


def sbx(P1, P2, rng, lb, ub, eta=20.0, pc=0.9):
    u = rng.random(P1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    mask = (rng.random(P1.shape) < 0.5) & (rng.random((P1.shape[0], 1)) < pc)
    beta = np.where(mask, beta, 1.0)
    C1 = 0.5 * ((1 + beta) * P1 + (1 - beta) * P2)
    C2 = 0.5 * ((1 - beta) * P1 + (1 + beta) * P2)
    return np.clip(np.vstack([C1, C2]), lb, ub)


def polynomial_mutation(X, rng, lb, ub, eta=20.0, pm=None):
    pm = 1.0 / X.shape[1] if pm is None else pm
    u = rng.random(X.shape)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    mask = rng.random(X.shape) < pm
    return np.clip(X + mask * delta * (ub - lb), lb, ub)


def nm_local(p, x0, step, maxfev):
    """Adaptive Nelder-Mead (Gao & Han, 2012) from x0 with an axis-aligned initial simplex."""
    if p.remaining <= 0:  # SciPy stops at maxfev without calling the objective again
        raise BudgetExhausted
    step = np.broadcast_to(step, (p.dim,)).astype(float)
    pts = x0 + np.diag(step)
    out = (pts > p.ub) | (pts < p.lb)
    pts = np.where(out, x0 - np.diag(step), pts)  # mirror vertices that fall outside the box
    sim = np.vstack([x0, np.clip(pts, p.lb, p.ub)])
    res = minimize(lambda x: p.evaluate(x)[0], x0, method="Nelder-Mead",
                   options=dict(initial_simplex=sim, maxfev=int(maxfev), adaptive=True,
                                xatol=1e-10, fatol=1e-10))
    return np.clip(res.x, p.lb, p.ub), float(res.fun)


def population_step(X, p):
    return np.maximum(X.std(axis=0), 1e-8 * (p.ub - p.lb))


# ----------------------------------------------------------------------------- parents
class GA:
    """Real-coded GA: binary tournament, SBX, polynomial mutation, (mu+lambda) survival."""

    def __init__(self, p, rng, n=100, X=None):
        self.p, self.rng = p, rng
        self.X = p.random(rng, n) if X is None else X
        self.F = p.evaluate(self.X)

    def offspring(self, X, F):
        n = len(X)
        m = (n + 1) // 2
        t = self.rng.integers(0, n, (2 * m, 2))
        win = np.where(F[t[:, 0]] <= F[t[:, 1]], t[:, 0], t[:, 1])
        C = sbx(X[win[:m]], X[win[m:]], self.rng, self.p.lb, self.p.ub)
        return polynomial_mutation(C, self.rng, self.p.lb, self.p.ub)[:n]

    def step(self):
        C = self.offspring(self.X, self.F)
        FC = self.p.evaluate(C)
        allX, allF = np.vstack([self.X, C]), np.concatenate([self.F, FC])
        keep = np.argsort(allF, kind="stable")[:len(self.X)]
        self.X, self.F = allX[keep], allF[keep]


class DE:
    """DE/rand/1/bin (Storn & Price, 1997)."""

    def __init__(self, p, rng, n=100, F=0.5, CR=0.9, X=None):
        self.p, self.rng, self.f, self.cr = p, rng, F, CR
        self.X = p.random(rng, n) if X is None else X
        self.F = p.evaluate(self.X)

    def trial(self, X):
        r = distinct_indices(len(X), 3, self.rng)
        V = X[r[:, 0]] + self.f * (X[r[:, 1]] - X[r[:, 2]])
        U = binomial_crossover(V, X, self.cr, self.rng)
        return midpoint_repair(U, X, self.p.lb, self.p.ub)

    def step(self):
        U = self.trial(self.X)
        FU = self.p.evaluate(U)
        better = FU <= self.F
        self.X[better], self.F[better] = U[better], FU[better]


class PSO:
    """Constricted PSO (Clerc & Kennedy, 2002) in the SPSO form of Bratton & Kennedy (2007)."""

    CHI, C = 0.72984, 2.05

    def __init__(self, p, rng, n=50, topology="ring", X=None):
        self.p, self.rng, self.topology = p, rng, topology
        self.pos = p.random(rng, n) if X is None else X
        self.vel = (p.random(rng, len(self.pos)) - self.pos) / 2.0
        self.fit = p.evaluate(self.pos)
        self.X, self.F = self.pos.copy(), self.fit.copy()  # personal bests

    def guide(self):
        if self.topology == "global":
            return np.broadcast_to(self.X[np.argmin(self.F)], self.X.shape)
        n = len(self.F)
        nb = np.stack([np.roll(np.arange(n), 1), np.arange(n), np.roll(np.arange(n), -1)], axis=1)
        return self.X[nb[np.arange(n), np.argmin(self.F[nb], axis=1)]]

    def move(self, idx=None, guide=None):
        idx = np.arange(len(self.F)) if idx is None else idx
        G = self.guide()[idx] if guide is None else guide
        x, v = self.pos[idx], self.vel[idx]
        r1, r2 = self.rng.random(x.shape), self.rng.random(x.shape)
        v = self.CHI * (v + self.C * r1 * (self.X[idx] - x) + self.C * r2 * (G - x))
        span = self.p.ub - self.p.lb
        v = np.clip(v, -span, span)
        x = x + v
        out = (x < self.p.lb) | (x > self.p.ub)
        x, v = np.clip(x, self.p.lb, self.p.ub), np.where(out, 0.0, v)
        f = self.p.evaluate(x)
        self.pos[idx], self.vel[idx], self.fit[idx] = x, v, f
        better = f < self.F[idx]
        self.X[idx[better]], self.F[idx[better]] = x[better], f[better]

    def step(self):
        self.move()


class LSHADE:
    """L-SHADE (Tanabe & Fukunaga, 2014) with linear population size reduction over `budget` FEs."""

    def __init__(self, p, rng, budget, H=6, r_arc=2.6, p_best=0.11):
        D = p.dim
        self.p, self.rng, self.budget = p, rng, budget
        self.n_init, self.n_min = int(round(18 * D)), 4
        self.H, self.r_arc, self.p_best = H, r_arc, p_best
        self.MF, self.MCR, self.k = np.full(H, 0.5), np.full(H, 0.5), 0
        self.A = np.empty((0, D))
        self.X = p.random(rng, self.n_init)
        self.F = p.evaluate(self.X)
        self.nfe = self.n_init

    def step(self):
        X, F, rng = self.X, self.F, self.rng
        n, D = X.shape
        r = rng.integers(0, self.H, n)
        CR = np.clip(rng.normal(self.MCR[r], 0.1), 0, 1)
        CR[self.MCR[r] == TERMINAL] = 0.0
        Fs = self.MF[r] + 0.1 * np.tan(np.pi * (rng.random(n) - 0.5))
        while (bad := Fs <= 0).any():
            Fs[bad] = self.MF[r[bad]] + 0.1 * np.tan(np.pi * (rng.random(bad.sum()) - 0.5))
        Fs = np.minimum(Fs, 1.0)

        order = np.argsort(F)
        pb = order[rng.integers(0, max(2, int(round(self.p_best * n))), n)]
        r1 = distinct_indices(n, 1, rng)[:, 0]
        PA = np.vstack([X, self.A]) if len(self.A) else X
        r2 = rng.integers(0, len(PA), n)
        while (bad := (r2 == np.arange(n)) | (r2 == r1)).any():
            r2[bad] = rng.integers(0, len(PA), bad.sum())
        Fc = Fs[:, None]
        V = X + Fc * (X[pb] - X) + Fc * (X[r1] - PA[r2])
        U = midpoint_repair(binomial_crossover(V, X, CR, rng), X, self.p.lb, self.p.ub)
        FU = self.p.evaluate(U)
        self.nfe += n

        imp = FU < F
        if imp.any():
            self.A = np.vstack([self.A, X[imp]])
            w = (F[imp] - FU[imp])
            w = w / w.sum()
            s_cr, s_f = CR[imp], Fs[imp]
            if self.MCR[self.k] == TERMINAL or s_cr.max() == 0:
                self.MCR[self.k] = TERMINAL
            else:
                self.MCR[self.k] = np.sum(w * s_cr ** 2) / np.sum(w * s_cr)
            self.MF[self.k] = np.sum(w * s_f ** 2) / np.sum(w * s_f)
            self.k = (self.k + 1) % self.H
        rep = FU <= F
        X[rep], F[rep] = U[rep], FU[rep]

        n_new = int(round((self.n_min - self.n_init) / self.budget * self.nfe + self.n_init))
        n_new = max(self.n_min, n_new)
        if n_new < n:
            keep = np.argsort(F, kind="stable")[:n_new]
            self.X, self.F = X[keep], F[keep]
        cap = int(round(self.r_arc * len(self.X)))
        if len(self.A) > cap:
            self.A = self.A[rng.choice(len(self.A), cap, replace=False)]


# ----------------------------------------------------------------------------- runners
def run_ga(p, rng):
    alg = GA(p, rng)
    while True:
        alg.step()


def run_de(p, rng):
    alg = DE(p, rng)
    while True:
        alg.step()


def run_pso(p, rng):
    alg = PSO(p, rng)
    while True:
        alg.step()


def run_nm(p, rng):
    """Nelder-Mead with random restarts until the budget is spent."""
    while True:
        nm_local(p, p.random(rng, 1)[0], 0.1 * (p.ub - p.lb), p.remaining)


def run_lshade(p, rng):
    alg = LSHADE(p, rng, p.max_fe)
    while True:
        alg.step()


def ipop_cmaes(p, rng, x0=None, sigma0=None):
    """IPOP-CMA-ES (Auger & Hansen, 2005) using pycma; the first run may start from x0/sigma0."""
    lam = 4 + int(3 * np.log(p.dim))
    while True:
        start = p.random(rng, 1)[0] if x0 is None else x0
        s0 = 0.3 * (p.ub - p.lb) if sigma0 is None else sigma0
        es = cma.CMAEvolutionStrategy(start, s0, {
            "bounds": [p.lb, p.ub], "popsize": lam, "seed": int(rng.integers(1, 2 ** 31 - 1)),
            "verbose": -9, "verb_disp": 0, "verb_log": 0, "maxfevals": p.remaining})
        while not es.stop():
            S = es.ask()
            es.tell(S, p.evaluate(np.asarray(S)).tolist())
        x0, sigma0, lam = None, None, lam * 2


def run_cmaes(p, rng):
    ipop_cmaes(p, rng)


# ----------------------------------------------------------------------------- hybrids
def memetic(alg, p, rho=RHO):
    """LRH: after each generation, refine the best member with Nelder-Mead while the
    local-search share of consumed evaluations is below rho (Lamarckian replacement)."""
    ls_used = 0
    while True:
        alg.step()
        if ls_used < rho * p.fe:
            i = int(np.argmin(alg.F))
            fe0 = p.fe
            x, f = nm_local(p, alg.X[i].copy(), population_step(alg.X, p), LS_CALL * p.dim)
            ls_used += p.fe - fe0
            if f < alg.F[i]:
                alg.X[i], alg.F[i] = x, f


def run_ma_ga(p, rng):
    memetic(GA(p, rng), p)


def run_ma_de(p, rng, rho=RHO):
    memetic(DE(p, rng), p, rho)


def run_ma_lshade(p, rng, rho=RHO):
    memetic(LSHADE(p, rng, (1 - rho) * p.max_fe), p, rho)


def run_de_to_nm(p, rng, rho=RHO):
    """HRH: DE for (1 - rho) of the budget, then restarted Nelder-Mead from the best solution."""
    alg = DE(p, rng)
    while p.fe < (1 - rho) * p.max_fe:
        alg.step()
    step = population_step(alg.X, p)
    while True:
        nm_local(p, p.best_x.copy(), step, p.remaining)


def run_lshade_to_cmaes(p, rng, rho=RHO):
    """HRH: L-SHADE for (1 - rho) of the budget, then IPOP-CMA-ES seeded with its best solution."""
    alg = LSHADE(p, rng, (1 - rho) * p.max_fe)
    while p.fe < (1 - rho) * p.max_fe:
        alg.step()
    sigma0 = float(max(np.mean(alg.X.std(axis=0)), 1e-8 * (p.ub - p.lb)))
    ipop_cmaes(p, rng, x0=p.best_x.copy(), sigma0=sigma0)


def run_ga_pso(p, rng, n=100):
    """LTH (after Kao & Zahara, 2008): each generation the better half is evolved by GA
    operators and the worse half is moved by PSO toward the global best."""
    ga = GA(p, rng, n=n)
    X, F = ga.X, ga.F
    V = np.zeros_like(X)
    P, PF = X.copy(), F.copy()
    h = n // 2
    while True:
        o = np.argsort(F, kind="stable")
        X, F, V, P, PF = X[o], F[o], V[o], P[o], PF[o]
        # elite half: GA offspring compete with the elites (mu + lambda)
        C = ga.offspring(X[:h], F[:h])
        FC = p.evaluate(C)
        allX, allF = np.vstack([X[:h], C]), np.concatenate([F[:h], FC])
        keep = np.argsort(allF, kind="stable")[:h]
        X[:h], F[:h] = allX[keep], allF[keep]
        V[:h], P[:h], PF[:h] = 0.0, X[:h], F[:h]
        # weaker half: PSO move toward the global best
        g = X[np.argmin(F)]
        x, v = X[h:], V[h:]
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = PSO.CHI * (v + PSO.C * r1 * (P[h:] - x) + PSO.C * r2 * (g - x))
        x = x + v
        out = (x < p.lb) | (x > p.ub)
        x, v = np.clip(x, p.lb, p.ub), np.where(out, 0.0, v)
        f = p.evaluate(x)
        X[h:], V[h:], F[h:] = x, v, f
        better = f < PF[h:]
        P[h:][better], PF[h:][better] = x[better], f[better]


def run_depso(p, rng):
    """LTH (after Zhang & Xie, 2003): PSO generations alternate with a DE/rand/1/bin
    operator applied to the personal-best memory."""
    pso = PSO(p, rng)
    de = DE.__new__(DE)
    de.p, de.rng, de.f, de.cr = p, rng, 0.5, 0.9
    gen = 0
    while True:
        if gen % 2 == 0:
            pso.step()
        else:
            U = de.trial(pso.X)
            FU = p.evaluate(U)
            better = FU < pso.F
            pso.X[better], pso.F[better] = U[better], FU[better]
        gen += 1


def run_nm_pso(p, rng):
    """LTH (after Fan & Zahara, 2007): population of 3D+1; one Nelder-Mead iteration on the
    best D+1 points, PSO (global best) on the remaining 2D points."""
    D = p.dim
    pso = PSO(p, rng, n=3 * D + 1, topology="global")
    while True:
        o = np.argsort(pso.fit, kind="stable")
        s = o[:D + 1]
        S, FS = pso.pos[s], pso.fit[s]
        w = s[-1]
        xo = S[:-1].mean(axis=0)
        xr = np.clip(2 * xo - S[-1], p.lb, p.ub)
        fr = p.evaluate(xr)[0]
        new_x, new_f = None, None
        if fr < FS[0]:
            xe = np.clip(3 * xo - 2 * S[-1], p.lb, p.ub)
            fe = p.evaluate(xe)[0]
            new_x, new_f = (xe, fe) if fe < fr else (xr, fr)
        elif fr < FS[-2]:
            new_x, new_f = xr, fr
        else:
            xc = xo + 0.5 * ((xr if fr < FS[-1] else S[-1]) - xo)
            fc = p.evaluate(xc)[0]
            if fc < min(fr, FS[-1]):
                new_x, new_f = xc, fc
            else:  # shrink toward the best vertex
                Sh = S[0] + 0.5 * (S[1:] - S[0])
                pso.pos[s[1:]], pso.fit[s[1:]] = Sh, p.evaluate(Sh)
        if new_x is not None:
            pso.pos[w], pso.fit[w] = new_x, new_f
        for i in s:  # keep personal-best memory consistent with simplex moves
            if pso.fit[i] < pso.F[i]:
                pso.X[i], pso.F[i] = pso.pos[i], pso.fit[i]
        rest = o[D + 1:]
        pso.move(rest, guide=np.broadcast_to(pso.X[np.argmin(pso.F)], (len(rest), D)))


def run_island(p, rng, epoch=20):
    """HTH: GA, DE and PSO islands evolve in parallel; every `epoch` generations the best
    member of each island replaces the worst member of the next island (ring)."""
    isl = [GA(p, rng), DE(p, rng), PSO(p, rng)]
    gen = 0
    while True:
        for a in isl:
            a.step()
        gen += 1
        if gen % epoch == 0:
            best = [(a.X[np.argmin(a.F)].copy(), a.F.min()) for a in isl]
            for j, a in enumerate(isl):
                x, f = best[j - 1]
                w = int(np.argmax(a.F))
                a.X[w], a.F[w] = x, f
                if isinstance(a, PSO):
                    a.pos[w], a.fit[w], a.vel[w] = x, f, 0.0


# name -> (runner, family, Talbi (2002) class, parent components)
ALGORITHMS = {
    "GA": (run_ga, "single", None, ()),
    "DE": (run_de, "single", None, ()),
    "PSO": (run_pso, "single", None, ()),
    "NM": (run_nm, "single", None, ()),
    "L-SHADE": (run_lshade, "single", None, ()),
    "CMA-ES": (run_cmaes, "single", None, ()),
    "GA-PSO": (run_ga_pso, "hybrid", "LTH", ("GA", "PSO")),
    "DEPSO": (run_depso, "hybrid", "LTH", ("DE", "PSO")),
    "NM-PSO": (run_nm_pso, "hybrid", "LTH", ("NM", "PSO")),
    "MA-GA": (run_ma_ga, "hybrid", "LRH", ("GA", "NM")),
    "MA-DE": (run_ma_de, "hybrid", "LRH", ("DE", "NM")),
    "MA-LSHADE": (run_ma_lshade, "hybrid", "LRH", ("L-SHADE", "NM")),
    "DE>NM": (run_de_to_nm, "hybrid", "HRH", ("DE", "NM")),
    "LSHADE>CMAES": (run_lshade_to_cmaes, "hybrid", "HRH", ("L-SHADE", "CMA-ES")),
    "Island": (run_island, "hybrid", "HTH", ("GA", "DE", "PSO")),
}


# Budget-split ablation: the four hybrids whose second component has an explicit share rho.
ABLATION = {}
for _rho in (0.05, 0.1, 0.4):
    for _name, _fn in (("MA-DE", run_ma_de), ("MA-LSHADE", run_ma_lshade), ("DE>NM", run_de_to_nm),
                       ("LSHADE>CMAES", run_lshade_to_cmaes)):
        ABLATION[f"{_name}@{_rho}"] = (lambda p, rng, _f=_fn, _r=_rho: _f(p, rng, rho=_r),
                                       "hybrid", ALGORITHMS[_name][2], ALGORITHMS[_name][3])
