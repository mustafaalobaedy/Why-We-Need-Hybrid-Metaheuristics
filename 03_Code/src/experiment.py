"""Run one (algorithm, function, dimension, run) task under the CEC2017 protocol."""
import time

import numpy as np

import cec17
from algorithms import ABLATION, ALGORITHMS
from core import CHECKPOINTS, BudgetExhausted, Problem

BUDGET_FACTOR = 10000  # MaxFEs = 10000 * D


def seed_for(fid, dim, run):
    # Common random numbers: every algorithm sees the same seed for a given (function, D, run).
    return int(np.random.SeedSequence([fid, dim, run]).generate_state(1)[0])


def run_task(alg_name, fid, dim, run, budget_factor=BUDGET_FACTOR):
    runner = {**ALGORITHMS, **ABLATION}[alg_name][0]
    p = Problem(cec17.CEC17(fid, dim), dim, cec17.LOWER, cec17.UPPER, budget_factor * dim)
    rng = np.random.default_rng(seed_for(fid, dim, run))
    t0 = time.perf_counter()
    try:
        runner(p, rng)
    except BudgetExhausted:
        pass
    elapsed = time.perf_counter() - t0
    if p.fe != p.max_fe or np.isnan(p.trace).any():
        raise RuntimeError(f"{alg_name} F{fid} D{dim} run {run}: used {p.fe}/{p.max_fe} FEs")
    return dict(alg=alg_name, fid=fid, dim=dim, run=run, error=p.best_f, time=elapsed,
                **{f"cp{c}": v for c, v in zip(CHECKPOINTS, p.trace)})
