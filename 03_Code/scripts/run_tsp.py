"""TSP case study: ACS, ILS and their hybrids on TSPLIB instances (CPU-time budget n/100 s, 51 runs)."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

import tsp  # noqa: E402
from core import CHECKPOINTS  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "04_Data", "tsplib")
FIELDS = ["alg", "instance", "n", "run", "length", "rpd"] + [f"cp{c}" for c in CHECKPOINTS]
_cache = {}


def _work(task):
    alg, inst, run, factor = task
    if inst not in _cache:
        _cache[inst] = tsp.load(os.path.join(DATA, f"{inst}.tsp"))
    D, NN = _cache[inst]
    seed = int(np.random.SeedSequence([len(D), run, 7]).generate_state(1)[0])  # common per (instance, run)
    best, trace, tour = tsp.run(D, NN, alg, factor * len(D), seed)
    assert sorted(tour.tolist()) == list(range(len(D)))
    opt = tsp.OPTIMA[inst]
    return dict(alg=alg, instance=inst, n=len(D), run=run, length=best, rpd=100.0 * (best - opt) / opt,
                **{f"cp{c}": 100.0 * (v - opt) / opt for c, v in zip(CHECKPOINTS, trace)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=51)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--factor", type=float, default=0.01)  # CPU seconds per city
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {(r["alg"], r["instance"], int(r["run"])) for r in csv.DictReader(open(a.out))}
    tasks = [(al, inst, r, a.factor) for inst in tsp.OPTIMA for al in tsp.TSP_ALGS for r in range(a.runs)
             if (al, inst, r) not in done]
    print(f"{len(tasks)} tasks", flush=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, res in enumerate(pool.imap_unordered(_work, tasks), 1):
            w.writerow(res)
            fh.flush()
            if k % 100 == 0:
                print(k, flush=True)


if __name__ == "__main__":
    main()
