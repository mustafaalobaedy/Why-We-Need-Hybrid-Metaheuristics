"""Parallel, resumable CEC2017 experiment: every (algorithm, function, D, run) task is one CSV row."""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from multiprocessing import Pool  # noqa: E402

import cec17  # noqa: E402
from algorithms import ABLATION, ALGORITHMS  # noqa: E402
from core import CHECKPOINTS  # noqa: E402
from experiment import run_task  # noqa: E402

FIELDS = ["alg", "fid", "dim", "run", "error", "time"] + [f"cp{c}" for c in CHECKPOINTS]


def _work(task):
    try:
        return run_task(*task)
    except Exception as exc:  # report and keep the pool alive
        return {"failed": task, "msg": repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algs", nargs="+", default=list(ALGORITHMS))
    ap.add_argument("--funcs", nargs="+", type=int, default=cec17.FUNCTIONS)
    ap.add_argument("--dims", nargs="+", type=int, default=[10, 30, 50])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            done = {(r["alg"], int(r["fid"]), int(r["dim"]), int(r["run"])) for r in csv.DictReader(fh)}
    tasks = [(al, f, d, r) for d in sorted(a.dims, reverse=True) for al in a.algs
             for f in a.funcs for r in range(a.runs) if (al, f, d, r) not in done]
    print(f"{len(done)} tasks done, {len(tasks)} to run", flush=True)

    new = not os.path.exists(a.out)
    t0 = time.time()
    with open(a.out, "a", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, res in enumerate(pool.imap_unordered(_work, tasks), 1):
            if "failed" in res:
                print("FAILED", res, flush=True)
                continue
            w.writerow(res)
            fh.flush()
            if k % 50 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"{k}/{len(tasks)}  elapsed {el / 3600:.2f} h  eta {el / k * (len(tasks) - k) / 3600:.2f} h",
                      flush=True)


if __name__ == "__main__":
    main()
