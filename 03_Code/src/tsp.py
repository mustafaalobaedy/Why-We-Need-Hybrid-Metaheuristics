"""Python interface to the C implementation of ACS, ILS and their hybrids for the TSP."""
import ctypes
import os

import numpy as np

from core import CHECKPOINTS

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsp")
_lib = ctypes.CDLL(os.path.join(_DIR, "libtsp.so"))
_P = np.ctypeslib.ndpointer
_lib.run_tsp.restype = ctypes.c_longlong
_lib.run_tsp.argtypes = [ctypes.c_int, ctypes.c_int, _P(np.int32, flags="C"), _P(np.int32, flags="C"),
                         ctypes.c_int, ctypes.c_double, ctypes.c_ulonglong, ctypes.c_double,
                         _P(np.float64, flags="C"), ctypes.c_int, _P(np.int64, flags="C"), _P(np.int32, flags="C")]

OPTIMA = {"eil51": 426, "kroA100": 21282, "ch150": 6528, "kroA200": 29368, "lin318": 42029,
          "pcb442": 50778, "rat783": 8806}
TSP_ALGS = {"ACS": 0, "ILS": 1, "ACS+2opt": 2, "ACS>ILS": 3}
TSP_HYBRIDS = {"ACS+2opt": ("LRH", ("ACS", "ILS")), "ACS>ILS": ("HRH", ("ACS", "ILS"))}
CAND = 20  # candidate-list size


def load(path):
    """TSPLIB EUC_2D instance -> integer distance matrix and nearest-neighbour lists."""
    xy, reading = [], False
    for line in open(path):
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            reading = True
            continue
        if reading:
            if line in ("EOF", ""):
                break
            _, x, y = line.split()[:3]
            xy.append((float(x), float(y)))
    xy = np.array(xy)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    D = np.floor(d + 0.5).astype(np.int32)  # TSPLIB nint()
    big = D + np.eye(len(D), dtype=np.int32) * np.iinfo(np.int32).max // 2
    NN = np.ascontiguousarray(np.argsort(big, axis=1, kind="stable")[:, :CAND].astype(np.int32))
    return np.ascontiguousarray(D), NN


def run(D, NN, alg, time_limit, seed, share=0.2):
    n = len(D)
    cpf = np.array(CHECKPOINTS, dtype=np.float64)
    trace = np.zeros(len(cpf), dtype=np.int64)
    tour = np.zeros(n, dtype=np.int32)
    best = _lib.run_tsp(n, NN.shape[1], D.ravel(), NN.ravel(), TSP_ALGS[alg], time_limit,
                        seed, share, cpf, len(cpf), trace, tour)
    return int(best), trace, tour
