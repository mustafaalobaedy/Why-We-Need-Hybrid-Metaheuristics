"""Python interface to the official CEC2017 bound-constrained benchmark (C++ reference code)."""
import ctypes
import os

import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cec17")
_lib = None

FUNCTIONS = [1] + list(range(3, 31))  # F2 is excluded by the CEC2017 organizers
LOWER, UPPER = -100.0, 100.0


def _load():
    global _lib
    if _lib is None:
        # The reference code reads "input_data/..." relative to the working directory.
        os.chdir(_DIR)
        _lib = ctypes.CDLL(os.path.join(_DIR, "libcec17.so"))
        _lib.cec17_eval.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib.cec17_eval.restype = None
    return _lib


def category(fid):
    if fid <= 3:
        return "Unimodal"
    if fid <= 10:
        return "Multimodal"
    if fid <= 20:
        return "Hybrid"
    return "Composition"


class CEC17:
    """Callable objective: f(X) with X of shape (n, D) or (D,); returns error f(x) - f*."""

    def __init__(self, fid, dim):
        self.fid, self.dim = fid, dim
        self.fopt = 100.0 * fid
        self.lb, self.ub = LOWER, UPPER
        self.lib = _load()

    def __call__(self, X):
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        n = X.shape[0]
        # The reference code expects column-major population layout: x[j*nx + i] for individual j.
        f = np.empty(n, dtype=np.float64)
        self.lib.cec17_eval(X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.dim, n, self.fid)
        return f - self.fopt
