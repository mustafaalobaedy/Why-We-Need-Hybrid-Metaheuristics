"""Budget-controlled problem wrapper shared by all algorithms."""
import numpy as np

# Fractions of MaxFEs at which the best error is recorded (CEC2017 protocol, Awad et al., 2016).
CHECKPOINTS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ERR_TOL = 1e-8  # errors below this value are reported as zero


class BudgetExhausted(Exception):
    """Raised as soon as the function-evaluation budget is used up."""


class Problem:
    def __init__(self, func, dim, lb, ub, max_fe):
        self.func, self.dim, self.lb, self.ub, self.max_fe = func, dim, lb, ub, max_fe
        self.fe = 0
        self.best_f, self.best_x = np.inf, None
        self.cp_fe = [int(round(c * max_fe)) for c in CHECKPOINTS]
        self.trace = np.full(len(CHECKPOINTS), np.nan)
        self._cp = 0

    @property
    def remaining(self):
        return self.max_fe - self.fe

    def random(self, rng, n):
        return rng.uniform(self.lb, self.ub, (n, self.dim))

    def evaluate(self, X):
        X = np.atleast_2d(X)
        if self.remaining <= 0:
            raise BudgetExhausted
        truncated = X.shape[0] > self.remaining
        if truncated:
            X = X[:self.remaining]
        X = np.clip(X, self.lb, self.ub)
        f = self.func(X)
        running = np.minimum.accumulate(f)
        start = self.fe
        while self._cp < len(self.cp_fe) and self.cp_fe[self._cp] <= start + len(f):
            k = self.cp_fe[self._cp] - start
            self.trace[self._cp] = min(self.best_f, running[k - 1])
            self._cp += 1
        i = int(np.argmin(f))
        if f[i] < self.best_f:
            self.best_f, self.best_x = float(f[i]), X[i].copy()
        self.fe += len(f)
        if truncated:
            raise BudgetExhausted
        return f
