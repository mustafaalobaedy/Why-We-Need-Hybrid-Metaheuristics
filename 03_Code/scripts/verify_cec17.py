import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import cec17

for D in (10, 30, 50):
    bad = []
    for fid in cec17.FUNCTIONS:
        o = np.loadtxt(os.path.join(cec17._DIR, "input_data", f"shift_data_{fid}.txt")).ravel()[:D]
        f = cec17.CEC17(fid, D)
        e = f(o)[0]
        if abs(e) > 1e-6:
            bad.append((fid, e))
        X = np.random.uniform(-100, 100, (200, D))
        batch = f(X); single = np.array([f(x)[0] for x in X[:5]])
        assert np.allclose(batch[:5], single), (fid, D)
    print(f"D={D}: optimum error > 1e-6 for {bad if bad else 'none'}"
          + ("  (F9: expected; the official Levy code has its minimum at M(x - o) = 1, not at o)" if [b for b in bad if b[0] == 9] == bad and bad else ""))

t = time.time(); f = cec17.CEC17(30, 50); X = np.random.uniform(-100, 100, (100, 50))
for _ in range(100): f(X)
print(f"F30 D=50: {(time.time()-t)/1e4*1e6:.2f} us per evaluation")
