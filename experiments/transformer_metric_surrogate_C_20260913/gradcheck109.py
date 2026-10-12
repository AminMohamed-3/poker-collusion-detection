import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/cf441075-1502-4bb5-9e98-d102782e30e6/scratchpad/s109')
import numpy as np
import tfm109 as M
rng = np.random.default_rng(7)
B, T, F, d, h, L, m = 3, 7, 4, 8, 2, 2, 12
X, P, off = rng.normal(size=(B, T, F)), np.tile(np.arange(T), (B, 1)), rng.normal(size=(B, T))
y = (rng.random((B, T)) < .4).astype(float)
y[:, 0] = 1.
n = np.minimum(5., y.sum(1) + 1.)
p = M.init(F, d, h, L, T, m, rng)
for k in p:
    p[k] = p[k] + rng.normal(0, .3, p[k].shape)
for loss in ('sap', 'lam', 'bce'):
    f = lambda: M.LOSS[loss](off + M.forward(p, X, P, L, h)[0], y, n)[0]
    out, cache = M.forward(p, X, P, L, h)
    g = M.backward(p, cache, M.LOSS[loss](off + out, y, n)[1], L, h)
    errs = []
    for k in p:
        flat = p[k].reshape(-1)
        for i in rng.choice(flat.size, min(5, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + 1e-6; fp = f()
            flat[i] = old - 1e-6; fm = f()
            flat[i] = old
            num, ana = (fp - fm) / 2e-6, g[k].reshape(-1)[i]
            errs.append((abs(num - ana) / max(1e-9, abs(num) + abs(ana)), k))
    s0 = off + out
    gs = M.LOSS[loss](s0, y, n)[1]
    es = []
    for b in range(B):
        for t in range(T):
            e = np.zeros((B, T)); e[b, t] = 1e-6
            num = (M.LOSS[loss](s0 + e, y, n)[0] - M.LOSS[loss](s0 - e, y, n)[0]) / 2e-6
            es.append(abs(num - gs[b, t]) / max(1e-9, abs(num) + abs(gs[b, t])))
    print(loss, 'param-gradient worst relative error', max(errs), '| score-gradient worst', max(es))
s = rng.normal(size=(400, 20)); y = (rng.random((400, 20)) < .2).astype(float); n = np.maximum(1., np.minimum(5., y.sum(1)))
print('exact AP@5', round(M.ap5_exact(s, y, n), 6), '| smooth tau .5', round(M.smooth_ap5(s, y, n)[0], 6), '| tau .05', round(M.smooth_ap5(s, y, n, .05, .05)[0], 6), '| tau .002', round(M.smooth_ap5(s, y, n, .002, .002)[0], 6))
