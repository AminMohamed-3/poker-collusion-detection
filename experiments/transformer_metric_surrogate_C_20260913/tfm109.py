"""Step 109 (Claude, training-only audit looks): a small transformer encoder over each pair's top-20 evidence candidates, in numpy with
explicit backprop, trained on differentiable surrogates of the leaderboard's Evidence MAP@5. Tokens are hands in time order; the output
is a per-hand margin added to a fixed offset (the incumbent's logit)."""
import numpy as np
from scipy.special import erf

SQ2, SQ2PI = np.sqrt(2.), np.sqrt(2. * np.pi)


def sig(x):
    return .5 * (1. + np.tanh(.5 * x))


def logit(v):
    v = np.clip(v, 1e-4, 1 - 1e-4)
    return np.log(v / (1 - v))


def gelu(x):
    return .5 * x * (1. + erf(x / SQ2))


def dgelu(x):
    return .5 * (1. + erf(x / SQ2)) + x * np.exp(-.5 * x * x) / SQ2PI


def init(F, d, h, L, T, m, rng):
    w = lambda a, b, k=1.: rng.normal(0., k * np.sqrt(2. / (a + b)), (a, b))
    p = {'Win': w(F, d), 'bin': np.zeros(d), 'pos': rng.normal(0., .1, (T, d)), 'gf': np.ones(d), 'bf': np.zeros(d),
         'wout': np.zeros(d), 'bout': np.zeros(1)}
    for l in range(L):
        p.update({f'g1_{l}': np.ones(d), f'b1_{l}': np.zeros(d), f'Wq_{l}': w(d, d), f'Wk_{l}': w(d, d), f'Wv_{l}': w(d, d),
                  f'Wo_{l}': w(d, d, .5), f'bo_{l}': np.zeros(d), f'g2_{l}': np.ones(d), f'b2_{l}': np.zeros(d),
                  f'W1_{l}': w(d, m), f'c1_{l}': np.zeros(m), f'W2_{l}': w(m, d, .5), f'c2_{l}': np.zeros(d)})
    return p


def ln_f(x, g, b, eps=1e-5):
    sd = np.sqrt(x.var(-1, keepdims=True) + eps)
    xh = (x - x.mean(-1, keepdims=True)) / sd
    return xh * g + b, (xh, sd, g)


def ln_b(dy, c):
    xh, sd, g = c
    k = xh.shape[-1]
    dxh = dy * g
    dx = (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True)) / sd
    return dx, (dy * xh).reshape(-1, k).sum(0), dy.reshape(-1, k).sum(0)


def forward(p, X, P, L, h):
    """X (B,T,F) standardized token features, P (B,T) time positions. Pre-LN blocks: x + MHSA(LN x), then x + FFN(LN x)."""
    B, T, _ = X.shape
    d = p['Win'].shape[1]
    dh = d // h
    H = X @ p['Win'] + p['bin'] + p['pos'][P]
    layers = []
    for l in range(L):
        A, c1 = ln_f(H, p[f'g1_{l}'], p[f'b1_{l}'])
        Q, K, V = [(A @ p[f'W{k}_{l}']).reshape(B, T, h, dh).transpose(0, 2, 1, 3) for k in 'qkv']
        z = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)
        z = np.exp(z - z.max(-1, keepdims=True))
        att = z / z.sum(-1, keepdims=True)
        O = (att @ V).transpose(0, 2, 1, 3).reshape(B, T, d)
        H = H + O @ p[f'Wo_{l}'] + p[f'bo_{l}']
        C, c2 = ln_f(H, p[f'g2_{l}'], p[f'b2_{l}'])
        U = C @ p[f'W1_{l}'] + p[f'c1_{l}']
        H = H + gelu(U) @ p[f'W2_{l}'] + p[f'c2_{l}']
        layers.append((A, c1, Q, K, V, att, O, C, c2, U))
    Fh, cf = ln_f(H, p['gf'], p['bf'])
    return Fh @ p['wout'] + p['bout'][0], (X, P, layers, Fh, cf)


def backward(p, cache, dout, L, h):
    """dout (B,T) = dJ/d(output margin). Returns dJ/d(parameter) for every parameter."""
    X, P, layers, Fh, cf = cache
    B, T, F = X.shape
    d = p['Win'].shape[1]
    dh = d // h
    fl = lambda M: M.reshape(-1, M.shape[-1])
    g = {'wout': fl(Fh * dout[..., None]).sum(0), 'bout': np.array([dout.sum()])}
    dH, g['gf'], g['bf'] = ln_b(dout[..., None] * p['wout'], cf)
    for l in reversed(range(L)):
        A, c1, Q, K, V, att, O, C, c2, U = layers[l]
        g[f'W2_{l}'] = fl(gelu(U)).T @ fl(dH)
        g[f'c2_{l}'] = fl(dH).sum(0)
        dU = (dH @ p[f'W2_{l}'].T) * dgelu(U)
        g[f'W1_{l}'] = fl(C).T @ fl(dU)
        g[f'c1_{l}'] = fl(dU).sum(0)
        dx, g[f'g2_{l}'], g[f'b2_{l}'] = ln_b(dU @ p[f'W1_{l}'].T, c2)
        dH = dH + dx
        g[f'Wo_{l}'] = fl(O).T @ fl(dH)
        g[f'bo_{l}'] = fl(dH).sum(0)
        dO = (dH @ p[f'Wo_{l}'].T).reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        datt = dO @ V.transpose(0, 1, 3, 2)
        dV = att.transpose(0, 1, 3, 2) @ dO
        dz = att * (datt - (datt * att).sum(-1, keepdims=True)) / np.sqrt(dh)
        merge = lambda M: M.transpose(0, 2, 1, 3).reshape(B, T, d)
        dA = 0.
        for k, dM in zip('qkv', (merge(dz @ K), merge(dz.transpose(0, 1, 3, 2) @ Q), merge(dV))):
            g[f'W{k}_{l}'] = fl(A).T @ fl(dM)
            dA = dA + dM @ p[f'W{k}_{l}'].T
        dx, g[f'g1_{l}'], g[f'b1_{l}'] = ln_b(dA, c1)
        dH = dH + dx
    g['Win'] = fl(X).T @ fl(dH)
    g['bin'] = fl(dH).sum(0)
    g['pos'] = np.zeros_like(p['pos'])
    np.add.at(g['pos'], P.reshape(-1), fl(dH))
    return g


def smooth_ap5(s, y, n, tau=.5, tauk=.5, K=5):
    """Smooth AP@5: 1[s_j > s_i] -> sigmoid((s_j - s_i)/tau), 1[rank <= 5] -> sigmoid((5.5 - soft rank)/tauk).
    s, y (B,T); n (B,) = min(5, |truth|) over ALL shared hands. Returns the batch mean and its gradient wrt s."""
    B, T = s.shape
    D = sig((s[:, None, :] - s[:, :, None]) / tau) * (1. - np.eye(T))
    Dp = D * (1. - D) / tau
    r = 1. + D.sum(-1)
    rp = 1. + (D * y[:, None, :]).sum(-1)
    gk = sig((K + .5 - r) / tauk)
    val = (y * gk * rp / r).sum(-1) / n
    a = y * (-gk * (1. - gk) / tauk * rp / r - gk * rp / r ** 2) / (n[:, None] * B)
    c = y * gk / r / (n[:, None] * B)
    M = (a[:, :, None] + c[:, :, None] * y[:, None, :]) * Dp
    return val.mean(), M.sum(1) - M.sum(2)


def lambda_ap5(s, y, n, tau=.5):
    """LambdaLoss surrogate: logistic loss on each (relevant i, irrelevant j), weighted by |AP@5 change if i and j swapped ranks|."""
    B, T = s.shape
    o = np.argsort(-s, axis=1, kind='stable')
    rank = np.empty((B, T), int)
    np.put_along_axis(rank, o, np.broadcast_to(np.arange(T), (B, T)).copy(), 1)
    y5, k = np.take_along_axis(y, o[:, :5], 1), np.arange(1., 6.)
    base = (y5 * np.cumsum(y5, 1) / k).sum(1)
    oh = (rank[:, :, None] == np.arange(5)).astype(float)
    W = np.maximum(y5[:, None, None, :] * (1. - oh[:, :, None, :]), oh[:, None, :, :])
    delta = np.abs((W * np.cumsum(W, -1) / k).sum(-1) - base[:, None, None]) / n[:, None, None] * (y[:, :, None] * (1. - y[:, None, :]))
    z = (s[:, :, None] - s[:, None, :]) / tau
    w = delta * sig(-z) / tau / B
    return -(delta * np.logaddexp(0., -z)).sum() / B, w.sum(2) - w.sum(1)


def bce(s, y, n=None):
    q = sig(s)
    return float(np.mean(y * np.log(q + 1e-12) + (1 - y) * np.log(1 - q + 1e-12))), (y - q) / s.size


def ap5_exact(s, y, n):
    o = np.argsort(-s, axis=1, kind='stable')[:, :5]
    ys = np.take_along_axis(y, o, 1)
    return float(((ys * np.cumsum(ys, 1) / np.arange(1., 6.)).sum(1) / n).mean())


LOSS = {'sap': smooth_ap5, 'lam': lambda_ap5, 'bce': bce}


def train(X, P, off, y, n, loss='sap', d=16, h=2, L=1, m=32, steps=300, lr=3e-3, wd=1e-2, seed=0, lk=None):
    """Full-batch AdamW ascent on the chosen objective; decoupled decay on matrices, positions and the output vector."""
    rng = np.random.default_rng(seed)
    p = init(X.shape[2], d, h, L, X.shape[1], m, rng)
    mom = {k: np.zeros_like(v) for k, v in p.items()}
    vel = {k: np.zeros_like(v) for k, v in p.items()}
    for t in range(1, steps + 1):
        out, cache = forward(p, X, P, L, h)
        g = backward(p, cache, LOSS[loss](off + out, y, n, **(lk or {}))[1], L, h)
        for k in p:
            mom[k] = .9 * mom[k] + .1 * g[k]
            vel[k] = .999 * vel[k] + .001 * g[k] ** 2
            p[k] += lr * (mom[k] / (1 - .9 ** t)) / (np.sqrt(vel[k] / (1 - .999 ** t)) + 1e-8)
            if k[0] in 'Wpw':
                p[k] -= lr * wd * p[k]
    return p
