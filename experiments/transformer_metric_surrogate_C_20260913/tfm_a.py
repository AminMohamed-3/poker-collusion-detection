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
