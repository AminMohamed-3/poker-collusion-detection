

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


def train(X, P, off, y, n, loss='sap', d=16, h=2, L=1, m=32, steps=300, lr=3e-3, wd=1e-2, seed=0):
    """Full-batch AdamW ascent on the chosen objective; decoupled decay on matrices, positions and the output vector."""
    rng = np.random.default_rng(seed)
    p = init(X.shape[2], d, h, L, X.shape[1], m, rng)
    mom = {k: np.zeros_like(v) for k, v in p.items()}
    vel = {k: np.zeros_like(v) for k, v in p.items()}
    for t in range(1, steps + 1):
        out, cache = forward(p, X, P, L, h)
        g = backward(p, cache, LOSS[loss](off + out, y, n)[1], L, h)
        for k in p:
            mom[k] = .9 * mom[k] + .1 * g[k]
            vel[k] = .999 * vel[k] + .001 * g[k] ** 2
            p[k] += lr * (mom[k] / (1 - .9 ** t)) / (np.sqrt(vel[k] / (1 - .999 ** t)) + 1e-8)
            if k[0] in 'Wpw':
                p[k] -= lr * wd * p[k]
    return p
