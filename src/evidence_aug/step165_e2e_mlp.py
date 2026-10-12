"""Step 165 (Claude): numpy end-to-end evidence network (no torch on this PC). One trunk over every hand of a pair feeds
(a) three family stage-1 heads trained with a pair-listwise loss over all hands and (b) a stage-2 head over the pair's top-20 candidates
that sees the medium layer h, h minus its top-20 pair mean, the stage-1 score and the top-20-only fields (contestable state, prefix
strength). Mode 'joint' back-propagates the stage-2 loss into the trunk and stage-1 heads (end to end); mode 'two_stage' trains the
trunk on the stage-1 loss first, freezes it, then trains the stage-2 head (the control). Full-batch AdamW with early stopping on
validation pairs held out (by table) from the training fold."""
import numpy as np

CONFIG = {'h1': 64, 'medium': 16, 'h2': 32, 'epochs_max': 400, 'patience': 40, 'lr': 1e-3, 'wd': 1e-2, 'lam_list2': .5, 'w_stage1': 1.,
          'clip': 6., 'val_share_tables': .2}


def relu(x):
    return np.maximum(x, 0)


class Batch:
    """Index structure: X (N x D) all hands of the training pairs, sorted by pair; fam (N,) family index; y (N,) listed;
    first (G,) start of each pair block; T (Nt,) row index of top-20 rows (sorted by pair), C (Nt x Dc) top-20-only fields,
    tfirst (G2,) start of each top-20 pair block inside T."""

    def __init__(self, X, fam, y, pair, T, C, dtype=np.float32):
        self.X, self.fam, self.y = X.astype(dtype), fam.astype(np.int64), y.astype(dtype)
        _, self.first = np.unique(pair, return_index=True)
        self.gid = np.repeat(np.arange(len(self.first)), np.diff(np.r_[self.first, len(pair)]))
        self.T, self.C = T, C.astype(dtype)
        tp = pair[T]
        _, self.tfirst = np.unique(tp, return_index=True)
        self.tgid = np.repeat(np.arange(len(self.tfirst)), np.diff(np.r_[self.tfirst, len(T)]))
        self.npos = np.add.reduceat(self.y, self.first)
        self.tnpos = np.add.reduceat(self.y[T], self.tfirst)
        self.tcount = np.diff(np.r_[self.tfirst, len(T)]).astype(dtype)


def init(D, Dc, seed, dtype=np.float32):
    c, r = CONFIG, np.random.default_rng(seed)
    he = lambda a, b: (r.standard_normal((a, b)) * np.sqrt(2. / a)).astype(dtype)
    return {'W1': he(D, c['h1']), 'b1': np.zeros(c['h1'], dtype), 'W2': he(c['h1'], c['medium']), 'b2': np.zeros(c['medium'], dtype),
            'U': (r.standard_normal((c['medium'], 3)) * .1).astype(dtype), 'c': np.zeros(3, dtype),
            'V1': he(2 * c['medium'] + 1 + Dc, c['h2']), 'e1': np.zeros(c['h2'], dtype),
            'v2': (r.standard_normal((c['h2'], 1)) * .1).astype(dtype), 'e2': np.full(1, -1., dtype)}


def group_softmax(s, first, gid):
    m = np.maximum.reduceat(s, first)[gid]
    e = np.exp(s - m)
    z = np.add.reduceat(e, first)[gid]
    return e / z, np.log(z) + m


def forward(P, B, with_stage2=True):
    a1 = B.X @ P['W1'] + P['b1']; z1 = relu(a1)
    a2 = z1 @ P['W2'] + P['b2']; h = relu(a2)
    S = h @ P['U'] + P['c']
    s1 = S[np.arange(len(h)), B.fam]
    cache = {'a1': a1, 'z1': z1, 'a2': a2, 'h': h, 'S': S, 's1': s1}
    if with_stage2:
        ht = h[B.T]
        hbar = np.add.reduceat(ht, B.tfirst, axis=0) / B.tcount[:, None]
        u = np.concatenate([ht, ht - hbar[B.tgid], s1[B.T, None], B.C], axis=1)
        b = u @ P['V1'] + P['e1']; r = relu(b)
        o = (r @ P['v2'] + P['e2'])[:, 0]
        cache.update(u=u, b=b, r=r, o=o)
    return cache


def losses(P, B, cache, mode):
    c = CONFIG
    sm, lz = group_softmax(cache['s1'], B.first, B.gid)
    G = len(B.first)
    L1 = float(-np.sum(B.y * (cache['s1'] - lz) / B.npos[B.gid]) / G)
    ds1 = (sm - B.y / B.npos[B.gid]) / G * c['w_stage1']
    out = {'L1': L1}
    grads = {}
    dh = np.zeros_like(cache['h'])
    if 'o' in cache:
        o, yt = cache['o'], B.y[B.T]
        p = 1 / (1 + np.exp(-o))
        Nt = len(o)
        bce = float(np.mean(np.logaddexp(0, o) - yt * o))
        ok = B.tnpos > 0
        tsm, tlz = group_softmax(o, B.tfirst, B.tgid)
        G2 = int(ok.sum())
        w = ok[B.tgid] / np.maximum(B.tnpos[B.tgid], 1)
        lst = float(-np.sum(w * yt * (o - tlz)) / G2)
        out.update(BCE=bce, L2list=lst)
        if mode not in ('joint', 'stage2'):
            return out, grads
        do = (p - yt) / Nt + c['lam_list2'] * (ok[B.tgid] * tsm - w * yt) / G2
        grads['v2'] = cache['r'].T @ do[:, None]; grads['e2'] = np.array([do.sum()], dtype=o.dtype)
        dr = do[:, None] * P['v2'].T
        db = dr * (cache['b'] > 0)
        grads['V1'] = cache['u'].T @ db; grads['e1'] = db.sum(0)
        if mode == 'joint':
            du = db @ P['V1'].T
            M = c['medium']
            dD = du[:, M:2 * M]
            dD = dD - (np.add.reduceat(dD, B.tfirst, axis=0) / B.tcount[:, None])[B.tgid]
            np.add.at(dh, B.T, du[:, :M] + dD)
            ds1 = ds1.copy(); np.add.at(ds1, B.T, du[:, 2 * M])
    if mode in ('joint', 'stage1'):
        dS = np.zeros_like(cache['S']); dS[np.arange(len(dS)), B.fam] = ds1
        grads['U'] = cache['h'].T @ dS; grads['c'] = dS.sum(0)
        dh = dh + dS @ P['U'].T
        da2 = dh * (cache['a2'] > 0)
        grads['W2'] = cache['z1'].T @ da2; grads['b2'] = da2.sum(0)
        dz1 = da2 @ P['W2'].T
        da1 = dz1 * (cache['a1'] > 0)
        grads['W1'] = B.X.T @ da1; grads['b1'] = da1.sum(0)
    return out, grads


def objective(out, mode):
    c = CONFIG
    s1 = c['w_stage1'] * out['L1'] if mode in ('joint', 'stage1') else 0.
    s2 = out['BCE'] + c['lam_list2'] * out['L2list'] if mode in ('joint', 'stage2') else 0.
    return s1 + s2


def adam_phase(P, B, mode, keys, Bv):
    """AdamW on keys; early stopping on the validation batch Bv (pairs held out from the training fold): the stage-2 objective for
    'joint'/'stage2', the stage-1 loss for 'stage1'. Returns the trace (first/best train objective, best val, best epoch)."""
    c = CONFIG
    m = {k: np.zeros_like(P[k]) for k in keys}; v = {k: np.zeros_like(P[k]) for k in keys}
    crit = 'stage1' if mode == 'stage1' else 'stage2'
    best, best_ep, best_P, first, bad = np.inf, 0, {k: P[k].copy() for k in keys}, None, 0
    for t in range(1, c['epochs_max'] + 1):
        cache = forward(P, B, with_stage2=mode != 'stage1')
        out, g = losses(P, B, cache, mode)
        if first is None:
            first = objective(out, mode)
        for k in keys:
            m[k] = .9 * m[k] + .1 * g[k]; v[k] = .999 * v[k] + .001 * g[k] ** 2
            step = (m[k] / (1 - .9 ** t)) / (np.sqrt(v[k] / (1 - .999 ** t)) + 1e-8)
            decay = c['wd'] * P[k] if P[k].ndim == 2 else 0.
            P[k] = (P[k] - c['lr'] * (step + decay)).astype(P[k].dtype)
        vout, _ = losses(P, Bv, forward(P, Bv, with_stage2=crit == 'stage2'), 'none')
        val = vout['L1'] if crit == 'stage1' else vout['BCE'] + c['lam_list2'] * vout['L2list']
        if val < best - 1e-6:
            best, best_ep, bad = val, t, 0
            best_P = {k: P[k].copy() for k in keys}
        else:
            bad += 1
            if bad >= c['patience']:
                break
    for k in keys:
        P[k] = best_P[k]
    return [float(first), float(best), float(best_ep)]


def train(B, mode, seed, Bv):
    P = init(B.X.shape[1], B.C.shape[1], seed, B.X.dtype)
    trunk = ['W1', 'b1', 'W2', 'b2', 'U', 'c']; head = ['V1', 'e1', 'v2', 'e2']
    if mode == 'joint':
        return P, adam_phase(P, B, 'joint', trunk + head, Bv)
    return P, adam_phase(P, B, 'stage1', trunk, Bv) + adam_phase(P, B, 'stage2', head, Bv)


def predict(P, B):
    """Stage-2 probability for the top-20 rows and the three family stage-1 scores (aligned outputs) for every row."""
    cache = forward(P, B, with_stage2=True)
    return 1 / (1 + np.exp(-cache['o'])), cache['S']


def gradcheck(seed=0):
    r = np.random.default_rng(seed)
    pair = np.repeat(np.array([f'p{i}' for i in range(5)]), 12)
    N = len(pair); D, Dc = 7, 4
    X = r.standard_normal((N, D)); fam = r.integers(0, 3, N); y = (r.random(N) < .3).astype(float)
    T = np.concatenate([np.arange(i * 12, i * 12 + 6) for i in range(5)])
    y[T[::5]] = 1
    C = r.standard_normal((len(T), Dc))
    B = Batch(X, fam, y, pair, T, C, dtype=np.float64)
    P = init(D, Dc, seed, np.float64)
    for k in P:
        P[k] = P[k] + r.standard_normal(P[k].shape) * .05
    worst = {}
    for mode in ('joint', 'stage1', 'stage2'):
        out, g = losses(P, B, forward(P, B, mode != 'stage1'), mode)
        for k, gk in g.items():
            num = np.zeros_like(P[k]); flat = P[k].reshape(-1)
            for j in range(min(flat.size, 25)):
                old = flat[j]
                flat[j] = old + 1e-6; fp = objective(losses(P, B, forward(P, B, mode != 'stage1'), mode)[0], mode)
                flat[j] = old - 1e-6; fm = objective(losses(P, B, forward(P, B, mode != 'stage1'), mode)[0], mode)
                flat[j] = old
                num.reshape(-1)[j] = (fp - fm) / 2e-6
            j = min(P[k].size, 25)
            worst[f'{mode}/{k}'] = float(np.max(np.abs(num.reshape(-1)[:j] - gk.reshape(-1)[:j])))
    return worst
