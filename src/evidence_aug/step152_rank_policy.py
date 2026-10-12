"""Step 152 (Claude): Plackett-Luce ranking policy over the top-20 evidence candidates of a pair, trained as a correction over the
incumbent decoder score. Arms: G1 = group-relative (GRPO-style) policy gradient with a dense ranking reward, G0 = the same with
AP@5 only, L = supervised listwise softmax on the listed hands (control). numpy only; this module reads no data."""
import numpy as np

W_POS = np.r_[1 / np.arange(1, 6), .05 / np.arange(6, 21)]
LAMBDA = {'pair': .25, 'pos': .25, 'cov': .25}
CONFIG = {'steps': 200, 'G': 32, 'lr': .02, 'beta_kl_first_position': .02, 'weight_decay': 1e-3, 'hidden': 8, 'init_W_sd': .3,
          'adam': [.9, .999, 1e-8], 'advantage_eps': 1e-4, 'score_clip': 50.}


def softmax(s):
    z = np.exp(s - s.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def sample_orders(s, G, rng):
    """Full orderings from PL(s) by Gumbel sorting: (Q, G, N) candidate indices, best first."""
    return np.argsort(-(s[:, None, :] + rng.gumbel(size=(s.shape[0], G, s.shape[1]))), axis=2, kind='stable')


def pl_logprob_grad(s, perm):
    """log PL probability of each full ordering and its gradient with respect to s. s (Q, N), perm (Q, G, N)."""
    sp = np.take_along_axis(np.broadcast_to(s[:, None, :], perm.shape), perm, axis=2)
    e = np.exp(sp - sp.max(axis=2, keepdims=True))
    C = np.cumsum(e[:, :, ::-1], axis=2)[:, :, ::-1]
    logp = (np.log(e) - np.log(C)).sum(axis=2)
    gp = 1.0 - e * np.cumsum(1.0 / C, axis=2)
    g = np.empty_like(gp)
    np.put_along_axis(g, perm, gp, axis=2)
    return logp, g


def rewards(perm, rel, n_norm, dense):
    """rel (Q, N) 1 = listed hand among the candidates; n_norm (Q,) = min(5, number of listed hands of the pair, all hands).
    AP@5 is the competition definition; dense adds pairwise order, position-weighted recall and top-5 coverage."""
    Q, G, N = perm.shape
    r = np.take_along_axis(np.broadcast_to(rel[:, None, :], perm.shape), perm, axis=2).astype(float)
    k = np.maximum(n_norm, 1)[:, None]
    ok = (n_norm > 0)[:, None]
    ap = np.where(ok, (r[:, :, :5] * np.cumsum(r[:, :, :5], axis=2) / np.arange(1, 6)).sum(axis=2) / k, 0.)
    if not dense:
        return ap
    nl = rel.sum(axis=1)
    npair = (nl * (N - nl))[:, None]
    pair = np.where(npair > 0, ((1 - r) * (np.cumsum(r, axis=2) - r)).sum(axis=2) / np.maximum(npair, 1), 0.)
    ideal = np.r_[0., np.cumsum(W_POS)][nl.astype(int)][:, None]
    pos = np.where(ideal > 0, (r * W_POS[:N]).sum(axis=2) / np.maximum(ideal, 1e-12), 0.)
    cov = np.where(ok, r[:, :, :5].sum(axis=2) / k, 0.)
    return ap + LAMBDA['pair'] * pair + LAMBDA['pos'] * pos + LAMBDA['cov'] * cov


class Policy:
    """s = w0 * base + X c + tanh(X W' + b) v; starts equal to the base score (c = v = 0, w0 = 1)."""

    def __init__(self, nx, rng, cfg=CONFIG):
        self.p = {'w0': np.array(1.), 'c': np.zeros(nx), 'W': rng.normal(0, cfg['init_W_sd'], (cfg['hidden'], nx)),
                  'b': np.zeros(cfg['hidden']), 'v': np.zeros(cfg['hidden'])}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.u = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.cfg = cfg

    def forward(self, base, X):
        H = np.tanh(X @ self.p['W'].T + self.p['b'])
        s = self.p['w0'] * base + X @ self.p['c'] + H @ self.p['v']
        self.cache = (base, X, H)
        return np.clip(s, -self.cfg['score_clip'], self.cfg['score_clip'])

    def backward(self, gs):
        base, X, H = self.cache
        gZ = gs[..., None] * self.p['v'] * (1 - H ** 2)
        return {'w0': np.array((gs * base).sum()), 'c': np.einsum('qn,qnk->k', gs, X), 'v': np.einsum('qn,qnh->h', gs, H),
                'W': np.einsum('qnh,qnk->hk', gZ, X), 'b': gZ.sum(axis=(0, 1))}

    def ascend(self, grads):
        b1, b2, eps = self.cfg['adam']
        self.t += 1
        for k, g in grads.items():
            if k in ('c', 'W', 'v'):
                g = g - self.cfg['weight_decay'] * self.p[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.u[k] = b2 * self.u[k] + (1 - b2) * g * g
            self.p[k] = self.p[k] + self.cfg['lr'] * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.u[k] / (1 - b2 ** self.t)) + eps)


def objective_grad(arm, s, base, rel, n_norm, rng, cfg=CONFIG):
    """Ascent direction on s (Q, N) for one step, plus the logged training objective."""
    Q = s.shape[0]
    if arm in ('G1', 'G0'):
        perm = sample_orders(s, cfg['G'], rng)
        R = rewards(perm, rel, n_norm, dense=arm == 'G1')
        sd = R.std(axis=1, keepdims=True)
        A = np.where(sd > 1e-9, (R - R.mean(axis=1, keepdims=True)) / (sd + cfg['advantage_eps']), 0.)
        _, g = pl_logprob_grad(s, perm)
        gs = (A[:, :, None] * g).mean(axis=1) / Q
        logged = float(rewards(perm, rel, n_norm, dense=False).mean())
    else:
        nl = rel.sum(axis=1, keepdims=True)
        sm = softmax(s)
        gs = np.where(nl > 0, rel / np.maximum(nl, 1) - sm, 0.) / Q
        logged = float((np.where(rel > 0, np.log(np.maximum(sm, 1e-300)), 0.).sum(axis=1) / np.maximum(nl[:, 0], 1)).mean())
    p, pr = softmax(s), softmax(base)
    lp, lpr = np.log(np.maximum(p, 1e-300)), np.log(np.maximum(pr, 1e-300))
    kl = (p * (lp - lpr)).sum(axis=1, keepdims=True)
    gs = gs - cfg['beta_kl_first_position'] * p * (lp - lpr - kl) / Q
    return gs, logged, float(kl.mean())


def train(arm, base, X, rel, n_norm, seed, cfg=CONFIG, log_every=50):
    rng = np.random.default_rng(seed)
    pol = Policy(X.shape[2], rng, cfg)
    trace = []
    for step in range(cfg['steps']):
        s = pol.forward(base, X)
        gs, logged, kl = objective_grad(arm, s, base, rel, n_norm, rng, cfg)
        pol.ascend(pol.backward(gs))
        if step % log_every == 0 or step == cfg['steps'] - 1:
            trace.append([step, round(logged, 5), round(kl, 6)])
    return pol, trace


def greedy_ap(s, rel, n_norm):
    perm = np.argsort(-s, axis=1, kind='stable')[:, None, :]
    return rewards(perm, rel, n_norm, dense=False)[:, 0]
