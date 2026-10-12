"""Shared beam search over conjunctions of thresholded atoms (used by S05 and S11)."""
import numpy as np

def build_literals(X, names, max_thr=20):
    cols, desc, spec = [], [], []
    for j, a in enumerate(names):
        x = X[:, j]
        u = np.unique(x)
        if len(u) < 2:
            continue
        if len(u) <= max_thr + 1:
            ge = u[1:]; le = u[:-1]
        else:
            q = np.unique(np.quantile(x, np.linspace(0.04, 0.96, max_thr)))
            ge = q; le = q
        for t in ge:
            cols.append((x >= t).astype(np.uint8)); desc.append(f"{a}>={t:.4g}"); spec.append((j, ">=", float(t)))
        for t in le:
            cols.append((x <= t).astype(np.uint8)); desc.append(f"{a}<={t:.4g}"); spec.append((j, "<=", float(t)))
    return np.stack(cols, axis=1), desc, spec

def apply_spec(X, lits, spec):
    m = np.ones(X.shape[0], dtype=bool)
    for i in lits:
        j, op, t = spec[i]
        m &= (X[:, j] >= t) if op == ">=" else (X[:, j] <= t)
    return m

def score(rec, fpr, obj):
    if obj == "J":
        return rec - fpr
    eps = float(obj)
    return np.where(fpr <= eps, rec, rec - 20.0 * (fpr - eps))

def beam_search(LIT, pos, neg, obj, beam=60, maxdepth=3, min_pos=5):
    """returns {depth: (lits, score, rec, fpr)} where each depth's entry is the best rule of depth <= that depth."""
    npos, nneg = pos.sum(), neg.sum()
    rec = LIT[pos].sum(0) / npos; fpr = LIT[neg].sum(0) / nneg
    sc = score(rec, fpr, obj)
    order = np.argsort(-sc)[:beam]
    items = [((int(i),), LIT[:, i].astype(bool), float(sc[i]), float(rec[i]), float(fpr[i])) for i in order]
    best = max(items, key=lambda t: t[2])
    out = {1: (best[0], best[2], best[3], best[4])}
    for depth in range(2, maxdepth + 1):
        cand = {}
        for lits, m, s0, r0, f0 in items:
            sub_pos = pos & m; sub_neg = neg & m
            rec = LIT[sub_pos].sum(0) / npos; fpr = LIT[sub_neg].sum(0) / nneg
            sc = score(rec, fpr, obj)
            sc[rec * npos < min_pos] = -9
            top = np.argsort(-sc)[:beam]
            for i in top:
                key = tuple(sorted(lits + (int(i),)))
                if int(i) in lits or key in cand or sc[i] <= s0 + 1e-9:
                    continue
                cand[key] = (key, None, float(sc[i]), float(rec[i]), float(fpr[i]))
        if not cand:
            out[depth] = out[depth - 1]
            continue
        items = sorted(cand.values(), key=lambda t: -t[2])[:beam]
        items = [(k, np.all(LIT[:, list(k)], axis=1), s, r, f) for k, _, s, r, f in items]
        b = max(items, key=lambda t: t[2])
        out[depth] = (b[0], b[2], b[3], b[4]) if b[2] > out[depth - 1][1] else out[depth - 1]
    return out
