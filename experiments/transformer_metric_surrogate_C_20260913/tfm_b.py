

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
