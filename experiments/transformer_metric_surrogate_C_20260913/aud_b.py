

def cv(loss, off, seeds=(0, 1, 2, 3, 4), inits=(0, 1)):
    oof, ins = np.zeros((Q, T)), []
    for sd in seeds:
        perm = np.random.RandomState(sd).permutation(len(ug))
        fmap = dict(zip(ug[perm], np.arange(len(ug)) % 5))
        fo = np.array([fmap[t] for t in tab])
        for k in range(5):
            tr, te = fo != k, fo == k
            Ft = X[tr].reshape(-1, X.shape[2])
            mu, sdv = Ft.mean(0), Ft.std(0) + 1e-9
            Xtr, Xte = np.clip((X[tr] - mu) / sdv, -5, 5), np.clip((X[te] - mu) / sdv, -5, 5)
            for ini in inits:
                prm = M.train(Xtr, P[tr], off[tr], Y[tr], NQ[tr], loss=loss, seed=1000 * sd + 10 * k + ini, **CFG)
                te_m = off[te] + M.forward(prm, Xte, P[te], CFG['L'], CFG['h'])[0]
                oof[te] += te_m / (len(seeds) * len(inits))
                if sd == 0 and ini == 0:
                    fit = off[tr] + M.forward(prm, Xtr, P[tr], CFG['L'], CFG['h'])[0]
                    ins.append(tuple(round(M.ap5_exact(*a), 4) for a in ((fit, Y[tr], NQ[tr]), (off[tr], Y[tr], NQ[tr]), (te_m, Y[te], NQ[te]), (off[te], Y[te], NQ[te]))))
    return oof, ins


def compare(name, arr):
    out = {}
    for bn, base in (('B0', B0), ('typed_S_star', decS)):
        c = evaluate(d, {'incumbent': base, name: arr})['comparisons'][name]
        r = json.loads(json.dumps({k: c[k] for k in ('mean_delta', 'ci95', 'by_family', 'halves', 'pairs_better', 'pairs_worse', 'gate_pass')}, default=lambda o: o.item()))
        out[bn] = r
        print(f'  {name} vs {bn}', round(r['mean_delta'], 6), [round(v, 6) for v in r['ci95']], {k[:5]: round(v, 4) for k, v in r['by_family'].items()},
              {k: round(v, 4) for k, v in r['halves'].items()}, r['pairs_better'], r['pairs_worse'], 'gate', r['gate_pass'], flush=True)
    return out


rep = evaluate(d, {'incumbent': B0, 'typed_S_star': decS})
print('replay typed_S_star vs B0', round(rep['comparisons']['typed_S_star']['mean_delta'], 6), '| sets', json.dumps(rep['sets'], default=lambda o: o.item())[:500], flush=True)
results = {'looks': LOOKS, 'cfg': CFG, 'tag': TAG}
for look in LOOKS:
    loss, offn = look.split('_')
    t0 = time.time()
    oof, ins = cv(loss, OFF[offn])
    print(look, 'seconds', round(time.time() - t0, 1), '| seed-0 folds (train fit, train offset, test fit, test offset) AP@5 on tokens:', ins, flush=True)
    results[look] = compare(f'tf_{look}', F6.to_scores(fs, x, oof.reshape(-1)))
    np.save(S + f'oof_{TAG}_{look}.npy', oof)
results['protected_unchanged'] = check_protected() == prot
json.dump(results, open(S + f'aud109_{TAG}.json', 'w', encoding='utf-8'), indent=2)
print('protected unchanged', results['protected_unchanged'], flush=True)
