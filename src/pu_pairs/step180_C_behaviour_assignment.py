"""Step 180 (Claude): how much BehaviorMAP is available from a better family assignment, with the risk ranking held fixed.
The behaviour column has been byte-identical to the v13 model of 2026-09-10 through the whole climb (OBS 19.130) and carries 10 % of
the metric. BehaviorMAP = mean over k in {DT,SP,CI} of AP(truth==k, risk * 1[assigned==k]), so a pair assigned to k enters k's ranking
at its risk position: a wrong assignment in the high-risk head is a false positive that damages every true member below it, and `none`
is a legal label that withholds a pair from all three rankings. Rules: R0 argmax (production style), R1 `none` below a posterior
threshold, R2 `none` outside the top-K risk, R3 MAP-aware greedy (include only when the posterior clears c x the list's running
precision; routing by argmax posterior or by posterior/precision ratio). Labels: development_labels.csv, disjoint from the 162-query
evidence audit that failed publicly today. Everything table-grouped OOF; honest cross-half selection; registration
experiments/behaviour_map_20260916/registration180.json. No uploads. Usage: step180_C_behaviour_assignment.py REGSHA"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
import sys, json, time, glob
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
import numpy as np, polars as pl

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/behaviour_map_20260916'
REG = EXP / 'registration180.json'
LAB = CV / 'experiments/crossphase_20260911/lab_on_dev.parquet'
EVF = CV / 'experiments/eval_frame_20260911/eval_frame.parquet'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
CLS = ['none'] + FAMS                       # 0 none, 1 DT, 2 SP, 3 CI
NFOLD = 5
T_GRID = [.02, .05, .1, .2, .3, .5, .7]
K_GRID = [250, 500, 1000, 2500, 5000, 10000, 25000]
C_GRID = [0., .25, .5, .75, 1., 1.25]


def stable_ap(truth, score):
    """The reference implementation (src/robust_pair_experiment.py): rows must already be in pair_id order."""
    truth = np.asarray(truth, dtype=np.float64)
    order = np.argsort(-np.asarray(score), kind='mergesort')
    pw = truth[order]
    if not pw.sum():
        return 0.
    precision = np.cumsum(pw) / np.arange(1, len(pw) + 1)
    return float(np.sum(precision * pw) / pw.sum())


def behaviour_map(y, risk, assign, sel=None):
    """mean over the three families of AP(truth==k, risk * 1[assign==k]); sel restricts the population."""
    if sel is not None:
        y, risk, assign = y[sel], risk[sel], assign[sel]
    out = []
    for k in (1, 2, 3):
        if not (y == k).any():
            continue
        out.append(stable_ap(y == k, np.where(assign == k, risk, 0.)))
    return float(np.mean(out)), [round(x, 5) for x in out]


def members():
    """32 members: 24 LightGBM multiclass (6 settings x 4 seeds) + 4 multinomial logistic + 4 HistGradientBoosting."""
    P = [dict(num_leaves=7, learning_rate=.05, n_estimators=200, min_child_samples=10, colsample_bytree=.3, subsample=.8, subsample_freq=1),
         dict(num_leaves=15, learning_rate=.05, n_estimators=150, min_child_samples=20, colsample_bytree=.5),
         dict(num_leaves=3, learning_rate=.1, n_estimators=250, min_child_samples=5, colsample_bytree=.2),
         dict(num_leaves=7, learning_rate=.03, n_estimators=400, min_child_samples=15, colsample_bytree=.4, reg_lambda=5.),
         dict(num_leaves=31, learning_rate=.05, n_estimators=120, min_child_samples=30, colsample_bytree=.7),
         dict(num_leaves=7, learning_rate=.08, n_estimators=200, min_child_samples=10, colsample_bytree=.3, reg_alpha=1.)]
    out = [('lgbm', dict(p, random_state=s)) for p in P for s in (7329, 11, 202, 4242)]
    out += [('logit', dict(C=c, sub=sub)) for c in (1., .1) for sub in ('all', 'base')]
    out += [('hgb', dict(random_state=s)) for s in (7329, 11, 202, 4242)]
    return out


def fit_predict(kind, par, Xtr, ytr, Xte_list, multiclass):
    """One member; returns a list of probability matrices, one per scoring frame."""
    if kind == 'lgbm':
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(objective='multiclass' if multiclass else 'binary', verbosity=-1, n_jobs=2, **par)
        m.fit(Xtr, ytr)
        return [m.predict_proba(X) for X in Xte_list]
    if kind == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_leaf_nodes=7, max_iter=150, learning_rate=.06, l2_regularization=1.,
                                           min_samples_leaf=10, random_state=par['random_state'])
        m.fit(Xtr, ytr)
        return [m.predict_proba(X) for X in Xte_list]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols = COLSETS[par['sub']]
    sc = StandardScaler().fit(Xtr[:, cols])
    m = LogisticRegression(C=par['C'], max_iter=2000, n_jobs=1)
    m.fit(sc.transform(Xtr[:, cols]), ytr)
    return [m.predict_proba(sc.transform(X[:, cols])) for X in Xte_list]


COLSETS = {}


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[1]
    from evidence_aug.step165_C_e2e_audit import sha
    from evidence_aug.paths import check_protected
    from evidence_aug.step5_evaluate import half
    assert sha(REG) == reg_sha, 'registration hash mismatch'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    dest = EXP / 'results180.json'
    assert not dest.exists(), 'preserve a completed read'
    prot = check_protected()

    lab = pl.read_csv(POKER / 'data/development_labels.csv')
    d = pl.read_parquet(LAB).sort('pair_id')
    assert d.height == 1860
    feats = [c for c in d.columns if c not in ('pair_id', 'table_id') and d.schema[c].is_numeric()]
    evcols = set(pl.scan_parquet(EVF).collect_schema().names())
    assert set(feats) <= evcols, 'features must exist on the evaluation frame too'
    COLSETS['all'] = np.arange(len(feats))
    COLSETS['base'] = np.array([i for i, c in enumerate(feats) if not c.startswith('ca_')])
    d = d.join(lab.select('pair_id', 'behavior_family', 'label'), on='pair_id', validate='1:1')
    y = np.array([CLS.index(f) if f in CLS else 0 for f in d['behavior_family']])
    assert (y > 0).sum() == 372 and int(d['label'].sum()) == 372
    X = np.nan_to_num(d.select(feats).to_numpy().astype(np.float32), posinf=0, neginf=0)
    tab = d['table_id'].to_numpy()
    print('rows', X.shape, 'families', np.bincount(y).tolist(), 'tables', len(set(tab)), flush=True)

    # ---- P_wide sensitivity population (unlabelled treated as negative)
    wf = sorted(glob.glob(str(CV / 'experiments/pu_pairs_20260911/fold1/batch_*-pairs.parquet'))) + \
         sorted(glob.glob(str(CV / 'experiments/pu_pairs_20260911/fold2/batch_*-pairs.parquet')))
    W = pl.read_parquet(wf, columns=['pair_id', 'table_id'] + feats).unique(subset='pair_id', keep='first').sort('pair_id')
    Wy = W.select('pair_id').join(lab.select('pair_id', 'behavior_family'), on='pair_id', how='left', maintain_order='left')
    wy = np.array([CLS.index(f) if f in CLS else 0 for f in Wy['behavior_family'].fill_null('none')])
    XW = np.nan_to_num(W.select(feats).to_numpy().astype(np.float32), posinf=0, neginf=0)
    wtab = W['table_id'].to_numpy()
    print('P_wide', XW.shape, 'positives', np.bincount(wy).tolist(), flush=True)

    # ---- table-grouped OOF: 4-class posteriors and binary risk, 32 members each
    from sklearn.model_selection import GroupKFold
    gkf = list(GroupKFold(n_splits=NFOLD).split(X, y, groups=tab))
    MEM = members()
    Pf = np.zeros((len(y), 4)); Rf = np.zeros(len(y))
    PW = np.zeros((NFOLD, len(wy), 4)); RW = np.zeros((NFOLD, len(wy)))
    for f, (tr, te) in enumerate(gkf):
        acc_p = np.zeros((len(te), 4)); acc_r = np.zeros(len(te))
        accw_p = np.zeros((len(wy), 4)); accw_r = np.zeros(len(wy))
        for kind, par in MEM:
            pr = fit_predict(kind, par, X[tr], y[tr], [X[te], XW], True)
            acc_p += pr[0]; accw_p += pr[1]
            br = fit_predict(kind, par, X[tr], (y[tr] > 0).astype(int), [X[te], XW], False)
            acc_r += br[0][:, 1]; accw_r += br[1][:, 1]
        Pf[te] = acc_p / len(MEM); Rf[te] = acc_r / len(MEM)
        PW[f] = accw_p / len(MEM); RW[f] = accw_r / len(MEM)
        print('fold', f, 'done', round(time.perf_counter() - t0), 's', flush=True)
    held = {t: f for f, (tr, te) in enumerate(gkf) for t in set(tab[te])}
    wf_idx = np.array([held.get(t, -1) for t in wtab])
    Pw = np.where(wf_idx[:, None] >= 0, PW[np.clip(wf_idx, 0, None), np.arange(len(wy))], PW.mean(0))
    Rw = np.where(wf_idx >= 0, RW[np.clip(wf_idx, 0, None), np.arange(len(wy))], RW.mean(0))
    np.savez_compressed(EXP / 'oof180.npz', P=Pf, R=Rf, y=y, Pw=Pw, Rw=Rw, wy=wy)

    # ---- assignment rules
    def R0(P):
        return 1 + np.argmax(P[:, 1:], axis=1)

    def R1(P, t):
        a = R0(P); return np.where(1 - P[:, 0] >= t, a, 0)

    def R2(P, R, K):
        a = R0(P); keep = np.zeros(len(a), bool); keep[np.argsort(-R, kind='mergesort')[:K]] = True
        return np.where(keep, a, 0)

    def R3(P, R, c, route):
        order = np.argsort(-R, kind='mergesort')
        E = np.ones(4); N = np.ones(4); out = np.zeros(len(R), dtype=int)
        for i in order:
            p = P[i, 1:]
            prec = E[1:] / N[1:]
            k = 1 + (int(np.argmax(p)) if route == 'p' else int(np.argmax(p / prec)))
            if p[k - 1] >= c * (E[k] / N[k]):
                out[i] = k; E[k] += p[k - 1]; N[k] += 1
        return out

    rules = {'R0': R0(Pf)}
    rules.update({f'R1_t{t}': R1(Pf, t) for t in T_GRID})
    rules.update({f'R2_K{K}': R2(Pf, Rf, K) for K in K_GRID})
    rules.update({f'R3_c{c}_{r}': R3(Pf, Rf, c, r) for c in C_GRID for r in ('p', 'ratio')})
    gate_r3 = int(np.abs(rules['R3_c0.0_p'] - rules['R0']).max()); assert gate_r3 == 0, 'R3 at c=0 must reduce to argmax'

    hv = np.array([half(t) for t in tab])
    names = list(rules)
    full = {n: behaviour_map(y, Rf, rules[n]) for n in names}
    byhalf = {h: {n: behaviour_map(y, Rf, rules[n], hv == h) for n in names} for h in (0, 1)}
    R = lambda x: round(float(x), 5)
    heldres = {}
    for h in (0, 1):
        sc = {n: byhalf[h][n][0] for n in names}
        pick = max(names, key=lambda n: (sc[n], n == 'R0'))
        o = 1 - h
        heldres[h] = {'selected_on_half': pick, 'selection_half_delta': R(sc[pick] - sc['R0']),
                      'held_half_delta': R(byhalf[o][pick][0] - byhalf[o]['R0'][0]), 'held_half_n': int((hv == o).sum())}
    G = sum(heldres[h]['held_half_delta'] * heldres[h]['held_half_n'] for h in (0, 1)) / len(y)
    fin = max(names, key=lambda n: (full[n][0], n == 'R0'))

    wrules = {'R0': R0(Pw)}
    wrules.update({f'R1_t{t}': R1(Pw, t) for t in T_GRID})
    wrules.update({f'R2_K{K}': R2(Pw, Rw, K) for K in K_GRID})
    wrules.update({f'R3_c{c}_{r}': R3(Pw, Rw, c, r) for c in C_GRID for r in ('p', 'ratio')})
    wide = {n: behaviour_map(wy, Rw, wrules[n])[0] for n in names}

    out = {'registration_sha256': reg_sha, 'author': 'claude',
           'gates': {'R3_c0_equals_argmax': gate_r3, 'features': len(feats), 'features_subset_of_eval_frame': True,
                     'risk_oof_pair_AP_P_lab': R(stable_ap(y > 0, Rf)), 'risk_oof_pair_AP_P_wide': R(stable_ap(wy > 0, Rw)),
                     'posterior_accuracy_on_positives': R(float((np.argmax(Pf[y > 0, 1:], axis=1) + 1 == y[y > 0]).mean()))},
           'incumbent_rule': 'R0', 'incumbent_behaviour_map': R(full['R0'][0]), 'incumbent_by_family': full['R0'][1],
           'cross_half': heldres, 'G_honest': R(G), 'final_rule': fin, 'final_behaviour_map': R(full[fin][0]),
           'final_vs_R0': R(full[fin][0] - full['R0'][0]), 'final_by_family': full[fin][1],
           'final_both_halves': [R(byhalf[h][fin][0] - byhalf[h]['R0'][0]) for h in (0, 1)],
           'P_wide_R0': R(wide['R0']), 'P_wide_final': R(wide[fin]), 'P_wide_delta': R(wide[fin] - wide['R0']),
           'grid': {n: {'map': R(full[n][0]), 'delta': R(full[n][0] - full['R0'][0]),
                        'half0': R(byhalf[0][n][0] - byhalf[0]['R0'][0]), 'half1': R(byhalf[1][n][0] - byhalf[1]['R0'][0]),
                        'wide_delta': R(wide[n] - wide['R0']), 'assigned': int((rules[n] > 0).sum())} for n in names},
           'ensemble': {'members': len(MEM), 'form': 'mean of 32 class-probability vectors: 24 LightGBM multiclass (6 settings x 4 seeds), 4 multinomial logistic (2 l2 x 2 feature subsets), 4 HistGradientBoosting seeds; same list with a binary objective for risk_oof'},
           'wall_seconds': round(time.perf_counter() - t0, 1)}
    proceed = bool(G > 0 and fin != 'R0' and min(out['final_both_halves']) > 0 and out['P_wide_delta'] > 0)
    out['ER_rule'] = 'G > 0 AND final != R0 AND both halves positive AND P_wide sign holds'
    out['proceed'] = proceed
    assert check_protected() == prot
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('G', out['G_honest'], 'final', fin, out['final_vs_R0'], 'halves', out['final_both_halves'],
          'wide', out['P_wide_delta'], 'PROCEED', proceed, flush=True)


if __name__ == '__main__':
    main()
