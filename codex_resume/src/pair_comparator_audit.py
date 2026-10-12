"""Frozen direct-comparison screen; stronger relational input than score differences.

Compared hands always belong to one player pair, and splits group whole pools.
This creates comparisons, not new independent labelled data. The comparison
model can be nontransitive; mean win probability supplies a fixed final order.
"""
from audit_common import *
import time
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT = HERE / 'experiments/pair_comparator_20260912'
ARMS = ['incumbent', 'pair_delta', 'pair_context', 'blend_context']
PARAMS = dict(CLF, min_child_samples=200, random_state=11)


def pair_features(x, left, right, context_n):
    delta = x[left] - x[right]
    if not context_n:
        return delta
    return np.column_stack([delta, (x[left, :context_n] + x[right, :context_n]) * .5])


def training_comparisons(x, y, pid, context_n):
    left, right, mass = [], [], []
    for p in np.unique(pid):
        yes = np.flatnonzero((pid == p) & (y == 1))
        no = np.flatnonzero((pid == p) & (y == 0))
        count = len(yes) * len(no)
        if not count:
            continue
        left.extend(np.repeat(yes, len(no)))
        right.extend(np.tile(no, len(yes)))
        # Each contributing query has total mass 20 over both orientations.
        mass.extend([10. / count] * count)
    a = np.asarray(left, dtype=int)
    b = np.asarray(right, dtype=int)
    assert len(a) > 0 and np.all(pid[a] == pid[b])
    assert np.all(y[a] == 1) and np.all(y[b] == 0)
    z = np.concatenate([pair_features(x, a, b, context_n), pair_features(x, b, a, context_n)])
    target = np.r_[np.ones(len(a)), np.zeros(len(a))]
    weights = np.tile(np.asarray(mass), 2)
    return z, target, weights


def tournament(model, x, pid, context_n):
    result = np.full(len(x), np.nan)
    max_anti_error = 0.
    for p in np.unique(pid):
        ix = np.flatnonzero(pid == p)
        n = len(ix)
        left = np.repeat(ix, n)
        right = np.tile(ix, n)
        raw = model.predict_proba(pair_features(x, left, right, context_n))[:, 1].reshape(n, n)
        sym = (raw + 1. - raw.T) * .5
        max_anti_error = max(max_anti_error, float(np.max(np.abs(sym + sym.T - 1.))))
        assert np.allclose(np.diag(sym), .5, rtol=0, atol=1e-14)
        result[ix] = (sym.sum(axis=1) - .5) / (n - 1)
    assert np.isfinite(result).all() and max_anti_error < 1e-14
    return result


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'freeze.json').exists(), 'Write-once screen already frozen'
    protected = check_protected()
    sources = [Path(__file__), HERE / 'src/audit_common.py', PREV / 'scores.parquet',
               HERE / 'experiments/error_diagnosis_20260912/diagnosis.json']
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'primary': 'pair_context', 'arms': ARMS,
        'source_hashes': {str(p): sha(p) for p in sources}, 'protected': protected,
        'hypothesis': 'Most DT/SP evidence is already in top20, yet about10% of positive-negative comparisons are inverted. Learn a nonlinear comparison of two hands, allowing relative feature interactions that are absent from independent scoring.',
        'reference': 'Exact saved current383 shortlist plus shipping r382DTSP/structCI classifier.',
        'features': {'pair_delta': 'Left minus right over identical per-family incumbent features.',
                     'pair_context': 'Same differences plus means of shared structural, chronology and first-stage features.',
                     'blend_context': 'Fixed equal within-query rank mean of incumbent and pair_context.'},
        'training': 'All listed-vs-unlisted comparisons within each training query, both orientations. Total weight20 per query. Queries with only one class contribute no comparison. Labels remain listing indicators.',
        'params': PARAMS, 'regularization_note': 'min_child_samples200 vs40 in pointwise classifier compensates for correlated comparison expansion; fixed before fit.',
        'inference': 'All top20 candidate comparisons; symmetrize p(a,b)=1-p(b,a), rank by mean win probability. Lower-ranked tail stays unchanged.',
        'validation': 'Same 162-query audit, whole-pool per-family GroupKFold5;15 exact incumbent controls;2 fixed candidate fits per fold; no outer evidence truth or evaluation data.',
        'gate': {'mean_map5_gain_min': .01, 'pool_bootstrap_lower_gt': 0, 'family_floor': -.01, 'both_pool_halves_nonnegative': True, 'negative_control': 'stage1_only fails'},
        'limits': ['Reused audit and shared stage1 OOF; exploratory, not nested.', 'Known-family routing on audit; predicted routing required for confirmation.', 'Comparisons do not increase independent sample size.', 'Nontransitive comparator; no theorem of MAP optimality.'],
        'background': 'https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/'}
    write(OUT / 'freeze.json', freeze)
    d, shared, extra, saved = load_current()
    pred = {n: d['oof'].to_numpy().copy() - 1e6 for n in ARMS}
    receipts = []
    for fam in FAMILIES:
        ix = np.flatnonzero(((d['behavior_family'] == fam) & (d['oof_rank'] <= 20)).to_numpy())
        q = d[ix]
        cols = shared + (extra if fam != 'coordinated_isolation' else [])
        x = q.select(cols).to_numpy().astype(float)
        assert np.isfinite(x).all()
        y = q['relevance'].to_numpy().astype(int)
        pid = q['pair_id'].to_numpy()
        pools = q['table_id'].to_numpy()
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(x, y, pools)):
            assert not set(pools[tr]) & set(pools[te]) and not set(pid[tr]) & set(pid[te])
            ref = lgb.LGBMClassifier(**CLF, random_state=11).fit(x[tr], y[tr])
            pp = ref.predict_proba(x[te])[:, 1]
            assert np.array_equal(pp + 10, saved['incumbent'].to_numpy()[ix[te]])
            pred['incumbent'][ix[te]] = pp + 10
            for name, context_n in [('pair_delta', 0), ('pair_context', len(shared))]:
                xx, yy, ww = training_comparisons(x[tr], y[tr], pid[tr], context_n)
                model = lgb.LGBMClassifier(**PARAMS).fit(xx, yy, sample_weight=ww)
                p = tournament(model, x[te], pid[te], context_n)
                if fold == 0:
                    order = np.arange(len(te))[::-1]
                    perm = tournament(model, x[te][order], pid[te][order], context_n)
                    assert np.allclose(p[order], perm, rtol=0, atol=1e-14), 'Candidate input-order invariance failed'
                path = OUT / f'{fam}-{fold}-{name}.txt'
                model.booster_.save_model(str(path))
                reload = lgb.Booster(model_file=str(path))
                assert np.array_equal(model.predict_proba(xx[:20])[:, 1], reload.predict(xx[:20]))
                pred[name][ix[te]] = p + 1000
                if name == 'pair_context':
                    pred['blend_context'][ix[te]] = rankmean(pid[te], pp, p) + 1000
                receipts.append({'family': fam, 'fold': fold, 'arm': name, 'comparison_rows': len(yy),
                                 'model_sha256': sha(path), 'parity_exact': True})
            print('Finished', fam, 'fold', fold, 'seconds', round(time.perf_counter() - start, 1), flush=True)
    sf = OUT / 'scores.parquet'
    d.select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(k, v) for k, v in pred.items()]).write_parquet(sf)
    write(OUT / 'predictions-frozen.json', {'scores_sha256': sha(sf), 'freeze_sha256': sha(OUT / 'freeze.json'), 'model_receipts': receipts, 'outer_truth_read': False})
    res = evaluate(d, dict(pred, stage1_only=d['oof'].to_numpy()))
    res['protected_unchanged'] = check_protected() == protected
    res['seconds'] = time.perf_counter() - start
    write(OUT / 'results.json', res)
    for arm, metrics in res['comparisons'].items():
        print(arm, json.dumps(metrics), flush=True)
    print('DONE', json.dumps({'passed': res['passed'], 'seconds': res['seconds']}), flush=True)


if __name__ == '__main__':
    main()
