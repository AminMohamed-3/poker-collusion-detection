"""Frozen residual-prior screen on the 162-query training-only audit.

The first-stage top-20 rank is calibrated inside each stage-2 training fold.
Its logit is passed to LightGBM as ``init_score``; LightGBM's tree-only raw
prediction is explicitly combined with that fold's prior at inference.
"""
from audit_common import *
import time
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

OUT = HERE / 'experiments' / 'residual_prior_20260912'
ARMS = ['incumbent', 'regularized_control', 'residual',
        'residual_regularized', 'rank_prior_only']
PRIMARY = 'residual_regularized'
REGULARIZED = 'regularized_control'
REG_PARAMS = dict(CLF, reg_lambda=5, random_state=11)


def sigmoid(raw):
    # All fitted values are bounded here, but this is stable for the fixed tail
    # sentinel as well.
    return np.where(raw >= 0, 1. / (1. + np.exp(-raw)),
                    np.exp(raw) / (1. + np.exp(raw)))


def calibrate(rank, label):
    """Fit the fold-training-only rank prior and return its 1..20 receipt."""
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
    iso.fit(rank, label)
    grid = iso.predict(np.arange(1., 21.))
    assert np.isfinite(grid).all()
    assert np.all(np.diff(grid) <= 1e-14), 'Rank calibration is not decreasing'
    clipped = np.clip(grid, .01, .99)
    return iso, grid, clipped


def prior_logit(iso, rank):
    probability = np.clip(iso.predict(rank), .01, .99)
    return np.log(probability) - np.log1p(-probability), probability


def raw_with_prior(model, x, prior):
    """LightGBM inference is tree-only; restore the separately fitted offset."""
    raw = model.booster_.predict(x, raw_score=True)
    wrapper = model.predict_proba(x)[:, 1]
    # This empirical API check proves sklearn prediction omits init_score.
    assert np.allclose(wrapper, sigmoid(raw), rtol=0, atol=2e-15)
    return raw + prior, raw


def no_update_check(booster, x, prior):
    """A request beginning after the final tree must be a zero-tree update."""
    n = booster.current_iteration()
    zero = booster.predict(x, raw_score=True, start_iteration=n, num_iteration=1)
    assert np.array_equal(zero, np.zeros_like(zero))
    no_update = sigmoid(zero + prior)
    direct = sigmoid(prior)
    assert np.array_equal(no_update, direct)
    return {'tree_count': int(n), 'zero_tree_raw_exact': True,
            'zero_update_reproduces_prior_exact': True}


def compare(a, b, families, ptab, pools):
    result = boot(a, b, ptab, pools)
    result['by_family'] = {f: float(np.mean([b[p] - a[p] for p in a if families[p] == f])) for f in FAMILIES}
    result['halves'] = {str(h): float(np.mean([b[p] - a[p] for p in a if half(ptab[p]) == h])) for h in (0, 1)}
    return result


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'freeze.json').exists(), 'Write-once study already started.'
    protected = check_protected()
    sources = [Path(__file__), HERE / 'src' / 'audit_common.py', PREV / 'scores.parquet',
               PREV / 'predictions-frozen.json', AUDIT / 'audit-oof.parquet',
               AUDIT / 'audit-structural-features.parquet']
    freeze = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'arms': ARMS + ['stage1_only'], 'primary': PRIMARY,
        'source_hashes': {str(p): sha(p) for p in sources}, 'protected': protected,
        'reference': 'Exact saved current chronology383 shipping hybrid on its unchanged top20 shortlist.',
        'data': 'Only the existing 162-query audit outside union outer0/1/2; per-family whole-pool GroupKFold5.',
        'params': {'incumbent': dict(CLF, random_state=11),
                   'regularized_control': REG_PARAMS,
                   'residual': dict(CLF, random_state=11),
                   'residual_regularized': REG_PARAMS},
        'prior': {'input': 'stage1 ordinal within-pair rank 1..20',
                  'fit': "IsotonicRegression(increasing=False, out_of_bounds='clip') on only each fold's training rows and listing labels",
                  'clip_probability': [.01, .99],
                  'training': 'Pass clipped-logit prior as LightGBM init_score.',
                  'inference': 'Add the same fold calibration logit to tree-only raw output, then sigmoid.',
                  'disclosure': 'The training residual prior is calibrated in-sample on each training partition; validation labels are never used.'},
        'budget': '15 exact incumbent fits; 15 regularization-control fits; 15 residual fits; 15 residual-regularized fits; two CPU threads; no early stopping or sweep.',
        'gate': {'mean_map5_gain_min': .01, 'pool_bootstrap_lower_gt': 0,
                 'family_floor': -.01, 'both_pool_halves_nonnegative': True,
                 'attribution': 'Primary residual_regularized must beat regularized_control in mean with paired pool-bootstrap interval reported.',
                 'eligible_arm': PRIMARY, 'negative_control': 'stage1_only must fail versus incumbent.'},
        'limits': ['Reused training-only audit and shared first-stage OOF; exploratory, not nested.',
                   'Known-family routing is audit-only.', 'No outer evidence labels, metrics, submissions, or public outcomes are read.']}
    write(OUT / 'freeze.json', freeze)

    d, shared, extra, saved = load_current()
    assert d.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id'))
    pred = {name: d['oof'].to_numpy().copy() - 1e6 for name in ARMS}
    receipts = []
    for family in FAMILIES:
        ix = np.flatnonzero(((d['behavior_family'] == family) & (d['oof_rank'] <= 20)).to_numpy())
        q = d[ix]
        cols = shared + (extra if family != 'coordinated_isolation' else [])
        x = q.select(cols).to_numpy().astype(float)
        assert np.isfinite(x).all()
        y = q['relevance'].to_numpy().astype(int)
        ranks = q['oof_rank'].to_numpy().astype(float)
        groups = q['table_id'].to_numpy()
        pid = q['pair_id'].to_numpy()
        assert np.array_equal(np.unique(ranks), np.arange(1., 21.))
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(x, y, groups)):
            assert not set(groups[tr]) & set(groups[te]) and not set(pid[tr]) & set(pid[te])
            incumbent = lgb.LGBMClassifier(**CLF, random_state=11).fit(x[tr], y[tr])
            incumbent_p = incumbent.predict_proba(x[te])[:, 1]
            assert np.array_equal(incumbent_p + 10, saved['incumbent'].to_numpy()[ix[te]]), 'Current reference prediction mismatch'
            pred['incumbent'][ix[te]] = incumbent_p + 10

            iso, unclip_grid, clip_grid = calibrate(ranks[tr], y[tr])
            init_tr, p_tr = prior_logit(iso, ranks[tr])
            init_te, p_te = prior_logit(iso, ranks[te])
            assert np.allclose(sigmoid(init_te), p_te, rtol=0, atol=2e-16)
            pred['rank_prior_only'][ix[te]] = p_te + 10
            receipt = {'family': family, 'fold': fold, 'train_pairs': int(len(np.unique(pid[tr]))),
                       'validation_pairs': int(len(np.unique(pid[te]))),
                       'calibration_1_to_20_unclipped': unclip_grid.tolist(),
                       'calibration_1_to_20_clipped': clip_grid.tolist(),
                       'monotone_decreasing': True}
            for name, params, residual in [
                    ('regularized_control', REG_PARAMS, False),
                    ('residual', dict(CLF, random_state=11), True),
                    (PRIMARY, REG_PARAMS, True)]:
                model = lgb.LGBMClassifier(**params).fit(x[tr], y[tr], init_score=init_tr if residual else None)
                path = OUT / f'{family}-{fold}-{name}.txt'
                model.booster_.save_model(str(path))
                reload = lgb.Booster(model_file=str(path))
                if residual:
                    total_raw, tree_raw = raw_with_prior(model, x[te], init_te)
                    reload_raw = reload.predict(x[te], raw_score=True)
                    assert np.array_equal(tree_raw, reload_raw)
                    reloaded_total = sigmoid(reload_raw + init_te)
                    score = sigmoid(total_raw)
                    assert np.array_equal(score, reloaded_total), 'Reload changed complete residual prediction'
                    no_update = no_update_check(model.booster_, x[te], init_te)
                    pred[name][ix[te]] = score + 10
                    saved_full_exact = True
                else:
                    score = model.predict_proba(x[te])[:, 1]
                    assert np.array_equal(score, reload.predict(x[te]))
                    pred[name][ix[te]] = score + 10
                    no_update = None
                    saved_full_exact = True
                receipt[name] = {'model_sha256': sha(path), 'reload_complete_predictions_exact': saved_full_exact,
                                 'init_score': residual, 'no_update_control': no_update}
            receipts.append(receipt)
            print('Finished', family, 'fold', fold, 'seconds', round(time.perf_counter() - started, 1), flush=True)

    assert all(np.isfinite(v).all() for v in pred.values())
    scorefile = OUT / 'scores.parquet'
    d.select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(name, values) for name, values in pred.items()]).write_parquet(scorefile)
    write(OUT / 'predictions-frozen.json', {'freeze_sha256': sha(OUT / 'freeze.json'),
                                             'scores_sha256': sha(scorefile), 'model_receipts': receipts,
                                             'outer_truth_read': False,
                                             'all_predictions_frozen_before_metrics': True})

    results = evaluate(d, dict(pred, stage1_only=d['oof'].to_numpy()))
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    families = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    # The requested attribution is paired on the identical audit queries/pools.
    _, maps_regularized = summarize(d.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred[REGULARIZED])), truths, families)
    _, maps_primary = summarize(d.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred[PRIMARY])), truths, families)
    attribution = compare(maps_regularized, maps_primary, families, ptab, pools)
    attribution['primary_mean_beats_regularized_control'] = attribution['mean_delta'] > 0
    primary = results['comparisons'][PRIMARY]
    results['primary_vs_regularized_control'] = attribution
    results['primary_eligible'] = bool(primary['gate_pass'] and attribution['primary_mean_beats_regularized_control'])
    results['passed'] = [PRIMARY] if results['primary_eligible'] else []
    results['negative_control_rejected'] = not results['comparisons']['stage1_only']['gate_pass']
    results['protected_unchanged'] = check_protected() == protected
    results['seconds'] = time.perf_counter() - started
    write(OUT / 'results.json', results)
    for arm, report in results['comparisons'].items():
        print(arm, json.dumps(report), flush=True)
    print('primary_vs_regularized_control', json.dumps(attribution), flush=True)
    print('DONE', json.dumps({'passed': results['passed'], 'negative_control_rejected': results['negative_control_rejected'],
                              'seconds': results['seconds']}), flush=True)


if __name__ == '__main__':
    main()
