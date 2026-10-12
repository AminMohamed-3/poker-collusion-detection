"""One frozen nested confirmation of equal_methods; no held evidence evaluation.

The three outer partitions are reused and overlap. Each fold excludes its held
pools from BOTH training stages; inner OOF supplies stage2 training scores.
"""
from audit_common import *
sys.path.insert(0, str(HERE / 'vendor/catboost'))
import time
from datetime import datetime, timezone
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from evidence_aug.frame import training_relevance, attach_relevance, family_columns, score
from evidence_aug.fold import fold_roles
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology, OUT as CHRONO_OUT
from evidence_aug.step14_audit_truncation import split_of, SPLITS
from symmetry_censor_audit import swapped
from catboost_evidence_audit import PARAMS as CAT_PARAMS
from pair_comparator_audit import training_comparisons, tournament, PARAMS as PAIR_PARAMS
from method_ensemble_audit import GROUPS, WEIGHTS, ranks

OUT = HERE / 'experiments/method_ensemble_confirmation_v2_20260912'
STRUCT_FILE = CV / 'experiments/family_stage2_20260911/dev-structural-features.parquet'
OLD = CV / 'experiments/family_stage2_r382_20260911'


def fit_predict_family(train, held, cols, shared_n, model_dir):
    """Labels may enter train only. Held must not have a relevance column."""
    assert 'relevance' not in held.columns
    assert not set(train['pair_id']) & set(held['pair_id'])
    model_dir.mkdir(parents=True, exist_ok=True)
    x = train.select(cols).to_numpy().astype(float)
    h = held.select(cols).to_numpy().astype(float)
    assert np.isfinite(x).all() and np.isfinite(h).all()
    xs, hs = swapped(x, cols), swapped(h, cols)
    y = train['relevance'].to_numpy().astype(int)
    pid, hp = train['pair_id'].to_numpy(), held['pair_id'].to_numpy()
    base = lgb.LGBMClassifier(**CLF, random_state=11).fit(x, y)
    base.booster_.save_model(str(model_dir / 'incumbent.txt'))
    p = base.predict_proba(h)[:, 1]
    raw = {'incumbent': p, 'swap_average': (p + base.predict_proba(hs)[:, 1]) * .5}
    cens = ((train['listed_count'] >= 5) & (train['relevance'] == 0) & (train['chrono_rank'] > train['last_listed'])).to_numpy()
    for name, sym, weight in [('symmetric_training', True, 1.), ('censor_half', False, .5), ('censor_quarter', False, .25), ('symmetric_censor_half', True, .5)]:
        w = np.where(cens, weight, 1.)
        for pair in np.unique(pid):
            q = pid == pair
            w[q] *= q.sum() / w[q].sum()
        xx, yy, params = x, y, dict(CLF)
        if sym:
            xx, yy, w = np.concatenate([x, xs]), np.tile(y, 2), np.tile(w / 2, 2)
            params['min_child_samples'] = 2 * CLF['min_child_samples']
        model = lgb.LGBMClassifier(**params, random_state=11).fit(xx, yy, sample_weight=w)
        model.booster_.save_model(str(model_dir / f'{name}.txt'))
        pp = model.predict_proba(h)[:, 1]
        raw[name] = (pp + model.predict_proba(hs)[:, 1]) * .5 if sym else pp
    for name, ranker in [('cat_logloss', False), ('cat_map', True)]:
        if ranker:
            model = CatBoostRanker(**CAT_PARAMS, loss_function='YetiRank:mode=MAP;top=5;permutations=10')
            model.fit(Pool(x, label=y, group_id=pid.tolist()))
            pp = model.predict(h)
        else:
            model = CatBoostClassifier(**CAT_PARAMS, loss_function='Logloss').fit(x, y)
            pp = model.predict_proba(h)[:, 1]
        model.save_model(str(model_dir / f'{name}.cbm'))
        raw[name] = pp
    for name, context_n in [('pair_delta', 0), ('pair_context', shared_n)]:
        xx, yy, ww = training_comparisons(x, y, pid, context_n)
        model = lgb.LGBMClassifier(**PAIR_PARAMS).fit(xx, yy, sample_weight=ww)
        model.booster_.save_model(str(model_dir / f'{name}.txt'))
        raw[name] = tournament(model, h, hp, context_n)
    gs = {'incumbent': ranks(hp, p)}
    for group, (_, arms) in GROUPS.items():
        gs[group] = np.mean([ranks(hp, raw[a]) for a in arms], axis=0)
    ensemble = sum(WEIGHTS['equal_methods'][k] * gs[k] for k in gs)
    write(model_dir / 'recipe.json', {'columns': cols, 'shared_n': shared_n, 'groups': GROUPS, 'weights': WEIGHTS['equal_methods'],
                                    'training_queries': len(set(pid)), 'held_queries': len(set(hp)),
                                    'model_sha256': {f.name: sha(f) for f in model_dir.iterdir() if f.suffix in ('.txt', '.cbm')}})
    return p, ensemble


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scores').mkdir(exist_ok=True)
    assert not (OUT / 'freeze.json').exists(), 'Confirmation already frozen'
    audit_dir = HERE / 'experiments/method_ensemble_20260912'
    audit = json.loads((audit_dir / 'results.json').read_text(encoding='utf-8'))
    assert audit['comparisons']['equal_methods']['gate_pass']
    protected = check_protected()
    source_files = [HERE / 'src' / name for name in ['audit_common.py', 'symmetry_censor_audit.py', 'catboost_evidence_audit.py', 'pair_comparator_audit.py', 'method_ensemble_audit.py', 'method_ensemble_confirm_fit_v2.py', 'method_ensemble_confirm_evaluate_v2.py']]
    source_files += [audit_dir / 'freeze.json', audit_dir / 'results.json', STRUCT_FILE, HERE / 'experiments/pot_feature_reconciliation_20260912/receipt.json']
    write(OUT / 'freeze.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'candidate': 'equal_methods',
        'source_hashes': {str(p): sha(p) for p in source_files}, 'protected': protected,
        'audit_result': audit['comparisons']['equal_methods'], 'groups': GROUPS, 'weights': WEIGHTS['equal_methods'],
        'selection': 'Primary fixed-weight ensemble only; do not evaluate other ensemble weights or individual components on these folds.',
        'pre_metric_amendment': 'Initial fit aborted before any stage2 model or held metric: pot precision equality was overly strict. Raw pot is float32, structural pot float64, mapping exactly agrees after float32 cast. All15 audit reference fits reproduce exactly under raw or structural pot. Retain original audit structural precision for every component, require saved outer incumbent score/rank parity. No gate, weight or parameter change.',
        'confirmation_folds': [0, 1, 2],
        'reference': 'Complete shipping hybrid: current383 chronology stage1; r382 stage2 for predicted DT/SP, structural stage2 for predicted CI. Exact chronological parity and <=2e-14 stage2 numerical parity plus exact within-query rank parity with saved Claude predictions required.',
        'fit': 'For each outer fold, both stages train outside held pools. Stage2 training candidates use five pool-hash OOF first-stage scores from within that outer training set. Fixed family predictions route held queries. Predicted none retains stage1.',
        'gate': {'primary_mean_unique_pair_delta_min': .01, 'raw_pool_bootstrap_lower_gt': 0, 'all_three_fold_deltas_nonnegative': True, 'pooled_true_family_floor': -.01, 'each_fold_true_family_floor': -.01, 'both_pool_halves_nonnegative': True},
        'pooling': 'The folds overlap. Average delta for each unique player pair across its held appearances, then bootstrap original pool IDs with all their pairs. Do not pretend fold:pool IDs are independent pools. Also report fold-appearance weighted mean descriptively.',
        'limits': 'Reused outer partitions; not pristine. Inner stacking crossfits share reference-policy features trained outside the OUTER fold, so inner OOF is not fully nested at policy level; outer pools still excluded.',
        'negative_control': 'Stage1-only should fail vs complete hybrid.',
        'progression': 'PASS -> freeze public prediction with uncertainty, refit production under identical recipe and build validated evidence-only candidate with incumbent risk and behavior. FAIL -> preserve results, no family/weight rescue and no submission.',
        'held_evidence_evaluation_before_freeze': False})
    feat = pl.read_parquet(STRUCT_FILE)
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    receipts = []
    for fold in (0, 1, 2):
        data, features, pools, identity = load_fold(fold)
        data = add_chronology(data)
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        evidence = training_relevance(set(train['table_id']))
        assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence)
        roles = fold_roles(fold)
        held = held.join(roles, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        model = fit_ranker(train, features + ['chrono_frac'])
        sh = score(model, family_columns(held, 'predicted_behavior'), features + ['chrono_frac'])
        old = pl.read_parquet(CHRONO_OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        assert held.select('pair_id', 'hand_id').equals(old.select('pair_id', 'hand_id'))
        assert np.array_equal(sh, old['score'].to_numpy()), 'First-stage reference mismatch'
        model.booster_.save_model(str(OUT / f'fold{fold}-stage1.txt'))
        split = np.asarray([split_of(p) for p in train['table_id']])
        st = np.full(train.height, np.nan)
        for s in range(SPLITS):
            te = np.flatnonzero(split == s)
            m = fit_ranker(train.filter(pl.Series(split != s)), features + ['chrono_frac'])
            st[te] = score(m, family_columns(train[te], 'behavior_family'), features + ['chrono_frac'])
        assert np.isfinite(st).all()
        # Fit stage1 before the structural join; preserve the production pot column.
        train = train.join(feat, on=['pair_id', 'hand_id'], validate='1:1').with_columns(pl.Series('oof', st))
        held = held.join(feat, on=['pair_id', 'hand_id'], validate='1:1').with_columns(pl.Series('oof', sh))
        # Raw cache stores float32; structural arithmetic is float64. Keep
        # the audited structural precision for ALL ensemble components.
        for frame in (train, held):
            assert np.array_equal(frame['pot_bb'].to_numpy().astype(np.float32), frame['pot_bb_right'].to_numpy().astype(np.float32))
        train = train.with_columns(pl.col('pot_bb_right').alias('pot_bb'))
        held = held.with_columns(pl.col('pot_bb_right').alias('pot_bb'))
        train = train.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
        held = held.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
        meta = train.group_by('pair_id').agg(pl.col('relevance').sum().alias('listed_count'), pl.col('chrono_rank').filter(pl.col('relevance') == 1).max().alias('last_listed'))
        train = train.join(meta, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        shared = struct + ['chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank']
        extra = [f for f in features if not f.startswith('family_') and f not in struct]
        train = train.with_columns(pl.col(extra).fill_null(-999.))
        held = held.with_columns(pl.col(extra).fill_null(-999.))
        base, candidate = sh.copy(), sh.copy()
        for fam in FAMILIES:
            ix = np.flatnonzero(((held['predicted_behavior'] == fam) & (held['oof_rank'] <= 20)).to_numpy())
            if not len(ix):
                continue
            tr = train.filter((pl.col('behavior_family') == fam) & (pl.col('oof_rank') <= 20))
            cols = shared + (extra if fam != 'coordinated_isolation' else [])
            pp, ep = fit_predict_family(tr, held[ix].drop('behavior_family'), cols, len(shared), OUT / 'models' / f'fold{fold}-{fam}')
            tail = ((held['predicted_behavior'] == fam) & (held['oof_rank'] > 20)).to_numpy()
            base[tail], candidate[tail] = sh[tail] - 1e6, sh[tail] - 1e6
            base[ix], candidate[ix] = pp + 10., ep + 1000.
            print('Finished confirmation fold', fold, fam, 'seconds', round(time.perf_counter() - start, 1), flush=True)
        old_struct = pl.read_parquet(OLD / 'scores' / f'fold{fold}-allfam_top20_struct.parquet')
        old_r382 = pl.read_parquet(OLD / 'scores' / f'fold{fold}-allfam_top20_r382.parquet')
        assert old_struct.select('pair_id', 'hand_id').equals(held.select('pair_id', 'hand_id'))
        saved = np.where((held['predicted_behavior'] == 'coordinated_isolation').to_numpy(), old_struct['score'].to_numpy(), old_r382['score'].to_numpy())
        diff = float(np.max(np.abs(base - saved)))
        assert diff <= 2e-14, f'Stage2 incumbent parity failed: {diff}'
        assert np.array_equal(ranks(held['pair_id'].to_numpy(), base), ranks(held['pair_id'].to_numpy(), saved))
        keys = held.select('pair_id', 'hand_id', 'table_id')
        for name, p in [('incumbent', saved), ('equal_methods', candidate), ('stage1_only', sh)]:
            assert np.isfinite(p).all()
            keys.with_columns(pl.Series('score', p)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
        roles.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        receipts.append({'fold': fold, 'identity': identity, 'held_pairs': held['pair_id'].n_unique(), 'training_pairs': train['pair_id'].n_unique(), 'incumbent_max_abs_diff': diff, 'incumbent_ranks_exact': True})
    write(OUT / 'predictions-frozen.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha(OUT / 'freeze.json'),
        'scores': {f.name: sha(f) for f in sorted((OUT / 'scores').iterdir())}, 'fold_receipts': receipts,
        'protected_unchanged': check_protected() == protected, 'seconds': time.perf_counter() - start,
        'held_evidence_evaluation': False})
    print('PREDICTIONS FROZEN', round(time.perf_counter() - start, 1), flush=True)


if __name__ == '__main__':
    main()
