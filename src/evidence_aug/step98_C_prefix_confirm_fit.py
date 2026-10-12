"""Step 98 (Claude): the single outer-fold 0-2 nested confirmation of S* = B0 + prefix_strength against nested B0.
Follows Codex contestable_state_confirm_fit.py step for step (Codex modules imported read-only, bytecode off) and adds
the 32 action-prefix strength fields. Registered: experiments/prefix_strength_confirmation_C_20260913/prediction-prefix-confirmation.json."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
import time
from datetime import datetime, timezone
import lightgbm as lgb
from evidence_aug.frame import training_relevance, attach_relevance, family_columns, score
from evidence_aug.fold import fold_roles
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology, OUT as CHRONO_OUT
from evidence_aug.step14_audit_truncation import split_of, SPLITS
from contestable_state_confirmation_features import build_for_keys
from contestable_state_features import FEATURES as STATE_FEATURES
from marginal_payout_features import inputs as strength_inputs
import retrospective_strength_features as RSF

OUT = CV / 'experiments/prefix_strength_confirmation_C_20260913'
PREREG = OUT / 'prediction-prefix-confirmation.json'
PREREG_SHA = 'd28237612dcd656509df2fc33dd9d74b99d8a9b77d8b1e7dfc68abcc87ea2e85'
CODEX_CONF = HERE / 'experiments/contestable_state_confirmation_20260912'
AUDIT_PREFIX = HERE / 'research/retrospective_strength_20260912/prefix.parquet'
AUDIT_PREFIX_SHA = 'ea3edf818bdb7943ed53bd38071edb2f2dba32bf2c17e01f30a6732a726f475d'
STRUCT_FILE = CV / 'experiments/family_stage2_20260911/dev-structural-features.parquet'
OLD = CV / 'experiments/family_stage2_r382_20260911'
KEYS = ['pair_id', 'hand_id']
CI = 'coordinated_isolation'
RS = list(RSF.FEATURES)
assert len(RS) == 32 and not set(RS) & set(STATE_FEATURES)


def ordinal(frame, values):
    return frame.select('pair_id').with_columns(pl.Series('s', values)).select(
        pl.col('s').rank('ordinal', descending=True).over('pair_id'))['s'].to_numpy()


def snapshot():
    """Size and mtime of every Codex file this step reads, plus any Codex bytecode; all must be unchanged at exit."""
    paths = sorted((HERE / 'src').glob('*.py')) + sorted((HERE / 'src/__pycache__').glob('*'))
    paths += sorted(p for p in CODEX_CONF.rglob('*') if p.is_file())
    paths += sorted((HERE / 'research/retrospective_strength_20260912').glob('*'))
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def prefix_for_keys(keys):
    """The frozen sec-14 extractor (retrospective_strength_features.extract) on explicit keys instead of target_keys()."""
    keys = keys.select(KEYS).sort(KEYS)
    pm, hm, sm, am = strength_inputs(keys)
    rows, swaps = [], 0
    for i, (pid, hid) in enumerate(keys.iter_rows()):
        feats, _ = RSF.extract(hm[hid], sm[hid], am[hid], pm[pid])
        if i < 20:
            swapped, _ = RSF.extract(hm[hid], sm[hid], am[hid], tuple(reversed(pm[pid])))
            assert swapped == feats
            swaps += 1
        rows.append({'pair_id': pid, 'hand_id': hid, **feats['prefix']})
    t = pl.DataFrame(rows).select(KEYS + RS).sort(KEYS)
    assert t.select(KEYS).equals(keys) and np.isfinite(t.select(RS).to_numpy()).all()
    return t, swaps


def main():
    start = time.perf_counter()
    assert sha(PREREG) == PREREG_SHA, 'registration changed'
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    assert not (OUT / 'freeze.json').exists() and not (OUT / 'predictions-frozen.json').exists(), 'write-once confirmation'
    (OUT / 'scores').mkdir(parents=True, exist_ok=False)
    protected = check_protected()
    snap0 = snapshot()
    codex_frozen = json.loads((CODEX_CONF / 'predictions-frozen.json').read_text(encoding='utf-8'))
    for name, digest in codex_frozen['scores'].items():
        assert sha(CODEX_CONF / 'scores' / name) == digest, name
    sources = [Path(__file__).resolve()] + [HERE / 'src' / n for n in (
        'audit_common.py', 'contestable_state_features.py', 'contestable_state_confirmation_features.py',
        'contestable_state_confirm_fit.py', 'contestable_state_confirm_evaluate.py', 'marginal_payout_features.py',
        'retrospective_strength_features.py')]
    sources += [PREREG, STRUCT_FILE, AUDIT_PREFIX, CODEX_CONF / 'predictions-frozen.json', CODEX_CONF / 'results.json']
    write(OUT / 'freeze.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'step': 98, 'author': 'Claude',
        'registration_sha256': PREREG_SHA, 'candidate': 'S_star = B0 + prefix_strength', 'comparator': 'B0',
        'source_hashes': {str(p): sha(p) for p in sources}, 'protected': protected, 'new_features': RS,
        'params': dict(CLF, random_state=11), 'gate': reg['gate'], 'confirmation_folds': [0, 1, 2],
        'held_evidence_evaluation_before_freeze': False})
    t0 = time.perf_counter()
    union = pl.concat([pl.read_parquet(CODEX_CONF / f'fold{f}-state-features.parquet').select(KEYS) for f in (0, 1, 2)]).unique().sort(KEYS)
    prefix, swaps = prefix_for_keys(union)
    audit_prefix = pl.read_parquet(AUDIT_PREFIX).sort(KEYS)
    assert sha(AUDIT_PREFIX) == AUDIT_PREFIX_SHA and audit_prefix.columns == KEYS + RS
    both = audit_prefix.join(prefix, on=KEYS, how='inner', suffix='_fold').sort(KEYS)
    a, b = both.select(RS).to_numpy(), both.select([c + '_fold' for c in RS]).to_numpy()
    c4 = {'union_keys': union.height, 'overlap_keys_with_audit_prefix': both.height, 'exact': bool(np.array_equal(a, b)),
          'max_abs_diff': float(np.max(np.abs(a - b))), 'pair_swap_checked_rows': swaps, 'seconds': round(time.perf_counter() - t0, 1)}
    assert c4['exact'] and both.height >= 3000, c4
    prefix.write_parquet(OUT / 'prefix-features-union.parquet')
    c4['sha256'] = sha(OUT / 'prefix-features-union.parquet')
    write(OUT / 'prefix-features-receipt.json', c4)
    print('prefix features', json.dumps(c4), flush=True)
    structural = pl.read_parquet(STRUCT_FILE)
    struct = [c for c in structural.columns if c not in ('pair_id', 'hand_id')]
    players = pl.read_csv(ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    receipts = []
    for fold in (0, 1, 2):
        data, features, pools, identity = load_fold(fold)
        data = add_chronology(data).join(structural, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        del data
        evidence = training_relevance(set(train['table_id']))
        assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence).sort('pair_id', 'hand_id')
        roles = fold_roles(fold)
        held = held.join(roles, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        model = fit_ranker(train, features + ['chrono_frac'])
        sh = score(model, family_columns(held, 'predicted_behavior'), features + ['chrono_frac'])
        old = pl.read_parquet(CHRONO_OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        assert held.select('pair_id', 'hand_id').equals(old.select('pair_id', 'hand_id'))
        assert np.array_equal(sh, old['score'].to_numpy()), 'Stage1 reference mismatch'
        split = np.asarray([split_of(p) for p in train['table_id']])
        st = np.full(train.height, np.nan)
        for s in range(SPLITS):
            te = np.flatnonzero(split == s)
            inner = fit_ranker(train.filter(pl.Series(split != s)), features + ['chrono_frac'])
            st[te] = score(inner, family_columns(train[te], 'behavior_family'), features + ['chrono_frac'])
        assert np.isfinite(st).all()
        train = train.with_columns(pl.Series('oof', st))
        held = held.with_columns(pl.Series('oof', sh))
        for frame in (train, held):
            assert np.array_equal(frame['pot_bb'].to_numpy().astype(np.float32), frame['pot_bb_right'].to_numpy().astype(np.float32))
        train = train.with_columns(pl.col('pot_bb_right').alias('pot_bb'))
        held = held.with_columns(pl.col('pot_bb_right').alias('pot_bb'))
        train = train.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
        held = held.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
        shared = struct + ['chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank']
        extra = [f for f in features if not f.startswith('family_') and f not in struct]
        train = train.with_columns(pl.col(extra).fill_null(-999.))
        held = held.with_columns(pl.col(extra).fill_null(-999.))
        old_struct = pl.read_parquet(OLD / 'scores' / f'fold{fold}-allfam_top20_struct.parquet')
        old_r382 = pl.read_parquet(OLD / 'scores' / f'fold{fold}-allfam_top20_r382.parquet')
        held_keys = held.select('pair_id', 'hand_id')
        assert held_keys.equals(old_struct.select('pair_id', 'hand_id')) and held_keys.equals(old_r382.select('pair_id', 'hand_id'))
        expected = np.where((held['predicted_behavior'] == CI).to_numpy(), old_struct['score'].to_numpy(), old_r382['score'].to_numpy())
        control = sh.copy()
        for fam in FAMILIES:
            cols = shared + (extra if fam != CI else [])
            tr = train.filter((pl.col('behavior_family') == fam) & (pl.col('oof_rank') <= 20))
            ix = np.flatnonzero(((held['predicted_behavior'] == fam) & (held['oof_rank'] <= 20)).to_numpy())
            ref = lgb.LGBMClassifier(**CLF, random_state=11).fit(tr.select(cols).to_numpy().astype(float), tr['relevance'].to_numpy())
            if len(ix):
                control[ix] = ref.predict_proba(held[ix].select(cols).to_numpy().astype(float))[:, 1] + 10
            tail = ((held['predicted_behavior'] == fam) & (held['oof_rank'] > 20)).to_numpy()
            control[tail] = sh[tail] - 1e6
        diff = float(np.max(np.abs(control - expected)))
        assert diff <= 2e-14 and np.array_equal(ordinal(held, control), ordinal(held, expected)), 'Complete reference mismatch'
        requested = pl.concat([train.filter(pl.col('oof_rank') <= 20).select(KEYS),
                               held.filter(pl.col('oof_rank') <= 20).select(KEYS)]).unique().sort(KEYS)
        new, accounting = build_for_keys(requested, players)
        saved_state = pl.read_parquet(CODEX_CONF / f'fold{fold}-state-features.parquet')
        assert new.columns == saved_state.columns and new.select(KEYS).equals(saved_state.select(KEYS))
        cs_exact = bool(np.array_equal(new.select(STATE_FEATURES).to_numpy(), saved_state.select(STATE_FEATURES).to_numpy()))
        assert cs_exact, 'cs_ rebuild differs from Codex saved fold features'
        pref = requested.join(prefix, on=KEYS, how='left', validate='1:1').sort(KEYS)
        assert pref.select(KEYS).equals(requested) and np.isfinite(pref.select(RS).to_numpy()).all()
        pref.write_parquet(OUT / f'fold{fold}-prefix-features.parquet')
        train_keys = train.select(KEYS)
        train = train.join(new, on=KEYS, how='left', validate='1:1').join(pref, on=KEYS, how='left', validate='1:1').sort(KEYS)
        held = held.join(new, on=KEYS, how='left', validate='1:1').join(pref, on=KEYS, how='left', validate='1:1').sort(KEYS)
        assert train_keys.equals(train.select(KEYS)) and held_keys.equals(held.select(KEYS))
        model_dir = OUT / 'models' / f'fold{fold}'
        model_dir.mkdir(parents=True, exist_ok=False)
        arms = {'B0': list(STATE_FEATURES), 'S_star': list(STATE_FEATURES) + RS}
        preds = {name: sh.copy() for name in arms}
        model_receipts = []
        for fam in FAMILIES:
            tr = train.filter((pl.col('behavior_family') == fam) & (pl.col('oof_rank') <= 20))
            ix = np.flatnonzero(((held['predicted_behavior'] == fam) & (held['oof_rank'] <= 20)).to_numpy())
            tail = ((held['predicted_behavior'] == fam) & (held['oof_rank'] > 20)).to_numpy()
            for name, added in arms.items():
                cols = shared + (extra if fam != CI else []) + added
                x = tr.select(cols).to_numpy().astype(float)
                assert np.isfinite(x).all()
                fitted = lgb.LGBMClassifier(**CLF, random_state=11).fit(x, tr['relevance'].to_numpy())
                path = model_dir / f'{fam}-{name}.txt'
                fitted.booster_.save_model(str(path))
                if len(ix):
                    h = held[ix].select(cols).to_numpy().astype(float)
                    assert np.isfinite(h).all()
                    p = fitted.predict_proba(h)[:, 1]
                    assert np.array_equal(p, lgb.Booster(model_file=str(path)).predict(h, num_threads=2)), 'Saved inference mismatch'
                    preds[name][ix] = p + 10
                preds[name][tail] = sh[tail] - 1e6
                model_receipts.append({'family': fam, 'arm': name, 'n_columns': len(cols), 'train_queries': tr['pair_id'].n_unique(),
                                       'held_queries': len(set(held['pair_id'].to_numpy()[ix])), 'model_sha256': sha(path), 'reload_exact': True})
        write(model_dir / 'recipe.json', model_receipts)
        codex_b0 = pl.read_parquet(CODEX_CONF / 'scores' / f'fold{fold}-contestable_state.parquet')
        codex_inc = pl.read_parquet(CODEX_CONF / 'scores' / f'fold{fold}-incumbent.parquet')
        assert codex_b0.select(KEYS).equals(held_keys) and codex_inc.select(KEYS).equals(held_keys)
        cb = codex_b0['score'].to_numpy()
        c3 = {'B0_bit_exact_vs_codex': bool(np.array_equal(preds['B0'], cb)), 'B0_max_abs_diff_vs_codex': float(np.max(np.abs(preds['B0'] - cb))),
              'B0_ordinal_exact_vs_codex': bool(np.array_equal(ordinal(held, preds['B0']), ordinal(held, cb))),
              'incumbent_bit_exact_vs_codex': bool(np.array_equal(expected, codex_inc['score'].to_numpy()))}
        assert c3['B0_max_abs_diff_vs_codex'] <= 2e-14 and c3['B0_ordinal_exact_vs_codex'] and c3['incumbent_bit_exact_vs_codex'], c3
        for name, p in [('incumbent', expected), ('B0', preds['B0']), ('S_star', preds['S_star']), ('stage1_only', sh)]:
            assert np.isfinite(p).all()
            held.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', p)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
        roles.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        receipts.append({'fold': fold, 'identity': identity, 'held_pairs': held['pair_id'].n_unique(), 'training_pairs': train['pair_id'].n_unique(),
                         'reference_max_abs_diff': diff, 'requested_keys': requested.height, 'cs_rebuild_exact_vs_codex': cs_exact,
                         'C3': c3, 'accounting': accounting, 'S_star_rows_differing_from_B0': int(np.sum(preds['S_star'] != preds['B0']))})
        print('Finished fold', fold, round(time.perf_counter() - start, 1), json.dumps(c3), flush=True)
        del train, held
    snap1 = snapshot()
    changed = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
    write(OUT / 'predictions-frozen.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha(OUT / 'freeze.json'),
        'registration_sha256': PREREG_SHA, 'scores': {f.name: sha(f) for f in sorted((OUT / 'scores').iterdir())},
        'prefix_features': c4, 'fold_receipts': receipts, 'protected_unchanged': check_protected() == protected,
        'codex_resume_unchanged': not changed, 'codex_changed_entries': changed, 'codex_entries_checked': len(snap0),
        'seconds': time.perf_counter() - start, 'held_evidence_evaluation': False})
    assert not changed, changed
    print('PREDICTIONS FROZEN', round(time.perf_counter() - start, 1), flush=True)


if __name__ == '__main__':
    main()
