"""Step 108 (Claude): the user-authorized exploratory production build of decode(S*) evidence rows onto the .89466 file. It never uploads.
Registered before any build: outputs_stacked_decoder_exploratory_20260913/prediction.json. Codex modules are imported read-only with bytecode
off, and nothing is written outside the output folder. Step 107 stays a registered FAIL."""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
sys.dont_write_bytecode = True
import gc
import json
import time
from pathlib import Path

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import contestable_state_production as CSP
import joblib
import numpy as np
import polars as pl
import retrospective_strength_features as RSF
from validation import validate_submission
from evidence_aug.paths import sha, check_protected, write_json
from evidence_aug.frame import FAMILIES
from evidence_aug import step98_C_prefix_confirm_fit as S98
from evidence_aug import step104_C_typed_decoder_build as S104

ROOT, DATA, CODEX_OUT = CSP.ROOT, CSP.ROOT / 'data', CSP.OUT
OUT = CSP.CV / 'outputs_stacked_decoder_exploratory_20260913'
PREREG, PREREG_SHA = OUT / 'prediction.json', '82fd0ce9300c6ec504743b44d80fafc7132a902a8a244a88e72e48ad9f759933'
AUTH, AUTH_SHA = OUT / 'exploratory-authorization.json', '718a4e78c9e1f2fb62453ef02b7ca2e0d9ebfd205f3032acc4090192b6e4286b'
BASE = POKER / 'Codex Resume/outputs_other_10k_exploratory_20260913/submission_other_conf065_total10000.csv'
BASE_SHA = 'a616889a76a6cf2f816fe60e455ce4fbc95ee7379e2f3cbbff2be69d835049bc'
KEYS, TOP, CI, CLF = list(CSP.KEYS), CSP.TOP, CSP.CI, CSP.CLF
STATE, RS = list(CSP.STATE_FEATURES), list(RSF.FEATURES)
assert len(RS) == 32 and not set(RS) & set(STATE)


def snapshot():
    """Size and mtime of the Codex files this build imports or reads; all must be unchanged at exit."""
    codex = POKER / 'Codex Resume'
    paths = sorted((codex / 'src').glob('*.py')) + sorted((codex / 'src/__pycache__').glob('*'))
    for d in ('outputs_contestable_exploratory_20260912', 'outputs_other_10k_exploratory_20260913'):
        paths += sorted(p for p in (codex / d).rglob('*') if p.is_file())
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def pair_inputs(keys, pairs):
    """marginal_payout_features.inputs line for line, with the pair-player frame supplied instead of development_labels.csv."""
    pairrows = pairs.select('pair_id', 'player_1', 'player_2').join(keys.select('pair_id').unique(), on='pair_id', how='semi')
    pairmap = {i: (a, b) for i, a, b in pairrows.iter_rows()}
    handids = keys['hand_id'].unique().to_list()
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(handids))
             .select('hand_id', 'button_seat', 'small_blind', 'big_blind', 'players_dealt', 'board_cards').collect())
    seats = (pl.scan_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(handids))
             .select('hand_id', 'player_id', 'seat_no', 'starting_stack', 'total_contribution', 'hole_card_1', 'hole_card_2').collect())
    actions = pl.scan_parquet(DATA / 'actions.parquet').filter(pl.col('hand_id').is_in(handids)).sort('hand_id', 'action_no').collect()
    hm = {x['hand_id']: x for x in hands.to_dicts()}
    sm = {k[0]: g.to_dicts() for k, g in seats.group_by('hand_id')}
    am = {k[0]: g.to_dicts() for k, g in actions.group_by('hand_id')}
    assert len(hm) == len(sm) == len(am) == len(handids)
    assert len(pairmap) == keys['pair_id'].n_unique()
    return pairmap, hm, sm, am


def prefix_rows(keys, inputs):
    """step-98 prefix_for_keys with its inputs supplied."""
    keys = keys.select(KEYS).sort(KEYS)
    pm, hm, sm, am = inputs
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


def chronology_eval(fd):
    """step-104 chronology() when no started_at tie sits inside a top-20 set; otherwise the registered hand_id tie-break."""
    hands = pl.scan_parquet(DATA / 'hands.parquet').select('hand_id', 'started_at').join(fd.select('hand_id').unique().lazy(), on='hand_id').collect()
    x = fd.join(hands, on='hand_id', validate='m:1', maintain_order='left')
    ok = x.filter(pl.col('score') > 5).group_by('pair_id').agg((pl.col('started_at').n_unique() == pl.len()).alias('ok'))
    ties = int((~ok['ok']).sum())
    if ties == 0:
        return S104.chronology(fd), 0
    out = x.sort(KEYS).with_columns(pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'))
    return out.drop('started_at').sort(KEYS), ties


def expand(lists):
    return lists.sort('pair_id').select('pair_id', *[pl.col('hands').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(c)
                                                     for i, c in enumerate(CSP.EVIDENCE_COLUMNS)])


def top5_changes(new, base):
    a, b = new.sort('pair_id'), base.sort('pair_id')
    assert a['pair_id'].equals(b['pair_id'])
    return float(np.mean([len(set(u) - set(v)) for u, v in zip(a.select(CSP.EVIDENCE_COLUMNS).rows(), b.select(CSP.EVIDENCE_COLUMNS).rows())]))


def fit_models():
    data, chrono, struct, r382, overlap = CSP.development_frame()
    players = pl.read_csv(ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    requested = data.filter(pl.col('s1_rank') <= TOP).select('pair_id', 'hand_id').unique().sort('pair_id', 'hand_id')
    state, accounting = CSP.build_for_keys(requested, players)
    assert state.select(KEYS).equals(requested)
    data = data.join(state, on=KEYS, how='left', validate='1:1')
    t0 = time.perf_counter()
    pref, swaps = S98.prefix_for_keys(requested)
    mirror, _ = prefix_rows(requested, pair_inputs(requested, players))
    assert mirror.equals(pref), 'prefix mirror differs from the step-98 extractor on development keys'
    union = pl.read_parquet(S98.OUT / 'prefix-features-union.parquet').sort(KEYS)
    both = pref.join(union, on=KEYS, how='inner', suffix='_s98')
    overlap_exact = bool(np.array_equal(both.select(RS).to_numpy(), both.select([c + '_s98' for c in RS]).to_numpy()))
    assert overlap_exact and both.height > 1000, both.height
    print('prefix features', requested.height, 'keys, mirror exact, step-98 overlap', both.height, round(time.perf_counter() - t0, 1), 's', flush=True)
    keys0 = data.select(KEYS)
    data = data.join(pref, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert data.select(KEYS).equals(keys0)
    assert data.filter(pl.col('s1_rank') <= TOP).select(STATE + RS).null_count().sum_horizontal().item() == 0
    r382b, prev = joblib.load(CSP.R382_OUT / 'stage2-models.joblib'), joblib.load(CSP.PREV_OUT / 'stage2-models.joblib')
    codex = joblib.load(CODEX_OUT / 'stage2-models.joblib')
    models, cols, receipts = {}, {}, []
    for family in FAMILIES:
        base_cols = prev['cols']['chrono3'] if family == CI else r382b['cols']['r382']
        ccols = next(r['columns'] for r in codex['receipts'] if r['family'] == family)
        assert ccols == base_cols + STATE
        top = data.filter((pl.col('behavior_family') == family) & (pl.col('s1_rank') <= TOP))
        refit, _, _ = CSP.stage2_fit(data, ccols, family, CLF)
        np.testing.assert_array_equal(CSP.stage2_predict(refit, top, ccols), CSP.stage2_predict(codex['models'][family], top, ccols))
        cols[family] = ccols + RS
        models[family], nq, npos = CSP.stage2_fit(data, cols[family], family, CLF)
        receipts.append({'family': family, 'n_columns': len(cols[family]), 'training_pairs': int(nq), 'training_listed': int(npos),
                         'seeds': list(CSP.SEEDS), 'b0_refit_equals_codex_saved': True})
        print('family', family, 'B0 refit equals Codex saved; S* fitted on', nq, 'pairs', flush=True)
    bundle = {'models': models, 'cols': cols, 'struct': struct, 'receipts': receipts}
    joblib.dump(bundle, OUT / 'stage2-models.joblib')
    restored = joblib.load(OUT / 'stage2-models.joblib')
    for family in FAMILIES:
        top = data.filter((pl.col('behavior_family') == family) & (pl.col('s1_rank') <= TOP))
        np.testing.assert_array_equal(CSP.stage2_predict(models[family], top, cols[family]),
                                      CSP.stage2_predict(restored['models'][family], top, cols[family]))
    write_json(OUT / 'fit.json', {'registration_sha256': PREREG_SHA, 'models_sha256': sha(OUT / 'stage2-models.joblib'), 'state_accounting': accounting,
                                  'struct_overlap': overlap, 'dev_top20_keys': requested.height, 'prefix_mirror_exact_on_dev_keys': True,
                                  'prefix_overlap_with_step98_union': both.height, 'prefix_overlap_exact': overlap_exact, 'prefix_swap_checks': swaps,
                                  'receipts': receipts, 'reload_exact': True})
    return bundle


def build_batch(index, path, bundle, ctx):
    """Codex build_batches line for line up to the state features; then prefix features, B0 replay, S*, T1 and decode."""
    name = path.stem
    receipt = OUT / 'batches' / (name + '.json')
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        assert saved['registration_sha256'] == PREREG_SHA and all(sha(OUT / 'batches' / p) == h for p, h in saved['files'].items())
        return saved
    t0 = time.perf_counter()
    raw = pl.read_parquet(path); events = [c for c in raw.columns if c.startswith('v2_')]
    eq = pl.read_parquet(ROOT / 'reports/evidence-public-calibration' / (name + '-features.parquet'))
    ca = pl.read_parquet(CSP.PACKAGE / 'results' / (name + '-evidence.parquet')).drop('table_id')
    policy_receipt = json.loads((CSP.PACKAGE / 'results' / (name + '.json')).read_text())
    assert all(sha(CSP.PACKAGE / 'results' / p) == h for p, h in policy_receipt['files'].items())
    cov = ROOT / 'outputs_v13_coverage' / 'batches'
    coverage_receipt = json.loads((cov / (name + '.json')).read_text()); assert all(sha(cov / p) == h for p, h in coverage_receipt['files'].items())
    caf, SEQ, EQ = ctx['caf'], CSP.SEQ, CSP.EQ
    extra = pl.concat([eq.join(ca, on=KEYS, validate='1:1').select(KEYS + SEQ + EQ + caf),
                       pl.read_parquet(cov / (name + '-features.parquet')).join(ctx['added'], on='pair_id', how='semi').select(KEYS + SEQ + EQ + caf)])
    sfile = CSP.FS2 / 'eval_structural' / (name + '.parquet'); assert sha(sfile) == ctx['structural_receipt']['files'][sfile.name]; sfeat = pl.read_parquet(sfile)
    frames = []; pools = sorted(raw['table_id'].unique())
    for offset in range(0, len(pools), 5):
        part = raw.filter(pl.col('table_id').is_in(pools[offset:offset + 5])).sort('pair_id', 'hand_id').with_columns(pl.Series('tie', np.random.default_rng(931207 + index * 100 + offset).random(raw.filter(pl.col('table_id').is_in(pools[offset:offset + 5])).height))).join(ctx['selected'], on='pair_id', how='semi')
        if not part.height: continue
        frame = part.join(extra, on=KEYS, validate='1:1').join(ctx['source'].select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id'); assert frame.height == part.height
        frame = CSP.add_chronology(CSP.family_columns(CSP.context(frame, events + SEQ + EQ + caf), 'predicted_behavior'), CSP.started_at('evaluation')).sort('pair_id', 'hand_id')
        s1 = ctx['chrono']['model'].predict(CSP.matrix(frame, ctx['chrono']['features']), num_threads=2)
        frame = CSP.with_rank(frame.with_columns(pl.Series('s1', s1)), 's1', 's1_rank').join(sfeat, on=KEYS, how='left', validate='1:1').sort('pair_id', 'hand_id')
        assert frame.select(pl.sum_horizontal(pl.col(bundle['struct']).null_count())).item() == 0; frames.append(frame)
    requested = pl.concat(frames).filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, accounting = CSP.build_for_keys(requested, ctx['players']); assert state.select(KEYS).equals(requested)
    prefix, swaps = prefix_rows(requested, pair_inputs(requested, ctx['players']))
    codex, pieces = ctx['codex'], []
    for frame in frames:
        frame = frame.join(state, on=KEYS, how='left', validate='1:1').sort(KEYS)
        keys0 = frame.select(KEYS)
        frame = frame.join(prefix, on=KEYS, how='left', validate='1:1', maintain_order='left'); assert frame.select(KEYS).equals(keys0)
        top = (frame['s1_rank'] <= TOP).to_numpy()
        assert frame.filter(pl.Series(top)).select(STATE + RS).null_count().sum_horizontal().item() == 0
        fams = frame['predicted_behavior'].to_numpy(); b0 = np.full(frame.height, np.nan); ss = np.full(frame.height, np.nan)
        for family in FAMILIES:
            mask = top & (fams == family)
            if mask.any():
                sub = frame.filter(pl.Series(mask))
                b0[mask] = CSP.stage2_predict(codex['models'][family], sub, next(r['columns'] for r in codex['receipts'] if r['family'] == family))
                ss[mask] = CSP.stage2_predict(bundle['models'][family], sub, bundle['cols'][family])
        s1v = frame['s1'].to_numpy()
        pieces.append(frame.select('pair_id', 'hand_id', 'tie').with_columns(pl.Series('b0', np.where(top, b0 + 10, s1v - 1e6)), pl.Series('ss', np.where(top, ss + 10, s1v - 1e6))))
    bf = pl.concat(pieces).sort(KEYS)
    assert np.isfinite(bf.select('b0', 'ss').to_numpy()).all() and bf.select(KEYS).n_unique() == bf.height
    fc, ties = chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
    assert fc.select(KEYS).equals(bf.select(KEYS))
    attrs = S104.type_attributes(fc.filter(pl.col('score') > 5).select(KEYS), ctx['players_map'])
    t1 = S104.align(fc, attrs)
    decoded = S104.decode(fc, t1)
    lb0, lss, ldec = expand(CSP.rank(bf, bf['b0'].to_numpy())), expand(CSP.rank(bf, bf['ss'].to_numpy())), expand(CSP.rank(bf, decoded))
    saved = pl.read_parquet(CODEX_OUT / 'batches' / (name + '-contestable.parquet')).sort('pair_id')
    assert lb0.equals(saved.select(lb0.columns)), 'B0 lists do not replay Codex saved lists'
    out = OUT / 'batches' / (name + '-decoded.parquet'); ldec.write_parquet(out)
    rec = {'registration_sha256': PREREG_SHA, 'files': {out.name: sha(out)}, 'selected_pairs': ldec.height, 'top20_pair_hands': requested.height,
           'T1_rows': int(t1.sum()), 'started_at_tie_pairs': ties, 'prefix_swap_checks': swaps, 'b0_lists_equal_codex_saved': True,
           'top5_changed_per_pair': {'S_star_vs_base': top5_changes(lss, lb0), 'decoded_vs_base': top5_changes(ldec, lb0), 'decoded_vs_S_star': top5_changes(ldec, lss)},
           'state_accounting': accounting, 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(receipt, rec)
    print(json.dumps({'batch': name, 'pairs': ldec.height, 'T1': rec['T1_rows'], 'ties': ties, 'changed': rec['top5_changed_per_pair'], 's': rec['seconds']}), flush=True)
    del raw, frames, state, bf; gc.collect()
    return rec


def main():
    start = time.perf_counter()
    assert sha(PREREG) == PREREG_SHA, 'registration changed'
    assert sha(AUTH) == AUTH_SHA and json.loads(AUTH.read_text(encoding='utf-8'))['user_new_request_exception'] is True
    assert sha(BASE) == BASE_SHA and sha(CSP.BEST) == CSP.BEST_SHA
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    for k, v in reg['inputs'].items():
        assert sha(Path(v['path'])) == v['sha256'], k
    protected, snap0 = check_protected(), snapshot()
    (OUT / 'batches').mkdir(exist_ok=True)
    mp, fp = OUT / 'stage2-models.joblib', OUT / 'fit.json'
    if mp.exists() or fp.exists():
        assert mp.exists() and fp.exists(), 'orphan model or fit receipt'
        fit = json.loads(fp.read_text(encoding='utf-8'))
        assert fit['registration_sha256'] == PREREG_SHA and fit['models_sha256'] == sha(mp), 'invalid model resume chain'
        bundle = joblib.load(mp)
    else:
        bundle = fit_models()
        gc.collect()
    source = pl.read_csv(CSP.BEST, infer_schema=False)
    assert pl.read_csv(CSP.BASE, infer_schema=False).select('pair_id', 'predicted_behavior').equals(source.select('pair_id', 'predicted_behavior'))
    selected = pl.read_parquet(ROOT / 'reports/replay-v13/selected-pairs.parquet'); assert selected.height == 16060
    old10 = pl.read_parquet(ROOT / 'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id')
    players = pl.read_csv(ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2'])
    ctx = {'source': source, 'selected': selected, 'added': selected.join(old10, on='pair_id', how='anti'), 'players': players,
           'players_map': {p: (a, b) for p, a, b in players.iter_rows()}, 'chrono': joblib.load(CSP.STAGE1),
           'structural_receipt': json.loads((CSP.FS2 / 'eval_structural/receipt.json').read_text()),
           'caf': joblib.load(CSP.CONDITIONAL / 'evidence-model.joblib')['conditional_features'], 'codex': joblib.load(CODEX_OUT / 'stage2-models.joblib')}
    batches = sorted((ROOT / 'outputs_v2/hands_evaluation').glob('batch_*.parquet')); assert len(batches) == 20
    recs = [build_batch(i, p, bundle, ctx) for i, p in enumerate(batches)]
    assert sum(r['selected_pairs'] for r in recs) == 16060
    ev = list(CSP.EVIDENCE_COLUMNS)
    repl = pl.concat([pl.read_parquet(OUT / 'batches' / (p.stem + '-decoded.parquet')) for p in batches]).sort('pair_id')
    sel = selected.select('pair_id').sort('pair_id')
    assert repl.height == 16060 and repl['pair_id'].n_unique() == 16060 and repl.select('pair_id').equals(sel)
    base = pl.read_csv(BASE, infer_schema=False)
    codex_lists = pl.concat([pl.read_parquet(CODEX_OUT / 'batches' / (p.stem + '-contestable.parquet')) for p in batches]).sort('pair_id')
    assert base.join(sel, on='pair_id', how='semi').sort('pair_id').select(codex_lists.columns).equals(codex_lists)
    result = pl.concat([base.join(sel, on='pair_id', how='anti'),
                        base.join(sel, on='pair_id', how='semi').drop(ev).join(repl, on='pair_id', validate='1:1')]).select(base.columns)
    result = base.select('pair_id').join(result, on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(base.select('pair_id', 'risk_score', 'predicted_behavior'))
    assert result.join(sel, on='pair_id', how='anti').sort('pair_id').equals(base.join(sel, on='pair_id', how='anti').sort('pair_id'))
    output = OUT / 'submission_stacked_decoder_exploratory.csv'
    assert not output.exists()
    result.write_csv(output)
    assert pl.read_csv(output, infer_schema=False).equals(result)
    validation = validate_submission(output, ROOT / 'data/evaluation_pairs.csv', ROOT / 'data/hands.parquet', ROOT / 'data/seats.parquet',
                                     ROOT / 'data/sample_submission.csv').to_dict()
    assert validation['valid'], validation
    write_json(OUT / 'validation.json', validation)
    cmp = result.select(['pair_id'] + ev).join(base.select(['pair_id'] + ev), on='pair_id', suffix='_base', validate='1:1')
    rows_changed = int(cmp.select(pl.any_horizontal([pl.col(c) != pl.col(c + '_base') for c in ev])).to_series().sum())
    weights = [r['selected_pairs'] for r in recs]
    changed = {k: float(np.average([r['top5_changed_per_pair'][k] for r in recs], weights=weights)) for k in recs[0]['top5_changed_per_pair']}
    write_json(OUT / 'receipt.json', {
        'registration_sha256': PREREG_SHA, 'authorization_sha256': AUTH_SHA, 'base_sha256': BASE_SHA, 'fit_sha256': sha(fp), 'models_sha256': sha(mp),
        'output': output.as_posix(), 'output_sha256': sha(output), 'validation_sha256': sha(OUT / 'validation.json'),
        'batch_receipts': {p.stem: sha(OUT / 'batches' / (p.stem + '.json')) for p in batches},
        'pairs_with_changed_evidence_rows': rows_changed, 'top5_changed_per_pair_weighted': changed,
        'T1_rows': sum(r['T1_rows'] for r in recs), 'started_at_tie_pairs': sum(r['started_at_tie_pairs'] for r in recs),
        'risk_behavior_exact': True, 'unselected_rows_exact': True, 'b0_lists_equal_codex_saved_all_batches': True,
        'protected_unchanged': check_protected() == protected, 'codex_files_unchanged': snapshot() == snap0,
        'no_submission_actions': True, 'seconds': round(time.perf_counter() - start, 1)})
    r = json.loads((OUT / 'receipt.json').read_text(encoding='utf-8'))
    assert r['protected_unchanged'] and r['codex_files_unchanged']
    print('DONE', json.dumps({k: r[k] for k in ('output_sha256', 'pairs_with_changed_evidence_rows', 'top5_changed_per_pair_weighted', 'T1_rows',
                                                'started_at_tie_pairs', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
