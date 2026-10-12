"""Step 128 (Claude): the user-authorized exploratory production build of decode(v1) evidence rows onto the .89466 file. It never uploads.
v1 = the step-108 S* production models averaged over the original pool and 40 label-free pool views (20 thinned, 20 padded), recomputing exactly
the fields the step-124 audit views recompute. Registered before any build: outputs_v1_view_averaged_exploratory_20260913/prediction.json.
Codex modules are imported read-only with bytecode off; nothing is written outside the output folder."""
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
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import joblib
import numpy as np
import polars as pl
from validation import validate_submission
from evidence_aug.paths import sha, check_protected, write_json
from evidence_aug.frame import FAMILIES
from evidence_aug import step104_C_typed_decoder_build as S104
from evidence_aug import step108_C_stacked_decoder_production as S108

CSP = S108.CSP
ROOT = CSP.ROOT
OUT = CSP.CV / 'outputs_v1_view_averaged_exploratory_20260913'
PREREG, PREREG_SHA = OUT / 'prediction.json', '7431e753a7b3115978365e413fa73ede5b2a496b57a6733677fd897e1636213e'
AUTH, AUTH_SHA = OUT / 'exploratory-authorization.json', 'bc8eff485e1e6b6c11e9132afd3ae4a79770f8b9c1b74a0118dfc2ff4a84c88e'
S108_OUT = S108.OUT
MODELS_SHA = '91c8fc809907e4d42d980b36a943aad65fef22421e36e74c83604f87eaae894a'
S108_CSV_SHA = 'c3c88c9f6c43ab15ef5b7db4631aa5616b007ef4a7e278a5174a2fbb36636ede'
KEYS, TOP = S108.KEYS, S108.TOP
U, NV = .5, 20
CHRONO_FIELDS = ['chrono_rank', 'chrono_frac', 'n_hands', 's1_rank']


def view(frame, src, pool, kind, u, rng):
    """One label-free pool view of a part frame; returns the top-20 rows with every pool-dependent field recomputed (step-124 audit recipe)."""
    top0 = (frame['s1_rank'] <= TOP).to_numpy()
    pick = (rng.random(frame.height) < u) & ~top0
    base = frame.select(KEYS + ['chrono_rank', 's1'] + src).with_columns(pl.Series('top0', top0))
    if kind == 'thin':
        v = base.filter(pl.Series(~pick))
    else:
        cmax = float(frame['chrono_rank'].max()) + 1
        v = pl.concat([base, base.filter(pl.Series(pick)).with_columns((pl.col('chrono_rank') + cmax * 10).alias('chrono_rank'), pl.lit(False).alias('top0'))])
    v = v.with_columns(pl.col('chrono_rank').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'),
                       pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'),
                       pl.col('s1').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('s1_rank'))
    v = v.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'))
    v = CSP.context(v, src)
    return v.filter(pl.col('top0')).select(KEYS + pool).sort(KEYS)


def assemble(index, path, bundle, ctx):
    """step-108 build_batch frame assembly, line for line (same tie seeds); returns part frames with their offsets."""
    name = path.stem
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
        assert frame.select(pl.sum_horizontal(pl.col(bundle['struct']).null_count())).item() == 0; frames.append((offset, frame))
    del raw, extra
    return frames, events + SEQ + EQ + caf


def build_batch(index, path, bundle, ctx):
    name = path.stem
    receipt = OUT / 'batches' / (name + '.json')
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        assert saved['registration_sha256'] == PREREG_SHA and all(sha(OUT / 'batches' / p) == h for p, h in saved['files'].items())
        return saved
    t0 = time.perf_counter()
    frames, src = assemble(index, path, bundle, ctx)
    assert len(src) == 136
    pool = CHRONO_FIELDS + ['context_' + c for c in src]
    for family in FAMILIES:
        ctxcols = {c for c in bundle['cols'][family] if c.startswith('context_')}
        assert ctxcols in (set(), set(pool[4:])), family
    requested = pl.concat([f for _, f in frames]).filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, accounting = CSP.build_for_keys(requested, ctx['players']); assert state.select(KEYS).equals(requested)
    prefix, swaps = S108.prefix_rows(requested, S108.pair_inputs(requested, ctx['players']))
    print(json.dumps({'batch': name, 'prepared_parts': [(o, f.height) for o, f in frames], 's': round(time.perf_counter() - t0, 1)}), flush=True)
    pieces, scores, nviews, u0 = [], [], 0, 0
    for offset, frame in frames:
        frame = frame.join(state, on=KEYS, how='left', validate='1:1').sort(KEYS)
        keys0 = frame.select(KEYS)
        frame = frame.join(prefix, on=KEYS, how='left', validate='1:1', maintain_order='left'); assert frame.select(KEYS).equals(keys0)
        top = (frame['s1_rank'] <= TOP).to_numpy()
        tf = frame.filter(pl.Series(top)).sort(KEYS)
        assert tf.select(S108.STATE + S108.RS).null_count().sum_horizontal().item() == 0
        fams = tf['predicted_behavior'].to_numpy()

        def score(t):
            out = np.full(t.height, np.nan)
            for family in FAMILIES:
                m = fams == family
                if m.any():
                    out[m] = CSP.stage2_predict(bundle['models'][family], t.filter(pl.Series(m)), bundle['cols'][family])
            assert np.isfinite(out).all()
            return out

        ss = score(tf); acc = ss.copy()
        ref = tf.select(KEYS + pool)
        for kind in ('thin', 'pad'):
            assert view(frame, src, pool, kind, 0., np.random.default_rng(0)).equals(ref), f'u=0 {kind} view does not replay {name} part {offset}'
        u0 += 1
        rest = tf.drop(pool)
        for k, kind in ((1, 'thin'), (2, 'pad')):
            for c in range(NV):
                v = view(frame, src, pool, kind, U, np.random.default_rng([128, k, index, offset, c]))
                t = rest.join(v, on=KEYS, how='left', validate='1:1', maintain_order='left')
                assert t.select(KEYS).equals(tf.select(KEYS)) and np.array_equal(t['s1_rank'].to_numpy(), tf['s1_rank'].to_numpy())
                acc += score(t); nviews += 1
                del v, t
        v1 = acc / (2 * NV + 1)
        sc = tf.select(KEYS).with_columns(pl.Series('ss', ss), pl.Series('v1', v1))
        scores.append(sc)
        full = frame.select('pair_id', 'hand_id', 'tie', 's1').join(sc, on=KEYS, how='left', validate='1:1', maintain_order='left')
        assert np.array_equal(full['ss'].is_not_null().to_numpy(), top)
        s1v = full['s1'].to_numpy()
        pieces.append(full.select('pair_id', 'hand_id', 'tie').with_columns(pl.Series('ss', np.where(top, full['ss'].fill_null(0).to_numpy() + 10, s1v - 1e6)),
                                                                              pl.Series('v1', np.where(top, full['v1'].fill_null(0).to_numpy() + 10, s1v - 1e6))))
        print(json.dumps({'batch': name, 'part': offset, 'top_rows': tf.height, 'views': nviews, 's': round(time.perf_counter() - t0, 1)}), flush=True)
        del frame, tf, rest; gc.collect()
    bf = pl.concat(pieces).sort(KEYS)
    assert np.isfinite(bf.select('ss', 'v1').to_numpy()).all() and bf.select(KEYS).n_unique() == bf.height
    fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
    assert fc.select(KEYS).equals(bf.select(KEYS))
    attrs = S104.type_attributes(fc.filter(pl.col('score') > 5).select(KEYS), ctx['players_map'])
    t1 = S104.align(fc, attrs)
    dec_ss = S104.decode(fc, t1)
    dec_v1 = S104.decode(fc.with_columns(pl.Series('score', bf['v1'].to_numpy())), t1)
    l_ss, l_v1 = S108.expand(CSP.rank(bf, dec_ss)), S108.expand(CSP.rank(bf, dec_v1))
    s108_receipt = json.loads((S108_OUT / 'batches' / (name + '.json')).read_text(encoding='utf-8'))
    s108_file = S108_OUT / 'batches' / (name + '-decoded.parquet')
    assert sha(s108_file) == s108_receipt['files'][s108_file.name]
    assert l_ss.equals(pl.read_parquet(s108_file)), 'no-view decode(S*) does not replay the step-108 saved lists'
    out = OUT / 'batches' / (name + '-v1-decoded.parquet'); l_v1.write_parquet(out)
    sout = OUT / 'batches' / (name + '-top20-scores.parquet'); pl.concat(scores).sort(KEYS).write_parquet(sout)
    rec = {'registration_sha256': PREREG_SHA, 'files': {out.name: sha(out), sout.name: sha(sout)}, 'selected_pairs': l_v1.height, 'top20_pair_hands': requested.height,
           'views_scored': nviews, 'parts': len(frames), 'u0_view_replay_exact_parts': u0, 'step108_decoded_replay_exact': True,
           'T1_rows': int(t1.sum()), 'started_at_tie_pairs': ties, 'prefix_swap_checks': swaps,
           'top5_changed_per_pair': {'decoded_v1_vs_decoded_S_star': S108.top5_changes(l_v1, l_ss)},
           'mean_abs_v1_minus_S_star_top20': float(np.mean(np.abs(bf['v1'].to_numpy() - bf['ss'].to_numpy())[bf['ss'].to_numpy() > 5])),
           'state_accounting': accounting, 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(receipt, rec)
    print(json.dumps({'batch': name, 'pairs': l_v1.height, 'views': nviews, 'changed': rec['top5_changed_per_pair'], 's': rec['seconds']}), flush=True)
    del frames, state, bf; gc.collect()
    return rec


def main():
    start = time.perf_counter()
    if os.environ.get('STEP128_DEBUG'):
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ['STEP128_DEBUG']), repeat=True)
    assert sha(PREREG) == PREREG_SHA, 'registration changed'
    assert sha(AUTH) == AUTH_SHA and json.loads(AUTH.read_text(encoding='utf-8'))['user_new_request_exception'] is True
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    for k, v in reg['inputs'].items():
        assert sha(Path(v['path'])) == v['sha256'], k
    assert sha(S108.BASE) == S108.BASE_SHA and sha(CSP.BEST) == CSP.BEST_SHA
    protected, snap0 = check_protected(), S108.snapshot()
    (OUT / 'batches').mkdir(exist_ok=True)
    mp = S108_OUT / 'stage2-models.joblib'
    assert sha(mp) == MODELS_SHA
    bundle = joblib.load(mp)
    source = pl.read_csv(CSP.BEST, infer_schema=False)
    selected = pl.read_parquet(ROOT / 'reports/replay-v13/selected-pairs.parquet'); assert selected.height == 16060
    old10 = pl.read_parquet(ROOT / 'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id')
    players = pl.read_csv(ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2'])
    ctx = {'source': source, 'selected': selected, 'added': selected.join(old10, on='pair_id', how='anti'), 'players': players,
           'players_map': {p: (a, b) for p, a, b in players.iter_rows()}, 'chrono': joblib.load(CSP.STAGE1),
           'structural_receipt': json.loads((CSP.FS2 / 'eval_structural/receipt.json').read_text()),
           'caf': joblib.load(CSP.CONDITIONAL / 'evidence-model.joblib')['conditional_features']}
    batches = sorted((ROOT / 'outputs_v2/hands_evaluation').glob('batch_*.parquet')); assert len(batches) == 20
    only = [int(x) for x in os.environ.get('STEP128_ONLY', '').split(',') if x]
    recs = [build_batch(i, p, bundle, ctx) for i, p in enumerate(batches) if not only or i in only]
    if only:
        print('partial run done', only, round(time.perf_counter() - start, 1), 's'); assert check_protected() == protected and S108.snapshot() == snap0
        return
    assert sum(r['selected_pairs'] for r in recs) == 16060
    ev = list(CSP.EVIDENCE_COLUMNS)
    repl = pl.concat([pl.read_parquet(OUT / 'batches' / (p.stem + '-v1-decoded.parquet')) for p in batches]).sort('pair_id')
    sel = selected.select('pair_id').sort('pair_id')
    assert repl.height == 16060 and repl['pair_id'].n_unique() == 16060 and repl.select('pair_id').equals(sel)
    base = pl.read_csv(S108.BASE, infer_schema=False)
    result = pl.concat([base.join(sel, on='pair_id', how='anti'),
                        base.join(sel, on='pair_id', how='semi').drop(ev).join(repl, on='pair_id', validate='1:1')]).select(base.columns)
    result = base.select('pair_id').join(result, on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(base.select('pair_id', 'risk_score', 'predicted_behavior'))
    assert result.join(sel, on='pair_id', how='anti').sort('pair_id').equals(base.join(sel, on='pair_id', how='anti').sort('pair_id'))
    s108csv = S108_OUT / 'submission_stacked_decoder_exploratory.csv'; assert sha(s108csv) == S108_CSV_SHA
    s108 = pl.read_csv(s108csv, infer_schema=False)
    output = OUT / 'submission_v1_view_averaged_exploratory.csv'
    assert not output.exists()
    result.write_csv(output)
    assert pl.read_csv(output, infer_schema=False).equals(result)
    validation = validate_submission(output, ROOT / 'data/evaluation_pairs.csv', ROOT / 'data/hands.parquet', ROOT / 'data/seats.parquet',
                                     ROOT / 'data/sample_submission.csv').to_dict()
    assert validation['valid'], validation
    write_json(OUT / 'validation.json', validation)

    def changed(other):
        cmp = result.select(['pair_id'] + ev).join(other.select(['pair_id'] + ev), on='pair_id', suffix='_o', validate='1:1')
        return int(cmp.select(pl.any_horizontal([pl.col(c) != pl.col(c + '_o') for c in ev])).to_series().sum())
    weights = [r['selected_pairs'] for r in recs]
    write_json(OUT / 'receipt.json', {
        'registration_sha256': PREREG_SHA, 'authorization_sha256': AUTH_SHA, 'base_sha256': S108.BASE_SHA, 'models_sha256': MODELS_SHA,
        'output': output.as_posix(), 'output_sha256': sha(output), 'validation_sha256': sha(OUT / 'validation.json'),
        'batch_receipts': {p.stem: sha(OUT / 'batches' / (p.stem + '.json')) for p in batches},
        'pairs_with_changed_evidence_rows_vs_base': changed(base), 'pairs_with_changed_evidence_rows_vs_step108_file': changed(s108),
        'top5_changed_per_pair_weighted_decoded_v1_vs_decoded_S_star': float(np.average([r['top5_changed_per_pair']['decoded_v1_vs_decoded_S_star'] for r in recs], weights=weights)),
        'views_scored': sum(r['views_scored'] for r in recs), 'u0_view_replay_exact_parts': sum(r['u0_view_replay_exact_parts'] for r in recs),
        'step108_decoded_replay_exact_all_batches': True, 'T1_rows': sum(r['T1_rows'] for r in recs),
        'started_at_tie_pairs': sum(r['started_at_tie_pairs'] for r in recs), 'risk_behavior_exact': True, 'unselected_rows_exact': True,
        'protected_unchanged': check_protected() == protected, 'codex_files_unchanged': S108.snapshot() == snap0,
        'no_submission_actions': True, 'seconds': round(time.perf_counter() - start, 1)})
    r = json.loads((OUT / 'receipt.json').read_text(encoding='utf-8'))
    assert r['protected_unchanged'] and r['codex_files_unchanged']
    print('DONE', json.dumps({k: r[k] for k in ('output_sha256', 'pairs_with_changed_evidence_rows_vs_base', 'pairs_with_changed_evidence_rows_vs_step108_file',
                                                'top5_changed_per_pair_weighted_decoded_v1_vs_decoded_S_star', 'views_scored', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
