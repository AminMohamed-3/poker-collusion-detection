"""Step 149 (Claude): decode(v1) evidence for directed_transfer-routed selected pairs with the 24 raw eq_* inputs at 4096 preflop runouts,
on the public .90173 file (STEP148). Never uploads. Registered before any build: outputs_v1_followups_20260914/prediction_step149.json.
Phases (each resumable from its receipt): equity (subprocess pool) -> directed_transfer refit -> per-batch STEP128 view scoring of routed
rows with the saved and the new model -> typed decode -> CSV. STEP149_ONLY=0,3 scores only those batches (no CSV)."""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
sys.dont_write_bytecode = True
import gc
import json
import subprocess
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
from evidence_aug import step104_C_typed_decoder_build as S104
from evidence_aug import step108_C_stacked_decoder_production as S108
from evidence_aug import step128_C_v1_view_averaged_production as S128
from evidence_aug import step98_C_prefix_confirm_fit as S98

CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
ROOT, DT = CSP.ROOT, 'directed_transfer'
V1 = S128.OUT
OUT = POKER / 'Claude Version/outputs_v1_followups_20260914'
WORK = OUT / 'step149'
PREREG, PREREG_SHA = OUT / 'prediction_step149.json', '6b8140ed386ef2cf3b7b73833b67dfabb068482e45a7104339469134f9ec1df8'
BASE = OUT / 'submission_v1_ci_untyped.csv'
CSV = OUT / 'submission_v1_ci_untyped_dt_equity4096.csv'
WORKER = POKER / 'Claude Version/src/evidence_aug/step149_C_equity4096_worker.py'
EQ, EV = list(CSP.EQ), list(CSP.EVIDENCE_COLUMNS)
assert len(EQ) == 24


def replace_eq(frame, values):
    """frame with the 24 raw eq_* taken from values where a key matches (same dtype, same column order)."""
    keys0 = frame.select(KEYS)
    g = frame.join(values.select(KEYS + EQ).rename({c: c + '__e' for c in EQ}), on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert g.select(KEYS).equals(keys0)
    return g.with_columns([pl.coalesce(pl.col(c + '__e'), pl.col(c)).cast(frame.schema[c]).alias(c) for c in EQ]).drop([c + '__e' for c in EQ])


def same_eq(frame, values):
    v = frame.select(KEYS).join(values.select(KEYS + EQ), on=KEYS, how='left', validate='1:1', maintain_order='left')
    return v.select(EQ).null_count().sum_horizontal().item() == 0 and np.array_equal(frame.select(EQ).to_numpy(), v.select(EQ).to_numpy())


def equity(tag, keys):
    """256 and 4096 eq_* for keys (pair_id, hand_id, player_1, player_2), computed once by the worker pool."""
    keys = keys.select(KEYS + ['player_1', 'player_2']).sort(KEYS)
    kp, p256, p4096, rec = (WORK / f'equity-{tag}-keys.parquet', WORK / f'equity-{tag}-256.parquet', WORK / f'equity-{tag}-4096.parquet', WORK / f'equity-{tag}.json')
    if rec.exists():
        r = json.loads(rec.read_text(encoding='utf-8'))
        assert r['registration_sha256'] == PREREG_SHA and r['keys_sha256'] == sha(kp) and r['out256_sha256'] == sha(p256) and r['out4096_sha256'] == sha(p4096)
        assert pl.read_parquet(kp).equals(keys)
    else:
        keys.write_parquet(kp)
        wr = WORK / f'equity-{tag}-worker.json'
        subprocess.run([sys.executable, str(WORKER), '--input', str(kp), '--out256', str(p256), '--out4096', str(p4096), '--receipt', str(wr), '--procs', '14'],
                       check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        r = dict(json.loads(wr.read_text(encoding='utf-8')), registration_sha256=PREREG_SHA, keys_sha256=sha(kp))
        write_json(rec, r)
    f256, f4096 = pl.read_parquet(p256), pl.read_parquet(p4096)
    assert f256.select(KEYS).equals(keys.select(KEYS)) and f4096.select(KEYS).equals(keys.select(KEYS))
    return f256, f4096, r


def fit(bundle, d256, d4096):
    mp, fp = WORK / 'dt-equity4096-model.joblib', WORK / 'fit.json'
    cols = bundle['cols'][DT]
    assert all(c in cols for c in EQ) and all('context_' + c in cols for c in EQ)
    if mp.exists() or fp.exists():
        assert mp.exists() and fp.exists(), 'orphan model or fit receipt'
        r = json.loads(fp.read_text(encoding='utf-8'))
        assert r['registration_sha256'] == PREREG_SHA and r['model_sha256'] == sha(mp)
        return joblib.load(mp), r
    t0 = time.perf_counter()
    data, _, _, _, _ = CSP.development_frame()
    players = pl.read_csv(ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    requested = data.filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, players)
    assert state.select(KEYS).equals(requested)
    data = data.join(state, on=KEYS, how='left', validate='1:1')
    pref, _ = S98.prefix_for_keys(requested)
    keys0 = data.select(KEYS)
    data = data.join(pref, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert data.select(KEYS).equals(keys0)
    dtrows = data.filter(pl.col('behavior_family') == DT).sort(KEYS)
    assert dtrows.select(KEYS).equals(d256.select(KEYS)), 'development DT rows differ from the equity keys'
    assert same_eq(dtrows, d256), '256 recompute differs from the cached development eq_* values'
    top = data.filter((pl.col('behavior_family') == DT) & (pl.col('s1_rank') <= TOP))
    refit, nq, npos = CSP.stage2_fit(data, cols, DT, CSP.CLF)
    saved = CSP.stage2_predict(bundle['models'][DT], top, cols)
    np.testing.assert_array_equal(CSP.stage2_predict(refit, top, cols), saved)
    data4 = replace_eq(data, d4096)
    top4 = data4.filter((pl.col('behavior_family') == DT) & (pl.col('s1_rank') <= TOP))
    assert top4.select(KEYS).equals(top.select(KEYS))
    other = [c for c in cols if c not in EQ]
    assert np.array_equal(top4.select(other).to_numpy().astype(float), top.select(other).to_numpy().astype(float), equal_nan=True) and same_eq(top4, d4096)
    model, nq4, npos4 = CSP.stage2_fit(data4, cols, DT, CSP.CLF)
    new = CSP.stage2_predict(model, top4, cols)
    joblib.dump(model, mp)
    np.testing.assert_array_equal(CSP.stage2_predict(joblib.load(mp), top4, cols), new)
    changed = int((np.abs(top4.select(EQ).to_numpy().astype(float) - top.select(EQ).to_numpy().astype(float)).max(axis=1) > 0).sum())
    r = {'registration_sha256': PREREG_SHA, 'model_sha256': sha(mp), 'training_pairs': int(nq4), 'training_listed': int(npos4), 'top20_rows': top.height,
         'top20_rows_eq_changed': changed, 'refit256_equals_saved_step108_model': True, 'dev_eq256_recompute_exact_rows': dtrows.height,
         'train_pred_mean_abs_change': float(np.mean(np.abs(new - saved))), 'train_pred_corr': float(np.corrcoef(new, saved)[0, 1]),
         'n_columns': len(cols), 'seconds': round(time.perf_counter() - t0, 1)}
    assert (nq4, npos4) == (nq, npos)
    write_json(fp, r)
    print('fit', json.dumps(r), flush=True)
    del data, data4; gc.collect()
    return model, r


def score_batch(index, path, bundle, model, ctx, e256, e4096, dt_pairs, rec128):
    name = path.stem
    receipt, sout = WORK / 'batches' / (name + '.json'), WORK / 'batches' / (name + '-dt-scores.parquet')
    if receipt.exists():
        r = json.loads(receipt.read_text(encoding='utf-8'))
        assert r['registration_sha256'] == PREREG_SHA and r['files'][sout.name] == sha(sout)
        return r
    t0 = time.perf_counter()
    frames, src = S128.assemble(index, path, bundle, ctx)
    pool = S128.CHRONO_FIELDS + ['context_' + c for c in src]
    cols = bundle['cols'][DT]
    assert {c for c in cols if c.startswith('context_')} == set(pool[4:]) and not set(EQ) & set(pool)
    requested = pl.concat([f for _, f in frames]).filter((pl.col('s1_rank') <= TOP) & (pl.col('predicted_behavior') == DT)).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, ctx['players'])
    assert state.select(KEYS).equals(requested)
    prefix, swaps = S108.prefix_rows(requested, S108.pair_inputs(requested, ctx['players']))
    print(json.dumps({'batch': name, 'dt_top_rows': requested.height, 'prepared_s': round(time.perf_counter() - t0, 1)}), flush=True)
    pieces, nviews, u0 = [], 0, 0
    for offset, frame in frames:
        frame = frame.sort(KEYS)
        dtm = ((frame['s1_rank'] <= TOP) & (frame['predicted_behavior'] == DT)).to_numpy()
        if not dtm.any():
            continue
        tf = frame.filter(pl.Series(dtm)).join(state, on=KEYS, how='left', validate='1:1').join(prefix, on=KEYS, how='left', validate='1:1').sort(KEYS)
        assert tf.select(S108.STATE + S108.RS).null_count().sum_horizontal().item() == 0
        assert same_eq(tf, e256), f'256 recompute differs from the frame eq_* values in {name} part {offset}'
        tf4 = replace_eq(tf, e4096)
        assert same_eq(tf4, e4096)
        ref = tf.select(KEYS + pool)
        for kind in ('thin', 'pad'):
            v0 = S128.view(frame, src, pool, kind, 0., np.random.default_rng(0))
            assert v0.join(tf.select(KEYS), on=KEYS, how='semi').sort(KEYS).equals(ref), f'u=0 {kind} view does not replay {name} part {offset}'
        u0 += 1
        ss_old, ss_new = CSP.stage2_predict(bundle['models'][DT], tf, cols), CSP.stage2_predict(model, tf4, cols)
        acc_old, acc_new = ss_old.copy(), ss_new.copy()
        rest, rest4 = tf.drop(pool), tf4.drop(pool)
        for k, kind in ((1, 'thin'), (2, 'pad')):
            for c in range(S128.NV):
                v = S128.view(frame, src, pool, kind, S128.U, np.random.default_rng([128, k, index, offset, c]))
                t, t4 = rest.join(v, on=KEYS, how='left', validate='1:1', maintain_order='left'), rest4.join(v, on=KEYS, how='left', validate='1:1', maintain_order='left')
                assert t.select(KEYS).equals(tf.select(KEYS)) and t4.select(KEYS).equals(tf.select(KEYS)) and np.array_equal(t['s1_rank'].to_numpy(), tf['s1_rank'].to_numpy())
                acc_old += CSP.stage2_predict(bundle['models'][DT], t, cols)
                acc_new += CSP.stage2_predict(model, t4, cols)
                nviews += 1
                del v, t, t4
        pieces.append(tf.select(KEYS).with_columns(pl.Series('ss', ss_old), pl.Series('v1', acc_old / (2 * S128.NV + 1)),
                                                   pl.Series('ss_e', ss_new), pl.Series('v1_e', acc_new / (2 * S128.NV + 1))))
        print(json.dumps({'batch': name, 'part': offset, 'dt_rows': tf.height, 'views': nviews, 's': round(time.perf_counter() - t0, 1)}), flush=True)
        del frame, tf, tf4, rest, rest4; gc.collect()
    bf = pl.concat(pieces).sort(KEYS)
    top_file = V1 / 'batches' / (name + '-top20-scores.parquet')
    assert sha(top_file) == rec128['files'][top_file.name]
    saved = pl.read_parquet(top_file).join(dt_pairs, on='pair_id', how='semi').sort(KEYS)
    assert bf.select(KEYS).equals(saved.select(KEYS)) and bf.select(KEYS).equals(requested)
    assert np.array_equal(bf['ss'].to_numpy(), saved['ss'].to_numpy()), 'saved-model no-view scores do not replay STEP128'
    assert np.array_equal(bf['v1'].to_numpy(), saved['v1'].to_numpy()), 'saved-model v1 does not replay STEP128'
    bf.write_parquet(sout)
    dv = bf['v1_e'].to_numpy() - bf['v1'].to_numpy()
    r = {'registration_sha256': PREREG_SHA, 'files': {sout.name: sha(sout)}, 'dt_pairs': bf['pair_id'].n_unique(), 'dt_top20_rows': bf.height,
         'views_scored': nviews, 'u0_view_replay_exact_parts': u0, 'eq256_frame_parity_exact': True, 'step128_ss_v1_replay_exact': True,
         'prefix_swap_checks': swaps, 'v1_mean_abs_change': float(np.abs(dv).mean()), 'v1_corr': float(np.corrcoef(bf['v1_e'].to_numpy(), bf['v1'].to_numpy())[0, 1]),
         'seconds': round(time.perf_counter() - t0, 1)}
    write_json(receipt, r)
    print(json.dumps({k: r[k] for k in ('dt_pairs', 'dt_top20_rows', 'views_scored', 'v1_mean_abs_change', 'seconds')}), flush=True)
    del frames, state, prefix, bf; gc.collect()
    return r


def main():
    start = time.perf_counter()
    assert sha(PREREG) == PREREG_SHA, 'registration changed'
    for k, v in json.loads(PREREG.read_text(encoding='utf-8'))['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], k
    protected, snap0 = check_protected(), S108.snapshot()
    (WORK / 'batches').mkdir(parents=True, exist_ok=True)
    rec128 = json.loads((V1 / 'receipt.json').read_text(encoding='utf-8'))
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    source = pl.read_csv(CSP.BEST, infer_schema=False)
    base = pl.read_csv(BASE, infer_schema=False)
    selected = pl.read_parquet(ROOT / 'reports/replay-v13/selected-pairs.parquet')
    assert selected.height == 16060
    dt_pairs = source.join(selected, on='pair_id', how='semi').filter(pl.col('predicted_behavior') == DT).select('pair_id').sort('pair_id')
    assert dt_pairs.equals(base.join(selected, on='pair_id', how='semi').filter(pl.col('predicted_behavior') == DT).select('pair_id').sort('pair_id'))
    assert dt_pairs.height == 9230
    batches = sorted((ROOT / 'outputs_v2/hands_evaluation').glob('batch_*.parquet'))
    assert len(batches) == 20
    b128 = {}
    for p in batches:
        rj = V1 / 'batches' / (p.stem + '.json')
        assert sha(rj) == rec128['batch_receipts'][p.stem]
        b128[p.stem] = json.loads(rj.read_text(encoding='utf-8'))
    eval_pairs = pl.read_csv(ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2'])
    dev_labels = pl.read_csv(ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2', 'behavior_family'])

    dev_cache = pl.read_parquet(ROOT / 'reports/equity-evidence-experiment/development.parquet')
    dev_keys = dev_cache.select(KEYS).join(dev_labels.filter(pl.col('behavior_family') == DT), on='pair_id', validate='m:1')
    d256, d4096, rdev = equity('dev', dev_keys)
    assert same_eq(dev_cache.join(d256.select(KEYS), on=KEYS, how='semi').sort(KEYS), d256), 'development 256 recompute differs from the cache'
    top_all = pl.concat([pl.read_parquet(V1 / 'batches' / (p.stem + '-top20-scores.parquet')).select(KEYS) for p in batches])
    ev_keys = top_all.join(dt_pairs, on='pair_id', how='semi').join(eval_pairs, on='pair_id', validate='m:1')
    assert ev_keys.height == 184600
    e256, e4096, rev = equity('eval', ev_keys)
    print('equity', json.dumps({'dev_rows': rdev['rows'], 'dev_changed': rdev['rows_any_field_changed'], 'eval_rows': rev['rows'],
                                'eval_changed': rev['rows_any_field_changed'], 's': round(time.perf_counter() - start, 1)}), flush=True)
    model, rfit = fit(bundle, d256, d4096)

    ctx = {'source': source, 'selected': selected, 'added': selected.join(pl.read_parquet(ROOT / 'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id'), on='pair_id', how='anti'),
           'players': eval_pairs, 'chrono': joblib.load(CSP.STAGE1), 'structural_receipt': json.loads((CSP.FS2 / 'eval_structural/receipt.json').read_text()),
           'caf': joblib.load(CSP.CONDITIONAL / 'evidence-model.joblib')['conditional_features']}
    only = [int(x) for x in os.environ.get('STEP149_ONLY', '').split(',') if x]
    recs = [score_batch(i, p, bundle, model, ctx, e256, e4096, dt_pairs, b128[p.stem]) for i, p in enumerate(batches) if not only or i in only]
    if only:
        assert check_protected() == protected and S108.snapshot() == snap0
        print('partial run done', only, round(time.perf_counter() - start, 1), 's', flush=True)
        return

    pmap = {p: (a, b) for p, a, b in eval_pairs.iter_rows()}
    news, olds, drecs = [], [], []
    for p in batches:
        name, tb = p.stem, time.perf_counter()
        lists_file, top_file = V1 / 'batches' / (name + '-v1-decoded.parquet'), V1 / 'batches' / (name + '-top20-scores.parquet')
        assert sha(lists_file) == b128[name]['files'][lists_file.name] and sha(top_file) == b128[name]['files'][top_file.name]
        saved = pl.read_parquet(lists_file).sort('pair_id')
        top = pl.read_parquet(top_file).sort(KEYS)
        dts = pl.read_parquet(WORK / 'batches' / (name + '-dt-scores.parquet')).sort(KEYS)
        joined = top.select(KEYS).join(dts.select(KEYS + ['v1_e']), on=KEYS, how='left', validate='1:1', maintain_order='left')
        routed = joined['v1_e'].is_not_null().to_numpy()
        assert int(routed.sum()) == dts.height
        bf = top.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'), (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1'))
        bf = bf.with_columns(pl.Series('v1e', np.where(routed, joined['v1_e'].fill_null(0).to_numpy() + 10, bf['v1'].to_numpy())))
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        dtb = saved.select('pair_id').join(dt_pairs, on='pair_id', how='semi')
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS).join(dtb, on='pair_id', how='semi'), pmap))
        old = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', bf['v1'].to_numpy())), t1))).join(dtb, on='pair_id', how='semi')
        chk = saved.join(dtb, on='pair_id', how='semi')
        assert old.equals(chk), 'typed replay mismatch ' + name
        new = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', bf['v1e'].to_numpy())), t1))).join(dtb, on='pair_id', how='semi')
        assert new.select('pair_id').equals(old.select('pair_id'))
        set_change = float(np.mean([len(set(u[1:]) - set(o[1:])) for u, o in zip(new.rows(), old.rows())]))
        new.write_parquet(WORK / 'batches' / (name + '-dt-equity4096-decoded.parquet'))
        news.append(new); olds.append(old)
        drecs.append({'batch': name, 'dt_pairs': new.height, 'T1_rows_in_scope': int(t1.sum()), 'dt_lists_changed': int(sum(u != o for u, o in zip(new.rows(), old.rows()))),
                      'dt_top5_set_change_mean': round(set_change, 4), 's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(drecs[-1]), flush=True)
    repl, prev = pl.concat(news).sort('pair_id'), pl.concat(olds).sort('pair_id')
    assert repl.select('pair_id').equals(dt_pairs) and repl['pair_id'].n_unique() == 9230
    assert base.join(dt_pairs, on='pair_id', how='semi').select(prev.columns).sort('pair_id').equals(prev), '.90173 DT evidence differs from the STEP128 lists'
    rest = base.join(dt_pairs, on='pair_id', how='anti')
    swap = base.join(dt_pairs, on='pair_id', how='semi').drop(EV).join(repl, on='pair_id', validate='1:1')
    result = base.select('pair_id').join(pl.concat([rest, swap.select(base.columns)]), on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(base.select('pair_id', 'risk_score', 'predicted_behavior'))
    assert not CSV.exists()
    result.write_csv(CSV)
    a, b = BASE.read_bytes().split(b'\n'), CSV.read_bytes().split(b'\n')
    diff = [k for k in range(len(a)) if a[k] != b[k]] if len(a) == len(b) else None
    assert diff is not None and {a[k].split(b',')[0].decode() for k in diff} <= set(dt_pairs['pair_id'])
    val = validate_submission(CSV, ROOT / 'data/evaluation_pairs.csv', ROOT / 'data/hands.parquet', ROOT / 'data/seats.parquet', ROOT / 'data/sample_submission.csv')
    vd = val.to_dict()
    assert vd['valid'], vd
    rec = {'registration_sha256': PREREG_SHA, 'base_sha256': sha(BASE), 'output': str(CSV), 'output_sha256': sha(CSV), 'rows_changed_vs_base': len(diff),
           'dt_pairs_routed': repl.height, 'fit': rfit, 'equity': {'dev': rdev, 'eval': rev}, 'score_batches': recs, 'decode_batches': drecs,
           'dt_top5_set_change_mean': float(np.average([d['dt_top5_set_change_mean'] for d in drecs], weights=[d['dt_pairs'] for d in drecs])),
           'validation': json.loads(json.dumps(vd, default=str)), 'protected_unchanged': check_protected() == protected,
           'codex_files_unchanged': S108.snapshot() == snap0, 'no_submission_actions': True, 'seconds': round(time.perf_counter() - start, 1)}
    (OUT / 'receipt_step149.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    assert rec['protected_unchanged'] and rec['codex_files_unchanged']
    print('DONE', json.dumps({k: rec[k] for k in ('output_sha256', 'rows_changed_vs_base', 'dt_pairs_routed', 'dt_top5_set_change_mean', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
