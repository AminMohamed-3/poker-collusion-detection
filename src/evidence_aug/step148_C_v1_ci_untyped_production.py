"""Step 148 (Claude): time-order decode(v1) for coordinated_isolation-labelled selected pairs on the public .90096 file. Never uploads.
STEP148_ONLY=0,3 runs only those batches (full typed replay there, no CSV)."""
import os, sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
from evidence_aug.paths import sha, check_protected
from validation import validate_submission
from evidence_aug import step104_C_typed_decoder_build as S104
from evidence_aug import step108_C_stacked_decoder_production as S108
CSP, KEYS = S108.CSP, S108.KEYS
ROOT = CSP.ROOT
V1 = POKER / 'Claude Version/outputs_v1_view_averaged_exploratory_20260913'
OUT = POKER / 'Claude Version/outputs_v1_followups_20260914'
PREREG, PREREG_SHA = OUT / 'prediction_step148.json', 'f9f9f6dfd723e5dcd25ebe1fc5ade58d7619dbf5465690ad9a5ae8f88f050b90'
BASE = V1 / 'submission_v1_view_averaged_original_labels.csv'
CSV = OUT / 'submission_v1_ci_untyped.csv'
EV = list(CSP.EVIDENCE_COLUMNS)


def flag(frame, pairs):
    return frame.select('pair_id').join(pairs.with_columns(pl.lit(True).alias('f')), on='pair_id', how='left', maintain_order='left')['f'].fill_null(False).to_numpy()


def main():
    t0 = time.perf_counter()
    assert sha(PREREG) == PREREG_SHA
    for k, v in json.loads(PREREG.read_text(encoding='utf-8'))['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], k
    prot = check_protected()
    base = pl.read_csv(BASE, infer_schema=False)
    ci = base.filter(pl.col('predicted_behavior') == 'coordinated_isolation').select('pair_id')
    pmap = {p: (a, b) for p, a, b in pl.read_csv(ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    only = [int(x) for x in os.environ.get('STEP148_ONLY', '').split(',') if x]
    (OUT / 'batches').mkdir(exist_ok=True)
    news, recs = [], []
    for i in [b for b in range(20) if not only or b in only]:
        name, tb = 'batch_%03d' % i, time.perf_counter()
        saved = pl.read_parquet(V1 / 'batches' / (name + '-v1-decoded.parquet')).sort('pair_id')
        top = pl.read_parquet(V1 / 'batches' / (name + '-top20-scores.parquet')).sort(KEYS)
        bf = top.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'), (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1'))
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        full = i == 0 or bool(only)
        cib = saved.select('pair_id').join(ci, on='pair_id', how='semi')
        keys = fc.select(KEYS) if full else fc.select(KEYS).join(cib, on='pair_id', how='semi')
        t1 = S104.align(fc, S104.type_attributes(keys, pmap))
        fv = fc.with_columns(pl.Series('score', bf['v1'].to_numpy()))
        typed = S108.expand(CSP.rank(bf, S104.decode(fv, t1)))
        chk = saved if full else saved.join(cib, on='pair_id', how='semi')
        assert typed.join(chk.select('pair_id'), on='pair_id', how='semi').equals(chk), 'typed replay mismatch ' + name
        is_ci = flag(fc, cib)
        untyped = S108.expand(CSP.rank(bf, S104.decode(fv, t1 & ~is_ci))).join(cib, on='pair_id', how='semi')
        old = saved.join(cib, on='pair_id', how='semi')
        set_change = float(np.mean([len(set(u[1:]) - set(o[1:])) for u, o in zip(untyped.rows(), old.rows())])) if old.height else 0.0
        order_change = int(sum(u != o for u, o in zip(untyped.rows(), old.rows())))
        untyped.write_parquet(OUT / 'batches' / (name + '-ci-untyped.parquet'))
        news.append(untyped)
        recs.append({'batch': name, 'ci_pairs': cib.height, 'full_typed_replay': full, 'replayed_pairs': chk.height, 'T1_rows_in_scope': int(t1.sum()),
                     'ci_lists_changed': order_change, 'ci_top5_set_change_mean': round(set_change, 4), 's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)
    if only:
        assert check_protected() == prot; print('partial run done'); return
    repl = pl.concat(news).sort('pair_id')
    sel_ci = ci.join(pl.concat([pl.read_parquet(V1 / 'batches' / ('batch_%03d-v1-decoded.parquet' % b)).select('pair_id') for b in range(20)]), on='pair_id', how='semi')
    assert repl.height == repl['pair_id'].n_unique() == sel_ci.height
    rest = base.join(repl.select('pair_id'), on='pair_id', how='anti')
    swap = base.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(repl, on='pair_id', validate='1:1')
    result = base.select('pair_id').join(pl.concat([rest, swap.select(base.columns)]), on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(base.select('pair_id', 'risk_score', 'predicted_behavior'))
    assert not CSV.exists()
    result.write_csv(CSV)
    a, b = BASE.read_bytes().split(b'\n'), CSV.read_bytes().split(b'\n')
    diff = [k for k in range(len(a)) if a[k] != b[k]] if len(a) == len(b) else None
    assert diff is not None and len(diff) <= repl.height and {a[k].split(b',')[0].decode() for k in diff} <= set(repl['pair_id'])
    val = validate_submission(CSV, ROOT / 'data/evaluation_pairs.csv', ROOT / 'data/hands.parquet', ROOT / 'data/seats.parquet', ROOT / 'data/sample_submission.csv')
    rec = {'registration_sha256': PREREG_SHA, 'base_sha256': sha(BASE), 'output': str(CSV), 'output_sha256': sha(CSV), 'rows_changed_vs_base': len(diff),
           'ci_pairs_routed': repl.height, 'batches': recs, 'validation': json.loads(json.dumps(val, default=str)), 'seconds': round(time.perf_counter() - t0, 1)}
    (OUT / 'receipt_step148.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    assert check_protected() == prot
    print(json.dumps({k: rec[k] for k in ('output_sha256', 'rows_changed_vs_base', 'ci_pairs_routed', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
