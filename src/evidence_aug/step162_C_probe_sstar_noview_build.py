"""Step 162 (Claude): information-gain probe file. Evidence lists of the 16,060 selected pairs decoded from the step-108 S* production scores
without views (STEP128 saved 'ss'), T1 off for predicted coordinated isolation (the STEP148 treatment), on the public .90472 file.
Risk scores and behaviour labels are byte-identical to the base; only evidence cells of selected pairs can differ.
Registered before any build: outputs_step162_20260915/prediction_step162.json. Never uploads.
Gate per batch: decode(v1) with the same T1 and CI rule reproduces the base file's evidence lists for every selected pair exactly."""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import sys, json, time, hashlib
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
OUT = POKER / 'Claude Version/outputs_step162_20260915'
PREREG = OUT / 'prediction_step162.json'
PREREG_SHA = sys.argv[1] if len(sys.argv) > 1 else ''
BASE = POKER / 'Claude Version/outputs_step156_20260915/submission_step156_blockM3x_tailSXN32.csv'
BASE_SHA = '6fdfaae403e97c074194b05e8e7d01aa506d74bcc369a92329477a70b4dcec94'
CSV = OUT / 'submission_step162_probe_Sstar_noview_evidence.csv'
EV = list(CSP.EVIDENCE_COLUMNS)


def flag(frame, pairs):
    return frame.select('pair_id').join(pairs.with_columns(pl.lit(True).alias('f')), on='pair_id', how='left', maintain_order='left')['f'].fill_null(False).to_numpy()


def main():
    t0 = time.perf_counter()
    assert sha(PREREG) == PREREG_SHA, 'pass the registration sha'
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    for k, v in reg['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], k
    assert sha(BASE) == BASE_SHA
    prot = check_protected()
    base = pl.read_csv(BASE, infer_schema=False)
    ci = base.filter(pl.col('predicted_behavior') == 'coordinated_isolation').select('pair_id')
    fam = dict(base.select('pair_id', 'predicted_behavior').iter_rows())
    pmap = {p: (a, b) for p, a, b in pl.read_csv(ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    (OUT / 'batches').mkdir(parents=True, exist_ok=True)
    news, recs = [], []
    for i in range(20):
        name, tb = 'batch_%03d' % i, time.perf_counter()
        bout = OUT / 'batches' / (name + '-sstar-noview.parquet')
        top = pl.read_parquet(V1 / 'batches' / (name + '-top20-scores.parquet')).sort(KEYS)
        sel = top.select('pair_id').unique().sort('pair_id')
        bf = top.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'), (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1'))
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS), pmap))
        typed = t1 & ~flag(fc, ci)
        inc = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', bf['v1'].to_numpy())), typed)))
        chk = base.join(sel, on='pair_id', how='semi').select(inc.columns).sort('pair_id')
        assert inc.equals(chk), 'v1 replay of the base evidence failed in ' + name
        new = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', bf['ss'].to_numpy())), typed)))
        assert new.select('pair_id').equals(inc.select('pair_id'))
        new.write_parquet(bout)
        news.append(new)
        sets = [len(set(u[1:]) - set(o[1:])) for u, o in zip(new.rows(), inc.rows())]
        fams = [fam[r[0]] for r in new.rows()]
        recs.append({'batch': name, 'selected_pairs': new.height, 'T1_rows_used': int(typed.sum()), 'lists_changed': int(sum(u != o for u, o in zip(new.rows(), inc.rows()))),
                     'top5_set_change_mean': round(float(np.mean(sets)), 4),
                     'top5_set_change_by_family': {f: round(float(np.mean([s for s, g in zip(sets, fams) if g == f])), 4) for f in sorted(set(fams))},
                     's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)
    repl = pl.concat(news).sort('pair_id')
    assert repl.height == repl['pair_id'].n_unique() == 16060
    rest = base.join(repl.select('pair_id'), on='pair_id', how='anti')
    swap = base.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(repl, on='pair_id', validate='1:1')
    result = base.select('pair_id').join(pl.concat([rest, swap.select(base.columns)]), on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(base.select('pair_id', 'risk_score', 'predicted_behavior'))
    assert not CSV.exists()
    result.write_csv(CSV)
    a, b = BASE.read_bytes().split(b'\n'), CSV.read_bytes().split(b'\n')
    diff = [k for k in range(len(a)) if a[k] != b[k]] if len(a) == len(b) else None
    assert diff is not None and {a[k].split(b',')[0].decode() for k in diff} <= set(repl['pair_id'])
    val = validate_submission(CSV, ROOT / 'data/evaluation_pairs.csv', ROOT / 'data/hands.parquet', ROOT / 'data/seats.parquet', ROOT / 'data/sample_submission.csv')
    vd = val.to_dict() if hasattr(val, 'to_dict') else val
    rec = {'registration_sha256': PREREG_SHA, 'base_sha256': BASE_SHA, 'output': str(CSV), 'output_sha256': sha(CSV), 'rows_changed_vs_base': len(diff),
           'batches': recs, 'lists_changed': int(sum(r['lists_changed'] for r in recs)),
           'top5_set_change_mean': float(np.average([r['top5_set_change_mean'] for r in recs], weights=[r['selected_pairs'] for r in recs])),
           'validation': json.loads(json.dumps(vd, default=str)), 'protected_unchanged': check_protected() == prot, 'no_submission_actions': True,
           'seconds': round(time.perf_counter() - t0, 1)}
    (OUT / 'receipt_step162.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    assert rec['protected_unchanged']
    print('DONE', json.dumps({k: rec[k] for k in ('output_sha256', 'rows_changed_vs_base', 'lists_changed', 'top5_set_change_mean', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
