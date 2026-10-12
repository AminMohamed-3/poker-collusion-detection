"""Step 175b (Claude): production of experiments/hp_search_20260915/registration175.json (decoder input calibration passed its rule:
final a = 1, b = .5). Build only, no fits: the saved STEP167 batch scores (outputs_step167_20260915/work/batches/*-lp-scores.parquet,
receipts checked against registration b1683a39) are re-decoded with p' = sigmoid(logit(p) + .5) applied to v1_lp before the step104
typed decoder (T1 off for predicted coordinated isolation, as STEP167). Gate: decode(v1_lp) without calibration replays the evidence
lists of the .90821 file (3ee17909) for all 16,060 selected pairs exactly. Files (never uploaded by Claude):
  submission_step175_calib_on_90821.csv       the .90821 file with only evidence cells changed (registered)
  submission_step175_calib_on_step173.csv     the STEP173 interleave file (873cdf32) with the same evidence swap (disclosed extra)
Usage: REGSHA"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
from evidence_aug.paths import sha, check_protected
from evidence_aug import step167_C_lp_production as S167
from evidence_aug.step175_C_decoder_calibration import calibrate

CV = POKER / 'Claude Version'
REG = CV / 'experiments/hp_search_20260915/registration175.json'
RES = CV / 'experiments/hp_search_20260915/calib175/results175.json'
REG167 = 'b1683a39646e2c964e93ffae5a3b81d43e90bc488c9a9ca5bd6b7d558a872c74'
BASE, BASE_SHA = CV / 'outputs_step167_20260915/submission_step167_lp_on_step169.csv', '3ee17909242f0d7ae2aaa30c3f26db401a2b2011a683e00b13ed2805905fec40'
ALT, ALT_SHA = CV / 'outputs_step173_20260915/submission_step173_interleave_on_90821.csv', '873cdf325489efdc91d787b3ce4e0f6db5548a764ec3d2917ed3583f2afd85c7'
OUTD = CV / 'outputs_step175_20260915'


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[1]
    assert sha(REG) == reg_sha and sha(BASE) == BASE_SHA and sha(ALT) == ALT_SHA
    res = json.loads(RES.read_text(encoding='utf-8'))
    assert res['registration_sha256'] == reg_sha and res['proceed'] and res['final_cell'] == 'a1.0_b0.5'
    a, b = 1.0, 0.5
    OUTD.mkdir(parents=True, exist_ok=True)
    S108, _ = S167.modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    from evidence_aug import step104_C_typed_decoder_build as S104
    from validation import validate_submission
    prot = check_protected()
    EV = list(CSP.EVIDENCE_COLUMNS)
    base = pl.read_csv(BASE, infer_schema=False); alt = pl.read_csv(ALT, infer_schema=False)
    assert alt.select(['pair_id', 'predicted_behavior'] + EV).equals(base.select(['pair_id', 'predicted_behavior'] + EV))
    ci = base.filter(pl.col('predicted_behavior') == 'coordinated_isolation').select('pair_id')
    fam = dict(base.select('pair_id', 'predicted_behavior').iter_rows())
    pmap = {p: (x, y) for p, x, y in pl.read_csv(CSP.ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    news, recs = [], []
    for i in range(20):
        name, tb = 'batch_%03d' % i, time.perf_counter()
        rj = S167.WORK / 'batches' / (name + '.json'); sc = S167.WORK / 'batches' / (name + '-lp-scores.parquet')
        r = json.loads(rj.read_text(encoding='utf-8')); assert r['registration_sha256'] == REG167 and r['files'][sc.name] == sha(sc)
        top = pl.read_parquet(sc).sort(KEYS)
        sel = top.select('pair_id').unique().sort('pair_id')
        bf = top.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'), (pl.col('ss') + 10).alias('ss'), (pl.col('v1_lp') + 10).alias('v1_lp'))
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS), pmap))
        cif = fc.select('pair_id').join(ci.with_columns(pl.lit(True).alias('f')), on='pair_id', how='left', maintain_order='left')['f'].fill_null(False).to_numpy()
        typed = t1 & ~cif
        v = bf['v1_lp'].to_numpy()
        inc = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', v)), typed)))
        chk = base.join(sel, on='pair_id', how='semi').select(inc.columns).sort('pair_id')
        assert inc.equals(chk), 'decode(v1_lp) does not replay the .90821 evidence in ' + name
        allrows = np.arange(len(v))
        vc = calibrate(v, allrows, a, b)
        new = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', vc)), typed)))
        assert new.select('pair_id').equals(inc.select('pair_id'))
        news.append(new)
        sets = [len(set(u[1:]) - set(o[1:])) for u, o in zip(new.rows(), inc.rows())]
        fams = [fam[x[0]] for x in new.rows()]
        recs.append({'batch': name, 'selected_pairs': new.height, 'lists_changed': int(sum(u != o for u, o in zip(new.rows(), inc.rows()))),
                     'top5_set_change_mean': round(float(np.mean(sets)), 4),
                     'top5_set_change_by_family': {f: round(float(np.mean([s for s, g in zip(sets, fams) if g == f])), 4) for f in sorted(set(fams))},
                     'mean_p_before_after': [round(float(v.mean() - 10), 4), round(float(vc.mean() - 10), 4)], 's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)
    repl = pl.concat(news).sort('pair_id')
    assert repl.height == repl['pair_id'].n_unique() == 16060
    outs = {}
    for tag, bdf, bpath in (('on_90821', base, BASE), ('on_step173', alt, ALT)):
        rest = bdf.join(repl.select('pair_id'), on='pair_id', how='anti')
        swap = bdf.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(repl, on='pair_id', validate='1:1')
        result = bdf.select('pair_id').join(pl.concat([rest, swap.select(bdf.columns)]), on='pair_id', validate='1:1', maintain_order='left')
        assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(bdf.select('pair_id', 'risk_score', 'predicted_behavior'))
        csv = OUTD / f'submission_step175_calib_{tag}.csv'
        assert not csv.exists()
        result.write_csv(csv)
        A, B = bpath.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
        diff = [k for k in range(len(A)) if A[k] != B[k]] if len(A) == len(B) else None
        assert diff is not None and {A[k].split(b',')[0].decode() for k in diff} <= set(repl['pair_id'])
        for k in diff:
            assert A[k].split(b',')[:3] == B[k].split(b',')[:3]
        val = validate_submission(csv, CSP.ROOT / 'data/evaluation_pairs.csv', CSP.ROOT / 'data/hands.parquet', CSP.ROOT / 'data/seats.parquet', CSP.ROOT / 'data/sample_submission.csv')
        assert not val.errors, val
        outs[tag] = {'output': str(csv), 'output_sha256': sha(csv), 'base_sha256': sha(bpath), 'rows_changed_vs_base': len(diff), 'validation': str(val)[:300]}
    assert check_protected() == prot
    rec = {'registration_sha256': reg_sha, 'calibration': {'a': a, 'b': b}, 'outputs': outs, 'batches': recs, 'lists_changed': int(sum(r['lists_changed'] for r in recs)),
           'top5_set_change_mean': float(np.average([r['top5_set_change_mean'] for r in recs], weights=[r['selected_pairs'] for r in recs])),
           'protected_unchanged': True, 'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    (OUTD / 'receipt_step175.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print('DONE', json.dumps({'outputs': outs, 'lists_changed': rec['lists_changed'], 'top5_set_change_mean': rec['top5_set_change_mean']}), flush=True)


if __name__ == '__main__':
    main()
