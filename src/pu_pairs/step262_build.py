"""STEP262 production: the tail of the STEP251 file re-keyed with the kept MECHPLUS type.

Only runs if experiments/step262_mech_tail_view_20260918/results262.json says build_authorised.

Lineage of the base file's risk column (verified here, not assumed):
  STEP163 gave the tail the FINAL2_32 key (outputs_step163_20260915/eval_views_FINAL2_32.parquet).
  STEP169 then applied the boundary amendment to the STEP168 file: with B = 909 block pairs and
  (N, M) from outputs_step169_20260915/prediction_step169_build.json, the global order became
      blk[:B-M] + topN(tail by k_FINAL2) + blk[B-M:] + (tail order minus those N)
  and the base file's risk strings were handed out by value in that order.
  STEP207 / STEP222 / STEP251 changed only the evidence and behaviour columns, so the STEP251 risk
  column is still exactly that.  The PARITY GATE below re-runs this builder with the new type's
  weight set to 0 (i.e. with the key = k_FINAL2) and requires the base risk column back, byte for byte.

The new key is k_NEW = mean over the ten FINAL2 type columns AND the kept STEP262 type column(s),
the new type brought onto the same scale by the registered affine map.  Only the tail is re-ordered;
predicted_behavior and the five evidence columns are byte-identical; the risk value multiset is
unchanged; the seven protected files are unchanged.  Never uploads.

Usage:  python step262_build.py <sha256 of outputs_step262_20260918/prediction262.json>
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
import sys, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
DATA = ROOT + 'data/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl, lightgbm as lgb, joblib                # noqa: E402
from evidence_aug.paths import ORIGINAL, sha, check_protected            # noqa: E402
from pu_pairs import step262_repro_tail as H                             # noqa: E402
from pu_pairs import step262_arms as A                                   # noqa: E402
from pu_pairs import step159c_C_combine_dev as S159                      # noqa: E402
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded          # noqa: E402
from pu_pairs.build_allpairs_fold import CONF                            # noqa: E402
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import MODEL_PARAMS                          # noqa: E402
from validation import validate_submission                               # noqa: E402

OUT = H.OUT
OUTD = C + 'outputs_step262_20260918/'
BASE = C + 'outputs_step251_20260917/submission_step251_ens4_dtfp3_on_step222.csv'
BASE_SHA = '804c52f2'
B168 = C + 'outputs_step163_20260915/submission_step163.csv'   # the STEP169 amendment's own base (STEP168 kept no type)
V163 = C + 'outputs_step163_20260915/eval_views_FINAL2_32.parquet'
REG169 = C + 'outputs_step169_20260915/prediction_step169_build.json'
EVALAGG = C + 'experiments/step257_eval_mech_scores_20260918/eval_pair_mech_aggregates.parquet'
EVALFRAME = C + 'experiments/eval_frame_20260911/eval_frame.parquet'
DEST = OUTD + 'submission_step262_mechtail_on_step251.csv'
AGG = A.AGG
DRAWS = A.DRAWS


def log(*a):
    print(*a, flush=True)


def eval_inputs(e):
    """the registered 7 inputs, on the evaluation aggregates (same transforms as the dev arms)."""
    return np.column_stack([
        np.log1p(e['cnt_q999'].to_numpy().astype(float)), np.log1p(e['cnt_q99'].to_numpy().astype(float)),
        e['top1_logit'].to_numpy().astype(float), e['top3_logit'].to_numpy().astype(float),
        e['top5_logit'].to_numpy().astype(float), np.log1p(e['noisy_or'].to_numpy().astype(float)),
        np.log(e['n_hands'].to_numpy().astype(float))])


def eval_raw(kind, e, tr, rec):
    """the raw (pre-scale) evaluation value of the kept type.  Evaluation models see every labelled
    development pair (held = set()), exactly as step163a's eval mode does."""
    if kind == 'MECH_T2':
        rec['eval_fit'] = 'none (MECHKEY is the standardised top3_logit)'
        return e['top3_logit'].to_numpy().astype(float)
    features = list(joblib.load(ROOT + 'outputs_v11_conditional/pair-model.joblib')['features']) if kind == 'MECH_T3' else None
    ef = pl.read_parquet(EVALFRAME, columns=list(dict.fromkeys(['pair_id'] + features))) if kind == 'MECH_T3' else None
    acc = np.zeros(e.height)
    cols8 = None
    for k in range(DRAWS):
        t = tr.filter(pl.col('k') == k)
        if kind == 'MECH_T3':
            fr = view_frames_seeded(CONF / 'conditional-fold1-v1' / 'policy-screen' / 'hand-features.parquet',
                                    set(), features, k)
            t = pl.concat([t.filter(pl.col('frac') == f).join(fr[f].select(['pair_id'] + features), on='pair_id', how='inner')
                           for f in (1.0, 0.7, 0.4)])
            del fr
        y = (t['label'] == 1).to_numpy().astype(int)
        Xtr = A.inputs(t, 'mechplus')
        Xte = eval_inputs(e)
        if kind == 'MECH_T3':
            if cols8 is None:
                m0 = lgb.LGBMClassifier(**{**MODEL_PARAMS, 'random_state': 7329, 'n_jobs': 4}).fit(t.select(features).to_numpy(), y)
                g = m0.booster_.feature_importance(importance_type='gain')
                cols8 = [features[i] for i in sorted(np.argsort(-g, kind='stable')[:8])]
                rec['eval_extra_columns'] = cols8
            j = e.select('pair_id').join(ef.select(['pair_id'] + cols8), on='pair_id', how='left', validate='1:1', maintain_order='left')
            assert j[cols8[0]].null_count() == 0
            Xtr = np.column_stack([Xtr, t.select(cols8).to_numpy().astype(float)])
            Xte = np.column_stack([Xte, j.select(cols8).to_numpy().astype(float)])
        m = lgb.LGBMClassifier(**{**MODEL_PARAMS, 'random_state': 7329 + k, 'n_jobs': 4}).fit(Xtr, y)
        acc += S159.lg(m.predict_proba(Xte)[:, 1])
        rec.setdefault('eval_fit', {})[f'draw{k}'] = {'n_train_rows': int(t.height), 'n_pos': int(y.sum())}
    return acc / DRAWS


HEAD = 932        # the M3x block (909) + the 23 STEP169-merged pairs, frozen by the STEP262 brief


def reorder_rest(rest_ids, key_of, promoted=()):
    """The registered production rule: the first HEAD positions of the base file keep their exact
    order AND values; the remaining rows are re-sorted by the new key (ties by pair_id) and handed
    back their own risk strings.  `promoted` is the CENSUS block, lifted to the top of the rest in
    its own order; everything else keeps key order."""
    pset = list(dict.fromkeys(promoted))
    rest_set = set(rest_ids)
    assert all(p in rest_set for p in pset), 'a promoted pair is not in the re-orderable region'
    ps = set(pset)
    bykey = sorted((p for p in rest_ids if p not in ps), key=lambda p: (-key_of[p], p))
    order = pset + bykey
    assert len(order) == len(rest_ids) and len(set(order)) == len(order)
    return order


def main():
    t0 = time.time()
    pred_sha = sys.argv[1]
    os.makedirs(OUTD, exist_ok=True)
    PRED = OUTD + 'prediction262.json'
    assert os.path.exists(PRED) and sha(PRED) == pred_sha, 'write prediction262.json BEFORE the CSV'
    assert not os.path.exists(DEST), DEST
    res = json.load(open(OUT + 'results262.json', encoding='utf-8'))
    cen = json.load(open(OUT + 'results262_census.json', encoding='utf-8')) if os.path.exists(OUT + 'results262_census.json') else {}
    kept = res['kept_types']
    census = bool(cen.get('build_authorised'))
    assert res['build_authorised'] or census, 'no STEP262 file: nothing joined under the registered gates'
    assert kept or census
    assert sha(BASE).startswith(BASE_SHA)
    prot = check_protected()
    rec = {'prediction262_sha256': pred_sha, 'results262_sha256': sha(OUT + 'results262.json'),
           'registration262_sha256': res['registration262_sha256'] if 'registration262_sha256' in res else res['registration_sha256'],
           'base': BASE, 'base_sha256': sha(BASE), 'kept_types': kept, 'census_applied': census}
    if census:
        rec['census_addendum_sha256'] = cen['addendum_sha256']

    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row').with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    assert base.height == 112540
    ranked = base.sort(['r', 'pair_id'], descending=[True, False])
    head_ids = ranked['pair_id'].to_list()[:HEAD]
    rest_ids = ranked['pair_id'].to_list()[HEAD:]
    rest_vals = ranked['risk_score'].to_list()[HEAD:]
    assert len(rest_ids) == 112540 - HEAD
    assert float(ranked['r'][HEAD - 1]) < .5 <= float(ranked['r'][908]), 'the 932/909 seam moved'
    rec['frozen_head'] = {'n': HEAD, 'n_at_or_above_half': int((ranked['r'] >= .5).sum())}

    V = pl.read_parquet(V163)
    tcols = [c for c in V.columns if c not in ('pair_id', 'k_FINAL', 'k_FINAL2')]
    assert len(tcols) == 10, tcols
    chk = V.select((pl.mean_horizontal(tcols) - pl.col('k_FINAL2')).abs().max()).item()
    assert chk < 1e-12, chk
    V = V.sort('pair_id')

    # ---- PARITY GATE: new type weight 0 AND an empty promotion set must give the base risk column
    kf = dict(zip(V['pair_id'].to_list(), V['k_FINAL2'].to_list()))
    order0 = reorder_rest(rest_ids, kf)
    got0 = pl.DataFrame({'pair_id': head_ids + order0, 'new_risk': ranked['risk_score'].to_list()[:HEAD] + rest_vals})
    p0 = base.join(got0, on='pair_id', how='left', validate='1:1', maintain_order='left')
    rec['parity_gate_rows_differing'] = int((p0['new_risk'] != p0['risk_score']).sum())
    assert rec['parity_gate_rows_differing'] == 0, 'PARITY GATE FAILED: weight 0 does not reproduce the base'
    log('parity gate PASS', round(time.time() - t0, 1), 's')

    # ---- the evaluation type column(s)
    e = pl.read_parquet(EVALAGG)
    assert e.height == 112540, e.height
    for c in ['pair_id', 'n_hands'] + AGG:
        assert c in e.columns, (c, e.columns)
    e = e.sort('pair_id')
    assert e['pair_id'].to_list() == V['pair_id'].to_list(), 'eval aggregate / FINAL2_32 pair sets differ'
    tr = pl.read_parquet(OUT + 'train_aggs262.parquet')
    mu = float(np.mean([float(V[c].mean()) for c in tcols]))
    sd = float(np.mean([float(V[c].std()) for c in tcols]))
    rec['eval_scale_map'] = {'mu': mu, 'sd': sd}
    newcols = []
    for kind in kept:
        sub = {}
        raw = eval_raw(kind, e, tr, sub)
        z = (raw - raw.mean()) / raw.std()
        V = V.with_columns(pl.Series(kind, z * sd + mu))
        newcols.append(kind)
        rec[f'eval_{kind}'] = sub
        log('eval type', kind, 'built', round(time.time() - t0, 1), 's')
    V = V.with_columns(pl.mean_horizontal(tcols + newcols).alias('k_NEW'))
    V.write_parquet(OUTD + 'eval_views_FINAL2M_32.parquet')
    rec['eval_views_sha256'] = sha(OUTD + 'eval_views_FINAL2M_32.parquet')
    from scipy.stats import spearmanr
    rec['eval_spearman_k_FINAL2_vs_k_NEW'] = round(float(spearmanr(V['k_FINAL2'], V['k_NEW'])[0]), 4)

    kn = dict(zip(V['pair_id'].to_list(), V['k_NEW'].to_list()))
    promoted = []
    if census:
        ea = e.join(pl.DataFrame({'pair_id': rest_ids}), on='pair_id', how='semi')
        ea = ea.filter(pl.col('cnt_q999') >= 3).sort(['cnt_q999', 'top3_logit'], descending=[True, True])
        promoted = ea['pair_id'].to_list()
        rec['census_promoted'] = {'n': len(promoted),
                                  'cnt_q999_range': [float(ea['cnt_q999'].min()), float(ea['cnt_q999'].max())] if ea.height else None}
    order = reorder_rest(rest_ids, kn, promoted)
    rec['rest_rows_whose_position_changed'] = int(sum(a != b for a, b in zip(order, rest_ids)))
    new = pl.DataFrame({'pair_id': head_ids + order, 'new_risk': ranked['risk_score'].to_list()[:HEAD] + rest_vals})
    res2 = base.join(new, on='pair_id', how='left', validate='1:1', maintain_order='left').sort('row')
    out = res2.with_columns(pl.col('new_risk').alias('risk_score')).select(cols)
    bc = base.sort('row').select(cols)
    assert out.drop('risk_score').equals(bc.drop('risk_score')), 'a non-risk column moved'
    assert sorted(out['risk_score'].to_list()) == sorted(bc['risk_score'].to_list()), 'value multiset changed'
    out.write_csv(DEST)
    Ab, Bb = open(BASE, 'rb').read().split(b'\n'), open(DEST, 'rb').read().split(b'\n')
    assert len(Ab) == len(Bb)
    diff = [i for i in range(len(Ab)) if Ab[i] != Bb[i]]
    for i in diff:
        x, y = Ab[i].split(b','), Bb[i].split(b',')
        assert x[0] == y[0] and x[2:] == y[2:], 'a cell other than risk_score changed'
    back = pl.read_csv(DEST, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    assert back.filter(pl.col('risk_score') >= .5).height == 909
    assert back['risk_score'].n_unique() == base['risk_score'].n_unique()
    val = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet',
                              ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv')
    assert check_protected() == prot
    rec.update({'output': DEST, 'output_sha256': sha(DEST), 'rows_changed_vs_base': len(diff),
                'only_risk_cells_changed': True, 'value_multiset_unchanged': True,
                'validation': json.loads(json.dumps(val, default=str)), 'protected_unchanged': True,
                'build_script_sha256': sha(os.path.abspath(__file__)), 'seconds': round(time.time() - t0, 1)})
    json.dump(rec, open(OUTD + 'receipt_step262.json', 'w', encoding='utf-8'), indent=1)
    log(json.dumps({k: v for k, v in rec.items() if k not in ('validation', 'eval_MECH_T1', 'eval_MECH_T3')}, indent=1))
    log('validation', str(val)[:300])


if __name__ == '__main__':
    main()
