"""STEP259-P: the production evidence file carrying the STEP254 MECHPLUS event fusion on DT and SP.

This is src/evidence_aug/step204_C_ci_causal_production.py:phase_build, re-implemented here (step204
is hash-registered and must not be edited) with ONE insertion: between the ENS4 x 40-view averaged
stage-2 probability `ens` and the typed decoder, the rows of pairs whose PREDICTED behaviour is
directed_transfer or soft_play are replaced by

    ens' = clip(ens, 1e-9, 1)**(1-w) * clip(prior_shift(p_evt, F=10), 1e-9, 1-1e-9)**w ,  w = .25

with p_evt = the STEP257 evaluation-phase MECHPLUS probability of that family for that (pair, hand)
(p_DT for DT-predicted pairs, p_SP for SP-predicted pairs).  That is exactly the transformation
src/evidence_aug/step259_A_fusion_composition.py measured on the 162-query audit, where the audit's
`REF[QIX] - 10.0` is the same quantity as production's `ens` (production stores the top-20 stage-2
rows only and adds the same +10 offset to all of them, so subtracting the offset is unnecessary here;
the equivalence is asserted below).  Coordinated isolation is NEVER fused - the audit says it hurts.

Everything else - the member averaging, the chronology ordering, S104.decode, CSP.rank, S108.expand,
the v1 replay gate, the evidence swap into the STEP169 base, validate_submission - is the production
path unchanged.

Usage:  python step259_P_fusion_build.py neutral      # w = 0, must reproduce the STEP251 file byte for byte
        python step259_P_fusion_build.py fused        # w = .25 on DT and SP
Never uploads.
"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
import glob
import json
import time
from pathlib import Path

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np
import polars as pl
from evidence_aug.paths import sha, check_protected, write_json
from evidence_aug import step167_C_lp_production as S167

CV = POKER / 'Claude Version'
S251 = CV / 'outputs_step251_20260917'
S257 = CV / 'experiments/step257_eval_mech_scores_20260918'
OUT = CV / 'outputs_step259_20260918'
BASE = CV / 'outputs_step169_20260915/submission_step169.csv'
BASE_SHA = '67060d0a53b32ef6c790ce0255f5dd5b6230abcae464ba0c3baed02b4543b9c1'
STEP251_FILE = S251 / 'submission_step251_ens4_dtfp3_on_step222.csv'
STEP251_SHA = '804c52f2d4a9e2fdffc91769e48edf713ac489d745874456025e2ee95e7761fb'
MEMBERS = ['INC', 'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3']
F_SHIFT = 10.0
W = .25
FUSED_FAMS = (('directed_transfer', 'p_DT'), ('soft_play', 'p_SP'))
OUTNAME = 'submission_step259_mechfusion_dt_sp_on_step251.csv'


def prior_shift(p, F):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    o = q / (1 - q) * float(F)
    return o / (1 + o)


def event_lookup(keys: pl.DataFrame) -> pl.DataFrame:
    files = sorted(glob.glob(str(S257 / 'eval_hand_scores_batch_*.parquet')))
    assert len(files) == 20, files
    parts = []
    for f in files:
        d = pl.read_parquet(f, columns=['pair_id', 'hand_id', 'p_DT', 'p_SP'])
        d = d.join(keys, on=['pair_id', 'hand_id'], how='semi')
        if d.height:
            parts.append(d)
    lk = pl.concat(parts)
    assert lk.height == keys.height, ('evidence rows without a STEP257 event score',
                                      keys.height, lk.height)
    return lk


def build(w: float, outname: str) -> dict:
    S108, _ = S167.modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    from evidence_aug import step104_C_typed_decoder_build as S104
    from validation import validate_submission
    t0 = time.perf_counter()
    prot = check_protected()
    EV = list(CSP.EVIDENCE_COLUMNS)
    assert sha(BASE) == BASE_SHA, 'base file changed'
    assert sha(STEP251_FILE) == STEP251_SHA, 'the STEP251 file changed'
    base = pl.read_csv(BASE, infer_schema=False)
    ci = base.filter(pl.col('predicted_behavior') == 'coordinated_isolation').select('pair_id')
    fam = dict(base.select('pair_id', 'predicted_behavior').iter_rows())
    pmap = {p: (a, b) for p, a, b in pl.read_csv(
        CSP.ROOT / 'data/evaluation_pairs.csv',
        columns=['pair_id', 'player_1', 'player_2']).iter_rows()}

    # ---- every key the decoder will see, and its event probability
    allkeys = []
    for i in range(20):
        p = S251 / 'work' / MEMBERS[0] / 'batches' / ('batch_%03d-lp-scores.parquet' % i)
        allkeys.append(pl.read_parquet(p, columns=KEYS))
    allkeys = pl.concat(allkeys).sort(KEYS)
    lk = event_lookup(allkeys)
    lkmap = {(a, b): (c, d) for a, b, c, d in lk.iter_rows()}
    print('event scores attached for', len(lkmap), 'evidence rows', flush=True)

    news, recs = [], []
    cov = {'directed_transfer': 0, 'soft_play': 0, 'other': 0}
    ens_min, ens_max = np.inf, -np.inf
    for i in range(20):
        name, tb = 'batch_%03d' % i, time.perf_counter()
        tops = []
        for m in MEMBERS:
            W_ = S251 / 'work' / m
            rj = W_ / 'batches' / (name + '.json')
            sc = W_ / 'batches' / (name + '-lp-scores.parquet')
            r = json.loads(rj.read_text(encoding='utf-8'))
            assert r['files'][sc.name] == sha(sc), (m, name)
            tops.append(pl.read_parquet(sc).sort(KEYS))
        ref = tops[0]
        for t in tops[1:]:
            assert t.select(KEYS).equals(ref.select(KEYS)), 'member key order differs in ' + name
            assert np.array_equal(t['v1'].to_numpy(), ref['v1'].to_numpy()), \
                'v1 must not depend on the member'
        ens = np.mean([t['v1_lp'].to_numpy() for t in tops], axis=0)
        ens_min, ens_max = min(ens_min, ens.min()), max(ens_max, ens.max())
        assert ens.min() >= 1e-9 and ens.max() <= 1.0, \
            'the clip inside the fusion would not be the identity at w = 0'

        # ---------------- the ONE insertion ----------------
        pids = ref['pair_id'].to_list()
        hids = ref['hand_id'].to_list()
        famrow = np.array([fam[p] for p in pids])
        ensf = ens.copy()
        for f, col in FUSED_FAMS:
            mrow = famrow == f
            if not mrow.any():
                continue
            k = 0 if col == 'p_DT' else 1
            pev = np.array([lkmap[(p, h)][k] for p, h, mm in zip(pids, hids, mrow) if mm],
                           dtype=np.float64)
            assert len(pev) == int(mrow.sum()) and np.isfinite(pev).all(), 'fusion coverage gap'
            pe = prior_shift(pev, F_SHIFT)
            ensf[mrow] = (np.clip(ens[mrow], 1e-9, 1.0) ** (1 - w)) * \
                np.clip(pe, 1e-9, 1 - 1e-9) ** w
            cov[f] += int(mrow.sum())
        cov['other'] += int((~np.isin(famrow, [f for f, _ in FUSED_FAMS])).sum())
        if w == 0.0:
            assert np.array_equal(ensf, ens), 'the w = 0 hook is not the identity'
        # ---------------------------------------------------

        bf = ref.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'),
                        (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1')
                        ).with_columns(pl.Series('v1_ens', ensf + 10))
        sel = bf.select('pair_id').unique().sort('pair_id')
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id',
                                                  pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS), pmap))
        cif = fc.select('pair_id').join(ci.with_columns(pl.lit(True).alias('f')), on='pair_id',
                                        how='left', maintain_order='left')['f'
                                                                           ].fill_null(False).to_numpy()
        typed = t1 & ~cif
        inc = S108.expand(CSP.rank(bf, S104.decode(
            fc.with_columns(pl.Series('score', bf['v1'].to_numpy())), typed)))
        chk = base.join(sel, on='pair_id', how='semi').select(inc.columns).sort('pair_id')
        assert inc.equals(chk), 'v1 replay of the base evidence failed in ' + name
        new = S108.expand(CSP.rank(bf, S104.decode(
            fc.with_columns(pl.Series('score', bf['v1_ens'].to_numpy())), typed)))
        assert new.select('pair_id').equals(inc.select('pair_id'))
        news.append(new)
        sets = [len(set(u[1:]) - set(o[1:])) for u, o in zip(new.rows(), inc.rows())]
        fams = [fam[x[0]] for x in new.rows()]
        recs.append({'batch': name, 'selected_pairs': new.height, 'T1_rows_used': int(typed.sum()),
                     'lists_changed': int(sum(u != o for u, o in zip(new.rows(), inc.rows()))),
                     'top5_set_change_mean': round(float(np.mean(sets)), 4),
                     's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)
    repl = pl.concat(news).sort('pair_id')
    assert repl.height == repl['pair_id'].n_unique() == 16060
    rest = base.join(repl.select('pair_id'), on='pair_id', how='anti')
    swap = base.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(
        repl, on='pair_id', validate='1:1')
    result = base.select('pair_id').join(pl.concat([rest, swap.select(base.columns)]),
                                         on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(
        base.select('pair_id', 'risk_score', 'predicted_behavior'))
    csv = OUT / outname
    assert not csv.exists(), 'preserve a built file'
    result.write_csv(csv)

    # ---- verification against STEP251, not against the STEP169 base
    A, B = STEP251_FILE.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
    assert len(A) == len(B)
    diff = [k for k in range(len(A)) if A[k] != B[k]]
    for k in diff:
        assert A[k].split(b',')[:3] == B[k].split(b',')[:3], 'risk or behaviour changed'
    fam_of = {}
    for k in diff:
        p = A[k].split(b',')[0].decode()
        fam_of[fam[p]] = fam_of.get(fam[p], 0) + 1
    ref251 = pl.read_csv(STEP251_FILE, infer_schema=False)
    ra, rb = result.select(EV).rows(), ref251.select(EV).rows()
    first_slot = sum(1 for a, b in zip(ra, rb) if a[0] != b[0])
    setchg = {}
    fams_all = result['predicted_behavior'].to_list()
    for a, b, f in zip(ra, rb, fams_all):
        if a != b:
            setchg[f] = setchg.get(f, [0, 0])
            setchg[f][0] += 1
            setchg[f][1] += len(set(a) - set(b))
    val = validate_submission(csv, POKER / 'data/evaluation_pairs.csv',
                              POKER / 'data/hands.parquet',
                              POKER / 'data/seats.parquet',
                              POKER / 'data/sample_submission.csv')
    errs = getattr(val, 'errors', None)
    assert not errs, str(val)
    assert check_protected() == prot
    nulls = sum(int((result[c] == '').sum() + result[c].null_count()) for c in EV)
    rec = {'w': w, 'output': str(csv.relative_to(CV)), 'output_sha256': sha(csv),
           'equals_step251': sha(csv) == STEP251_SHA,
           'base_sha256': BASE_SHA, 'step251_sha256': STEP251_SHA,
           'rows_differing_from_step251': len(diff),
           'rows_differing_by_predicted_behavior': fam_of,
           'lists_changed_by_family': {k: v[0] for k, v in setchg.items()},
           'top5_set_changes_by_family': {k: v[1] for k, v in setchg.items()},
           'first_slot_changed': first_slot,
           'risk_behaviour_identical_to_step251': result.select(
               'pair_id', 'risk_score', 'predicted_behavior').equals(
                   ref251.select('pair_id', 'risk_score', 'predicted_behavior')),
           'null_evidence_cells': nulls,
           'fusion_rows': cov, 'ens_min': float(ens_min), 'ens_max': float(ens_max),
           'validation': str(val)[:400], 'protected_unchanged': True,
           'batches': recs, 'members': MEMBERS,
           'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    print('DONE', json.dumps({k: rec[k] for k in
                              ('output_sha256', 'equals_step251', 'rows_differing_from_step251',
                               'rows_differing_by_predicted_behavior', 'first_slot_changed',
                               'null_evidence_cells', 'seconds')}, indent=1), flush=True)
    return rec


def main():
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == 'neutral':
        rec = build(0.0, 'neutral_w0_check.csv')
        write_json(OUT / 'receipt259_neutral.json', rec)
        assert rec['equals_step251'], 'the w = 0 hook does NOT reproduce the STEP251 file'
        print('NEUTRALITY PROVED: w = 0 reproduces', STEP251_SHA)
        return
    if mode == 'fused':
        assert (OUT / 'receipt259_neutral.json').exists(), 'prove the hook neutral first'
        assert json.loads((OUT / 'receipt259_neutral.json').read_text(
            encoding='utf-8'))['equals_step251']
        assert (OUT / 'prediction259.json').exists(), 'register the prediction first'
        rec = build(W, OUTNAME)
        write_json(OUT / 'receipt259.json', rec)
        return
    raise SystemExit('usage: neutral|fused')


if __name__ == '__main__':
    main()
