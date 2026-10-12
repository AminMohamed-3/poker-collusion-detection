"""STEP276-G: gamma_CI = 2 in the typed listing decoder. Evidence-only, on the STEP272 pair side.

What changes, in one line: the decoder scores `10 + p*q**gamma + 1e-9*p` instead of `10 + p*q + 1e-9*p`, with
gamma = 2 for coordinated_isolation pairs and gamma = 1 (i.e. production, unchanged) for everything else.

Why (OBSERVATIONS 19.234(e)(h)). `q = P(fewer than five qualified ahead of me in traversal order)` is the
largest single effect in the evidence decoder: dropping it costs -.047 audit MAP@5. Its exponent had never been
varied. Swept jointly over all families it is a sharp global optimum at 1 -- production's cell is rank 1 of 300
over traversal order x cap x temperature x exponent, and honest split-half selection over that grid returns
-.00778 with 0/400 positive. Per family it is not: CI reads +.000741 / +.001152 / +.001461 / +.001276 / +.000288
at gamma 1.5 / 1.75 / 2.0 / 2.5 / 3.0 (P(>0) .624 / .782 / .869 / .805 / .567), a smooth unimodal plateau worth
+.0055 on the 43 CI audit queries with both halves positive, while DT and SP both want exactly 1.

The mechanism is structural, not fitted, which is why this is worth a file at all. Production forces T1 OFF for
CI (`typed = t1 & ~cif`), so CI alone has no block term and its traversal is pure chronology -- the censoring
prior is CI's only ordering signal, and applying it harder partly substitutes for the T1 term the other two
families get. Consistent with that, extending the STEP259b event fusion to CI is negative in all 24 (w, F) cells
tested, so CI has no other handle.

Size, stated honestly: honest split-half selection of gamma_CI gives +.000482 against the +.001461 in-sample
read, so the audit value is +.0005 to +.0015 and the composite value is about +.0001. This is a CARRY to ride on
a larger file, not a submission that can be resolved on its own.

Construction. The evidence lists are rebuilt exactly as step259b_P_fusion_build.py does -- same four LP members,
same arithmetic ensemble, same geometric-source event fusion at w = .25 and F = 10 on DT and SP, same chronology,
same type attributes, same CI mask -- and only the decoder's exponent differs. `risk_score` and
`predicted_behavior` are copied from the STEP272 file byte-for-byte; nothing on the pair side is recomputed.

Three gates, all of which must pass:
  GATE 1  v1 replay: decoding the base v1 score must reproduce the STEP169 base file's evidence, per batch.
  GATE 2  identity: at gamma = 1 the new decoder must equal the production decoder's output exactly.
  GATE 3  end-to-end: at gamma = 1 the written file must be BYTE-IDENTICAL to STEP272 d5891a68.

Usage:  python step276_G_ci_gamma_build.py neutral    # GATE 3: must reproduce d5891a68...
        python step276_G_ci_gamma_build.py fused      # the STEP276 candidate, gamma_CI = 2
Never uploads.
"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
sys.dont_write_bytecode = True
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
OUT = CV / 'outputs_step276_20260919'
BASE = CV / 'outputs_step169_20260915/submission_step169.csv'
BASE_SHA = '67060d0a53b32ef6c790ce0255f5dd5b6230abcae464ba0c3baed02b4543b9c1'
PAIRBASE = CV / 'outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv'
PAIRBASE_SHA = 'd5891a68919d51e2204d72c1515950e7b01b64b6c1b3242a5bc3b37ca18a53f5'
SCORES = CV / 'outputs_step259b_20260918/scope_scores_geo.parquet'
MEMBERS = ['INC', 'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3']
F_SHIFT = 10.0
W = .25
FUSED_FAMS = (('directed_transfer', 'p_DT'), ('soft_play', 'p_SP'))
CI_FAM = 'coordinated_isolation'
GAMMA_CI = 2.0
OUTNAME = 'submission_step276_cigamma2_on_step272.csv'


def prior_shift(p, F):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    o = q / (1 - q) * float(F)
    return o / (1 + o)


def event_lookup(keys: pl.DataFrame) -> pl.DataFrame:
    d = pl.read_parquet(SCORES, columns=['pair_id', 'hand_id', 'p_DT_geo', 'p_SP_geo'])
    d = d.rename({'p_DT_geo': 'p_DT', 'p_SP_geo': 'p_SP'}).select('pair_id', 'hand_id', 'p_DT', 'p_SP')
    lk = d.join(keys, on=['pair_id', 'hand_id'], how='semi')
    assert lk.height == keys.height, ('evidence rows without a re-scored event score', keys.height, lk.height)
    return lk


def decode_gamma(frame, t1, isci, gamma):
    """step104 decode with a per-family exponent on q.  At gamma = 1 this is bit-identical to it."""
    from evidence_aug.step104_C_typed_decoder_build import less_than_five
    score = frame['score'].to_numpy()
    out = score.copy()
    pid, cr = frame['pair_id'].to_numpy(), frame['chrono_rank'].to_numpy()
    idx = np.flatnonzero(score > 5)
    _, first = np.unique(pid[idx], return_index=True)
    for g in np.split(idx, first[1:]):
        p = np.clip(score[g] - 10, 0, 1)
        order = np.lexsort((cr[g], ~t1[g]))
        q = np.empty(len(g))
        q[order] = less_than_five(p[order])
        gm = float(gamma) if bool(isci[g[0]]) else 1.0
        out[g] = 10 + p * (q ** gm) + 1e-9 * p
    return out


def build(gamma: float, outname: str) -> dict:
    S108, _ = S167.modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    from evidence_aug import step104_C_typed_decoder_build as S104
    from validation import validate_submission
    t0 = time.perf_counter()
    prot = check_protected()
    EV = list(CSP.EVIDENCE_COLUMNS)
    assert sha(BASE) == BASE_SHA, 'the STEP169 base file changed'
    assert sha(PAIRBASE) == PAIRBASE_SHA, 'the STEP272 pair base changed'
    OUT.mkdir(exist_ok=True)
    base = pl.read_csv(BASE, infer_schema=False)
    pairbase = pl.read_csv(PAIRBASE, infer_schema=False)
    assert base.select('pair_id').equals(pairbase.select('pair_id')), 'pair_id order differs between bases'
    ci = base.filter(pl.col('predicted_behavior') == CI_FAM).select('pair_id')
    fam = dict(base.select('pair_id', 'predicted_behavior').iter_rows())
    assert fam == dict(pairbase.select('pair_id', 'predicted_behavior').iter_rows()), \
        'predicted_behavior differs between the STEP169 base and the STEP272 pair base'
    pmap = {p: (a, b) for p, a, b in pl.read_csv(
        CSP.ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}

    allkeys = []
    for i in range(20):
        p = S251 / 'work' / MEMBERS[0] / 'batches' / ('batch_%03d-lp-scores.parquet' % i)
        allkeys.append(pl.read_parquet(p, columns=KEYS))
    allkeys = pl.concat(allkeys).sort(KEYS)
    lkmap = {(a, b): (c, d) for a, b, c, d in event_lookup(allkeys).iter_rows()}
    print('event scores attached for', len(lkmap), 'evidence rows', flush=True)

    news, prods, recs = [], [], []
    cov = {'directed_transfer': 0, 'soft_play': 0, 'other': 0}
    ci_rows = 0
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
            assert np.array_equal(t['v1'].to_numpy(), ref['v1'].to_numpy()), 'v1 must not depend on the member'
        ens = np.mean([t['v1_lp'].to_numpy() for t in tops], axis=0)
        assert ens.min() >= 1e-9 and ens.max() <= 1.0, 'the fusion clip would not be the identity at w = 0'

        pids = ref['pair_id'].to_list()
        hids = ref['hand_id'].to_list()
        famrow = np.array([fam[p] for p in pids])
        ensf = ens.copy()
        for f, col in FUSED_FAMS:
            mrow = famrow == f
            if not mrow.any():
                continue
            k = 0 if col == 'p_DT' else 1
            pev = np.array([lkmap[(p, h)][k] for p, h, mm in zip(pids, hids, mrow) if mm], dtype=np.float64)
            assert len(pev) == int(mrow.sum()) and np.isfinite(pev).all(), 'fusion coverage gap'
            pe = prior_shift(pev, F_SHIFT)
            ensf[mrow] = (np.clip(ens[mrow], 1e-9, 1.0) ** (1 - W)) * np.clip(pe, 1e-9, 1 - 1e-9) ** W
            cov[f] += int(mrow.sum())
        cov['other'] += int((~np.isin(famrow, [f for f, _ in FUSED_FAMS])).sum())

        bf = ref.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'),
                        (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1')
                        ).with_columns(pl.Series('v1_ens', ensf + 10))
        sel = bf.select('pair_id').unique().sort('pair_id')
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS), pmap))
        cif = fc.select('pair_id').join(ci.with_columns(pl.lit(True).alias('f')), on='pair_id', how='left',
                                        maintain_order='left')['f'].fill_null(False).to_numpy()
        typed = t1 & ~cif
        ci_rows += int(cif.sum())

        # GATE 1 -- the v1 replay of the base evidence
        inc = S108.expand(CSP.rank(bf, S104.decode(
            fc.with_columns(pl.Series('score', bf['v1'].to_numpy())), typed)))
        chk = base.join(sel, on='pair_id', how='semi').select(inc.columns).sort('pair_id')
        assert inc.equals(chk), 'GATE 1 v1 replay of the base evidence failed in ' + name

        fe = fc.with_columns(pl.Series('score', bf['v1_ens'].to_numpy()))
        # the production decoder = STEP259b's evidence, which is what STEP272 carries
        prod = S108.expand(CSP.rank(bf, S104.decode(fe, typed)))
        # GATE 2 -- identity of my decoder at gamma = 1
        mine1 = S108.expand(CSP.rank(bf, decode_gamma(fe, typed, cif, 1.0)))
        assert mine1.equals(prod), 'GATE 2 gamma = 1 is not the production decoder in ' + name
        new = prod if gamma == 1.0 else S108.expand(CSP.rank(bf, decode_gamma(fe, typed, cif, gamma)))
        assert new.select('pair_id').equals(prod.select('pair_id'))
        news.append(new)
        prods.append(prod)
        chg = sum(1 for u, o in zip(new.rows(), prod.rows()) if u != o)
        recs.append({'batch': name, 'selected_pairs': new.height, 'CI_rows': int(cif.sum()),
                     'T1_rows_used': int(typed.sum()), 'lists_changed_vs_step259b': chg,
                     's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)

    repl = pl.concat(news).sort('pair_id')
    rprod = pl.concat(prods).sort('pair_id')
    assert repl.height == repl['pair_id'].n_unique() == 16060

    # ---- the STEP272 evidence columns must be exactly what the production decoder produces
    tgt = pairbase.join(repl.select('pair_id'), on='pair_id', how='semi').select(rprod.columns).sort('pair_id')
    assert rprod.equals(tgt), 'the production decoder does not reproduce the STEP272 evidence columns'
    print('STEP259b/STEP272 evidence reproduced exactly for all 16,060 listed pairs', flush=True)

    rest = pairbase.join(repl.select('pair_id'), on='pair_id', how='anti')
    swap = pairbase.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(
        repl, on='pair_id', validate='1:1')
    result = pairbase.select('pair_id').join(pl.concat([rest, swap.select(pairbase.columns)]),
                                            on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(
        pairbase.select('pair_id', 'risk_score', 'predicted_behavior')), 'the pair side moved'
    csv = OUT / outname
    assert not csv.exists(), 'preserve a built file'
    result.write_csv(csv)
    out_sha = sha(csv)

    A = PAIRBASE.read_bytes().split(b'\n')
    B = csv.read_bytes().split(b'\n')
    assert len(A) == len(B)
    diff = [k for k in range(len(A)) if A[k] != B[k]]
    for k in diff:
        assert A[k].split(b',')[:3] == B[k].split(b',')[:3], 'risk or behaviour changed'
    byfam = {}
    for k in diff:
        p = A[k].split(b',')[0].decode()
        byfam[fam[p]] = byfam.get(fam[p], 0) + 1
    val = validate_submission(csv, POKER / 'data/evaluation_pairs.csv', POKER / 'data/hands.parquet',
                              POKER / 'data/seats.parquet', POKER / 'data/sample_submission.csv')
    assert not getattr(val, 'errors', None), str(val)
    assert check_protected() == prot
    nulls = sum(int((result[c] == '').sum() + result[c].null_count()) for c in EV)
    rec = {'step': 'STEP276', 'gamma_CI': gamma, 'output': str(csv.relative_to(CV)), 'output_sha256': out_sha,
           'equals_step272': out_sha == PAIRBASE_SHA, 'pair_base': str(PAIRBASE.relative_to(CV)),
           'pair_base_sha256': PAIRBASE_SHA, 'step169_base_sha256': BASE_SHA,
           'scores_file': str(SCORES.relative_to(CV)), 'scores_sha256': sha(SCORES), 'members': MEMBERS,
           'rows_differing_from_step272': len(diff), 'rows_differing_by_family': byfam,
           'CI_candidate_rows': ci_rows, 'fusion_rows': cov, 'null_evidence_cells': nulls,
           'risk_behaviour_identical_to_step272': result.select(
               'pair_id', 'risk_score', 'predicted_behavior').equals(
                   pairbase.select('pair_id', 'risk_score', 'predicted_behavior')),
           'validation': str(val)[:400], 'protected_unchanged': True, 'batches': recs,
           'audit_evidence': {'gamma1_decoded': 0.733740, 'gamma2_decoded': 0.735201,
                              'd_all': 0.001461, 'P_gt_0_all': 0.8685,
                              'd_CI_subgroup': 0.005504, 'P_gt_0_CI': 0.8715,
                              'halves': [0.00100, 0.00200],
                              'splithalf_honest_mean': 0.000482, 'splithalf_positive': '176/400',
                              'CI_audit_queries': 43},
           'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / ('receipt_step276_gamma%g.json' % gamma), rec)
    print('DONE', json.dumps({k: rec[k] for k in ('output_sha256', 'equals_step272',
                                                 'rows_differing_from_step272', 'rows_differing_by_family',
                                                 'null_evidence_cells', 'seconds')}, indent=1), flush=True)
    return rec


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fused'
    assert mode in ('neutral', 'fused'), mode
    if mode == 'neutral':
        rec = build(1.0, 'gate_step276_neutral_must_equal_step272.csv')
        assert rec['equals_step272'], 'GATE 3 FAILED: gamma = 1 did not reproduce STEP272 byte-for-byte'
        print('GATE 3 PASSED: gamma = 1 reproduces STEP272 d5891a68 exactly', flush=True)
    else:
        rec = build(GAMMA_CI, OUTNAME)
        assert not rec['equals_step272'], 'gamma = 2 produced an identical file; the change did nothing'
        assert rec['rows_differing_by_family'].get('directed_transfer', 0) == 0 and \
            rec['rows_differing_by_family'].get('soft_play', 0) == 0, \
            'only coordinated_isolation lists may change'
        print('STEP276 built. sha256', rec['output_sha256'], flush=True)


if __name__ == '__main__':
    main()
