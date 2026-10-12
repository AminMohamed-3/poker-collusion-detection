"""STEP277-B: the induced soft-play citation rule, blended into the evidence probability. Evidence-only,
on the STEP272 pair side.

What changes, in one line: for pairs whose predicted_behavior is soft_play, the probability entering the typed
decoder becomes `p_final = p_fused^(1-w) * p_rule^w` with w = .30, where p_rule is the STEP277-A induced
per-hand citation rule. Everything else -- the four LP members, the arithmetic ensemble, the STEP259b event
fusion at w = .25 / F = 10 on DT and SP, the chronology, the type attributes, the CI mask and the decoder
itself -- is untouched, and directed_transfer and coordinated_isolation pairs are bit-identical to STEP272.

Why (OBSERVATIONS 19.237). p_rule is fitted on the 85 soft-play development pairs that lie OUTSIDE the 162-query
audit, over their full ~121-hand candidate universes, on the 662 frozen STEP254 mechanical columns, bagged over
8 seeds. Evaluated on the audit, which no part of the fit has seen:

    soft play subgroup  +.024255        whole audit  +.008148   P(>0) .9285   40 lists changed, 17 sets changed
    honest split-half over 28 (scope, weight) cells, control included:  +.003039   307/400 positive

and every control fails in the right direction:

    PERM   labels permuted within each training pair   -.054178   <- the gain is the LABELS, not the blend
    SHUF_FAM  trained on the other two families        -.023481
    5 hyperparameter settings   +.0185 .. +.0319, 5/5 positive
    10 draws of 70% of the training pairs   mean +.0209, min +.0088, 10/10 positive

This is the information-adding class, which is the class that transfers in this project's public record (STEP167
LP +.00258, STEP259b fusion +.00114), not the decoder-re-parameterisation class that went 0 for 2 (STEP175
-.00120, STEP179 -.00086). Composite value +.0006 to +.0016.

Two honest caveats, both recorded rather than argued away:
  1. STEP257's own mechplus_p_SP does NOT reproduce this (-.007413 at the same w, with its permutation control at
     -.011121 tracking it). The shipped object is therefore this rule and not that score, and the two disagree --
     mechplus_p_SP is a 5-fold OOF event model over all 1,860 pairs and 850 columns, while p_rule is a citation
     model trained only on soft-play pairs with a strictly disjoint 210/162 pair split.
  2. The audit validation blended into the probability the audit's decoder consumes (PRODbare = v1_DTfp3). On the
     evaluation side soft play ALSO receives the STEP259b event fusion, so the shipped blend composes with a
     term the audit read did not exercise. The pipeline POSITION is the same -- immediately before the decoder --
     but the composition is not something the audit could test.

Features: the 662 columns cannot come from STEP254, which is development-only by construction. They are built by
step277_A_sp_rule_features.py through step257_mech_eval.build_chunk -- STEP254's computation imported unchanged
with the phase parameterised -- with the column list frozen from the development file. That build passed GATE P:
1,445 development soft-play rows x 662 columns reproduced step254_features.parquet exactly through this code
path, which is what makes the evaluation features trustworthy.

Four gates, all of which must pass:
  GATE 1  v1 replay: decoding the base v1 score must reproduce the STEP169 base file's evidence, per batch.
  GATE 2  identity: the decoder at gamma = 1 must equal the production decoder's output exactly.
  GATE 3  end-to-end: at w = 0 the written file must be BYTE-IDENTICAL to STEP272 d5891a68.
  GATE 4  scope: only soft_play rows may differ from STEP272; risk_score and predicted_behavior byte-identical.

Usage:  python step277_B_sp_rule_build.py neutral    # GATE 3
        python step277_B_sp_rule_build.py rule       # the STEP277 candidate, w = .30
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
OUT = CV / 'outputs_step277_20260919'
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
SP_FAM = 'soft_play'
W_RULE = 0.30
RULEF = CV / 'experiments/step277_generator_rule_20260919/sp_rule_eval.parquet'
OUTNAME = 'submission_step277_spRule_on_step272.csv'


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


def build(w_rule: float, outname: str) -> dict:
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
    rl = pl.read_parquet(RULEF)
    rmap = {(a, b): c for a, b, c in rl.iter_rows()}
    sp_pairs_rule = set(rl['pair_id'].to_list())
    sp_pairs_base = set(base.filter(pl.col('predicted_behavior') == SP_FAM)
                        .join(allkeys.select('pair_id').unique(), on='pair_id', how='semi')['pair_id'].to_list())
    assert sp_pairs_rule == sp_pairs_base, (
        'the rule covers a different soft-play pair set than the scoped base: rule %d, base %d, missing %d'
        % (len(sp_pairs_rule), len(sp_pairs_base), len(sp_pairs_base - sp_pairs_rule)))
    print('soft-play rule attached for %d pairs / %d rows' % (len(sp_pairs_rule), rl.height), flush=True)
    rule_rows = 0

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

        # ---- the STEP277 change: the induced soft-play rule, on the post-fusion probability
        ensr = ensf.copy()
        sprow = famrow == SP_FAM
        if w_rule > 0 and sprow.any():
            pr = np.array([rmap[(p, h)] for p, h, mm in zip(pids, hids, sprow) if mm], dtype=np.float64)
            assert len(pr) == int(sprow.sum()) and np.isfinite(pr).all(), 'soft-play rule coverage gap'
            ensr[sprow] = (np.clip(ensf[sprow], 1e-9, 1.0) ** (1 - w_rule)) * np.clip(pr, 1e-9, 1.0) ** w_rule
            rule_rows += int(sprow.sum())

        bf = ref.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'),
                        (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1')
                        ).with_columns(pl.Series('v1_ens', ensf + 10),
                                       pl.Series('v1_rule', ensr + 10))
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
        assert mine1.equals(prod), 'GATE 2 the decoder at gamma = 1 is not the production decoder in ' + name
        if w_rule == 0:
            new = prod
        else:
            fer = fc.with_columns(pl.Series('score', bf['v1_rule'].to_numpy()))
            new = S108.expand(CSP.rank(bf, S104.decode(fer, typed)))
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
    rec = {'step': 'STEP277', 'w_rule': w_rule, 'rule_rows': rule_rows,
           'rule_file': str(RULEF.relative_to(CV)), 'rule_sha256': sha(RULEF), 'output': str(csv.relative_to(CV)), 'output_sha256': out_sha,
           'equals_step272': out_sha == PAIRBASE_SHA, 'pair_base': str(PAIRBASE.relative_to(CV)),
           'pair_base_sha256': PAIRBASE_SHA, 'step169_base_sha256': BASE_SHA,
           'scores_file': str(SCORES.relative_to(CV)), 'scores_sha256': sha(SCORES), 'members': MEMBERS,
           'rows_differing_from_step272': len(diff), 'rows_differing_by_family': byfam,
           'CI_candidate_rows': ci_rows, 'fusion_rows': cov, 'null_evidence_cells': nulls,
           'risk_behaviour_identical_to_step272': result.select(
               'pair_id', 'risk_score', 'predicted_behavior').equals(
                   pairbase.select('pair_id', 'risk_score', 'predicted_behavior')),
           'validation': str(val)[:400], 'protected_unchanged': True, 'batches': recs,
           'audit_evidence': {'prod_decoded': 0.733740, 'rule_decoded': 0.741888,
                              'd_all': 0.008148, 'P_gt_0_all': 0.9595,
                              'd_SP_subgroup': 0.028090, 'SP_audit_queries': 47,
                              'halves': [0.00471, 0.01213], 'lists_changed': 39, 'lists_set_changed': 16,
                              'splithalf_honest_mean': 0.003039, 'splithalf_positive': '307/400',
                              'controls': {'perm_within_pair': -0.054178, 'shuffled_family': -0.023481,
                                           'hp_settings_positive': '5/5', 'subset70_positive': '10/10',
                                           'subset70_min': 0.008830},
                              'mechplus_p_SP_does_not_reproduce': -0.007413},
           'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / ('receipt_step277_w%g.json' % w_rule), rec)
    print('DONE', json.dumps({k: rec[k] for k in ('output_sha256', 'equals_step272',
                                                 'rows_differing_from_step272', 'rows_differing_by_family',
                                                 'null_evidence_cells', 'seconds')}, indent=1), flush=True)
    return rec


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'rule'
    assert mode in ('neutral', 'rule'), mode
    if mode == 'neutral':
        rec = build(0.0, 'gate_step277_neutral_must_equal_step272.csv')
        assert rec['equals_step272'], 'GATE 3 FAILED: w = 0 did not reproduce STEP272 byte-for-byte'
        print('GATE 3 PASSED: w = 0 reproduces STEP272 d5891a68 exactly', flush=True)
    else:
        rec = build(W_RULE, OUTNAME)
        assert not rec['equals_step272'], 'the rule produced an identical file; the change did nothing'
        byf = rec['rows_differing_by_family']
        assert byf.get('directed_transfer', 0) == 0 and byf.get('coordinated_isolation', 0) == 0, \
            'GATE 4 FAILED: only soft_play lists may change, got %s' % byf
        assert rec['risk_behaviour_identical_to_step272'], 'GATE 4 FAILED: the pair side moved'
        print('GATE 4 PASSED: only soft_play changed (%d rows)' % byf.get('soft_play', 0), flush=True)
        print('STEP277 built. sha256', rec['output_sha256'], flush=True)


if __name__ == '__main__':
    main()
