"""STEP259b-P: STEP259 with the MECHPLUS fold average changed from ARITHMETIC to GEOMETRIC.

This is src/evidence_aug/step259_P_fusion_build.py (hash-registered, must not be edited) copied with
exactly two changes:

  * event_lookup reads outputs_step259b_20260918/scope_scores_geo.parquet - the STEP259 fusion scope
    re-scored by step259b_rescore_scope.py with the same five saved boosters per head and the same 850
    columns - and takes p_DT / p_SP from either the arithmetic mean (--source arith, the STEP257
    production form, used to reproduce STEP259) or the geometric mean
    sigmoid(mean_k logit(clip(p_k, 1e-7, 1 - 1e-7))) (--source geo, the STEP266 BUILD_G4 form);
  * OUTNAME.

w = .25, F = 10, the members, the scope, the decoder, the v1 replay gate, the evidence swap into the
STEP169 base and validate_submission are all unchanged, so risk_score and predicted_behavior stay
byte-identical to STEP251 and to STEP259.

Usage:  python step259b_P_fusion_build.py neutral                 # GATE B: w = 0 must reproduce 804c52f2...
        python step259b_P_fusion_build.py fused --source arith    # GATE C: must reproduce 938e9e71... (STEP259)
        python step259b_P_fusion_build.py fused --source geo      # the STEP259b candidate
Never uploads.
"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
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
OUT = CV / 'outputs_step259b_20260918'
BASE = CV / 'outputs_step169_20260915/submission_step169.csv'
BASE_SHA = '67060d0a53b32ef6c790ce0255f5dd5b6230abcae464ba0c3baed02b4543b9c1'
STEP251_FILE = S251 / 'submission_step251_ens4_dtfp3_on_step222.csv'
STEP251_SHA = '804c52f2d4a9e2fdffc91769e48edf713ac489d745874456025e2ee95e7761fb'
STEP259_FILE = CV / 'outputs_step259_20260918/submission_step259_mechfusion_dt_sp_on_step251.csv'
STEP259_SHA = '938e9e71997617c4c91b1003051aca00505300c384753bbcc6325d44ea5b89b9'
SCORES = OUT / 'scope_scores_geo.parquet'
MEMBERS = ['INC', 'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3']
F_SHIFT = 10.0
W = .25
FUSED_FAMS = (('directed_transfer', 'p_DT'), ('soft_play', 'p_SP'))
OUTNAME = 'submission_step259b_geofusion_dt_sp_on_step251.csv'


def prior_shift(p, F):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    o = q / (1 - q) * float(F)
    return o / (1 + o)


def event_lookup(keys: pl.DataFrame, source: str) -> pl.DataFrame:
    assert source in ('arith', 'geo'), source
    d = pl.read_parquet(SCORES, columns=['pair_id', 'hand_id',
                                         'p_DT_' + source, 'p_SP_' + source])
    d = d.rename({'p_DT_' + source: 'p_DT', 'p_SP_' + source: 'p_SP'})
    d = d.select('pair_id', 'hand_id', 'p_DT', 'p_SP')
    lk = d.join(keys, on=['pair_id', 'hand_id'], how='semi')
    assert lk.height == keys.height, ('evidence rows without a re-scored event score',
                                      keys.height, lk.height)
    return lk


def compare(result: pl.DataFrame, ref_path: Path, csv: Path, EV, fam) -> dict:
    """rows / lists / top-5 set changes of `result` (already written to csv) against a reference file."""
    A, B = ref_path.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
    assert len(A) == len(B)
    diff = [k for k in range(len(A)) if A[k] != B[k]]
    for k in diff:
        assert A[k].split(b',')[:3] == B[k].split(b',')[:3], 'risk or behaviour changed'
    fam_of = {}
    for k in diff:
        p = A[k].split(b',')[0].decode()
        fam_of[fam[p]] = fam_of.get(fam[p], 0) + 1
    ref = pl.read_csv(ref_path, infer_schema=False)
    ra, rb = result.select(EV).rows(), ref.select(EV).rows()
    first_slot = sum(1 for a, b in zip(ra, rb) if a[0] != b[0])
    setchg = {}
    for a, b, f in zip(ra, rb, result['predicted_behavior'].to_list()):
        if a != b:
            setchg[f] = setchg.get(f, [0, 0])
            setchg[f][0] += 1
            setchg[f][1] += len(set(a) - set(b))
    return {'reference': str(ref_path.relative_to(CV)), 'reference_sha256': sha(ref_path),
            'rows_differing': len(diff), 'rows_differing_by_predicted_behavior': fam_of,
            'lists_changed_by_family': {k: v[0] for k, v in setchg.items()},
            'top5_set_changes_by_family': {k: v[1] for k, v in setchg.items()},
            'first_slot_changed': first_slot,
            'risk_behaviour_identical': result.select(
                'pair_id', 'risk_score', 'predicted_behavior').equals(
                    ref.select('pair_id', 'risk_score', 'predicted_behavior'))}


def build(w: float, outname: str, source: str) -> dict:
    S108, _ = S167.modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    from evidence_aug import step104_C_typed_decoder_build as S104
    from validation import validate_submission
    t0 = time.perf_counter()
    prot = check_protected()
    EV = list(CSP.EVIDENCE_COLUMNS)
    assert sha(BASE) == BASE_SHA, 'base file changed'
    assert sha(STEP251_FILE) == STEP251_SHA, 'the STEP251 file changed'
    assert sha(STEP259_FILE) == STEP259_SHA, 'the STEP259 file changed'
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
    lk = event_lookup(allkeys, source)
    lkmap = {(a, b): (c, d) for a, b, c, d in lk.iter_rows()}
    print('event scores attached for', len(lkmap), 'evidence rows  source =', source, flush=True)

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

    # ---- verification against STEP251 AND against STEP259
    vs251 = compare(result, STEP251_FILE, csv, EV, fam)
    vs259 = compare(result, STEP259_FILE, csv, EV, fam)
    val = validate_submission(csv, POKER / 'data/evaluation_pairs.csv',
                              POKER / 'data/hands.parquet',
                              POKER / 'data/seats.parquet',
                              POKER / 'data/sample_submission.csv')
    errs = getattr(val, 'errors', None)
    assert not errs, str(val)
    assert check_protected() == prot
    nulls = sum(int((result[c] == '').sum() + result[c].null_count()) for c in EV)
    rec = {'w': w, 'source': source, 'output': str(csv.relative_to(CV)), 'output_sha256': sha(csv),
           'equals_step251': sha(csv) == STEP251_SHA,
           'equals_step259': sha(csv) == STEP259_SHA,
           'base_sha256': BASE_SHA, 'step251_sha256': STEP251_SHA, 'step259_sha256': STEP259_SHA,
           'scores_file': str(SCORES.relative_to(CV)), 'scores_sha256': sha(SCORES),
           'vs_step251': vs251, 'vs_step259': vs259,
           'rows_differing_from_step251': vs251['rows_differing'],
           'rows_differing_by_predicted_behavior': vs251['rows_differing_by_predicted_behavior'],
           'lists_changed_by_family': vs251['lists_changed_by_family'],
           'top5_set_changes_by_family': vs251['top5_set_changes_by_family'],
           'first_slot_changed': vs251['first_slot_changed'],
           'risk_behaviour_identical_to_step251': vs251['risk_behaviour_identical'],
           'null_evidence_cells': nulls,
           'fusion_rows': cov, 'ens_min': float(ens_min), 'ens_max': float(ens_max),
           'validation': str(val)[:400], 'protected_unchanged': True,
           'batches': recs, 'members': MEMBERS,
           'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    print('DONE', json.dumps({k: rec[k] for k in
                              ('output_sha256', 'equals_step251', 'equals_step259',
                               'rows_differing_from_step251', 'null_evidence_cells', 'seconds')},
                             indent=1), flush=True)
    print('VS_STEP259', json.dumps(vs259, indent=1), flush=True)
    return rec


def main():
    mode = sys.argv[1]
    source = 'geo'
    if '--source' in sys.argv:
        source = sys.argv[sys.argv.index('--source') + 1]
    assert source in ('arith', 'geo'), source
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT / 'prediction259b.json').exists(), 'register the prediction first'
    assert SCORES.exists(), 'run step259b_rescore_scope.py first'
    gateA = json.loads((OUT / 'gateA259b.json').read_text(encoding='utf-8'))
    assert gateA['gate_A']['PASS'], 'GATE A did not pass'

    if mode == 'neutral':                                              # GATE B
        rec = build(0.0, 'neutral_check_259b.csv', source)
        write_json(OUT / 'receipt259b_neutral.json', rec)
        assert rec['equals_step251'], 'GATE B FAILED: w = 0 does NOT reproduce the STEP251 file'
        print('GATE B PASS: w = 0 reproduces', STEP251_SHA)
        return
    if mode == 'fused' and source == 'arith':                          # GATE C
        assert (OUT / 'receipt259b_neutral.json').exists(), 'prove the hook neutral first'
        assert json.loads((OUT / 'receipt259b_neutral.json').read_text(
            encoding='utf-8'))['equals_step251']
        rec = build(W, 'arith_check_259b.csv', 'arith')
        write_json(OUT / 'receipt259b_arith.json', rec)
        assert rec['equals_step259'], 'GATE C FAILED: the arithmetic path does NOT reproduce STEP259'
        print('GATE C PASS: --source arith reproduces', STEP259_SHA)
        return
    if mode == 'fused' and source == 'geo':                            # the candidate
        for g, f in (('B', 'receipt259b_neutral.json'), ('C', 'receipt259b_arith.json')):
            assert (OUT / f).exists(), 'GATE ' + g + ' has not been run'
        assert json.loads((OUT / 'receipt259b_neutral.json').read_text(
            encoding='utf-8'))['equals_step251'], 'GATE B did not pass'
        assert json.loads((OUT / 'receipt259b_arith.json').read_text(
            encoding='utf-8'))['equals_step259'], 'GATE C did not pass'
        rec = build(W, OUTNAME, 'geo')
        rec['gates'] = {
            'A_feature_rebuild_max_abs_diff': {h: gateA['gate_A'][h]['max_abs_diff']
                                               for h in ('DT', 'SP')},
            'A_PASS': gateA['gate_A']['PASS'],
            'B_neutral_sha256': json.loads((OUT / 'receipt259b_neutral.json').read_text(
                encoding='utf-8'))['output_sha256'],
            'B_PASS': True,
            'C_arith_sha256': json.loads((OUT / 'receipt259b_arith.json').read_text(
                encoding='utf-8'))['output_sha256'],
            'C_PASS': True}
        assert not rec['equals_step251'] and not rec['equals_step259'], \
            'the geometric file is identical to a previous file - nothing changed'
        write_json(OUT / 'receipt259b.json', rec)
        return
    raise SystemExit('usage: neutral | fused --source arith|geo')


if __name__ == '__main__':
    main()
