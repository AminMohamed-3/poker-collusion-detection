"""STEP266 stage 4: apply the registered rule, write results266.json and REPORT.txt.

Reads registration266.json, gate266.json, arms266.json, fusion_check266.json and
feasibility_probe266.json from the STEP266 folder; writes results266.json and REPORT.txt.
No measurement happens here - the rule is the one registered before any number was computed.

Usage: python step266_report.py      Never uploads.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

CV = r'C:\Data\Projects\Money Making\work\poker\Claude Version'
OUT = CV + r'\experiments\step266_fold_average_calibration_20260918'
SRC = CV + r'\src\evidence_aug'
REF_COMP, REF_H0, REF_H1 = 0.011147, 0.011887, 0.010289
ARMS = ['REF', 'S3', 'A4', 'G4', 'M4', 'SINGLE_j0', 'SINGLE_j1', 'SINGLE_j2', 'SINGLE_j3']
NEW_SCRIPTS = ['step266_nested_folds.py', 'step266_make_arms.py', 'step266_fusion_check.py',
               'step266_gate_merge.py', 'step266_feasibility_probe.py', 'step266_report.py']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def row(name, d, w=22):
    cells = ''.join((f'{d[k]:>+11.6f}' if k in d else f'{"-":>11}')
                    for k in ('delta', 'h0', 'h1', 'DT', 'SP'))
    ud = d.get('up_down')
    return f'  {name:<{w}}{cells}   ' + (f'{ud[0]:>3}/{ud[1]:<3}' if ud else f'{"-":>3} {"":<3}')


def main() -> None:
    reg = json.load(open(OUT + r'\registration266.json', encoding='utf-8'))
    gate = json.load(open(OUT + r'\gate266.json', encoding='utf-8'))
    arms = json.load(open(OUT + r'\arms266.json', encoding='utf-8'))
    fc = json.load(open(OUT + r'\fusion_check266.json', encoding='utf-8'))
    probe = json.load(open(OUT + r'\feasibility_probe266.json', encoding='utf-8')) \
        if os.path.exists(OUT + r'\feasibility_probe266.json') else {}
    bt = fc['_bootstrap']

    # ---- gate 2: the registered fusion value must come back on arm REF
    ref = fc['REF:COMP_DT25_SP25']
    g2 = {'expected': {'delta': REF_COMP, 'h0': REF_H0, 'h1': REF_H1},
          'got': {k: ref[k] for k in ('delta', 'h0', 'h1')},
          'max_abs_diff': max(abs(ref['delta'] - REF_COMP), abs(ref['h0'] - REF_H0),
                              abs(ref['h1'] - REF_H1))}
    g2['result'] = 'PASS' if g2['max_abs_diff'] < 1e-6 else 'FAIL'

    # ---- the registered rule, applied
    ms = fc['MEAN_SINGLE:COMP_DT25_SP25']['delta']

    def clauses(cand):
        d = bt[f'{cand}_minus_A4']
        return {'a_pooled_gt_0': bool(d['point'] > 0),
                'b_both_halves_ge_0': bool(d['h0'] >= 0 and d['h1'] >= 0),
                'c_P_gt_0_ge_075': bool(d['P_gt_0'] >= .75),
                'd_ge_mean_single_minus_001': bool(fc[f'{cand}:COMP_DT25_SP25']['delta']
                                                   >= ms - .001),
                'values': {'point_vs_A4': d['point'], 'h0': d['h0'], 'h1': d['h1'],
                           'P_gt_0': d['P_gt_0'],
                           'arm_delta': fc[f'{cand}:COMP_DT25_SP25']['delta'],
                           'mean_single_delta': ms}}

    cl = {c: clauses(c) for c in ('G4', 'M4')}
    for c in cl:
        cl[c]['all_pass'] = all(cl[c][k] for k in ('a_pooled_gt_0', 'b_both_halves_ge_0',
                                                   'c_P_gt_0_ge_075',
                                                   'd_ge_mean_single_minus_001'))
    a4_faithful = abs(fc['A4:COMP_DT25_SP25']['delta'] - fc['REF:COMP_DT25_SP25']['delta']) < .001
    if cl['G4']['all_pass'] and cl['M4']['all_pass']:
        verdict = 'BUILD_M4' if (fc['M4:COMP_DT25_SP25']['delta'] >
                                 fc['G4:COMP_DT25_SP25']['delta']) else 'BUILD_G4'
    elif cl['G4']['all_pass']:
        verdict = 'BUILD_G4'
    elif cl['M4']['all_pass']:
        verdict = 'BUILD_M4'
    else:
        verdict = 'NO_BUILD'
    vtext = {'BUILD_G4': 'production changes from the arithmetic to the GEOMETRIC mean (STEP259b)',
             'BUILD_M4': 'production changes from the arithmetic mean to the MEDIAN (STEP259b)',
             'NO_BUILD': 'production averaging is NOT changed'}[verdict]
    faithful = ('production path is faithful; nothing to change' if a4_faithful else
                'A4 differs from REF by more than .001: the production averaging path is NOT '
                'neutral, and the size of that gap is the headline number of this step')

    pred = reg['prediction_parsed']
    scored = {
     'A4_minus_REF': {'predicted': pred['A4_minus_REF_point'], 'observed': bt['A4_minus_REF']['point'],
                      'predicted_P_below_REF': pred['P_A4_below_REF'],
                      'observed_P_below_REF': bt['A4_minus_REF']['P_lt_0']},
     'G4_minus_A4': {'predicted': pred['G4_minus_A4_point'], 'observed': bt['G4_minus_A4']['point'],
                     'predicted_P_gt_0': pred['P_G4_gt_A4'], 'observed_P_gt_0': bt['G4_minus_A4']['P_gt_0']},
     'S3_minus_REF': {'predicted': pred['S3_minus_REF_point'], 'observed': bt['S3_minus_REF']['point']},
     'M4_between_A4_and_G4': {'predicted': True,
                              'observed': bool(min(fc['A4:COMP_DT25_SP25']['delta'],
                                                   fc['G4:COMP_DT25_SP25']['delta'])
                                               <= fc['M4:COMP_DT25_SP25']['delta'] <=
                                               max(fc['A4:COMP_DT25_SP25']['delta'],
                                                   fc['G4:COMP_DT25_SP25']['delta']))}}

    res = {'step': 'STEP266', 'date': '2026-09-18',
           'gate266_G1': gate['result'], 'gate266_G2': g2,
           'fusion': {k: v for k, v in fc.items() if not k.startswith('_')},
           'head_ap': {a: arms['arms'][a] for a in ARMS},
           'arm_distance': arms['arm_distance'],
           'rule_clauses': cl, 'verdict': verdict, 'verdict_text': vtext,
           'A4_within_001_of_REF': a4_faithful, 'faithfulness_statement': faithful,
           'prediction_scored': scored,
           'bootstrap': bt,
           'feasibility_probe': probe}
    json.dump(res, open(OUT + r'\results266.json', 'w', encoding='utf-8'), indent=1)

    # ------------------------------------------------------------------ REPORT.txt
    L = []
    A = L.append
    A('STEP266  fold-average calibration of the MECHPLUS fusion  (2026-09-18, development only)')
    A('=' * 110)
    A('')
    A('1. REGISTRATION ECHO  (registration266.json, written before any model was fitted)')
    A('-' * 110)
    A('QUESTION')
    for ln in _wrap(reg['question']):
        A('  ' + ln)
    A('')
    A('ARMS')
    for k, v in reg['arms'].items():
        A(f'  {k:<22}' + _wrap(v, 84)[0])
        for ln in _wrap(v, 84)[1:]:
            A(' ' * 24 + ln)
    A('')
    A('RULE')
    for ln in _wrap(reg['rule']):
        A('  ' + ln)
    A('')
    A('PREDICTION (verbatim, Amin)')
    for ln in _wrap(reg['prediction_verbatim']):
        A('  ' + ln)
    A('')
    A('2. GATES')
    A('-' * 110)
    A(f'  G1 reproduce the registered OOF                 {gate["result"]}')
    for h in ('DT', 'SP'):
        g = gate['heads'][h]
        A(f'     {h}: max abs diff vs oof_mechplus.npy = {g["max_abs_diff_vs_registered"]:.3g}, '
          f'spearman {g["spearman"]}, AP refit {g["refit"]["ap"]} vs registered '
          f'{g["registered"]["ap"]}, AUC {g["refit"]["auc"]} vs {g["registered"]["auc"]}')
    A(f'     {gate["note"]}')
    A(f'  G2 reproduce the registered fusion on arm REF   {g2["result"]}   '
      f'COMP_DT25_SP25 {ref["delta"]:+.6f} (expected {REF_COMP:+.6f}), halves '
      f'{ref["h0"]:+.6f} / {ref["h1"]:+.6f} (expected {REF_H0:+.6f} / {REF_H1:+.6f})')
    A('')
    A('3. HEAD AP  (STEP254 definition: listed-of-family vs every hand of a label-0 pair)')
    A('-' * 110)
    A(f'  {"arm":<22}{"DT AP":>9}{"DT AUC":>10}{"SP AP":>9}{"SP AUC":>10}'
      f'{"DT clean med p":>17}{"SP clean med p":>17}')
    for a in ARMS:
        d = arms['arms'][a]
        A(f'  {a:<22}{d["DT"]["ap"]:>9.4f}{d["DT"]["auc"]:>10.5f}{d["SP"]["ap"]:>9.4f}'
          f'{d["SP"]["auc"]:>10.5f}{d["DT"]["clean_p_q50"]:>17.8f}{d["SP"]["clean_p_q50"]:>17.8f}')
    A('')
    A('4. FUSION TABLE  (delta = mean over the 162 audit queries of MAP@5 minus the unfused '
      'v1_DTfp3 base)')
    A('-' * 110)
    for tag in ('DT_w025', 'SP_w025', 'COMP_DT25_SP25'):
        A(f'  {tag}')
        A(f'  {"arm":<22}{"delta":>11}{"h0":>11}{"h1":>11}{"DT":>11}{"SP":>11}   up/down')
        for a in ARMS:
            A(row(a, fc[f'{a}:{tag}']))
        if f'MEAN_SINGLE:{tag}' in fc:
            A(row('MEAN_SINGLE', fc[f'MEAN_SINGLE:{tag}']))
        A('')
    A('  (CI is never fused; the any and CI rows of every arm file are the registered ones, so the '
      'CI columns above are identical across arms by construction.)')
    A('')
    A('5. PAIRED BOOTSTRAP over the 162 queries, 2,000 draws, seed 0, on COMP_DT25_SP25')
    A('-' * 110)
    A(f'  {"contrast":<26}{"point":>11}{"P(>0)":>9}{"P(<0)":>9}{"p05":>11}{"p50":>11}{"p95":>11}'
      f'{"h0":>11}{"h1":>11}')
    for nm in ('A4_minus_REF', 'G4_minus_A4', 'G4_minus_REF', 'M4_minus_A4', 'M4_minus_G4',
               'S3_minus_REF', 'A4_minus_MEAN_SINGLE', 'G4_minus_MEAN_SINGLE',
               'MEAN_SINGLE_minus_REF'):
        b = bt[nm]
        A(f'  {nm:<26}{b["point"]:>+11.6f}{b["P_gt_0"]:>9.4f}{b["P_lt_0"]:>9.4f}{b["p05"]:>+11.6f}'
          f'{b["p50"]:>+11.6f}{b["p95"]:>+11.6f}{b["h0"]:>+11.6f}{b["h1"]:>+11.6f}')
    A('')
    A(f'  {"arm vs zero":<26}{"point":>11}{"P(>0)":>9}{"p05":>11}{"p50":>11}{"p95":>11}')
    for a in ARMS + ['MEAN_SINGLE']:
        b = bt[f'{a}_vs_zero']
        A(f'  {a:<26}{b["point"]:>+11.6f}{b["P_gt_0"]:>9.4f}{b["p05"]:>+11.6f}{b["p50"]:>+11.6f}'
          f'{b["p95"]:>+11.6f}')
    A('')
    A('6. VERDICT  (the rule as registered, applied without modification)')
    A('-' * 110)
    for c in ('G4', 'M4'):
        v = cl[c]
        A(f'  {c}:  (a) pooled > 0            {v["a_pooled_gt_0"]}   '
          f'[{v["values"]["point_vs_A4"]:+.6f} vs A4]')
        A(f'       (b) both halves >= 0     {v["b_both_halves_ge_0"]}   '
          f'[h0 {v["values"]["h0"]:+.6f}, h1 {v["values"]["h1"]:+.6f}]')
        A(f'       (c) P(>0) >= .75         {v["c_P_gt_0_ge_075"]}   [{v["values"]["P_gt_0"]:.4f}]')
        A(f'       (d) >= mean-single-.001  {v["d_ge_mean_single_minus_001"]}   '
          f'[arm {v["values"]["arm_delta"]:+.6f} vs mean-single '
          f'{v["values"]["mean_single_delta"]:+.6f}]')
        A(f'       ALL PASS: {v["all_pass"]}')
    A(f'  VERDICT: {verdict} - {vtext}')
    A(f'  A4 within .001 of REF: {a4_faithful}. {faithful}')
    A('')
    A('  READING THE NUMBERS - capacity is confounded with averaging, and the controls separate them')
    A('  ' + '-' * 106)
    A(f'    REF            {fc["REF:COMP_DT25_SP25"]["delta"]:+.6f}   one 4-fold model per scoring '
      f'fold - the audited path, which PRODUCTION CANNOT USE (it has five models, not one).')
    A(f'    S3             {fc["S3:COMP_DT25_SP25"]["delta"]:+.6f}   one 3-fold model  '
      f'(S3 - REF {bt["S3_minus_REF"]["point"]:+.6f}, P(<0) {bt["S3_minus_REF"]["P_lt_0"]:.3f}) - '
      f'the pure CAPACITY penalty of the nested design.')
    A(f'    MEAN_SINGLE    {ms:+.6f}   the four 3-fold models averaged AS DELTAS, i.e. the '
      f'single-model level at nested capacity.')
    A(f'    A4             {fc["A4:COMP_DT25_SP25"]["delta"]:+.6f}   arithmetic mean of the same '
      f'four models = the PRODUCTION form.')
    A(f'    G4             {fc["G4:COMP_DT25_SP25"]["delta"]:+.6f}   geometric mean of the same '
      f'four models.')
    A('')
    A(f'    A4 - REF   {bt["A4_minus_REF"]["point"]:+.6f} is NOT all averaging: S3 - REF is '
      f'{bt["S3_minus_REF"]["point"]:+.6f}, so almost all of it is the 3-fold capacity of the nested'
      ' models,')
    A('    which production does not suffer (its five models each see 4/5 of the development data, '
      'like REF).')
    A(f'    At FIXED capacity the two clean contrasts are: arithmetic averaging '
      f'A4 - MEAN_SINGLE {bt["A4_minus_MEAN_SINGLE"]["point"]:+.6f} '
      f'(P(<0) {bt["A4_minus_MEAN_SINGLE"]["P_lt_0"]:.3f}),')
    A(f'    geometric averaging G4 - MEAN_SINGLE '
      f'{bt["G4_minus_MEAN_SINGLE"]["point"]:+.6f} '
      f'(P(>0) {bt["G4_minus_MEAN_SINGLE"]["P_gt_0"]:.3f}), and the decisive head-to-head at '
      f'identical models and identical information,')
    A(f'    G4 - A4 {bt["G4_minus_A4"]["point"]:+.6f} '
      f'(P(>0) {bt["G4_minus_A4"]["P_gt_0"]:.3f}, halves {bt["G4_minus_A4"]["h0"]:+.6f} / '
      f'{bt["G4_minus_A4"]["h1"]:+.6f}).')
    A('    So the transferable claims are: (i) the production ARITHMETIC mean costs roughly '
      f'{abs(bt["A4_minus_MEAN_SINGLE"]["point"]):.4f} against a single model of the same')
    A(f'    capacity; (ii) switching to the GEOMETRIC mean is worth about '
      f'{bt["G4_minus_A4"]["point"]:.4f} on top of that, and puts the ensemble ABOVE the '
      'single-model level.')
    A(f'    What is NOT claimed: that a geometric production file recovers the whole audited '
      f'{REF_COMP:+.6f} - G4 - REF is still '
      f'{bt["G4_minus_REF"]["point"]:+.6f} (P(<0) {bt["G4_minus_REF"]["P_lt_0"]:.3f}) at nested '
      'capacity.')
    A('')
    A('  MECHANISM - how small the probability change is, and why it still moves MAP@5')
    ad = arms['arm_distance']
    for h in ('DT', 'SP'):
        A(f'    {h}: mean |A4 - G4| = {ad[f"{h}:A4_vs_G4"]["mean_abs_diff"]}, and in the lowest '
          f'decile of A4 the median A4/G4 ratio is '
          f'{ad[f"{h}:A4_vs_G4"]["A4_over_G4_median_low_decile"]} - the arithmetic mean sits under '
          'one percent')
        A(f'        above the geometric one on non-event hands.  A4 vs REF rank correlation '
          f'{ad[f"{h}:A4_vs_REF"]["spearman_like_corr"]}, G4 vs REF '
          f'{ad[f"{h}:G4_vs_REF"]["spearman_like_corr"]}.')
    A('    The geometric mean is the arithmetic mean minus a term that grows with the DISAGREEMENT '
      'between the models in logit space,')
    A('    so it is not a monotone transform of A4: it demotes hands that one model likes and three '
      'do not.  After prior_shift(., 10) ** .25')
    A('    that is a sub-percent change in the multiplier, but the fused score decides a top-5 '
      'ordering, so a handful of boundary flips per query is')
    A('    all it takes.  The same dispersion appears in the five PRODUCTION models '
      '(feasibility probe below: mean |A5 - G5| of the same size),')
    A('    which is the reason to expect the contrast to transfer.')
    A('')
    A('  PREDICTION SCORED')
    for k, v in scored.items():
        A(f'    {k:<22} ' + json.dumps(v))
    A('')
    A('7. FEASIBILITY OF A STEP259b GEOMETRIC-MEAN PRODUCTION RUN  (described only, NOT built)')
    A('-' * 110)
    for ln in FEAS.strip('\n').split('\n'):
        A('  ' + ln if ln else '')
    if probe:
        A('')
        A('  MEASURED PREDICT COST (step266_feasibility_probe.py, 50,000 development rows, '
          f'{probe["num_threads"]} threads)')
        for h in ('DT', 'SP'):
            p = probe['heads'][h]
            A(f'    {h}: five boosters ({p["n_trees"]} trees each) over 50,000 x 850 rows in '
              f'{p["seconds_5_models"]} s = {p["seconds_per_1k_rows_5_models"]} s / 1k rows')
        A(f'    both heads, five models each: '
          f'{probe["seconds_per_1k_rows_both_heads_5_models"]} s / 1k rows  ->  '
          f'{probe["seconds_per_1k_rows_both_heads_5_models"] * 321.2:.0f} s for the '
          f'321,200 scope rows')
        A('')
        A('  IN-SAMPLE DISPERSION of the five PRODUCTION models (not an accuracy measurement)')
        for h in ('DT', 'SP'):
            d = probe['in_sample_five_model_dispersion'][h]
            A(f'    {h}: mean |A5 - G5| {d["mean_abs_diff_A5_G5"]}, low-decile median A5/G5 '
              f'{d["A5_over_G5_median_low_decile"]}; shift(p,10)^.25 low-decile median: '
              f'A5 {d["shift_pow_A5_median_low_decile"]}, G5 {d["shift_pow_G5_median_low_decile"]}, '
              f'single fold0 {d["shift_pow_single_fold0_median_low_decile"]}')
    A('')
    A('8. FILES CREATED BY STEP266  (sha256; no existing file was modified or deleted)')
    A('-' * 110)
    files = []
    for f in sorted(os.listdir(OUT)):
        p = OUT + '\\' + f
        if os.path.isfile(p) and f != 'REPORT.txt':
            files.append(('experiments/step266_fold_average_calibration_20260918/' + f, p))
    for s in NEW_SCRIPTS:
        p = SRC + '\\' + s
        if os.path.exists(p):
            files.append(('src/evidence_aug/' + s, p))
    for rel, p in files:
        A(f'  {sha(p)}  {os.path.getsize(p):>10}  {rel}')
    A('')
    A(f'  written {time.strftime("%Y-%m-%d %H:%M:%S")}')
    open(OUT + r'\REPORT.txt', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L[-40:]))
    print('\nwrote', OUT + r'\REPORT.txt', 'and results266.json')


def _wrap(s, w=104):
    out, cur = [], ''
    for word in s.split():
        if len(cur) + len(word) + 1 > w:
            out.append(cur)
            cur = word
        else:
            cur = (cur + ' ' + word).strip()
    if cur:
        out.append(cur)
    return out or ['']


FEAS = """
WHAT THE FUSION SCOPE IS
  src/evidence_aug/step259_P_fusion_build.py builds `allkeys` from the twenty
  outputs_step251_20260917/work/INC/batches/batch_%03d-lp-scores.parquet files: 321,200 (pair_id,
  hand_id) rows over 239,845 distinct evaluation hands and 16,060 pairs (measured, this step).  Only
  the pairs whose predicted_behavior is directed_transfer or soft_play are fused: 184,600 + 90,580 =
  275,180 rows over 216,225 distinct hands (27.0 % of the 800,000 evaluation hands); the 46,020
  coordinated_isolation rows are never fused.  (receipt259.json fusion_rows agrees exactly.)

WHY THE SCOPE MUST BE RE-SCORED
  step259_P_fusion_build.event_lookup reads only p_DT and p_SP from
  experiments/step257_eval_mech_scores_20260918/eval_hand_scores_batch_*.parquet, and
  step257_eval_build.score_batch wrote those columns as `acc / 5.0`, the ARITHMETIC mean of the five
  fold boosters.  The five per-model probabilities were never stored, so a geometric mean cannot be
  recovered from the existing files: the scope rows have to be scored again with the saved boosters.

WHAT A RE-SCORE OF ONLY THE SCOPE ROWS NEEDS  (a new src/evidence_aug/step266b_geo_scores.py; the
STEP257 scripts are hash-registered and must not be edited)
  0. keys      the 321,200 rows above (or just the 275,180 DT/SP rows), read exactly the way
               step259_P_fusion_build does, plus pair -> (player_1, player_2) from
               data/evaluation_pairs.csv and p1_is_A from the chronology block.
  1. columns   models/feature_order.json - 850 names in three blocks (163 production / 25 STEP237 /
               662 mechanical); step257_eval_build asserts each block's layout, keep those asserts.
  2. 163       step257_prep_eval.batch_files('evaluation') + the read_filtered semi-join on hand_id.
               One pass over the twenty outputs_v2/hands_evaluation/batch_*.parquet files.  Measured
               in the full build: 263.9 s for twenty passes, i.e. ~15-30 s for one.
  3. 25        evidence_aug.step246_invisible_features.build(keys, pmap, 'evaluation', tmp, workers).
               Row-proportional: 924.3 s for 9,651,820 rows = .096 s / 1k rows -> ~31 s for the scope.
               (step257_eval_build deletes its temporary file at the end of each batch, so none of
               these survive; they must be rebuilt.)
  4. 662       step257_mech_eval.build_chunk(ks, 'evaluation', PCT169, EQ169) in chunks of
               STEP257_SUBHANDS hands, joined to the t4 chronology block.  IMPORTANT: the chronology
               block is NOT row-local - step257_prep_eval.chrono_for computes cumulative chip flow and
               shared_idx over ALL of a pair's hands ordered by started_at - so it must be built for
               the pair's FULL hand list and then filtered to the scope.  The twenty
               chrono_evaluation/chrono_b*.parquet files from the STEP257 run are still in this
               session's scratchpad (9,651,820 rows; receipt says 50.5 s to rebuild by running
               `python step257_prep_eval.py evaluation` unchanged).  build_chunk itself is row-local
               (its equity enumeration and board work need only the row's own hand), so the mechanical
               block CAN legitimately be built on the subset.
  5. score     load mechplus_DT_fold{0..4}.txt and mechplus_SP_fold{0..4}.txt (ten boosters; the 'any'
               and CI heads are not needed) and replace `acc / 5.0` by
               sigmoid(mean_k logit(clip(p_k, 1e-7, 1 - 1e-7))).  Write a NEW parquet
               (eval_hand_scores_geo_*.parquet); never touch the STEP257 files.
  6. build     a new src/evidence_aug/step259b_P_fusion_build.py - step259_P_fusion_build.py with
               event_lookup pointed at the new parquet and a new OUTNAME.  Its w = 0 neutrality check
               against the STEP251 sha (804c52f2...) and validate_submission come along unchanged and
               should be re-run first.  Measured runtime of the STEP259 fused build: 1,252.6 s.

EXPECTED RUNTIME
  Measured STEP257 full build: 10,544.4 s of lane time for 9,651,820 rows over 799,900 hands, run as
  two concurrent lanes (batches 0-9 and 10-19), ~5,900 s wall.  That is 1.092 s per 1k rows or 13.18 s
  per 1k hands.  The scope is HAND-SPARSE - 1.34 rows per hand against 12.07 in the full build - so the
  per-hand fixed costs (the hands / seats / actions scans and the python side tables inside
  build_chunk) are amortised nine times less well.  Bounds: 351 s if the cost were purely
  row-proportional, 3,161 s if purely hand-proportional; expect near the upper bound, ~2,000-3,500 s of
  lane time, i.e. ~20-30 min wall on two lanes.  Total for a STEP259b file: ~50 s chronology (reusable)
  + ~30 s invisible features + ~20-30 min feature build + a few minutes of booster prediction +
  ~21 min for the fusion build itself = roughly 45-60 min wall.  No new data, no new model fits, no
  download, no upload.  (The brief's '~3,100 s' for STEP257 is lower than the logged 10,544 s of lane
  time / ~5,900 s wall; the numbers above are the ones in the STEP257 log_batch_*.json receipts.)
"""

if __name__ == '__main__':
    main()
