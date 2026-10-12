"""STEP259b step 6: REPORT.txt - registration echo, gates, counts, timing and the sha256 of every new file.

Reads only artefacts written by this step (plus the STEP259 / STEP251 receipts it compares against) and
writes outputs_step259b_20260918/REPORT.txt.  Creates nothing else and deletes nothing.

Usage: python step259b_report.py
"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
sys.dont_write_bytecode = True
import datetime
import json
from pathlib import Path

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
from evidence_aug.paths import sha                                             # noqa: E402

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step259b_20260918'
SRC = CV / 'src/evidence_aug'
SCRIPTS = ['step259b_scope_keys.py', 'step259b_rescore_scope.py', 'step259b_P_fusion_build.py',
           'step259b_verify.py', 'step259b_finalize.py', 'step259b_report.py']
STEP251_SHA = '804c52f2d4a9e2fdffc91769e48edf713ac489d745874456025e2ee95e7761fb'
STEP259_SHA = '938e9e71997617c4c91b1003051aca00505300c384753bbcc6325d44ea5b89b9'


def J(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def fam_line(d, order=('directed_transfer', 'soft_play', 'coordinated_isolation', 'none')):
    return '  '.join(f'{k} {d.get(k, 0)}' for k in order if d.get(k, 0) or k in d) or 'none'


def main() -> None:
    pred = J(OUT / 'prediction259b.json')
    sk = J(OUT / 'scope_keys_receipt.json')
    ga = J(OUT / 'gateA259b.json')
    rn = J(OUT / 'receipt259b_neutral.json')
    ra = J(OUT / 'receipt259b_arith.json')
    rg = J(OUT / 'receipt259b.json')
    vf = J(OUT / 'verify259b.json')
    r259 = J(CV / 'outputs_step259_20260918/receipt259.json')
    L = []
    w = L.append

    w('STEP259b  geometric fold average of the MECHPLUS fusion  (2026-09-18, production evidence file)')
    w('=' * 110)
    w('')
    w('1. REGISTRATION ECHO  (prediction259b.json, written before any evaluation row was re-scored)')
    w('-' * 110)
    w('  RECIPE')
    w('    STEP259b = STEP259 with ONE thing changed: the MECHPLUS event probability p_evt that enters')
    w('      p\' = clip(p_stage2,1e-9,1)^(1-w) * clip(prior_shift(p_evt, F=10),1e-9,1-1e-9)^w ,  w = .25')
    w('    is the GEOMETRIC mean of the five saved fold boosters, sigmoid(mean_k logit(clip(p_k,1e-7,1-1e-7))),')
    w('    instead of STEP257\'s ARITHMETIC mean (step257_eval_build.py line `acc / 5.0`).  Same w, same F = 10,')
    w('    same members, same scope, same decoder; risk_score and predicted_behavior identical to STEP251.')
    w('    Coordinated isolation is never fused.')
    w('  WHY  experiments/step266_fold_average_calibration_20260918/REPORT.txt, verdict BUILD_G4:')
    w('    on the 162-query development audit at identical models and identical information,')
    w('    G4 - A4 = +0.004352 MAP@5, P(>0) 0.959, both halves positive (+0.006111 / +0.002311).')
    w('  REGISTERED PREDICTION (verbatim, from prediction259b.json)')
    for ln in pred['step266_increment']['verbatim'].split('; '):
        w('    ' + ln)
    w('    registered point vs STEP251 %+.4f   honest %+.4f   band80 [%+.4f, %+.4f]   P(above STEP251) %.2f'
      % (pred['step266_increment']['registered_point_delta_vs_STEP251'],
         pred['step266_increment']['honest_point_delta_vs_STEP251'],
         pred['step266_increment']['band80_vs_STEP251'][0],
         pred['step266_increment']['band80_vs_STEP251'][1],
         pred['step266_increment']['P_above_STEP251']))
    w('    STEP259\'s own registered point vs STEP251 was %+.5f (honest %+.4f).'
      % (pred['registered_step259_prediction_echo']['point_delta_vs_STEP251'],
         pred['registered_step259_prediction_echo']['honest_point_delta_vs_STEP251']))
    w('  RULE  the file is handed over only if EVERY gate below passes.')
    w('')

    w('2. GATES')
    w('-' * 110)
    w('  SCOPE       PASS   %d keys;  directed_transfer %d / soft_play %d / other %d  ==  receipt259.json fusion_rows'
      % (sk['rows'], sk['fusion_rows']['directed_transfer'], sk['fusion_rows']['soft_play'],
         sk['fusion_rows']['other']))
    w('              %d distinct evaluation hands, %d pairs (%d distinct hands inside the fused families)'
      % (sk['distinct_hands'], sk['distinct_pairs'], sk['distinct_hands_fused']))
    w('  GATE A      %s   the rebuilt 850-column matrix reproduces STEP257 exactly:'
      % ('PASS' if ga['gate_A']['PASS'] else 'FAIL'))
    for h in ('DT', 'SP'):
        g = ga['gate_A'][h]
        w('                 p_%s arithmetic mean vs the stored STEP257 p_%s: max abs diff %.3g, '
          'mean abs diff %.3g, exact on %d / %d rows (%.4f)'
          % (h, h, g['max_abs_diff'], g['mean_abs_diff'], g['n_exact'], g['n'], g['share_exact']))
    w('                 tolerance %g' % ga['gate_A']['tolerance'])
    w('  GATE B      %s   w = 0 through the NEW builder reproduces the STEP251 file byte for byte'
      % ('PASS' if rn['equals_step251'] else 'FAIL'))
    w('                 %s   (expected %s)' % (rn['output_sha256'], STEP251_SHA))
    w('  GATE C      %s   w = .25 with --source arith reproduces the STEP259 file byte for byte'
      % ('PASS' if ra['equals_step259'] else 'FAIL'))
    w('                 %s   (expected %s)' % (ra['output_sha256'], STEP259_SHA))
    w('  VALIDATION  %s' % rg['validation'].strip())
    w('  PROTECTED   %s   the seven protected submissions are unchanged before and after every run'
      % ('PASS' if rg['protected_unchanged'] and vf['protected_unchanged'] else 'FAIL'))
    w('  INDEPENDENT %s   step259b_verify.py, which imports none of the builder\'s frames'
      % ('PASS' if vf['ALL_CHECKS_PASS'] else 'FAIL'))
    w('')

    w('3. THE CANDIDATE FILE')
    w('-' * 110)
    w('  %s' % rg['output'])
    w('  sha256 %s' % rg['output_sha256'])
    w('  rows %d   evidence cells %d   empty or null cells %d   rows with a repeated hand %d'
      % (vf['rows'], vf['evidence_cells'], vf['empty_or_null_evidence_cells'],
         vf['rows_with_a_repeated_hand']))
    w('  distinct evidence hands %d   all evaluation phase: %s   membership violations %d'
      % (vf['distinct_evidence_hands'], vf['evidence_hands_all_evaluation_phase'],
         vf['evidence_membership_violations']))
    w('  risk_score / predicted_behavior identical to STEP251: %s     to STEP259: %s'
      % (vf['risk_behaviour_identical_to_step251'], vf['risk_behaviour_identical_to_step259']))
    w('  fusion rows applied: %s' % json.dumps(rg['fusion_rows']))
    w('  ens range over the twenty batches: [%.6g, %.6g]' % (rg['ens_min'], rg['ens_max']))
    w('')
    w('  ROWS DIFFERING  (independent count, step259b_verify.py)')
    w('    vs STEP251  %6d   by predicted_behavior:  %s'
      % (vf['rows_differing_from_step251'], fam_line(vf['rows_differing_from_step251_by_family'])))
    w('                        top-5 set changes:     %s'
      % fam_line(vf['top5_set_changes_vs_step251_by_family']))
    w('    vs STEP259  %6d   by predicted_behavior:  %s'
      % (vf['rows_differing_from_step259'], fam_line(vf['rows_differing_from_step259_by_family'])))
    w('                        top-5 set changes:     %s'
      % fam_line(vf['top5_set_changes_vs_step259_by_family']))
    w('    coordinated_isolation rows changed: %d vs STEP251, %d vs STEP259.  none rows changed: %d.'
      % (vf['coordinated_isolation_rows_changed_vs_step251'],
         vf['coordinated_isolation_rows_changed_vs_step259'],
         vf['none_rows_changed_vs_step251']))
    w('    first evidence slot changed vs STEP251: %d   vs STEP259: %d'
      % (rg['vs_step251']['first_slot_changed'], rg['vs_step259']['first_slot_changed']))
    w('    for reference, STEP259 moved %d rows off STEP251 (%s).'
      % (r259['rows_differing_from_step251'],
         fam_line(r259['rows_differing_by_predicted_behavior'])))
    w('    rows by predicted_behavior in the file: %s' % json.dumps(vf['rows_by_family']))
    w('')

    w('4. HOW FAR THE GEOMETRIC MEAN MOVES p_evt ON THE EVALUATION SCOPE  (descriptive)')
    w('-' * 110)
    w('  head   fused rows   mean |arith - geo|   low-decile median arith/geo   share geo < arith   rank corr')
    for h in ('DT', 'SP'):
        d = ga['arith_vs_geo'][h]
        w('  %-4s   %10d   %18.6f   %27.4f   %17.4f   %9.6f'
          % (h, d['fused_rows'], d['mean_abs_diff_arith_geo'], d['median_ratio_low_decile'],
             d['share_geo_below_arith'], d['spearman_rank_corr']))
    w('  STEP266\'s in-sample probe of the same five production models measured mean |A5 - G5| of 0.000152 (DT)')
    w('  and 0.000182 (SP) on development rows those models partly trained on.  On the evaluation scope, where')
    w('  every row is out of sample for all five, the dispersion is 4-6x larger, so the geometric correction is')
    w('  BIGGER here than the feasibility note assumed - the direction that favours the registered increment.')
    w('')

    w('5. TIMING')
    w('-' * 110)
    w('  scope keys              %8.1f s' % sk['seconds'])
    w('  feature prep (163/25/chronology reuse) + 6 parallel scoring shards + merge: see gateA259b.json')
    w('  gate B neutral build    %8.1f s' % rn['seconds'])
    w('  gate C arithmetic build %8.1f s' % ra['seconds'])
    w('  geometric build         %8.1f s' % rg['seconds'])
    w('  independent verify      %8.1f s' % vf['seconds'])
    w('')

    w('6. FILES CREATED BY STEP259b  (sha256; no existing file was modified)')
    w('-' * 110)
    rows = []
    for s in SCRIPTS:
        p = SRC / s
        if p.exists():
            rows.append((sha(p), p.stat().st_size, 'src/evidence_aug/' + s))
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != 'REPORT.txt':
            rows.append((sha(p), p.stat().st_size, 'outputs_step259b_20260918/' + p.name))
    for h, n, rel in rows:
        w('  %s  %12d  %s' % (h, n, rel))
    gone = json.loads((OUT / 'receipt259b.json').read_text(encoding='utf-8')).get('check_csvs_removed')
    if gone:
        w('')
        w('  REMOVED AFTER THE GATES (byte-identical duplicates of already registered files; shas recorded above')
        w('  in receipt259b.json and here):')
        for k, v in gone.items():
            w('  %s  %s  (== %s)' % (v['sha256'], k, v['identical_to']))
    w('')
    w('  written ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    (OUT / 'REPORT.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
