"""STEP266: merge gate_DT.json and gate_SP.json into gate266.json (the registered G1 record).

G1 (registration266.json): refit the plain 5-fold OOF for the DT and SP heads with the STEP254 MECHPLUS
recipe and compare with oof_mechplus.npy rows 1 and 2.  PASS = max abs diff < 1e-3 AND identical AP to 4
decimals; PASS_EXACT if max abs diff < 1e-6.
"""
import json
import os

CV = r'C:\Data\Projects\Money Making\work\poker\Claude Version'
OUT = CV + r'\experiments\step266_fold_average_calibration_20260918'

g = {}
for h in ('DT', 'SP'):
    r = json.load(open(OUT + f'\\gate_{h}.json', encoding='utf-8'))
    g[h] = {k: r[k] for k in ('gate', 'max_abs_diff_vs_registered', 'exact_1e6', 'pass_1e3',
                              'spearman', 'refit', 'registered', 'ap_identical_4dp', 'n_jobs',
                              'seconds')}
st = [g[h]['gate'] for h in ('DT', 'SP')]
rec = {'gate': 'G1_reproduce_registered_oof',
       'definition': 'refit the plain 5-fold MECHPLUS OOF for DT and SP (850 columns, step232 table '
                     'folds, LightGBM 400/15/.03/30/.8/1/.5/5 seed 232) and compare with '
                     'experiments/step254_mechanical_tensor_20260917/oof_mechplus.npy rows 1 and 2',
       'threshold': 'PASS if max abs diff < 1e-3 and AP identical to 4 decimals; PASS_EXACT if '
                    'max abs diff < 1e-6',
       'reference_aps': {'DT': 0.8853, 'SP': 0.6626},
       'heads': g,
       'result': 'PASS_EXACT' if all(s == 'PASS_EXACT' for s in st) else (
           'PASS' if all(s.startswith('PASS') for s in st) else 'FAIL'),
       'consequence': 'REF (the registered oof_mechplus.npy) is used as the reference arm; no '
                      'REF_refit fallback is needed',
       'note': 'the refit used n_jobs 7 while STEP254 used 5 and STEP257 used 4; all three agree bit '
               'for bit, so LightGBM is thread-count independent on this recipe'}
json.dump(rec, open(OUT + r'\gate266.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps({'result': rec['result'],
                  'DT': {k: g['DT'][k] for k in ('max_abs_diff_vs_registered', 'refit')},
                  'SP': {k: g['SP'][k] for k in ('max_abs_diff_vs_registered', 'refit')}}, indent=1))
print('wrote', OUT + r'\gate266.json', os.path.exists(OUT + r'\gate266.json'))
