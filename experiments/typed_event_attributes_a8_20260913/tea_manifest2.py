"""poker-a8 typed-event attributes: manifest part 2 (coverage, step106 duplicates, STEP112 field map, statements). Reads no labels, scores nothing."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, hashlib
sys.dont_write_bytecode = True
import polars as pl
E = 'C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/typed_event_attributes_a8_20260913/'
NM = ['pre', 'flop', 'turn', 'river']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
man = json.load(open(E + 'MANIFEST_attrs-audit.json', encoding='utf-8'))
xa = pl.read_parquet(E + 'attrs-audit.parquet')
x = xa.filter(pl.col('top20'))
cov = {}
for i, s in enumerate(NM):
    reached = pl.col(f'pot_bb_start_{s}').is_not_null()
    inn = {R: reached & (pl.col(f'{R}_fold_street').is_null() | (pl.col(f'{R}_fold_street') >= i)) for R in 'WL'}
    for R in 'WL':
        cov[f'{R}_S_{s}'] = {'n_in': x.filter(inn[R]).height, 'null_where_in': x.filter(inn[R] & pl.col(f'{R}_S_{s}').is_null()).height,
                             'nonnull_where_not_in': x.filter(~inn[R] & pl.col(f'{R}_S_{s}').is_not_null()).height}
    cov[f'L_eq_{s}'] = {'n_both_in': x.filter(inn['W'] & inn['L']).height, 'null_where_both_in': x.filter(inn['W'] & inn['L'] & pl.col(f'L_eq_{s}').is_null()).height,
                        'nonnull_otherwise': x.filter(~(inn['W'] & inn['L']) & pl.col(f'L_eq_{s}').is_not_null()).height}
EQ = [c for c in xa.columns if c.startswith(('L_eq_', 'W_S_', 'L_S_', 'g6_')) or c in ('W_fold_eq', 'L_fold_eq', 'R1', 'T1')
      or any(c.startswith(f'{R}_{k}') for R in 'WL' for k in ('vt_', 'ps_'))]
off = xa.filter(~pl.col('top20'))
man['coverage_top20'] = cov
man['coverage_ok'] = all(v.get('null_where_in', 0) == 0 and v.get('null_where_both_in', 0) == 0 for v in cov.values())
man['equity_stage_columns'] = EQ
man['equity_stage_all_null_off_top20'] = all(off[c].null_count() == off.height for c in EQ)
man['duplicates_of_step106_exact'] = {k: v['ref'] for k, v in man['overlap_vs_step106_reranker_attrs'].items() if not k.endswith(('_br_sum', '_calls_sum'))}
man['derived_equal_to_step106'] = {'sum of W_{street}_br': 'W_aggr', 'sum of L_{street}_br': 'L_aggr', 'sum of W_{street}_calls': 'W_calls', 'sum of L_{street}_calls': 'L_calls'}
man['related_not_identical_to_step106'] = {'W_ps_n/L_ps_*': 'passive_strong (street-level, equity > .7)', 'W_vt_call_*/L_vt_call_*': 'weak_call_partner (equity < .35 flag)',
                                           'R1': 'step106_features R1 (formula replayed exactly)', 'g6_*/T1': 'step104 type attributes (exact)'}
man['step112_16_fields_in_this_table'] = {'L_fold_eq': 'L_fold_eq', 'L_fold_s': 'L_fold_street', 'L_fold_pot_bb': 'L_fold_pot_bb', 'L_fold_to_call_bb': 'L_fold_to_call_bb',
    'L_inv_bb': None, 'L_eq_pre': 'L_eq_pre', 'L_S_preflop': 'L_S_pre', 'W_aggr': 'sum of W_{street}_br', 'L_calls': 'sum of L_{street}_calls', 'W_S_turn': 'W_S_turn',
    'W_S_river': 'W_S_river', 'L_eq_turn': 'L_eq_turn', 'L_eq_river': 'L_eq_river', 'L_weak_call_partner': None, 'W_passive_strong': None, 'both_sd': 'both_showdown'}
man['earlier_looks_on_these_162_queries'] = ['step 106 residual reranker (overlapping fields; failed on held folds)', 'STEP112 (16 step-106 fields, T1-gated)',
    'STEP111G / poker-09 STEP123 hard-rule screens (111G predicates)', 'step 104 T1 decoder (G6)']
man['statement'] = ('No field uses relevance, evidence_rank, labels (label, label_status, behavior_family), scores, or evaluation-phase hands; within-hand gameplay '
                    'from hands/seats/actions only (showdown hole cards used for equity/strength). top20 is membership in the step-104 top-20 key set.')
man['definitions_file'] = {'path': 'FIELD_DEFINITIONS.md', 'sha256': sha(E + 'FIELD_DEFINITIONS.md')}
man['builder_sha256']['tea_manifest2.py'] = sha(E + 'tea_manifest2.py')
json.dump(man, open(E + 'MANIFEST_attrs-audit.json', 'w', encoding='utf-8'), indent=1)
print('coverage ok', man['coverage_ok'], '| equity cols', len(EQ), 'all null off top20', man['equity_stage_all_null_off_top20'])
print({k: v for k, v in cov.items() if v.get('null_where_in', 0) or v.get('null_where_both_in', 0) or v.get('nonnull_where_not_in', 0) or v.get('nonnull_otherwise', 0)})
