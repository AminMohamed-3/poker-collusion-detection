"""poker-a8 typed-event attributes: validation manifest (sha256, frozen column list, G6/R1 replay, step106 overlap diffs). Reads no labels, scores nothing."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, hashlib, glob
sys.dont_write_bytecode = True
import polars as pl
E = 'C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/typed_event_attributes_a8_20260913/'
CV = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
KEYS, NM, N6 = ['pair_id', 'hand_id'], ['pre', 'flop', 'turn', 'river'], ['preflop', 'flop', 'turn', 'river']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
x = pl.read_parquet(E + 'attrs-audit.parquet')
ref = pl.read_parquet(CV + 'experiments/residual_reranker_C_20260913/audit-reranker-attributes.parquet')
top = x.filter(pl.col('top20')).with_columns(*[pl.sum_horizontal([pl.col(f'{R}_{s}_{k}') for s in NM]).alias(f'{R}_{k}_sum') for R in 'WL' for k in ('br', 'calls')])
P = [('players_dealt', 'players_dealt', None), ('players_at_showdown', 'sd_players', None), ('pot_bb', 'pot_bb', None), ('W_net_bb', 'W_net_bb', None),
     ('L_net_bb', 'L_net_bb', None), ('both_showdown', 'both_sd', None)] + [(f'L_eq_{s}', f'L_eq_{s}', None) for s in NM]
P += [(f'{R}_S_{s}', f'{R}_S_{n}', None) for R in 'WL' for s, n in zip(NM, N6)] + [(f'{R}_fold_street', f'{R}_fold_s', -1) for R in 'WL']
P += [(f'{R}_fold_facing', f'{R}_fold_facing', 'nofold') for R in 'WL'] + [(f'{R}_{k}', f'{R}_{k}', 0.) for R in 'WL' for k in ('fold_to_call_bb', 'fold_pot_bb')]
P += [(f'{R}_fold_eq', f'{R}_fold_eq', None) for R in 'WL'] + [(f'{R}_br_sum', f'{R}_aggr', None) for R in 'WL'] + [(f'{R}_calls_sum', f'{R}_calls', None) for R in 'WL']
P += [(c, c, None) for c in ['fold_partner_hu', 'fold_partner_multi', 'fold_partner_after_outfold', 'pair_reraise', 'sandwich', 'caught', 'out_fold_to_pair',
                             'out_aggr', 'out_call_pair', 'out_contrib_bb', 'pair_contrib_bb']]
j = top.join(ref, on=KEYS, how='inner', suffix='_ref')
assert j.height == 3240 == top.height
ov = {}
for mine, rc, fill in P:
    m, rr = j[mine], (j[rc + '_ref'] if rc + '_ref' in j.columns else j[rc])
    if fill is not None:
        m = m.fill_null(fill)
    if m.dtype == pl.String or rr.dtype == pl.String:
        ov[mine] = {'ref': rc, 'mismatch': int((m.cast(pl.String).fill_null('<n>') != rr.cast(pl.String).fill_null('<n>')).sum())}
        continue
    m, rr = m.cast(pl.Float64), rr.cast(pl.Float64)
    both = m.is_not_null() & rr.is_not_null()
    dif = (m - rr).abs().filter(both)
    ov[mine] = {'ref': rc, 'n_both': int(both.sum()), 'max_abs_diff': float(dif.max()) if dif.len() else None, 'n_diff_gt_1e-9': int((dif > 1e-9).sum()),
                'ref_only_nonnull': int((rr.is_not_null() & m.is_null()).sum()), 'mine_only_nonnull': int((m.is_not_null() & rr.is_null()).sum())}
sf = lambda R: pl.coalesce([pl.when(pl.col(f'{R}_fold_s') == i).then(pl.col(f'{R}_S_{n}')) for i, n in enumerate(N6)])
r1 = lambda R: (pl.col(f'{R}_fold_facing') == 'partner') & (pl.col(f'{R}_fold_eq').fill_null(0) > .5) & (sf(R).fill_null(0) > .6)
r1m = top.select(KEYS + ['R1']).join(ref.select(KEYS + [(r1('W') | r1('L')).alias('R1_ref')]), on=KEYS).filter(pl.col('R1') != pl.col('R1_ref')).height
bad = {k: v for k, v in ov.items() if v.get('mismatch', 0) or v.get('n_diff_gt_1e-9', 0) or v.get('ref_only_nonnull', 0)}
S = CV + 'src/evidence_aug/'
man = {'deliverable': 'typed_event_attributes_a8 for the STEP112 interface (audit split)', 'label_free': True, 'held_read': False, 'scored_on_audit': False,
       'file': 'attrs-audit.parquet', 'file_sha256': sha(E + 'attrs-audit.parquet'), 'rows': x.height, 'cols': x.width,
       'keys_unique': bool(x.select(KEYS).is_unique().all()), 'top20_rows': top.height, 'build_run': json.load(open(E + 'build_run.json', encoding='utf-8')),
       'builder_sha256': {os.path.basename(p): sha(p) for p in sorted(glob.glob(E + 'tea_*.py'))},
       'imported_sha256': {f: sha(S + f) for f in ('step104_C_typed_decoder_build.py', 'step106_attrs.py')},
       'R1_mismatch_vs_step106_features_formula_on_reranker_attrs': r1m, 'overlap_vs_step106_reranker_attrs': ov, 'overlap_not_exact': bad,
       'nulls_all_rows': {c: v for c, v in x.null_count().row(0, named=True).items() if v},
       'nulls_top20_rows': {c: v for c, v in x.filter(pl.col('top20')).null_count().row(0, named=True).items() if v},
       'columns': {c: str(t) for c, t in x.schema.items()}}
json.dump(man, open(E + 'MANIFEST_attrs-audit.json', 'w', encoding='utf-8'), indent=1)
print('rows', x.height, 'cols', x.width, 'sha', man['file_sha256'][:16], 'g6', man['build_run']['g6_exact_vs_step104'], 'R1 mismatch', r1m, 'overlap fields', len(ov),
      'not exact', json.dumps(bad)[:1500])
