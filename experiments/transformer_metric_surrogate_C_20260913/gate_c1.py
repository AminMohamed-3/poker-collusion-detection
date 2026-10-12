# ---- part C1: Part 1 necessity table (true family used ONLY as a descriptive grouping) + cross-checks
OWN = {'directed_transfer': ['opp_sign'], 'coordinated_isolation': ['ci_strict', 'ci_relaxed', 'ci_reraise', 'ci_sandwich'],
       'soft_play': ['sp_hu_street', 'sp_river_check_any', 'sp_river_check_both']}
fr = fr.with_columns(grp=pl.when(pl.col('relevance') == 1).then(pl.when(pl.col('top20')).then(pl.lit('listed_top20')).otherwise(pl.lit('listed_out')))
                     .otherwise(pl.when(pl.col('top20')).then(pl.lit('unlisted_top20')).otherwise(pl.lit('unlisted_out'))))
res = {'study': 'STEP111G_type_gate', 'audit_only': True, 'held_read': False,
       'predeclare_sha256': '2f566b2abf2e39fe3aa703d04186007406d4937f8263fec2d6d7ebc12dbfa378',
       'overall': {c: round(float(fr[c].mean()), 4) for c in PRED}}
part1 = {}
for fam in FAMILIES:
    g = fr.filter(pl.col('behavior_family') == fam)
    part1[fam] = {}
    subs = [('listed_all', g.filter(pl.col('relevance') == 1))] + [(k, g.filter(pl.col('grp') == k)) for k in ('listed_top20', 'listed_out', 'unlisted_top20', 'unlisted_out')]
    for name, sub in subs:
        part1[fam][name] = {'n': sub.height, **{c: (round(float(sub[c].mean()), 4) if sub.height else None) for c in PRED}}
res['part1_descriptive_true_family'] = part1
res['viability'] = {c: {'family': fam, 'listed_rate': part1[fam]['listed_all'][c], 'listed_top20_rate': part1[fam]['listed_top20'][c],
                        'unlisted_top20_rate': part1[fam]['unlisted_top20'][c], 'viable': part1[fam]['listed_all'][c] >= .98}
                    for fam, cs in OWN.items() for c in cs}
print('PART 1 (true family used ONLY as a descriptive grouping)')
for fam in FAMILIES:
    for k, v in part1[fam].items():
        print(f'  {fam[:5]} {k:15s} n={v["n"]:5d} ' + ' '.join(f'{c}={v[c]}' for c in PRED))
for c, v in res['viability'].items():
    print(f'  viability {c:20s} {v["family"][:5]} listed {v["listed_rate"]} (top20 {v["listed_top20_rate"]}) unlisted_top20 {v["unlisted_top20_rate"]} viable {v["viable"]}')
f2x = pl.read_parquet(AUDIT / 'audit-f2-features.parquet', columns=KEYS + ['f2_opposite_signs'])
ra = pl.read_parquet(R106 + 'audit-reranker-attributes.parquet', columns=KEYS + ['sandwich', 'out_fold_to_pair', 'pair_reraise', 'fold_partner_after_outfold'])
cx = fr.select(KEYS + PRED).join(f2x, on=KEYS, how='left').join(ra, on=KEYS, how='left')
assert cx.height == fr.height


def agree(mine, theirs):
    t = cx[theirs].cast(pl.Float64).fill_null(0).to_numpy() > 0
    m = cx[mine].to_numpy().astype(bool)
    return {'both': int((m & t).sum()), 'mine_only': int((m & ~t).sum()), 'theirs_only': int((~m & t).sum()), 'neither': int((~m & ~t).sum()), 'theirs_null': int(cx[theirs].null_count())}


res['crosscheck'] = {f'{m}~{t}': agree(m, t) for m, t in [('opp_sign', 'f2_opposite_signs'), ('ci_sandwich', 'sandwich'), ('ci_strict', 'out_fold_to_pair'),
                                                           ('ci_relaxed', 'out_fold_to_pair'), ('ci_reraise', 'pair_reraise'), ('ci_relaxed', 'fold_partner_after_outfold')]}
for k, v in res['crosscheck'].items():
    print('  crosscheck', k, v, flush=True)
