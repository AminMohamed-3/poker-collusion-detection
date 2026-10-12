# ---- part C2: Part 2 (top-5 violators; true family DESCRIPTIVE ONLY) and Part 3 G_union (label-free)
GATE = {'directed_transfer': 'opp_sign', 'coordinated_isolation': 'ci_relaxed', 'soft_play': 'sp_hu_street'}
fr = fr.with_columns(union=(pl.col('opp_sign') | pl.col('ci_relaxed') | pl.col('sp_hu_street')))
srt = fr.sort(['pair_id', 'dec', 'chrono_rank'], descending=[False, True, False])
top5 = srt.group_by('pair_id', maintain_order=True).head(5)
ties = srt.group_by('pair_id', maintain_order=True).agg(t=(pl.col('dec').gather(4) == pl.col('dec').gather(5)).first())
p2 = {}
for fam, gc in GATE.items():
    g = top5.filter(pl.col('behavior_family') == fam)
    v = g.filter(~pl.col(gc))
    p2[fam] = {'gate': gc, 'picks': g.height, 'picks_listed': int((g['relevance'] == 1).sum()), 'violators': v.height, 'violators_unlisted': int((v['relevance'] == 0).sum())}
v = top5.filter(~pl.col('union'))
p2['label_free_union_all_pairs'] = {'picks': top5.height, 'picks_listed': int((top5['relevance'] == 1).sum()), 'violators': v.height, 'violators_unlisted': int((v['relevance'] == 0).sum())}
res['part2_descriptive_true_family'] = p2
res['part2_top5_boundary_ties'] = int(ties['t'].fill_null(False).sum())
print('PART 2 (DESCRIPTIVE ONLY: true family grouping)')
for k, x in p2.items():
    print('  ', k, x)
print('  top-5 boundary ties', res['part2_top5_boundary_ties'], flush=True)
BIG = 1e8
assert BIG > 10 * (float(np.abs(decS).max()) + 1)
elig, rel, top20 = fr['union'].to_numpy().astype(bool), fr['relevance'].to_numpy() == 1, fr['top20'].to_numpy().astype(bool)
gu = decS + BIG * elig


def show(name, rep_, base):
    c = rep_['comparisons'][name]
    r = json.loads(json.dumps({k: c[k] for k in ('mean_delta', 'ci95', 'by_family', 'halves', 'pairs_better', 'pairs_worse', 'gate_pass')}, default=lambda o: o.item()))
    print(f'  {name} vs {base}', r, flush=True)
    return r


print('PART 3 G_union (label-free): union rate listed', round(float(elig[rel].mean()), 4), 'unlisted top20', round(float(elig[~rel & top20].mean()), 4),
      'unlisted out', round(float(elig[~rel & ~top20].mean()), 4))
rs = evaluate(d, {'incumbent': decS, 'G_union': gu})
rb = evaluate(d, {'incumbent': B0, 'G_union': gu})
res['part3'] = {'BIG': BIG, 'union_listed_rate': float(elig[rel].mean()), 'union_unlisted_top20_rate': float(elig[~rel & top20].mean()),
                'union_unlisted_out_rate': float(elig[~rel & ~top20].mean()), 'sets': json.loads(json.dumps(rs['sets'], default=lambda o: o.item())),
                'vs_typed_S_star': show('G_union', rs, 'typed_S_star'), 'vs_B0': show('G_union', rb, 'B0'),
                'G_route': 'not run: no label-free routing found in audit files; true family never used for routing'}
print('  sets map5', {k: round(v['map5'], 6) for k, v in res['part3']['sets'].items()})
fr.select(KEYS + PRED + ['top20', 'union']).write_parquet(S + 'gate111G_predicates.parquet')
res['protected_unchanged'] = check_protected() == prot
res['seconds'] = round(time.time() - t0, 1)
open(S + 'gate111G.json', 'w', encoding='utf-8').write(json.dumps(res, indent=1, default=lambda o: o.item()))
print('protected unchanged', res['protected_unchanged'], 'seconds', res['seconds'], flush=True)
