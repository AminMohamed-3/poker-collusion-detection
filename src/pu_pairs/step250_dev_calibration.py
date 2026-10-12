"""STEP250 part A -- development-only calibration of the pair statistic "number of hands above the honest q99.9 of the
clean event classifier" (cnt999), at evaluation-like length (65 % random thinning, 3 seeds).  Out-of-fold per-hand
scores from STEP232 (`hand_oof.parquet`), so every hand is scored by a model that never saw its table.
Questions: P(cnt >= k) for positives (by family, by listed count) and for confirmed non-targets (split by whether the
pair shares a player with a labelled positive pair = spill-over), and how the statistic relates to a pair-aggregate
model's misses (STEP247 OLD arm, T2 out-of-fold score)."""
import json
import numpy as np, polars as pl
ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step250_event_count_tail_20260917'
THR999, THR99 = .4662, .0852
h = pl.read_parquet(CV + r'\experiments\step232_event_detector_20260917\hand_oof.parquet')
lab = pl.read_csv(ROOT + r'\data\development_labels.csv')
pos_players = set(lab.filter(pl.col('label') == 1)['player_1'].to_list()) | set(lab.filter(pl.col('label') == 1)['player_2'].to_list())
lab = lab.with_columns(spill=(pl.col('player_1').is_in(list(pos_players)) | pl.col('player_2').is_in(list(pos_players))) & (pl.col('label') == 0))
nl = h.group_by('pair_id').agg(pl.col('listed').sum().alias('n_listed'), pl.len().alias('n_full'))
meta = lab.select('pair_id', 'label', 'behavior_family', 'spill').join(nl, on='pair_id')
z = np.load(CV + r'\experiments\step247_pair_new_block_20260917\cache\oof_OLD_236.npz', allow_pickle=True)
sm = json.load(open(CV + r'\experiments\step247_pair_new_block_20260917\cache\store_meta.json', encoding='utf-8'))
proxy = pl.DataFrame({'pair_id': sm['pair_ids'], 'proxy_T2': z['T2']})
meta = meta.join(proxy, on='pair_id', how='left')
res = {'thresholds': {'q999': THR999, 'q99': THR99}, 'seeds': {}}
for seed in (250, 251, 252):
    rng = np.random.default_rng(seed)
    keep = rng.random(h.height) < .65
    g = h.filter(pl.Series(keep)).group_by('pair_id').agg(
        pl.len().alias('n'), (pl.col('p_fmax') > THR999).sum().alias('cnt999'), (pl.col('p_fmax') > THR99).sum().alias('cnt99'),
        ((pl.col('p_fmax') > THR999) & (pl.col('listed') == 0)).sum().alias('cnt999_unlisted'))
    d = meta.join(g, on='pair_id', how='left').with_columns(pl.col('cnt999').fill_null(0), pl.col('cnt99').fill_null(0))
    out = {}
    def rates(df):
        c = df['cnt999'].to_numpy()
        return {'n': int(len(c)), **{f'ge{k}': round(float((c >= k).mean()), 4) for k in (1, 2, 3, 4, 5)}, 'median_hands': float(df['n'].median() or 0)}
    out['negatives_all'] = rates(d.filter(pl.col('label') == 0))
    out['negatives_no_spill'] = rates(d.filter((pl.col('label') == 0) & ~pl.col('spill')))
    out['negatives_spill'] = rates(d.filter((pl.col('label') == 0) & pl.col('spill')))
    for f in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
        out['pos_' + f] = rates(d.filter((pl.col('label') == 1) & (pl.col('behavior_family') == f)))
    for k in (3, 4, 5):
        out[f'pos_listed{k}'] = rates(d.filter((pl.col('label') == 1) & (pl.col('n_listed') == k)))
    # relation to the pair-aggregate proxy: threshold = 99th percentile of negatives' proxy score
    thr = float(np.quantile(d.filter(pl.col('label') == 0)['proxy_T2'].to_numpy(), .99))
    missed = d.filter((pl.col('label') == 1) & (pl.col('proxy_T2') < thr)); caught = d.filter((pl.col('label') == 1) & (pl.col('proxy_T2') >= thr))
    out['proxy'] = {'thr_neg_q99': round(thr, 4), 'pos_missed_by_proxy': rates(missed), 'pos_caught_by_proxy': rates(caught),
                    'neg_above_proxy_thr': rates(d.filter((pl.col('label') == 0) & (pl.col('proxy_T2') >= thr)))}
    res['seeds'][seed] = out
    print(seed, json.dumps(out, indent=0)[:3000], flush=True)
json.dump(res, open(OUT + r'\dev_calibration.json', 'w', encoding='utf-8'), indent=1)
