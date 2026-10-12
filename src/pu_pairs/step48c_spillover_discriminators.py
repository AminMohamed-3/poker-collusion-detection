"""Step 48c: two discriminators between 'spill-over false positive' and 'undisclosed ring member' for dominated block pairs.
(1) family agreement with the dominator: a spill-over pair inherits the dominator's manipulated hands, so its predicted
    family should match the dominator's far above the ring rate (labelled wedges share the family in 44/106, 14.18);
(2) overlap of the pair's five most anomalous hands (proxy listing: top-5 by ca_fold_benefit_absmax, the same proxy as
    step 48b) with the dominator's five; genuine ring pairs never share listed hands (0/53 truth wedges, 1/225 predicted).
Populations: dev dominated labelled positives, dev R1 pairs, dev dominated unlabelled non-R1, eval R1, eval dominated non-R1.
Evaluation labels are never read; dev labels only group. Output: experiments/spillover_20260911/discriminators.json."""
import os, sys, json
os.environ['POLARS_MAX_THREADS'] = '2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import polars as pl, numpy as np

CV = ORIGINAL / 'Claude Version'
OUT = CV / 'experiments' / 'spillover_20260911'
PROXY = 'ca_fold_benefit_absmax'


def top5(ca):
    return ca.sort(['pair_id', PROXY], descending=[False, True]).group_by('pair_id', maintain_order=True).agg(pl.col('hand_id').head(5).alias('top5'))


def overlap_stats(df, list_a, list_b):
    ov = [len(set(a) & set(b)) if a is not None and b is not None else None for a, b in zip(df[list_a].to_list(), df[list_b].to_list())]
    ov = [o for o in ov if o is not None]
    if not ov: return {'n': 0}
    ov = np.array(ov)
    return {'n': int(len(ov)), 'mean_overlap': round(float(ov.mean()), 3), 'share_ge1': round(float((ov >= 1).mean()), 3), 'share_ge2': round(float((ov >= 2).mean()), 3), 'share_ge3': round(float((ov >= 3).mean()), 3)}


def agree_stats(df, fam_a, fam_b):
    d = df.filter(pl.col(fam_a).is_not_null() & pl.col(fam_b).is_not_null())
    if d.height == 0: return {'n': 0}
    a = (d[fam_a] == d[fam_b]).to_numpy()
    fams = d[fam_b].value_counts().sort('count', descending=True)
    p = (fams['count'] / fams['count'].sum()).to_numpy()
    return {'n': int(d.height), 'share_same_family': round(float(a.mean()), 3), 'chance_same_family': round(float((p ** 2).sum()), 3),
            'dominator_families': dict(zip(fams[fam_b].to_list(), fams['count'].to_list()))}


def main():
    out = {}
    # ---- evaluation (label-free)
    ev = pl.read_parquet(OUT / 'eval_rescoring.parquet')
    ca = pl.read_parquet(OUT / 'eval_block_ca.parquet').select('pair_id', 'hand_id', PROXY)
    t5 = top5(ca)
    sub = pl.read_csv(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv').select('pair_id', 'predicted_behavior', pl.concat_list([f'evidence_hand_{i}' for i in range(1, 6)]).alias('listed'))
    ev = ev.join(t5, on='pair_id', how='left').join(sub, on='pair_id', how='left')
    dom = ev.select(pl.col('pair_id').alias('dom_pair'), pl.col('top5').alias('dom_top5'), pl.col('predicted_behavior').alias('dom_family'), pl.col('listed').alias('dom_listed'), pl.col('risk_noT').alias('dom_risk_noT'), pl.col('dominated').alias('dom_dominated'))
    ev = ev.join(dom, on='dom_pair', how='left')
    R1 = ev['dominated'] & (ev['risk_noT'].fill_nan(0) < .1)
    pops = {'eval R1 (dominated & risk_noT<.1)': ev.filter(R1), 'eval dominated non-R1': ev.filter(ev['dominated'] & ~R1),
            'eval dominated non-R1 band<.99': ev.filter(ev['dominated'] & ~R1 & (ev['risk_v13'] < .99))}
    for name, d in pops.items():
        out[name] = {'family': agree_stats(d, 'predicted_behavior', 'dom_family'), 'proxy_top5_overlap_with_dominator': overlap_stats(d, 'top5', 'dom_top5'),
                     'submitted_list_overlap_with_dominator': overlap_stats(d, 'listed', 'dom_listed'),
                     'dominator_risk_noT_median': round(float(np.nanmedian(d['dom_risk_noT'].to_numpy())), 4), 'dominator_share_risk_noT_ge_.5': round(float(np.nanmean(d['dom_risk_noT'].fill_nan(0).to_numpy() >= .5)), 3),
                     'dominator_itself_dominated_share': round(float(d['dom_dominated'].fill_null(False).mean()), 3)}
    # chance level for proxy overlap: pair vs a random other block pair of the same table that shares NO player
    pairs = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2')
    ev2 = ev.join(pairs, on='pair_id', how='left')
    rng = np.random.default_rng(3)
    ovs = []
    by_table = {t: g for t, g in ev2.group_by('table_id')}
    for row in ev2.filter(R1).iter_rows(named=True):
        g = by_table[(row['table_id'],)] if (row['table_id'],) in by_table else by_table[row['table_id']]
        cand = g.filter((pl.col('pair_id') != row['pair_id']) & (pl.col('player_1') != row['player_1']) & (pl.col('player_1') != row['player_2']) & (pl.col('player_2') != row['player_1']) & (pl.col('player_2') != row['player_2']))
        if cand.height == 0: continue
        other = cand.row(int(rng.integers(cand.height)), named=True)
        if row['top5'] is not None and other['top5'] is not None:
            ovs.append(len(set(row['top5']) & set(other['top5'])))
    ovs = np.array(ovs)
    out['eval R1 proxy overlap with a random same-table block pair sharing no player (chance)'] = {'n': int(len(ovs)), 'mean_overlap': round(float(ovs.mean()), 3), 'share_ge1': round(float((ovs >= 1).mean()), 3)} if len(ovs) else {'n': 0}
    # ---- development (labels only to group)
    dv = pl.read_parquet(OUT / 'dev_rescoring.parquet')
    frames = []
    for k in (1, 2):
        cak = pl.read_parquet(OUT / f'dev_fold{k}_ca.parquet').select('pair_id', 'hand_id', PROXY)
        sc = pl.read_parquet(CV / 'experiments' / 'pu_pairs_20260911' / f'fold{k}' / 'scores.parquet').select('pair_id', 'predicted_class', 'behavior_family')
        f = dv.filter(pl.col('fold') == k).join(top5(cak), on='pair_id', how='left').join(sc, on='pair_id', how='left', suffix='_sc')
        frames.append(f)
    dv = pl.concat(frames, how='vertical_relaxed')
    truth = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').group_by('pair_id').agg(pl.col('hand_id').alias('truth_list'))
    dv = dv.join(truth, on='pair_id', how='left')
    # predicted family: predicted_class code -> name via the labelled rows' mapping if numeric
    if dv['predicted_class'].dtype != pl.Utf8:
        lab = dv.filter(pl.col('label') == 1).group_by('predicted_class').agg(pl.col('behavior_family').mode().first().alias('fam'))
        mp = dict(zip(lab['predicted_class'].to_list(), lab['fam'].to_list()))
        dv = dv.with_columns(pl.col('predicted_class').replace_strict(mp, default=None).alias('pred_family'))
    else:
        dv = dv.with_columns(pl.col('predicted_class').alias('pred_family'))
    domd = dv.select(pl.col('fold'), pl.col('pair_id').alias('dom_pair'), pl.col('top5').alias('dom_top5'), pl.col('pred_family').alias('dom_pred_family'), pl.col('behavior_family').alias('dom_truth_family'), pl.col('truth_list').alias('dom_truth_list'), pl.col('risk_noT').alias('dom_risk_noT'))
    dv = dv.join(domd, on=['fold', 'dom_pair'], how='left')
    R1d = dv['dominated'] & (dv['risk_noT'].fill_nan(0) < .1)
    dpops = {'dev dominated labelled positives': dv.filter(dv['dominated'] & (dv['label'] == 1)),
             'dev R1 (dominated & risk_noT<.1, all unlabelled)': dv.filter(R1d),
             'dev dominated unlabelled non-R1': dv.filter(dv['dominated'] & ~R1d & (dv['label'] != 1))}
    for name, d in dpops.items():
        out[name] = {'family_predicted_vs_dominator_predicted': agree_stats(d, 'pred_family', 'dom_pred_family'),
                     'family_truth_vs_dominator_truth (both labelled)': agree_stats(d.filter(pl.col('dom_label') == 1), 'behavior_family', 'dom_truth_family'),
                     'proxy_top5_overlap_with_dominator': overlap_stats(d, 'top5', 'dom_top5'),
                     'truth_list_overlap_with_dominator_truth_list': overlap_stats(d.filter(pl.col('dom_label') == 1), 'truth_list', 'dom_truth_list'),
                     'proxy_top5_overlap_with_dominator_truth_list': overlap_stats(d.filter(pl.col('dom_label') == 1), 'top5', 'dom_truth_list'),
                     'dominator_labelled_positive_share': round(float((d['dom_label'] == 1).mean()), 3),
                     'dominator_risk_noT_median': round(float(np.nanmedian(d['dom_risk_noT'].to_numpy())), 4)}
    write_json(OUT / 'discriminators.json', out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()