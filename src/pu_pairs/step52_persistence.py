"""Step 52 (label-free on evaluation; development labels only to validate): cross-phase PERSISTENCE as a false-positive
discriminator for the block. Every pair exists in both phases (same table, same two players; 3,000 development and
2,000 evaluation hands) and collusion is planted independently per phase (14.10, 14.11: development colluders are clean
in evaluation, planting is uncorrelated across phases). A genuine evaluation colluder therefore has a clean development
record (risk_dev ~ null), whereas a pair whose collusion-like statistics come from the two bots' PERSISTENT personal
styles (not from a plant) scores high in both phases. The spill-over false positives (steps 48-50) are NOT persistent
either (the partner plant is phase-specific), so this test isolates a different false-positive class: style artefacts.

Part 1 (development -> evaluation, instant): dev block pairs (held-fold risk >= .5, folds 1+2 deduped) joined by player
pair to the v13 EVALUATION risk. Labelled positives cannot be joined (their players are excluded from the evaluation
pair list); unlabelled block pairs by band and by R1 membership vs matched tail pairs (risk < .1).
Part 2 (evaluation -> development): a single-phase LightGBM on the 655-column base+v2 aggregates (means + stds, step 32
recipe) trained on all 1,860 labelled development pairs (lab_on_dev), applied to eval_on_dev = the development-phase
features of the 112,478 evaluation pairs. Reports risk_dev for the 909 block pairs by band / R1 / dominated vs the tail,
and the same model's evaluation-phase score (eval_on_eval) as a parity check against v13.
Outputs: experiments/persistence_20260911/{persistence.json, eval_devphase_scores.parquet}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

CV = ORIGINAL / 'Claude Version'
XP = CV / 'experiments' / 'crossphase_20260911'
SP = CV / 'experiments' / 'spillover_20260911'
OUT = CV / 'experiments' / 'persistence_20260911'
BASE = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
PARAMS = dict(objective='binary', n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
              reg_lambda=5.0, random_state=7329, deterministic=True, force_col_wise=True, n_jobs=2, verbosity=-1)


def key(df):
    return df.with_columns(pl.when(pl.col('player_1') < pl.col('player_2')).then(pl.col('player_1') + '|' + pl.col('player_2')).otherwise(pl.col('player_2') + '|' + pl.col('player_1')).alias('pkey'))


def dist(x, name):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return {'n': 0}
    return {'n': int(len(x)), 'mean': round(float(x.mean()), 4), 'median': round(float(np.median(x)), 4), 'q90': round(float(np.quantile(x, .9)), 4),
            'share_ge_.1': round(float((x >= .1).mean()), 4), 'share_ge_.5': round(float((x >= .5).mean()), 4), 'share_ge_.9': round(float((x >= .9).mean()), 4)}


def part1():
    out = {}
    dev = pl.concat([pl.read_parquet(CV / 'experiments' / 'pu_pairs_20260911' / f'fold{k}' / 'scores.parquet').select('pair_id', 'player_1', 'player_2', 'table_id', 'risk', 'label', 'shared_hands') for k in (1, 2)]).unique(subset=['pair_id'], keep='first', maintain_order=True)
    flags = pl.read_parquet(SP / 'neighbour_dev_pairs.parquet').select('pair_id', 'dominated', 'R1', 'R4any').unique(subset=['pair_id'], keep='first')
    dev = key(dev).join(flags, on='pair_id', how='left')
    ev = key(pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2', 'shared_hands').rename({'shared_hands': 'shared_hands_eval', 'pair_id': 'pair_id_eval'}))
    base = pl.read_csv(BASE).select('pair_id', 'risk_score').rename({'pair_id': 'pair_id_eval', 'risk_score': 'risk_eval'})
    ev = ev.join(base, on='pair_id_eval', how='inner')
    j = dev.join(ev.select('pkey', 'pair_id_eval', 'shared_hands_eval', 'risk_eval'), on='pkey', how='left')
    out['dev_pairs'] = int(dev.height); out['dev_pairs_with_eval_counterpart'] = int(j['risk_eval'].is_not_null().sum())
    out['same_pair_id_across_phases_share'] = round(float((j.filter(pl.col('pair_id_eval').is_not_null())['pair_id'] == j.filter(pl.col('pair_id_eval').is_not_null())['pair_id_eval']).mean()), 4)
    jj = j.filter(pl.col('risk_eval').is_not_null())
    band = pl.when(pl.col('risk') >= .99).then(pl.lit('[.99,1]')).when(pl.col('risk') >= .9).then(pl.lit('[.9,.99)')).when(pl.col('risk') >= .5).then(pl.lit('[.5,.9)')).when(pl.col('risk') >= .1).then(pl.lit('[.1,.5)')).otherwise(pl.lit('<.1')).alias('band')
    jj = jj.with_columns(band)
    groups = {}
    for (lab, b), g in jj.group_by('label', 'band'):
        groups[f'label={lab} band={b}'] = dist(g['risk_eval'].to_numpy(), '')
    out['eval_risk_of_dev_pairs_by_label_band'] = dict(sorted(groups.items()))
    blk = jj.filter(pl.col('risk') >= .5)
    out['eval_risk_of_dev_block_unlabelled_by_R1'] = {f'R1={r}': dist(g['risk_eval'].to_numpy(), '') for (r,), g in blk.filter(pl.col('label') == -1).group_by('R1')}
    out['eval_risk_of_dev_block_unlabelled_by_dominated'] = {f'dominated={r}': dist(g['risk_eval'].to_numpy(), '') for (r,), g in blk.filter(pl.col('label') == -1).group_by('dominated')}
    # matched tail control: tail unlabelled pairs (dev risk < .1) with eval shared hands in the same bins as the block
    tail = jj.filter((pl.col('risk') < .1) & (pl.col('label') == -1))
    out['eval_risk_of_dev_tail_unlabelled'] = dist(tail['risk_eval'].to_numpy(), '')
    out['eval_shared_hands_median_block_vs_tail'] = [float(blk['shared_hands_eval'].median()), float(tail['shared_hands_eval'].median())]
    # listing of the persistent ones
    pers = blk.filter(pl.col('risk_eval') >= .1).select('pair_id', 'table_id', 'risk', 'risk_eval', 'label', 'R1', 'dominated', 'shared_hands', 'shared_hands_eval').sort('risk_eval', descending=True)
    out['dev_block_pairs_with_eval_risk_ge_.1'] = pers.to_dicts()
    return out


def part2():
    out = {}
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    lab_dev = pl.read_parquet(XP / 'lab_on_dev.parquet').join(lab, on='pair_id', how='inner')
    feats = [c for c in lab_dev.columns if c.endswith('_mean') or c.endswith('_std')]
    y = lab_dev['label'].to_numpy().astype(int); groups = lab_dev['table_id'].to_numpy()
    Xl = np.nan_to_num(lab_dev.select(feats).to_numpy().astype(np.float32), nan=0, posinf=0, neginf=0)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Xl, y, groups):
        oof[te] = lgb.LGBMClassifier(**PARAMS).fit(Xl[tr], y[tr]).predict_proba(Xl[te])[:, 1]
    out['single_phase_655_cv'] = {'ap': round(float(average_precision_score(y, oof)), 4), 'auc': round(float(roc_auc_score(y, oof)), 4), 'n_features': len(feats),
                                  'neg_oof_share_ge_.1_.5_.9': [round(float((oof[y == 0] >= t).mean()), 4) for t in (.1, .5, .9)]}
    model = lgb.LGBMClassifier(**PARAMS).fit(Xl, y)
    # other-phase (evaluation) score of the labelled development pairs: positives are presumed clean there
    lab_ev = pl.read_parquet(XP / 'lab_on_eval.parquet').join(lab, on='pair_id', how='inner')
    s_ev = model.predict_proba(np.nan_to_num(lab_ev.select(feats).to_numpy().astype(np.float32), nan=0, posinf=0, neginf=0))[:, 1]
    out['labelled_pairs_other_phase_score'] = {f'label={l}': dist(s_ev[lab_ev['label'].to_numpy() == l], '') for l in (0, 1)}
    # development-phase score of every evaluation pair
    ev_dev = pl.read_parquet(XP / 'eval_on_dev.parquet')
    s_dev = model.predict_proba(np.nan_to_num(ev_dev.select(feats).to_numpy().astype(np.float32), nan=0, posinf=0, neginf=0))[:, 1]
    ev_ev = pl.read_parquet(XP / 'eval_on_eval.parquet')
    s_evev = model.predict_proba(np.nan_to_num(ev_ev.select(feats).to_numpy().astype(np.float32), nan=0, posinf=0, neginf=0))[:, 1]
    sc = ev_dev.select('pair_id', 'table_id', pl.col('shared_hands').alias('shared_hands_dev')).with_columns(pl.Series('risk_devphase', s_dev)).join(
        ev_ev.select('pair_id', pl.col('shared_hands').alias('shared_hands_eval')).with_columns(pl.Series('risk_evalphase_655', s_evev)), on='pair_id', how='left')
    base = pl.read_csv(BASE).select('pair_id', pl.col('risk_score').alias('risk_v13'))
    sc = sc.join(base, on='pair_id', how='left')
    flags = pl.read_parquet(SP / 'neighbour_eval_pairs.parquet').select('pair_id', 'dominated', 'R1', 'R4any')
    sc = sc.join(flags, on='pair_id', how='left').with_columns(pl.col('dominated').fill_null(False), pl.col('R1').fill_null(False), pl.col('R4any').fill_null(False))
    v = sc.filter(pl.col('risk_v13').is_not_null())
    hi = v.filter(pl.col('risk_v13') >= .9); lo = v.filter(pl.col('risk_v13') < .1)
    out['parity_655_evalphase_vs_v13'] = {'auc_hi_vs_lo': round(float(roc_auc_score(np.r_[np.ones(hi.height), np.zeros(lo.height)], np.r_[hi['risk_evalphase_655'].to_numpy(), lo['risk_evalphase_655'].to_numpy()])), 4),
                                         'spearman_block': round(float(np.corrcoef(np.argsort(np.argsort(v.filter(pl.col('risk_v13') >= .5)['risk_v13'].to_numpy())), np.argsort(np.argsort(v.filter(pl.col('risk_v13') >= .5)['risk_evalphase_655'].to_numpy())))[0, 1]), 4)}
    band = pl.when(pl.col('risk_v13') >= .99).then(pl.lit('[.99,1]')).when(pl.col('risk_v13') >= .9).then(pl.lit('[.9,.99)')).when(pl.col('risk_v13') >= .5).then(pl.lit('[.5,.9)')).when(pl.col('risk_v13') >= .1).then(pl.lit('[.1,.5)')).otherwise(pl.lit('<.1')).alias('band')
    v = v.with_columns(band)
    out['devphase_risk_by_v13_band'] = {b: dist(g['risk_devphase'].to_numpy(), '') for (b,), g in sorted(v.group_by('band'), key=lambda kv: kv[0])}
    blk = v.filter(pl.col('risk_v13') >= .5)
    out['devphase_risk_block_by_R1'] = {f'R1={r}': dist(g['risk_devphase'].to_numpy(), '') for (r,), g in blk.group_by('R1')}
    out['devphase_risk_block_by_dominated'] = {f'dominated={r}': dist(g['risk_devphase'].to_numpy(), '') for (r,), g in blk.group_by('dominated')}
    # tail control matched on development shared hands (block pairs have longer dev histories? report medians)
    out['dev_shared_hands_median_block_vs_tail'] = [float(blk['shared_hands_dev'].median()), float(v.filter(pl.col('risk_v13') < .1)['shared_hands_dev'].median())]
    # expected chance rate: undisclosed development colluders among evaluation pairs ~ share of tail pairs >= .9 in dev phase
    out['tail_devphase_share_ge_.9'] = round(float((v.filter(pl.col('risk_v13') < .1)['risk_devphase'] >= .9).mean()), 5)
    pers = blk.filter(pl.col('risk_devphase') >= .1).select('pair_id', 'table_id', 'risk_v13', 'risk_devphase', 'risk_evalphase_655', 'R1', 'dominated', 'shared_hands_dev', 'shared_hands_eval').sort('risk_devphase', descending=True)
    out['block_pairs_with_devphase_risk_ge_.1'] = {'n': pers.height, 'by_v13_band': pers.with_columns(band.alias('band') if False else pl.when(pl.col('risk_v13') >= .99).then(pl.lit('[.99,1]')).when(pl.col('risk_v13') >= .9).then(pl.lit('[.9,.99)')).otherwise(pl.lit('[.5,.9)')).alias('band')).group_by('band').len().sort('band').to_dicts(),
                                                   'in_R1': int(pers['R1'].sum()), 'rows': pers.head(60).to_dicts()}
    # joint: is devphase risk correlated with the evaluation-phase 655 score among tail pairs? (style persistence at the population level)
    t = v.filter(pl.col('risk_v13') < .1)
    out['tail_corr_devphase_vs_evalphase655_spearman'] = round(float(pl.DataFrame({'a': t['risk_devphase'], 'b': t['risk_evalphase_655']}).select(pl.corr('a', 'b', method='spearman')).item()), 4)
    sc.write_parquet(OUT / 'eval_devphase_scores.parquet')
    return out


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    res = {'part1_dev_block_to_eval': part1()}
    print(json.dumps({k: v for k, v in res['part1_dev_block_to_eval'].items() if k != 'dev_block_pairs_with_eval_risk_ge_.1'}, indent=1), flush=True)
    print('persistent dev block pairs:', json.dumps(res['part1_dev_block_to_eval']['dev_block_pairs_with_eval_risk_ge_.1'][:40]), flush=True)
    res['part2_eval_block_to_dev'] = part2()
    print(json.dumps({k: v for k, v in res['part2_eval_block_to_dev'].items() if k != 'block_pairs_with_devphase_risk_ge_.1'}, indent=1), flush=True)
    p = res['part2_eval_block_to_dev']['block_pairs_with_devphase_risk_ge_.1']
    print('eval block pairs with dev-phase risk >= .1:', p['n'], p['by_v13_band'], 'in_R1', p['in_R1'], flush=True)
    print(json.dumps(p['rows'][:40]), flush=True)
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'persistence.json', res)


if __name__ == '__main__':
    main()