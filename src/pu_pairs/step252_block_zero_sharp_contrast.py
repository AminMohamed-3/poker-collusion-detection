"""STEP252 (exploratory, label-free, NO submission) -- what marks the hands of block pairs that have no sharp
known-family event hand?

Background (OBS 19.196/19.197): 261 of the 909 pairs with STEP207 risk >= .5 have cnt999 = 0 under the clean known-family
event classifier; development positives of the same length predict ~98.  Group contrast, partner-specific:
  positives = every evaluation pair-hand of a group's pairs (A, B)
  negatives = pair-hands of CONTROL pairs (A, C) and (B, C): same players, another partner C who is in no block pair and
              whose pair sits below risk .01  -> player style is held fixed, only the partner changes.
A LightGBM model with table-grouped folds separates the two; the out-of-fold hand score is the "partner-specific
contrast".  Three groups so the instrument is calibrated:
  Z  = block (risk >= .5) and cnt999 = 0              (the question)
  A  = risk >= .99 and cnt999 >= 3                    (positive control: known events must be found)
  H  = random tail pairs, risk < .01, members in no block pair   (placebo: AUC must be ~.5)
Ensemble form: 3 seeds x 5 table-grouped folds per group, mean of seeds.  Single phase: every hand asserted evaluation.
No evaluation label exists; nothing here reads the leaderboard; no file is produced.
"""
import os, json, glob, time, sys
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step252_zero_sharp_contrast_20260917'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
rng = np.random.default_rng(252)
x = pl.read_parquet(CV + r'\experiments\step250_event_count_tail_20260917\eval_pair_excess.parquet')
block_players = set(x.filter(pl.col('risk_score') >= .5)['player_1'].to_list()) | set(x.filter(pl.col('risk_score') >= .5)['player_2'].to_list())
x = x.with_columns(p1b=pl.col('player_1').is_in(list(block_players)), p2b=pl.col('player_2').is_in(list(block_players)))
groups = {
    'Z': x.filter((pl.col('risk_score') >= .5) & (pl.col('cnt999') == 0)),
    'A': x.filter((pl.col('risk_score') >= .99) & (pl.col('cnt999') >= 3)),
    'H': x.filter((pl.col('risk_score') < .01) & ~pl.col('p1b') & ~pl.col('p2b')).sample(300, seed=252),
}
clean_tail = x.filter(pl.col('risk_score') < .01)          # control pairs: below .01 and partner C in no block pair
by_player = {}
for pid, a, b, ab, bb in clean_tail.select('pair_id', 'player_1', 'player_2', 'p1b', 'p2b').iter_rows():
    if not bb: by_player.setdefault(a, []).append(pid)      # (a, C=b) usable as a control for a if b is clean
    if not ab: by_player.setdefault(b, []).append(pid)
NCTRL = 4                                                   # control pairs per member
assign = []                                                 # (group, pair_id, role, source_pair)
for gname, g in groups.items():
    for pid, a, b in g.select('pair_id', 'player_1', 'player_2').iter_rows():
        assign.append((gname, pid, 1, pid))
        for m in (a, b):
            cands = [c for c in by_player.get(m, []) if c != pid]
            for c in rng.choice(cands, min(NCTRL, len(cands)), replace=False) if cands else []:
                assign.append((gname, str(c), 0, pid))
asg = pl.DataFrame(assign, schema=['grp', 'pair_id', 'y', 'source_pair'], orient='row')
print({g: (asg.filter((pl.col('grp') == g) & (pl.col('y') == 1)).height, asg.filter((pl.col('grp') == g) & (pl.col('y') == 0)).height) for g in groups}, flush=True)
need = asg['pair_id'].unique().to_list()
phase = pl.read_parquet(ROOT + r'\data\hands.parquet', columns=['hand_id', 'phase'])
evp = phase.filter(pl.col('phase') == 'evaluation').select('hand_id')
parts = []
for fp in sorted(glob.glob(ROOT + r'\outputs_v2\hands_evaluation\batch_*.parquet')):
    parts.append(pl.scan_parquet(fp).filter(pl.col('pair_id').is_in(need)).collect())
    print(os.path.basename(fp), parts[-1].height, round(time.time() - t0), 's', flush=True)
h = pl.concat(parts); del parts
assert h.select('hand_id').unique().join(evp, on='hand_id', how='anti').height == 0, 'non-evaluation hand'
feats = [c for c in h.columns if c not in ('pair_id', 'hand_id', 'table_id')]
ev = pl.read_parquet(glob.glob(CV + r'\experiments\step250_event_count_tail_20260917\sharp_hands_batch_*.parquet')).select('pair_id', 'hand_id', 'p_fmax')
P = dict(n_estimators=300, num_leaves=15, learning_rate=.04, min_child_samples=40, subsample=.8, subsample_freq=1,
         colsample_bytree=.5, reg_lambda=5.0, n_jobs=4, verbosity=-1)
res = {}
for gname in groups:
    a = asg.filter(pl.col('grp') == gname)
    # a control pair can serve two sources; keep one row per (pair, y)
    d = h.join(a.select('pair_id', 'y').unique(), on='pair_id')
    X = d.select(feats).to_numpy().astype(np.float32); y = d['y'].to_numpy()
    tables = d['table_id'].to_numpy(); ut = np.unique(tables); fold_of = {t: i % 5 for i, t in enumerate(rng.permutation(ut))}
    fold = np.array([fold_of[t] for t in tables])
    oof = np.zeros(len(y)); gain = np.zeros(len(feats))
    for seed in (1, 2, 3):
        for k in range(5):
            tr, te = fold != k, fold == k
            m = lgb.LGBMClassifier(**P, random_state=252 + seed).fit(X[tr], y[tr])
            oof[te] += m.predict_proba(X[te])[:, 1] / 3
            gain += m.booster_.feature_importance('gain')
    auc = float(roc_auc_score(y, oof))
    d = d.select('pair_id', 'hand_id', 'table_id', 'y').with_columns(pl.Series('contrast', oof.astype(np.float32))).join(ev, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('p_fmax').fill_null(0.))
    d.write_parquet(OUT + f'\\hand_contrast_{gname}.parquet')
    thr = float(np.quantile(oof[y == 0], .999)); thr99 = float(np.quantile(oof[y == 0], .99))
    pp = d.group_by('pair_id', 'y').agg(pl.len().alias('n'), (pl.col('contrast') > thr).sum().alias('c999'), (pl.col('contrast') > thr99).sum().alias('c99'), pl.col('contrast').max().alias('mx'))
    pos, neg = pp.filter(pl.col('y') == 1), pp.filter(pl.col('y') == 0)
    top = [(feats[i], round(float(gain[i] / gain.sum()), 4)) for i in np.argsort(-gain)[:25]]
    hi = d.filter((pl.col('y') == 1) & (pl.col('contrast') > thr))
    res[gname] = {'hands_pos': int((y == 1).sum()), 'hands_neg': int((y == 0).sum()), 'hand_auc': round(auc, 4),
                  'pos_hand_rate_above_neg_q999': round(float((oof[y == 1] > thr).mean()), 5), 'pos_hand_rate_above_neg_q99': round(float((oof[y == 1] > thr99).mean()), 5),
                  'pairs_pos_with_ge1_q999': round(float((pos['c999'] >= 1).mean()), 4), 'pairs_neg_with_ge1_q999': round(float((neg['c999'] >= 1).mean()), 4),
                  'pairs_pos_with_ge2_q999': round(float((pos['c999'] >= 2).mean()), 4), 'pairs_neg_with_ge2_q999': round(float((neg['c999'] >= 2).mean()), 4),
                  'pairs_pos_with_ge3_q99': round(float((pos['c99'] >= 3).mean()), 4), 'pairs_neg_with_ge3_q99': round(float((neg['c99'] >= 3).mean()), 4),
                  'high_contrast_pos_hands': hi.height, 'of_which_known_event_p_fmax_gt_q99': int((hi['p_fmax'] > .0852).sum()),
                  'top_gain': top}
    print(gname, json.dumps({k: v for k, v in res[gname].items() if k != 'top_gain'}), flush=True)
    print('   top gain:', top[:12], flush=True)
    # standardised mean difference of each feature on the high-contrast positive hands vs the group's control hands
    if hi.height >= 30:
        Xi = X[(y == 1) & (oof > thr)]; Xn = X[y == 0]
        smd = (np.nanmean(Xi, 0) - np.nanmean(Xn, 0)) / (np.nanstd(Xn, 0) + 1e-9)
        o = np.argsort(-np.abs(smd))[:25]
        res[gname]['high_contrast_smd'] = [(feats[i], round(float(smd[i]), 2), round(float(np.nanmean(Xi[:, i])), 3), round(float(np.nanmean(Xn[:, i])), 3)) for i in o]
        print('   high-contrast hands SMD:', res[gname]['high_contrast_smd'][:15], flush=True)
json.dump(res, open(OUT + r'\results252.json', 'w', encoding='utf-8'), indent=1)
print('DONE', round(time.time() - t0), 's')
