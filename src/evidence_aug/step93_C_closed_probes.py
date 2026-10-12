"""Step 93 [C]: two descriptive probes, training-only, no model fit, no held-fold read.

(1) Cross-pair evidence exclusivity inside the 162-pair audit set (outside every held fold): when two
    audit pairs share a player, is a hand listed for one pair ever listed, or ranked high, for the other?
(2) Account attributes (players.parquet, used by no script in either tree) of confirmed targets against
    confirmed non-targets and unknown pairs, inside Codex's frozen 169-pool allowed boundary.
"""
import hashlib, io, json
import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from evidence_aug.paths import check_protected

ROOT = 'C:/Data/Projects/Money Making/work/poker/'
CV = ROOT + 'Claude Version/'
OUT = CV + 'experiments/closed_probes_20260913/probes.json'
AUDIT = CV + 'experiments/label_process_20260911/audit-oof.parquet'
INV = ROOT + 'Codex Resume/research/unlabeled_development_inventory_20260913/development-pair-inventory.parquet'
LAB, PLAY = ROOT + 'data/development_labels.csv', ROOT + 'data/players.parquet'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


before = check_protected()
R = {'inputs_sha256': {p.split('/')[-1]: sha(p) for p in (AUDIT, INV, LAB, PLAY)}, 'held_folds_touched': False}

A = pl.read_parquet(AUDIT)
lab = pl.read_csv(LAB).join(A.select('pair_id').unique(), on='pair_id', how='inner')
long = pl.concat([lab.select('pair_id', pl.col('player_1').alias('p')), lab.select('pair_id', pl.col('player_2').alias('p'))])
deg = long.group_by('p').len()
w = long.join(long, on='p').filter(pl.col('pair_id') != pl.col('pair_id_right')).select('pair_id', pl.col('pair_id_right').alias('nb'))
h = A.select('pair_id', 'hand_id', 'relevance', 'oof_rank')
x = w.join(h, on='pair_id').join(h.rename({'pair_id': 'nb', 'relevance': 'rel_nb', 'oof_rank': 'rank_nb'}), on=['nb', 'hand_id'])
dec = x.filter((pl.col('relevance') == 0) & (pl.col('rel_nb') == 1))
R['cross_pair'] = {
    'audit_pairs': lab.height, 'players': deg.height, 'players_degree_1': int((deg['len'] == 1).sum()),
    'players_degree_2': int((deg['len'] == 2).sum()), 'pairs_in_a_wedge': w['pair_id'].n_unique(),
    'shared_hands_undirected': x.height // 2, 'listed_for_both': x.filter((pl.col('relevance') == 1) & (pl.col('rel_nb') == 1)).height // 2,
    'listed_for_one_side_only': dec.height, 'neighbour_listed_in_stage1_top5': dec.filter(pl.col('oof_rank') <= 5).height,
    'neighbour_listed_in_stage1_top20': dec.filter(pl.col('oof_rank') <= 20).height,
    'pairs_with_neighbour_listed_hand_in_top20': dec.filter(pl.col('oof_rank') <= 20)['pair_id'].n_unique(),
    'all_unlisted_stage1_top5_picks': A.filter((pl.col('oof_rank') <= 5) & (pl.col('relevance') == 0)).height,
    'listed_hands': int(A['relevance'].sum()), 'listed_in_stage1_top20': A.filter((pl.col('oof_rank') <= 20) & (pl.col('relevance') == 1)).height}

I = pl.read_parquet(INV).filter(pl.col('shared_hands') >= 38)
P = pl.read_parquet(PLAY)
a = P.rename({c: c + '_a' for c in P.columns}); b = P.rename({c: c + '_b' for c in P.columns})
X = I.join(a, left_on='player_1', right_on='player_id_a').join(b, left_on='player_2', right_on='player_id_b').with_columns(
    same_region=pl.col('region_bucket_a') == pl.col('region_bucket_b'), same_client=pl.col('client_family_a') == pl.col('client_family_b'),
    same_stake=pl.col('preferred_stake_a') == pl.col('preferred_stake_b'), same_exp=pl.col('experience_hands_bucket_a') == pl.col('experience_hands_bucket_b'),
    age_min=pl.min_horizontal('account_age_days_a', 'account_age_days_b'), age_max=pl.max_horizontal('account_age_days_a', 'account_age_days_b'),
    age_gap=(pl.col('account_age_days_a') - pl.col('account_age_days_b')).abs())
X = X.drop('behavior_family').join(pl.read_csv(LAB).select('pair_id', 'behavior_family'), on='pair_id', how='left')
BIN, CONT = ['same_region', 'same_client', 'same_stake', 'same_exp'], ['age_min', 'age_max', 'age_gap']
pos, neg, unk = X.filter(pl.col('label') == 1), X.filter(pl.col('label') == 0), X.filter(pl.col('label') == -1)
m = {'counts': {'positives': pos.height, 'confirmed_negatives': neg.height, 'unknown': unk.height}, 'auc_pos_vs_neg': {}, 'auc_pos_vs_unknown': {}, 'rates': {}, 'family_tests': []}
for c in BIN + CONT:
    f = lambda d: d[c].cast(pl.Float64).to_numpy()
    m['auc_pos_vs_neg'][c] = round(roc_auc_score(np.r_[np.ones(pos.height), np.zeros(neg.height)], np.r_[f(pos), f(neg)]), 3)
    m['auc_pos_vs_unknown'][c] = round(roc_auc_score(np.r_[np.ones(pos.height), np.zeros(unk.height)], np.r_[f(pos), f(unk)]), 3)
for g, d in [('confirmed_negatives', neg), ('unknown', unk)] + [(k, pos.filter(pl.col('behavior_family') == k)) for k in sorted(pos['behavior_family'].unique())]:
    m['rates'][g] = {'n': d.height, **{c: round(float(d[c].mean()), 3) for c in BIN}}
for k in sorted(pos['behavior_family'].unique()):
    d = pos.filter(pl.col('behavior_family') == k)
    for c in BIN:
        p1, p0, n1, n0 = float(d[c].mean()), float(neg[c].mean()), d.height, neg.height
        pp = (p1 * n1 + p0 * n0) / (n1 + n0)
        z = (p1 - p0) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
        m['family_tests'].append({'family': k, 'attr': c, 'z': round(float(z), 2), 'p_two_sided': round(float(2 * norm.sf(abs(z))), 4)})
m['n_family_tests'] = len(m['family_tests'])
m['min_bonferroni_p'] = round(min(1.0, min(t['p_two_sided'] for t in m['family_tests']) * len(m['family_tests'])), 3)
m['max_abs_auc_minus_half'] = round(max(abs(v - .5) for v in list(m['auc_pos_vs_neg'].values()) + list(m['auc_pos_vs_unknown'].values())), 3)
R['account_attributes'] = m
assert check_protected() == before, 'protected files changed'
R['protected_identical'] = True
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(R, indent=1))
print(json.dumps(R['cross_pair']))
print(json.dumps({k: m[k] for k in ('counts', 'auc_pos_vs_neg', 'auc_pos_vs_unknown', 'n_family_tests', 'min_bonferroni_p', 'max_abs_auc_minus_half')}))
print(json.dumps(m['rates']))
print(sorted(m['family_tests'], key=lambda t: t['p_two_sided'])[:3])
