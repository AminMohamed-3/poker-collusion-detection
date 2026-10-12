"""Census follow-up 2 (label-free, evaluation).  For each tail candidate (A, X) with p < 1e-3: are its sharp hands
PAIR-SPECIFIC, or does A (or X) produce sharp hands with everybody (a 'hot player' -- over-dispersion of the null)?
Measures per candidate: rate of sharp hands in A's other pairings and X's other pairings (hands above q99.9 / hands),
against the deep-tail honest rate .00078; and whether the candidate's own sharp hands are also sharp in A's / X's other
pairings that contain the same hand (hand-level spill within a player).  Also the same statistics for a random sample
of deep-tail pairs, as the calibration."""
import glob
import numpy as np, polars as pl
ROOT = r'C:\Data\Projects\Money Making\work\poker'; CV = ROOT + r'\Claude Version'
Q999 = .44939; RATE = .00078
res = pl.read_parquet(CV + r'\experiments\step257_eval_mech_scores_20260918\census_tail_candidates.parquet')
ev = pl.read_csv(ROOT + r'\data\evaluation_pairs.csv'); pc = [c for c in ev.columns if c.startswith('player')]
sub = pl.read_csv(CV + r'\outputs_step251_20260917\submission_step251_ens4_dtfp3_on_step222.csv', columns=['pair_id', 'risk_score'])
ev = ev.select(['pair_id'] + pc).join(sub, on='pair_id')
cand = res.join(ev, on='pair_id')
rng = np.random.default_rng(7)
deep = ev.filter(pl.col('risk_score') < .01)
ctrl = deep[rng.choice(deep.height, 150, replace=False)]
players = set(cand[pc[0]]) | set(cand[pc[1]]) | set(ctrl[pc[0]]) | set(ctrl[pc[1]])
pairs_of = ev.filter(pl.col(pc[0]).is_in(list(players)) | pl.col(pc[1]).is_in(list(players)))
print('players', len(players), 'pairs touching them', pairs_of.height)
hs = pl.concat([pl.read_parquet(f, columns=['pair_id', 'hand_id', 'p_fmax']).filter(pl.col('pair_id').is_in(pairs_of['pair_id'].to_list()))
                for f in sorted(glob.glob(CV + r'\experiments\step257_eval_mech_scores_20260918\eval_hand_scores_batch_*.parquet'))])
hs = hs.join(pairs_of.select(['pair_id'] + pc + ['risk_score']), on='pair_id')
print('rows', hs.height)
# per (player, pair): hands and sharp hands
long = pl.concat([hs.select(pl.col(pc[0]).alias('p'), pl.col(pc[1]).alias('o'), 'pair_id', 'hand_id', 'p_fmax', 'risk_score'),
                  hs.select(pl.col(pc[1]).alias('p'), pl.col(pc[0]).alias('o'), 'pair_id', 'hand_id', 'p_fmax', 'risk_score')])
long = long.with_columns(sharp=(pl.col('p_fmax') > Q999))

def player_stats(p, exclude_pair):
    g = long.filter((pl.col('p') == p) & (pl.col('pair_id') != exclude_pair) & (pl.col('risk_score') < .5))   # other TAIL pairings only
    return g.height, int(g['sharp'].sum()), g['pair_id'].n_unique()

def hand_spill(pid, p, hand_ids):
    # of the candidate's sharp hands, how many are ALSO sharp in some other pairing of player p that contains the hand
    g = long.filter((pl.col('p') == p) & (pl.col('pair_id') != pid) & pl.col('hand_id').is_in(hand_ids) & pl.col('sharp'))
    return g['hand_id'].n_unique()

def table(df, label):
    rows = []
    for r in df.iter_rows(named=True):
        pid, a, x = r['pair_id'], r[pc[0]], r[pc[1]]
        own = hs.filter((pl.col('pair_id') == pid) & (pl.col('p_fmax') > Q999))['hand_id'].to_list()
        ha, sa, ka = player_stats(a, pid); hx, sx, kx = player_stats(x, pid)
        rows.append({'pair_id': pid, 'risk': r['risk_score'], 'own_sharp': len(own),
                     'A_other_rate_x': (sa / max(ha, 1)) / RATE, 'X_other_rate_x': (sx / max(hx, 1)) / RATE, 'A_other_pairs': ka, 'X_other_pairs': kx,
                     'own_sharp_also_sharp_for_A_elsewhere': hand_spill(pid, a, own) if own else 0, 'own_sharp_also_sharp_for_X_elsewhere': hand_spill(pid, x, own) if own else 0})
    t = pl.DataFrame(rows)
    t = t.with_columns(hot=(pl.max_horizontal('A_other_rate_x', 'X_other_rate_x') >= 3.0),
                       hand_spill_share=(pl.max_horizontal('own_sharp_also_sharp_for_A_elsewhere', 'own_sharp_also_sharp_for_X_elsewhere') / pl.col('own_sharp').clip(1)))
    print(f'\n== {label}: n {t.height} | median own sharp {t["own_sharp"].median()} | share with a HOT member (other-pairing sharp rate >= 3x honest) {t["hot"].mean():.2f} | '
          f'median max other-pairing rate x honest {float(t.select(pl.max_horizontal("A_other_rate_x", "X_other_rate_x")).to_series().median()):.2f} | '
          f'median share of own sharp hands also sharp in a member\'s other pairing {t["hand_spill_share"].median():.2f} | share of pairs with hand_spill_share == 0: {(t["hand_spill_share"] == 0).mean():.2f}')
    return t

tc = table(cand, 'CANDIDATES p<1e-3 (192)')
print(tc.filter(pl.col('pair_id').is_in(res.filter(pl.col('pv') < 1e-4)['pair_id'].to_list())).select('pair_id', 'risk', 'own_sharp', 'A_other_rate_x', 'X_other_rate_x', 'hand_spill_share', 'hot').sort('own_sharp', descending=True).head(40))
tu = tc.filter(pl.col('pair_id').is_in(res.filter(pl.col('cls') == 'UNLINKED')['pair_id'].to_list()))
print('UNLINKED candidates: n', tu.height, '| hot share', round(float(tu['hot'].mean()), 2), '| hand_spill_share == 0:', round(float((tu['hand_spill_share'] == 0).mean()), 2), '| pair-specific (not hot AND hand_spill_share == 0):', int((~tu['hot'] & (tu['hand_spill_share'] == 0)).sum()))
tctrl = table(ctrl.with_columns(pl.lit(0.0).alias('pv')), 'CONTROL: 150 random deep-tail pairs')
tc.write_parquet(CV + r'\experiments\step257_eval_mech_scores_20260918\census_tail_candidates_player.parquet')
