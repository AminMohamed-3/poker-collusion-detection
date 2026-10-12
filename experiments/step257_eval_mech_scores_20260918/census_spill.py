"""Census follow-up (label-free, evaluation): the tail holds 40 pairs at binomial p < 1e-4 (chance 11) and 192 at p < 1e-3
(chance 112) by the count of hands above the development honest q99.9.  Are they SPILL-OVER (a colluder's other pairings,
sharing the true pair's event hands) or pairs that share no player with any block pair?
For each candidate: block partners = block pairs (STEP251 risk >= .5) sharing a player; for each sharp hand of the
candidate, was the shared player's block partner dealt in?  Ownership: is the block pair's own p_fmax on that hand higher?"""
import glob
import numpy as np, polars as pl
from scipy.stats import binom
ROOT = r'C:\Data\Projects\Money Making\work\poker'; CV = ROOT + r'\Claude Version'
Q999 = .44939; RATE = .00078
a = pl.read_parquet(CV + r'\experiments\step257_eval_mech_scores_20260918\eval_pair_mech_aggregates.parquet')
sub = pl.read_csv(CV + r'\outputs_step251_20260917\submission_step251_ens4_dtfp3_on_step222.csv', columns=['pair_id', 'risk_score'])
ev = pl.read_csv(ROOT + r'\data\evaluation_pairs.csv')
pc = [c for c in ev.columns if c.startswith('player')]
d = a.join(sub, on='pair_id').join(ev.select(['pair_id'] + pc), on='pair_id')
d = d.with_columns(pl.Series('pv', binom.sf(d['cnt_q999'].to_numpy() - 1, d['n_hands'].to_numpy(), RATE)))
cand = d.filter((pl.col('risk_score') < .5) & (pl.col('pv') < 1e-3)).sort('pv')
block = d.filter(pl.col('risk_score') >= .5)
print('candidates', cand.height, '| block pairs', block.height)
# players of block pairs
bp = {}
for r in block.select(pc + ['pair_id', 'risk_score']).iter_rows():
    bp.setdefault(r[0], []).append((r[1], r[2], r[3])); bp.setdefault(r[1], []).append((r[0], r[2], r[3]))
cand = cand.with_columns(pl.Series('n_block_links', [len(bp.get(p1, [])) + len(bp.get(p2, [])) for p1, p2 in zip(cand[pc[0]], cand[pc[1]])]))
print('candidates sharing a player with a block pair:', (cand['n_block_links'] > 0).sum(), 'of', cand.height, '| p<1e-4 subset:', cand.filter(pl.col('pv') < 1e-4).height, 'linked', cand.filter((pl.col('pv') < 1e-4) & (pl.col('n_block_links') > 0)).height)
# per-hand scores of candidates and of their block partners
need = set(cand['pair_id'].to_list())
for p1, p2 in zip(cand[pc[0]], cand[pc[1]]):
    for q in (p1, p2):
        for other, bpid, _ in bp.get(q, []):
            need.add(bpid)
hs = pl.concat([pl.read_parquet(f, columns=['pair_id', 'hand_id', 'p_fmax']).filter(pl.col('pair_id').is_in(list(need))) for f in sorted(glob.glob(CV + r'\experiments\step257_eval_mech_scores_20260918\eval_hand_scores_batch_*.parquet'))])
print('per-hand rows loaded', hs.height, 'pairs', hs['pair_id'].n_unique())
seats = pl.read_parquet(ROOT + r'\data\seats.parquet', columns=['hand_id', 'player_id'])
sharp = hs.filter(pl.col('p_fmax') > Q999)
sharp_hands = sharp.filter(pl.col('pair_id').is_in(cand['pair_id'].to_list()))
present = seats.filter(pl.col('hand_id').is_in(sharp_hands['hand_id'].unique().to_list()))
pres = {}
for h, p in present.iter_rows():
    pres.setdefault(h, set()).add(p)
bsc = {(r[0], r[1]): r[2] for r in hs.filter(pl.col('pair_id').is_in([b for v in bp.values() for _, b, _ in v])).iter_rows()}
rows = []
for r in cand.iter_rows(named=True):
    pid, p1, p2 = r['pair_id'], r[pc[0]], r[pc[1]]
    sh = sharp_hands.filter(pl.col('pair_id') == pid)
    n_sharp = sh.height; n_partner_present = 0; n_owned_by_block = 0
    for h, pf in zip(sh['hand_id'], sh['p_fmax']):
        links = [(q, other, bpid) for q in (p1, p2) for other, bpid, _ in bp.get(q, [])]
        pp = any(other in pres.get(h, set()) for _, other, _ in links)
        n_partner_present += pp
        n_owned_by_block += any(bsc.get((bpid, h), -1) >= pf for _, other, bpid in links if other in pres.get(h, set()))
    rows.append({'pair_id': pid, 'risk': round(r['risk_score'], 4), 'n_hands': r['n_hands'], 'cnt': int(r['cnt_q999']), 'pv': r['pv'], 'links': r['n_block_links'], 'sharp': n_sharp, 'partner_present': n_partner_present, 'owned_by_block': n_owned_by_block})
res = pl.DataFrame(rows)
res = res.with_columns(cls=pl.when(pl.col('links') == 0).then(pl.lit('UNLINKED')).when(pl.col('owned_by_block') >= pl.col('sharp') - 0).then(pl.lit('spill_all_owned')).when(pl.col('partner_present') >= pl.col('sharp')).then(pl.lit('spill_partner_present')).otherwise(pl.lit('linked_but_own_hands')))
print(res.group_by('cls').agg(pl.len(), pl.col('risk').median(), pl.col('cnt').mean()).sort('len', descending=True))
print('p < 1e-4 subset:'); print(res.filter(pl.col('pv') < 1e-4).group_by('cls').len().sort('len', descending=True))
print(res.filter(pl.col('cls').is_in(['UNLINKED', 'linked_but_own_hands'])).sort('pv').head(40))
# how many UNLINKED would be expected by chance: the unlinked share of the deep tail x chance count
res.write_parquet(CV + r'\experiments\step257_eval_mech_scores_20260918\census_tail_candidates.parquet')
