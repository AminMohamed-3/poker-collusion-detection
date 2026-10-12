"""Step 51 (label-free on evaluation; development labels only to validate): the positive-pair graph is a forest.
Development labels: 372 positive pairs over 693 players form 321 connected components (276 pairs, 40 triples, 4 quads,
1 quintet); every multi-player component is a tree (3 players -> exactly 2 positive pairs, never 3). So among any set of
players the planted pairs never close a cycle. In the evaluation block (risk >= .5) an edge that closes a cycle against
higher-risk edges is therefore a candidate false positive. This script measures, on both phases, the maximum-risk
spanning forest (Kruskal, descending risk) of the block graph per table and reports the excluded (cycle-closing) edges:
how many, which band, how many are labelled positives in development (should be ~0 if the structure holds), and the
overlap with the step-49 R1 demotion set.
Outputs: experiments/spillover_20260911/block_cycles.json"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import polars as pl
import numpy as np

CV = ORIGINAL / 'Claude Version'
EXP = CV / 'experiments' / 'pu_pairs_20260911'
OUT = CV / 'experiments' / 'spillover_20260911'
BASE = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
R1 = CV / 'outputs_spillover_20260911' / 'submission_spillover_demotion_R1.csv'
BLOCK = 0.5


class DSU:
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        self.p[ra] = rb; return True


def spanning_forest(edges):
    """edges: list of (risk, pair_id, a, b). Returns set of excluded pair_ids (close a cycle against higher-risk edges)
    and, for each excluded edge, the max-risk path edges are not needed; we also record triangle closure."""
    dsu = DSU(); excluded = []
    for risk, pid, a, b in sorted(edges, key=lambda e: -e[0]):
        if not dsu.union(a, b):
            excluded.append(pid)
    return set(excluded)


def triangles(edges):
    """pair_ids of edges that are the minimum-risk edge of at least one triangle in the block graph."""
    adj = collections.defaultdict(dict)
    for risk, pid, a, b in edges:
        adj[a][b] = (risk, pid); adj[b][a] = (risk, pid)
    tri_min = set(); n_tri = 0
    for risk, pid, a, b in edges:
        for c in set(adj[a]) & set(adj[b]):
            if c in (a, b): continue
            # each triangle counted 3 times; count once by requiring pid to be the min edge
            r2, p2 = adj[a][c]; r3, p3 = adj[b][c]
            if risk <= r2 and risk <= r3:
                tri_min.add(pid); n_tri += 1
    return tri_min, n_tri // 1


def analyse(df, name, r1_ids=None):
    """df: pair_id, player_1, player_2, table_id, risk, label (label -1 unknown / 0 / 1)."""
    blk = df.filter(pl.col('risk') >= BLOCK)
    out = {'pairs_in_block': int(blk.height), 'tables': int(blk['table_id'].n_unique())}
    excluded, tri_min = set(), set()
    comp_stats = collections.Counter(); n_cycles = 0
    for tid, g in blk.group_by('table_id'):
        edges = [(float(r['risk']), r['pair_id'], r['player_1'], r['player_2']) for r in g.iter_rows(named=True)]
        ex = spanning_forest(edges); excluded |= ex
        tm, _ = triangles(edges); tri_min |= tm
        n_cycles += len(ex)
        # component sizes (players)
        dsu = DSU()
        for _, _, a, b in edges: dsu.union(a, b)
        sizes = collections.Counter(dsu.find(a) for _, _, a, b in edges for a in (a, b))
        # sizes counts endpoints (2 per edge); use players
        players = collections.defaultdict(set)
        for _, _, a, b in edges: players[dsu.find(a)].update((a, b))
        for c, s in players.items(): comp_stats[len(s)] += 1
    out['components_by_player_count'] = dict(sorted((int(k), int(v)) for k, v in comp_stats.items()))
    out['cycle_closing_edges'] = int(len(excluded))
    out['triangle_min_edges'] = int(len(tri_min))
    out['triangle_min_edges_also_cycle_closing'] = int(len(tri_min & excluded))
    blk = blk.with_columns(pl.col('pair_id').is_in(list(excluded)).alias('cycle_close'), pl.col('pair_id').is_in(list(tri_min)).alias('tri_min'))
    band = pl.when(pl.col('risk') >= .99).then(pl.lit('[.99,1]')).when(pl.col('risk') >= .9).then(pl.lit('[.9,.99)')).otherwise(pl.lit('[.5,.9)')).alias('band')
    blk = blk.with_columns(band)
    out['cycle_closing_by_band'] = blk.filter('cycle_close').group_by('band').len().sort('band').to_dicts()
    out['block_by_band'] = blk.group_by('band').len().sort('band').to_dicts()
    if 'label' in blk.columns:
        out['cycle_closing_by_label'] = blk.filter('cycle_close').group_by('label').len().sort('label').to_dicts()
        out['triangle_min_by_label'] = blk.filter('tri_min').group_by('label').len().sort('label').to_dicts()
        out['block_by_label'] = blk.group_by('label').len().sort('label').to_dicts()
        # positives that are cycle-closing: their risk and the competing edges
        pos_ex = blk.filter(pl.col('cycle_close') & (pl.col('label') == 1))
        out['positives_cycle_closing'] = pos_ex.select('pair_id', 'table_id', 'risk', 'behavior_family').to_dicts() if pos_ex.height else []
        # AP within block before/after demoting cycle-closing edges to just below .5 (labelled rows only, as in step 50)
        lab = blk.filter(pl.col('label') >= 0)
        def ap(y, s):
            o = np.argsort(-s, kind='stable'); y = y[o]
            if y.sum() == 0: return float('nan')
            prec = np.cumsum(y) / (np.arange(len(y)) + 1)
            return float((prec * y).sum() / y.sum())
        # dev AP over all pairs (block + rest) with labels
        allp = df.filter(pl.col('label') >= 0).with_columns(pl.col('pair_id').is_in(list(excluded)).alias('cycle_close'))
        y = allp['label'].to_numpy().astype(float); s = allp['risk'].to_numpy().astype(float)
        s2 = np.where(allp['cycle_close'].to_numpy(), np.minimum(s, .4999), s)
        out['dev_AP_labelled_before_after'] = [round(ap(y, s), 5), round(ap(y, s2), 5)]
    if r1_ids is not None:
        out['cycle_closing_in_R1'] = int(blk.filter(pl.col('cycle_close') & pl.col('pair_id').is_in(list(r1_ids))).height)
        out['cycle_closing_not_in_R1_by_band'] = blk.filter(pl.col('cycle_close') & ~pl.col('pair_id').is_in(list(r1_ids))).group_by('band').len().sort('band').to_dicts()
        out['triangle_min_not_in_R1'] = int(blk.filter(pl.col('tri_min') & ~pl.col('pair_id').is_in(list(r1_ids))).height)
    print(name, json.dumps({k: v for k, v in out.items() if k != 'positives_cycle_closing'}), 'positives_cycle_closing n=', len(out.get('positives_cycle_closing', [])))
    return out, blk


def main():
    t0 = time.time()
    res = {}
    # development: held-fold scores, folds 1 and 2
    dev = pl.concat([pl.read_parquet(EXP / f'fold{k}' / 'scores.parquet').select('pair_id', 'player_1', 'player_2', 'table_id', 'risk', 'label', 'behavior_family') for k in (1, 2)]).unique(subset=['pair_id'], keep='first', maintain_order=True)  # folds share 29 held tables
    res['dev'], dev_blk = analyse(dev, 'dev')
    # evaluation: base risk and player ids
    base = pl.read_csv(BASE).select('pair_id', 'risk_score').rename({'risk_score': 'risk'})
    pairs = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2').join(pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet').select('pair_id', 'table_id'), on='pair_id', how='inner')
    ev = pairs.join(base, on='pair_id', how='inner')
    r1 = pl.read_csv(R1).select('pair_id', 'risk_score').join(base, on='pair_id').filter(pl.col('risk_score') != pl.col('risk'))['pair_id'].to_list()
    res['eval'], ev_blk = analyse(ev, 'eval', set(r1))
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'block_cycles.json', res)
    ev_blk.write_parquet(OUT / 'block_cycles_eval.parquet'); dev_blk.write_parquet(OUT / 'block_cycles_dev.parquet')


if __name__ == '__main__':
    main()