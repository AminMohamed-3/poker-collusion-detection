"""HF5: is the concentrated candidate population a coordinating RING, or one odd PLAYER?

hf4's G4 is the strongest signal in this investigation: 38 candidate pairs span 20 tables
against a random null of 29.9 +- 2.3, and one player sits in 8 of them against a null of
1.8 +- 0.5.  That count alone cannot distinguish the two explanations, and they mean opposite
things:

  RING  -- three or more players coordinating as a group.  The competition scores PAIRS, so a
           ring of k players emits k(k-1)/2 positive pairs.  All three disclosed families are
           described pairwise; a multi-way mechanism is the most natural thing for a fourth,
           undisclosed one to be.  Signature: the candidate pairs among a table's involved
           players form TRIANGLES, and each member looks ordinary with everyone outside the ring.
  STAR  -- one player with an unusual individual policy.  Every one of their ~29 partners then
           looks anomalous in pair features, because ca_ features are pair-level but driven by
           individual behaviour.  This is a DECOY, explicitly listed by the host ("unusual but
           non-coordinated play patterns").  Signature: high degree with no triangles, and the
           hub is anomalous with EVERY partner, not a select few.

The separating statistic is therefore per-player selectivity: for each player in a candidate
pair, compare the anomaly of the pairs they are in against the anomaly of all their OTHER
pairs.  A ring member is anomalous only with fellow members; a star hub is anomalous with
everyone.

Also: (a) the same graph statistics on the LABELLED positives, which tells us what a mechanism
the host does disclose actually looks like structurally; (b) an interpretable per-feature
signature of the candidates against each disclosed family, to answer "what does it look like";
(c) the null calibration hf4's T2 was missing -- bootstrap the mixture L1 residual under the
fitted model, so the residual can be read as signal or as the sampling noise of 38 points.

Read-only.  Consumes hf3/hf4 frozen outputs.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '4'
import glob
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import CV, write_json

OUT = CV / 'experiments' / 'hidden_family_20260912'
PU = CV / 'experiments' / 'pu_pairs_20260911'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
RNG = np.random.default_rng(20260912)


def graph_stats(p1, p2):
    """Degree profile, triangles and density of the pair graph."""
    edges = set()
    deg = {}
    for a, b in zip(p1, p2):
        e = (a, b) if a < b else (b, a)
        edges.add(e)
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    nodes = sorted(deg)
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    tri = 0
    for a, b in edges:
        tri += len(adj[a] & adj[b])
    tri //= 3
    # how many of the hub's partners are connected to each other
    hub = max(deg, key=deg.get) if deg else None
    hub_partners = adj[hub] if hub else set()
    hub_closed = sum(1 for x, y in combinations(sorted(hub_partners), 2) if y in adj[x])
    hub_possible = len(hub_partners) * (len(hub_partners) - 1) // 2
    return {'n_edges': len(edges), 'n_nodes': len(nodes), 'triangles': tri,
            'max_degree': max(deg.values()) if deg else 0,
            'mean_degree': float(np.mean(list(deg.values()))) if deg else 0.0,
            'hub_partner_closure': hub_closed,
            'hub_partner_possible': hub_possible,
            'hub_closure_rate': hub_closed / hub_possible if hub_possible else float('nan')}


def main():
    fr = pl.read_parquet(OUT / 'hf3-dev-eligible.parquet')
    Z = np.load(OUT / 'hf3-dev-z.npy').astype(np.float64)
    y = fr['label'].to_numpy()
    fam = fr['behavior_family'].to_numpy()
    tab = fr['table_id'].to_numpy()
    pct = fr['anom_pct'].to_numpy()
    rk = fr['risk_known'].to_numpy()
    P1 = fr['player_1'].to_numpy()
    P2 = fr['player_2'].to_numpy()
    pos, neg, unl = y == 1, y == 0, y == -1

    print('=' * 96)
    print('A.  GRAPH SHAPE: candidates vs labelled positives vs random, at several bands')
    print('=' * 96)
    print(f'{"set":<26}{"pairs":>7}{"nodes":>7}{"tri":>6}{"maxdeg":>8}'
          f'{"hubclose":>10}{"null_tri":>10}{"null_maxdeg":>13}')
    graph = {}
    sets = {'labelled positives': pos}
    for q in [0.0025, 0.0050, 0.0100]:
        sets[f'candidates top-{q:.4f}'] = (pct >= 1 - q) & unl & (rk < 0.5)
    for name, m in sets.items():
        g = graph_stats(P1[m], P2[m])
        nt, nd = [], []
        for _ in range(300):
            pick = RNG.choice(len(fr), size=int(m.sum()), replace=False)
            gn = graph_stats(P1[pick], P2[pick])
            nt.append(gn['triangles'])
            nd.append(gn['max_degree'])
        g['null_triangles_mean'] = float(np.mean(nt))
        g['null_max_degree_mean'] = float(np.mean(nd))
        g['null_max_degree_sd'] = float(np.std(nd))
        graph[name] = g
        print(f'{name:<26}{g["n_edges"]:>7}{g["n_nodes"]:>7}{g["triangles"]:>6}'
              f'{g["max_degree"]:>8}{g["hub_closure_rate"]:>10.3f}'
              f'{np.mean(nt):>10.2f}{np.mean(nd):>8.1f}+-{np.std(nd):<4.1f}')

    # ---------------- B. per-player selectivity: ring member or star hub? ----------------
    print()
    print('=' * 96)
    print('B.  SELECTIVITY: is a busy player anomalous with EVERYONE, or only with a few?')
    print('=' * 96)
    cand = (pct >= 0.9950) & unl & (rk < 0.5)   # .0050 band: 137 pairs, enough to graph
    by_player = {}
    for i in np.where(cand)[0]:
        for p in (P1[i], P2[i]):
            by_player.setdefault(p, []).append(i)
    busy = sorted(by_player, key=lambda p: -len(by_player[p]))[:8]

    # every eligible pair each player appears in
    player_rows = {}
    for i in range(len(fr)):
        player_rows.setdefault(P1[i], []).append(i)
        player_rows.setdefault(P2[i], []).append(i)

    print(f'{"player":<16}{"cand pairs":>11}{"all pairs":>11}{"med pct ALL":>13}'
          f'{"med pct CAND":>14}{"med pct OTHER":>15}{"verdict":>10}')
    sel_rows = []
    for p in busy:
        rows = np.array(player_rows[p])
        cr = np.array(by_player[p])
        other = np.array([i for i in rows if i not in set(cr.tolist())])
        med_all = float(np.median(pct[rows]))
        med_c = float(np.median(pct[cr]))
        med_o = float(np.median(pct[other])) if len(other) else float('nan')
        verdict = 'RING' if med_o < 0.90 else 'STAR'
        sel_rows.append({'player': str(p), 'n_cand': len(cr), 'n_all': len(rows),
                         'median_pct_all': med_all, 'median_pct_cand': med_c,
                         'median_pct_other': med_o, 'verdict': verdict})
        print(f'{str(p):<16}{len(cr):>11}{len(rows):>11}{med_all:>13.4f}'
              f'{med_c:>14.4f}{med_o:>15.4f}{verdict:>10}')

    # the same statistic for players in labelled positive pairs -- the reference
    pos_by_player = {}
    for i in np.where(pos)[0]:
        for p in (P1[i], P2[i]):
            pos_by_player.setdefault(p, []).append(i)
    ref = []
    for p, cr in pos_by_player.items():
        rows = np.array(player_rows[p])
        other = np.array([i for i in rows if i not in set(cr)])
        if len(other):
            ref.append(float(np.median(pct[other])))
    print()
    print(f'  reference: players in LABELLED positive pairs, median anomaly percentile of '
          f'their OTHER pairs = {np.median(ref):.4f} (n={len(ref)} players)')
    print('  a confirmed colluder looks ORDINARY with non-partners: that is the ring signature.')

    # ---------------- C. interpretable signature ----------------
    print()
    print('=' * 96)
    print('C.  WHAT IT LOOKS LIKE: per-feature z against ordinary play')
    print('=' * 96)
    files = sorted(glob.glob(str(PU / 'fold1' / 'batch_*-pairs.parquet')))
    sch = pl.scan_parquet(files[0]).collect_schema().names()
    names = [c for c in sch if c.startswith('ca_') and c.endswith('_mean')]
    base = Z[neg].mean(axis=0)
    sd = Z[neg].std(axis=0) + 1e-9
    prof = {'candidates': (Z[cand].mean(axis=0) - base) / sd}
    for f in FAMS:
        prof[f] = (Z[fam == f].mean(axis=0) - base) / sd
    order = np.argsort(-np.abs(prof['candidates']))[:12]
    print(f'{"feature":<34}{"CAND":>8}{"DT":>8}{"SP":>8}{"CI":>8}')
    for j in order:
        print(f'{names[j].replace("ca_", "").replace("_mean", ""):<34}'
              f'{prof["candidates"][j]:>8.2f}{prof["directed_transfer"][j]:>8.2f}'
              f'{prof["soft_play"][j]:>8.2f}{prof["coordinated_isolation"][j]:>8.2f}')
    sig = {k: {names[j]: float(v[j]) for j in range(len(names))} for k, v in prof.items()}

    # ---------------- D. T2 null calibration ----------------
    print()
    print('=' * 96)
    print("D.  hf4's T2 residual: signal, or the sampling noise of 38 points?")
    print('=' * 96)
    d_off = None
    import json
    hf4 = json.load(open(OUT / 'hf4-fourth-or-unlisted.json'))['0.0050']
    w_hat = hf4['T2']['disclosed_positive_share']
    l1_obs = hf4['T2']['l1_residual']
    n_c = hf4['n_candidates']
    # regenerate the off-manifold statistic exactly as hf4 did, for the null draw
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(Z[neg])
    Pm = lw.get_precision()
    cent_all = {f: Z[pos][fam[pos] == f].mean(axis=0) for f in FAMS}
    D = np.stack([((Z - cent_all[f]) @ Pm * (Z - cent_all[f])).sum(1) for f in FAMS])
    d_off = D.min(axis=0)
    edges = np.quantile(np.concatenate([d_off[pos], d_off[neg]]), np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf

    def hist(a):
        h = np.histogram(a, bins=edges)[0].astype(float)
        return h / h.sum()

    h_pos, h_neg = hist(d_off[pos]), hist(d_off[neg])
    ws = np.linspace(0, 1, 201)
    pool_pos, pool_neg = d_off[pos], d_off[neg]
    null_l1 = []
    for _ in range(400):
        k = RNG.binomial(n_c, w_hat)
        draw = np.concatenate([RNG.choice(pool_pos, k, replace=True),
                               RNG.choice(pool_neg, n_c - k, replace=True)])
        hb = hist(draw)
        null_l1.append(min(np.abs(w * h_pos + (1 - w) * h_neg - hb).sum() for w in ws))
    p_val = float(np.mean(np.array(null_l1) >= l1_obs))
    print(f'  observed L1 residual = {l1_obs:.4f}')
    print(f'  null (38 points drawn from the fitted mixture): '
          f'mean {np.mean(null_l1):.4f}, 95th pct {np.quantile(null_l1, .95):.4f}')
    print(f'  p = {p_val:.3f}  -> ' +
          ('the two-component fit is ADEQUATE; no third population needed'
           if p_val > 0.05 else 'a third component IS required'))

    write_json(OUT / 'hf5-ring-or-star.json',
               {'graph': graph, 'selectivity': sel_rows,
                'labelled_positive_other_pair_median_pct': float(np.median(ref)),
                'signature_z': sig,
                'T2_null': {'l1_observed': l1_obs, 'l1_null_mean': float(np.mean(null_l1)),
                            'l1_null_p95': float(np.quantile(null_l1, .95)), 'p_value': p_val}})
    print()
    print(f'written {OUT / "hf5-ring-or-star.json"}')


if __name__ == '__main__':
    main()
