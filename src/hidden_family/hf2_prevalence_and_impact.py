"""HF2: how much of the evaluation truth can the hidden family be, and what does it cost us?

Two independent estimators, both anchored on measured numbers only.

(A) EVIDENCE ALGEBRA.  Evidence MAP@5 is averaged over ALL truth positives, hidden family
    included (host metric, OBS 11.2).  So
        E_public = (1 - pi) * E_disclosed + pi * E_hidden,   E_hidden = rho * E_disclosed
    with rho measured by the LOFO omit runs (unseen-family MAP@5 / trained MAP@5).
    Public E is known exactly from the probe uploads (.608).  E_disclosed is the local
    development MAP@5 of the same production ranker.  Solve for pi.

(B) PAIR-AP SIMULATION.  Rank 112,540 pairs.  Disclosed positives draw their risk from the
    empirical distribution of labelled positives under the trained model; hidden positives
    draw from the empirical distribution of held positives whose family was REMOVED from
    training (the omit runs, shortened history = evaluation-like); negatives draw from the
    empirical evaluation risk distribution of our own submission.  Sweep pi, compute the
    host's stable Pair AP, and report which pi reproduces the measured public P = .938.
    This is an UPPER bound on pi: the pair deficit has other documented causes
    (length-induced false positives, short-history recall, OBS 14.23-14.35).

Read-only on the original tree.  Writes to experiments/hidden_family_20260912/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import polars as pl

from evidence_aug.paths import ORIGINAL, write_json

OUT = Path(__file__).resolve().parents[2] / 'experiments' / 'hidden_family_20260912'
CONF = ORIGINAL / 'reports' / 'conditional-action-confirmation' / 'results'
FAMILIES = ['directed_transfer', 'soft_play', 'coordinated_isolation']

# measured public components (probe uploads, OBS 14.9 / memory)
PUBLIC = {'P': 0.938, 'B': 0.976, 'E_chrono': 0.608, 'E_v13': 0.529}
WEIGHTS = {'P': 0.70, 'E': 0.20, 'B': 0.10}


def stable_ap(scores, truth, pair_ids):
    order = np.argsort(pair_ids, kind='mergesort')
    s, y = np.asarray(scores)[order], np.asarray(truth)[order]
    idx = np.argsort(-s, kind='mergesort')
    y = y[idx]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def evidence_algebra():
    trained, unseen = {}, {}
    for f in ['fold1', 'fold2']:
        d = json.load(open(CONF / f / 'metrics.json'))
        for r in d['evidence_results']:
            if r['candidate'] == 'conditional':
                for k, v in r['by_family'].items():
                    trained.setdefault(k, []).append(v)
    for fam in FAMILIES:
        d = json.load(open(CONF / f'omit_{fam}' / 'metrics.json'))
        for r in d['evidence_results']:
            if r['candidate'] == 'conditional':
                unseen[fam] = r['map5']
    rho = {f: unseen[f] / float(np.mean(trained[f])) for f in FAMILIES}
    rho_pooled = float(np.mean(list(unseen.values())) / np.mean([np.mean(v) for v in trained.values()]))

    print('=' * 96)
    print('(A) EVIDENCE: what an UNSEEN family costs the evidence ranker')
    print('=' * 96)
    print(f'{"family":<24}{"trained MAP@5":>15}{"unseen MAP@5":>15}{"retention rho":>15}')
    for f in FAMILIES:
        print(f'{f:<24}{np.mean(trained[f]):>15.4f}{unseen[f]:>15.4f}{rho[f]:>15.3f}')
    print(f'{"pooled":<24}{np.mean([np.mean(v) for v in trained.values()]):>15.4f}'
          f'{np.mean(list(unseen.values())):>15.4f}{rho_pooled:>15.3f}')

    # solve pi for a grid of assumptions about E_disclosed on evaluation
    rows = []
    print(f'\npi implied by  E_public = (1-pi)*E_disc + pi*rho*E_disc,  E_public = {PUBLIC["E_chrono"]:.3f}')
    print(f'{"E_disc(eval)":>14}' + ''.join(f'{f"rho={r:.2f}":>12}' for r in [0.4, 0.57, 0.8, 1.0]))
    for e_disc in [0.60, 0.62, 0.65, 0.68, 0.70]:
        cells = []
        for r in [0.4, 0.57, 0.8, 1.0]:
            denom = e_disc * (1 - r)
            pi = (e_disc - PUBLIC['E_chrono']) / denom if denom > 1e-9 else float('nan')
            cells.append(pi)
            rows.append({'E_disc': e_disc, 'rho': r, 'pi': pi})
        print(f'{e_disc:>14.2f}' + ''.join(f'{c:>12.3f}' if np.isfinite(c) else f'{"n/a":>12}' for c in cells))
    print('\nNegative pi means the public evidence score is already at or above the local')
    print('disclosed-family level, i.e. no room for a badly-served hidden population.')
    return {'trained_by_family': {f: float(np.mean(v)) for f, v in trained.items()},
            'unseen_by_family': unseen, 'rho_by_family': rho, 'rho_pooled': rho_pooled,
            'pi_grid': rows}


def pair_simulation(rng):
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    screen = pl.read_csv(ORIGINAL / 'reports' / 'conditional-action-experiment' /
                         'screen-results' / 'pair-predictions.csv')
    screen = screen.filter((pl.col('candidate') == 'conditional') & (pl.col('history') == 'shortened'))
    risk = dict(zip(screen['pair_id'].to_list(), screen['risk'].to_list()))
    pos_ids = labels.filter(pl.col('label') == 1)['pair_id'].to_list()
    disclosed_pool = np.array([risk[p] for p in pos_ids if p in risk])

    hidden_pool = []
    for fam in FAMILIES:
        d = pl.read_csv(CONF / f'omit_{fam}' / 'pair-predictions.csv').filter(
            (pl.col('candidate') == 'conditional') & (pl.col('history') == 'shortened'))
        fam_ids = set(labels.filter(pl.col('behavior_family') == fam)['pair_id'])
        hidden_pool += d.filter(pl.col('pair_id').is_in(list(fam_ids)))['risk'].to_list()
    hidden_pool = np.array(hidden_pool)

    sub = pl.read_csv(ORIGINAL / 'Claude Version' / 'outputs_multiview_20260911' /
                      'submission_M4_stage2_chrono3.csv', columns=['risk_score'])
    neg_pool = sub['risk_score'].to_numpy()
    neg_pool = neg_pool[neg_pool < 0.5]           # the tail: overwhelmingly negatives

    print('\n' + '=' * 96)
    print('(B) PAIR AP SIMULATION at evaluation scale')
    print('=' * 96)
    print(f'disclosed-positive risk pool  n={len(disclosed_pool):>5}  median={np.median(disclosed_pool):.4f}'
          f'  share>=.5={np.mean(disclosed_pool >= .5):.3f}')
    print(f'hidden-positive risk pool     n={len(hidden_pool):>5}  median={np.median(hidden_pool):.4f}'
          f'  share>=.5={np.mean(hidden_pool >= .5):.3f}')
    print(f'negative risk pool            n={len(neg_pool):>6} median={np.median(neg_pool):.2e}')

    N = 112540
    results = []
    print(f'\n{"n_pos":>7}{"pi":>7}{"n_hidden":>10}{"PairAP":>10}{"vs P=.938":>12}')
    for n_pos in [850, 930, 1010]:
        for pi in [0.0, 0.02, 0.04, 0.06, 0.10, 0.15, 0.25]:
            aps = []
            for _ in range(12):
                n_h = int(round(n_pos * pi))
                n_d = n_pos - n_h
                s = np.concatenate([rng.choice(disclosed_pool, n_d, replace=True),
                                    rng.choice(hidden_pool, n_h, replace=True),
                                    rng.choice(neg_pool, N - n_pos, replace=True)])
                y = np.concatenate([np.ones(n_pos), np.zeros(N - n_pos)])
                ids = np.arange(N)
                rng.shuffle(ids)
                aps.append(stable_ap(s, y, ids))
            ap = float(np.mean(aps))
            results.append({'n_pos': n_pos, 'pi': pi, 'pair_ap': ap})
            print(f'{n_pos:>7}{pi:>7.2f}{int(round(n_pos*pi)):>10}{ap:>10.4f}{ap - PUBLIC["P"]:>+12.4f}')
    return {'disclosed_pool_n': len(disclosed_pool), 'hidden_pool_n': len(hidden_pool),
            'hidden_share_ge_50': float(np.mean(hidden_pool >= .5)),
            'disclosed_share_ge_50': float(np.mean(disclosed_pool >= .5)),
            'simulation': results}


def cost_table(rho_pooled):
    print('\n' + '=' * 96)
    print('(C) WHAT THE HIDDEN FAMILY COSTS US, per unit of prevalence')
    print('=' * 96)
    print('Assumes E_disclosed=.62, pair AP on hidden ~.70 vs ~.99 disclosed (LOFO, short history),')
    print('and that behaviour is unaffected (see hf3: other_coordination is metrically inert).')
    print(f'\n{"pi":>6}{"n_hidden":>10}{"dP":>9}{"dE":>9}{"composite cost":>17}{"recoverable":>13}')
    rows = []
    for pi in [0.02, 0.04, 0.06, 0.10, 0.15, 0.25]:
        d_p = pi * (0.99 - 0.70)
        d_e = pi * 0.62 * (1 - rho_pooled)
        cost = WEIGHTS['P'] * d_p + WEIGHTS['E'] * d_e
        rows.append({'pi': pi, 'delta_P': d_p, 'delta_E': d_e, 'composite_cost': cost})
        print(f'{pi:>6.2f}{int(round(930*pi)):>10}{d_p:>9.4f}{d_e:>9.4f}{cost:>17.4f}{cost:>13.4f}')
    return rows


def main():
    rng = np.random.default_rng(20260912)
    ev = evidence_algebra()
    pair = pair_simulation(rng)
    cost = cost_table(ev['rho_pooled'])
    write_json(OUT / 'hf2-prevalence-and-impact.json',
               {'public': PUBLIC, 'evidence': ev, 'pair': pair, 'cost': cost})
    print(f'\nwritten {OUT / "hf2-prevalence-and-impact.json"}')


if __name__ == '__main__':
    main()
