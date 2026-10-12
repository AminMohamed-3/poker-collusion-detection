"""STEP281 stage C: the two-population read, and the discovery on evaluation.

Both nulls built for stage B failed their matched placebo, so no parametric tail is used here.
DEVELOPMENT ITSELF is the null -- a real population from the same generator, same tables, same
players, same session structure, same host decoys, and labels that contain only the three
disclosed families.  Every pair, development or evaluation, is scored against the SAME reference
(development, labelled positives removed, stratified by joint-hand count), so the two flag rates
are directly comparable and any mis-specification cancels.

The size of the error bar is measured, not assumed: splitting the 400 development tables in half
and running the identical comparison gives the spread of the excess when the two populations ARE
the same.  The evaluation excess is read against that spread.

Verified before use: `n_shared` here equals the competition's `shared_hands` for all 112,540
pairs exactly, and the evaluation list is a uniform ~88 % sample of the pairs with 38 or more
shared hands (.883 / .880 / .877 / .873 across length bins), so length reweighting is the only
adjustment the comparison needs.
"""
import json
import os

os.environ['POLARS_MAX_THREADS'] = '6'
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Claude Version' / 'experiments' / 'step281_victim_instrument_20260919'
PROD = (ROOT / 'Claude Version' / 'outputs_step272_20260919'
        / 'submission_step272_blockM3xKM_on_step259b.csv')

CELLS = [('v_fold_excess', +1), ('v_foldany_excess', +1), ('v_bleed_excess', +1),
         ('v_take_excess', +1), ('v_pressure_excess', +1), ('v_squeeze', +1),
         ('v_bothaggr', +1), ('v_conc', +1), ('v_victim_take', +1),
         ('v_jointnet_z', +1), ('v_jointnet_z', -1), ('v_conflict', -1)]
PLACEBO = [(n, -s) for n, s in CELLS if n != 'v_jointnet_z']
# cells that carry a disclosed family (AUC >= .60 on at least one, stage B): the calibrated
# subset.  Track A below uses ALL cells and knows nothing about the families.
SELECTED = [('v_pressure_excess', +1), ('v_squeeze', +1), ('v_bothaggr', +1),
            ('v_jointnet_z', -1), ('v_conflict', -1), ('v_fold_excess', +1),
            ('v_bleed_excess', +1), ('v_victim_take', +1), ('v_jointnet_z', +1)]
N_STRATA = 10
FAMILIES = ['directed_transfer', 'soft_play', 'coordinated_isolation']


def strat(x, edges):
    return np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)


class Scorer:
    """One-sided chi-square over cells, each read as a percentile inside the pair's n_B stratum.

    score = sum over cells of max(z, 0)^2, z = normal quantile of the cell's percentile in the
    reference.  Continuous, so it has no ties at the small flag budgets where a max-of-cells
    statistic runs out of resolution.
    """

    def __init__(self, ref: pl.DataFrame, cells):
        self.cells = cells
        nb = ref['n_B'].to_numpy()
        e = np.unique(np.quantile(nb, np.linspace(0, 1, N_STRATA + 1)))
        e[-1] = np.inf
        self.edges = e
        s = strat(nb, e)
        self.ref = {(n, g): [np.sort(ref[n].to_numpy()[s == k] * g)
                             for k in range(len(e) - 1)] for n, g in cells}

    def __call__(self, df: pl.DataFrame):
        s = strat(df['n_B'].to_numpy(), self.edges)
        tot = np.zeros(df.height)
        for (name, sign) in self.cells:
            v = df[name].to_numpy() * sign
            z = np.zeros(df.height)
            for k, ref in enumerate(self.ref[(name, sign)]):
                m = s == k
                if not m.any() or len(ref) < 50:
                    continue
                p = np.searchsorted(ref, v[m], side='left') / len(ref)
                z[m] = ndtri(np.clip(p, 0.5, 1 - 0.5 / len(ref)))
            tot += np.maximum(z, 0.0) ** 2
        return tot


def excess_statistic(keys: pl.DataFrame, score):
    """OBS 15.3's decoy control: is the PAIR odd, or are its two members odd with everyone?"""
    pct = (np.argsort(np.argsort(score)) + 0.5) / len(score)
    d = keys.with_columns(pl.Series('pct', pct))
    long = pl.concat([d.select(pl.col('p1').alias('p'), 'pct'),
                      d.select(pl.col('p2').alias('p'), 'pct')])
    med = long.group_by('p').agg(pl.col('pct').median().alias('m'))
    d = (d.join(med.rename({'p': 'p1', 'm': 'm1'}), on='p1', how='left')
          .join(med.rename({'p': 'p2', 'm': 'm2'}), on='p2', how='left'))
    return pct - np.maximum(d['m1'].to_numpy(), d['m2'].to_numpy())


def matched_rate(ref_s, ref_sh, w, thr, n_bins):
    """Reference flag rate reweighted to the target population's history-length mix."""
    f = ref_s >= thr
    return float(np.sum([w[k] * (f & (ref_sh == k)).sum() / len(ref_s)
                         for k in range(n_bins)]))


def run():
    dev = pl.read_parquet(OUT / 'victim_development_trunc2000.parquet') \
            .filter(pl.col('n_shared') >= 38)
    ev = pl.read_parquet(OUT / 'victim_evaluation.parquet')
    pairs = pl.read_csv(ROOT / 'data' / 'evaluation_pairs.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p1'),
        pl.max_horizontal('player_1', 'player_2').alias('p2'))
    ev = ev.join(pairs.select('pair_id', 'p1', 'p2'), on=['p1', 'p2'], how='inner')

    lab = pl.read_csv(ROOT / 'data' / 'development_labels.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p1'),
        pl.max_horizontal('player_1', 'player_2').alias('p2'))
    fam = dev.join(lab.select('p1', 'p2', 'behavior_family'), on=['p1', 'p2'], how='left')[
        'behavior_family'].fill_null('unlabelled').to_numpy()
    # Reference choice matters and both are reported.
    #   FULL    : every eligible development pair, disclosed positives INCLUDED.  This is the
    #             like-for-like population -- evaluation also carries its own disclosed
    #             positives, unlabelled -- so an excess here is something development does NOT
    #             have, which is the fourth-family question.
    #   CLEANED : labelled positives removed.  An excess here is the disclosed families of the
    #             evaluation phase PLUS anything else, so it answers "how much of the test data
    #             does not fit clean play" rather than "is there a fourth mechanism".
    refs = {'FULL': dev, 'CLEANED': dev.filter(pl.Series(~np.isin(fam, FAMILIES)))}
    print(f'reference FULL {refs["FULL"].height} / CLEANED {refs["CLEANED"].height}; '
          f'evaluation {ev.height}')

    results = {}
    for (track, cells), (rname, ref) in [((t, c), (rn, rf)) for t, c in
                                         (('A_all_cells', CELLS), ('B_selected', SELECTED))
                                         for rn, rf in refs.items()]:
        track = f'{track}|{rname}'
        sc, pls = Scorer(ref, cells), Scorer(ref, PLACEBO)
        dev_s, ev_s, ref_s = sc(dev), sc(ev), sc(ref)
        ev_p, ref_p = pls(ev), pls(ref)

        neg = dev_s[fam == 'none']
        power = {}
        for f in FAMILIES:
            pos = dev_s[fam == f]
            r = np.concatenate([pos, neg]).argsort().argsort()
            power[f'auc_{f}'] = round(float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                                            / (len(pos) * len(neg))), 3)
        t = np.quantile(neg, 0.99)
        rec = {f: round(float((dev_s[fam == f] >= t).mean()), 3) for f in FAMILIES}
        power['recall_at_1pct_fpr'] = {**rec, 'macro': round(float(np.mean(list(rec.values()))), 3)}

        sh_e = np.unique(np.quantile(ev['n_shared'].to_numpy(), np.linspace(0, 1, 11)))
        sh_e[0], sh_e[-1] = -np.inf, np.inf
        nbin = len(sh_e) - 1
        ref_sh, ev_sh = strat(ref['n_shared'].to_numpy(), sh_e), strat(ev['n_shared'].to_numpy(), sh_e)
        w = np.array([(ev_sh == k).mean() / max((ref_sh == k).mean(), 1e-9) for k in range(nbin)])

        # --- how big is an excess between two samples of the SAME population? ---------
        rng = np.random.default_rng(11)
        tabs = ref['table_id'].to_numpy()
        utab = np.unique(tabs)
        rows = []
        for budget in (100, 300, 1000, 3000):
            thr = np.quantile(ref_s, 1 - budget / 112_540)
            exp = matched_rate(ref_s, ref_sh, w, thr, nbin) * 112_540
            obs = int((ev_s >= thr).sum())
            thr_p = np.quantile(ref_p, 1 - budget / 112_540)
            null = []
            for _ in range(200):
                half = set(rng.choice(utab, len(utab) // 2, replace=False).tolist())
                m = np.fromiter((t in half for t in tabs), bool, len(tabs))
                ta, tb = ~m, m
                thr_a = np.quantile(ref_s[ta], 1 - budget / 112_540)
                wa = np.array([(ref_sh[tb] == k).mean() / max((ref_sh[ta] == k).mean(), 1e-9)
                               for k in range(nbin)])
                e_a = matched_rate(ref_s[ta], ref_sh[ta], wa, thr_a, nbin) * 112_540
                o_b = (ref_s[tb] >= thr_a).sum() / tb.sum() * 112_540
                null.append(o_b - e_a)
            null = np.array(null)
            rows.append({'budget': budget, 'eval_flagged': obs,
                         'dev_expected_length_matched': round(exp, 1),
                         'excess': round(obs - exp, 1),
                         'same_population_excess_sd': round(float(null.std()), 1),
                         'z': round(float((obs - exp - null.mean()) / max(null.std(), 1e-9)), 2),
                         'placebo_excess': round(float((ev_p >= thr_p).sum() - budget), 1)})
            print(track, rows[-1], flush=True)
        results[track] = {'power': power, 'two_population': rows}
        if track == 'A_all_cells|FULL':
            keep_s, keep_dev = ev_s, dev_s
    print('\nPOWER:', json.dumps({k: v['power'] for k, v in results.items()}, indent=1))

    # --- where the evaluation flags sit in the production ranking -----------------------
    prod = pl.read_csv(PROD).select('pair_id', 'risk_score')
    ev2 = ev.select('pair_id', 'p1', 'p2', 'n_shared', 'n_B') \
            .with_columns(pl.Series('vscore', keep_s)).join(prod, on='pair_id', how='left')
    r = ev2['risk_score'].to_numpy()
    ev2 = ev2.with_columns(pl.Series('risk_pct', (np.argsort(np.argsort(r)) + 0.5) / len(r)),
                           pl.Series('decoy_excess',
                                     excess_statistic(ev2.select('p1', 'p2'), keep_s)))
    dev_ex = excess_statistic(dev.select('p1', 'p2'), keep_dev)
    q25 = float(np.quantile(dev_ex[np.isin(fam, FAMILIES)], 0.25))
    print(f'\ndecoy gate: 25th pct of labelled positives\' excess = {q25:.3f}; '
          f'confirmed non-targets median {np.median(dev_ex[fam == "none"]):.3f}')

    sc = Scorer(refs['FULL'], CELLS)
    ref_s = sc(refs['FULL'])
    print('\ndecoy excess by group (higher = the PAIR is odd, not its members):')
    for g, m in (('labelled positives', np.isin(fam, FAMILIES)),
                 ('confirmed non-targets', fam == 'none'),
                 ('all development', np.ones(len(fam), bool))):
        print(f'  {g:24s} median {np.median(dev_ex[m]):+.3f}  q25 {np.quantile(dev_ex[m], .25):+.3f}')

    bands = []
    for budget in (300, 1000, 3000):
        thr = np.quantile(ref_s, 1 - budget / 112_540)
        f = ev2.filter(pl.col('vscore') >= thr)
        mid = (f['risk_pct'] >= .90) & (f['risk_pct'] < .995) & (f['risk_score'] < .5)
        # how many of development's OWN labelled positives clear the same bar?  That is what
        # evaluation's (unlabelled) disclosed positives are expected to contribute.
        devflag = keep_dev >= thr
        bands.append({'budget': budget, 'flagged': f.height,
                      'block_risk_ge_.5': int((f['risk_score'] >= .5).sum()),
                      'midband': int(mid.sum()),
                      'low_risk_pct_lt_.90': int((f['risk_pct'] < .90).sum()),
                      'dev_labelled_positives_flagged':
                          {g: int((devflag & (fam == g)).sum()) for g in FAMILIES},
                      'dev_positive_recall_here':
                          round(float(devflag[np.isin(fam, FAMILIES)].mean()), 3),
                      'decoy_pass': int((f['decoy_excess'] >= q25).sum()),
                      'midband_and_decoy': int((mid & (f['decoy_excess'] >= q25)).sum())})
        print(bands[-1])

    res = {'tracks': results, 'evaluation_bands': bands, 'decoy_q25_positives': q25,
           'n_reference': ref.height, 'n_eval': ev.height}
    (OUT / 'discover281.json').write_text(json.dumps(res, indent=2))
    ev2.sort('vscore', descending=True).head(5000).write_parquet(OUT / 'eval_top_victim.parquet')
    np.save(OUT / 'eval_vscore.npy', keep_s)
    np.save(OUT / 'dev_vscore.npy', keep_dev)
    print('\nwrote discover281.json')


if __name__ == '__main__':
    run()
