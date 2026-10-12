"""Step 50b: does a block pair that collapses without a NON-block third player T point at a missed genuine pair (A,T) or (C,T)?
CONTROL class (step 48b): non-dominated block pair P=(A,C) with risk_noT < .1 where T = most co-present third player over
all shared hands. If P is spill-over from a genuine pair that is NOT in the block, the source (A,T) or (C,T) is a missed
positive sitting in the tail. Development: label and risk of the source pairs (labels only to validate); evaluation: risk
distribution of the source pairs (label-free). Baselines: source pairs for NON-collapsing non-dominated block pairs, and
for a random third player. Output: experiments/spillover_20260911/source_pairs.json."""
import os, sys, json
os.environ['POLARS_MAX_THREADS'] = '2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np, polars as pl
from collections import defaultdict
from pu_pairs.build_allpairs_fold import EXP

CV = ORIGINAL / 'Claude Version'
OUT = CV / 'experiments' / 'spillover_20260911'


def source_stats(rows, lookup, name):
    """rows: list of (A, C, T, label_P). lookup: frozenset(players) -> (risk, label)."""
    best_risk, best_label, exists = [], [], 0
    for A, C, T, _ in rows:
        cands = [lookup.get(frozenset((A, T))), lookup.get(frozenset((C, T)))]
        cands = [c for c in cands if c is not None]
        if not cands: continue
        exists += 1
        r, l = max(cands)
        best_risk.append(r); best_label.append(l)
    br = np.array(best_risk); bl = np.array(best_label)
    out = {'n_pairs': len(rows), 'n_with_source_pair_in_list': exists}
    if len(br):
        out.update({'source_risk_median': round(float(np.median(br)), 4), 'source_risk_q75_q90': [round(float(x), 4) for x in np.quantile(br, [.75, .9])],
                    'share_source_ge_.1': round(float((br >= .1).mean()), 3), 'share_source_ge_.3': round(float((br >= .3).mean()), 3), 'share_source_ge_.5': round(float((br >= .5).mean()), 3),
                    'share_source_labelled_positive': round(float((bl == 1).mean()), 3), 'n_source_labelled_positive': int((bl == 1).sum()), 'n_source_labelled_negative': int((bl == 0).sum())})
    return {name: out}


def main():
    res = {}
    # ---------------- development
    dv = pl.read_parquet(OUT / 'dev_rescoring.parquet')
    rng = np.random.default_rng(50)
    for fold in (1, 2):
        s = pl.read_parquet(EXP / f'fold{fold}' / 'scores.parquet').select('pair_id', 'player_1', 'player_2', 'risk', 'label')
        lookup = {frozenset((a, b)): (r, l) for a, b, r, l in s.select('player_1', 'player_2', 'risk', 'label').iter_rows()}
        players = dict(s.select('pair_id', pl.concat_list(['player_1', 'player_2'])).iter_rows())
        d = dv.filter(pl.col('fold') == fold)
        nd = d.filter(~pl.col('dominated'))
        ctrl = nd.filter(pl.col('risk_noT').fill_nan(0) < .1)
        keep = nd.filter(pl.col('risk_noT').fill_nan(0) >= .5)
        def rows_of(df):
            return [(players[p][0], players[p][1], T, l) for p, T, l in df.select('pair_id', 'T', 'label').iter_rows()]
        r = {}
        r.update(source_stats(rows_of(ctrl), lookup, 'CONTROL non-dominated & risk_noT<.1 (T = most co-present third)'))
        r.update(source_stats(rows_of(ctrl.filter(pl.col('label') != 1)), lookup, 'CONTROL, P unlabelled'))
        r.update(source_stats(rows_of(keep), lookup, 'baseline: non-dominated & risk_noT>=.5'))
        # random third player baseline: T = a random player from a random other pair at the same table
        tab_players = defaultdict(set)
        for pid, (a, b) in players.items():
            tab_players[pid[:0]].add(a)  # placeholder (not used)
        allp = list({p for ab in players.values() for p in ab})
        rr = [(A, C, allp[rng.integers(len(allp))], l) for A, C, T, l in rows_of(ctrl)]
        r.update(source_stats(rr, lookup, 'baseline: random third player (same fold pool)'))
        # missed labelled positives: are they sources of a collapsing block pair?
        missed = s.filter((pl.col('label') == 1) & (pl.col('risk') < .5))
        src_sets = set()
        for A, C, T, _ in rows_of(ctrl):
            src_sets.add(frozenset((A, T))); src_sets.add(frozenset((C, T)))
        hit = [m for m in missed.select('player_1', 'player_2', 'risk').iter_rows() if frozenset((m[0], m[1])) in src_sets]
        r['missed labelled positives (risk<.5)'] = {'n': missed.height, 'n_that_are_a_CONTROL_source': len(hit), 'their_risks': [round(float(h[2]), 4) for h in hit]}
        r['control_labelled_positive_P'] = ctrl.filter(pl.col('label') == 1).select('pair_id', 'risk_full', 'risk_noT', 'T').to_dicts()
        res[f'dev fold{fold}'] = r
    # ---------------- evaluation (label-free)
    ev = pl.read_parquet(OUT / 'eval_rescoring.parquet')
    ep = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2')
    sub = pl.read_csv(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv').select('pair_id', 'risk_score')
    s = ep.join(sub, on='pair_id')
    lookup = {frozenset((a, b)): (r, -1) for a, b, r in s.select('player_1', 'player_2', 'risk_score').iter_rows()}
    players = dict(s.select('pair_id', pl.concat_list(['player_1', 'player_2'])).iter_rows())
    nd = ev.filter(~pl.col('dominated'))
    ctrl = nd.filter(pl.col('risk_noT').fill_nan(0) < .1)
    keep = nd.filter(pl.col('risk_noT').fill_nan(0) >= .5)
    def rows_of(df):
        return [(players[p][0], players[p][1], T, -1) for p, T in df.select('pair_id', 'T').iter_rows()]
    r = {}
    r.update(source_stats(rows_of(ctrl), lookup, 'CONTROL non-dominated & risk_noT<.1'))
    r.update(source_stats(rows_of(ctrl.filter(pl.col('risk_v13') < .99)), lookup, 'CONTROL, band < .99'))
    r.update(source_stats(rows_of(keep), lookup, 'baseline: non-dominated & risk_noT>=.5'))
    allp = list({p for ab in players.values() for p in ab})
    rr = [(A, C, allp[rng.integers(len(allp))], l) for A, C, T, l in rows_of(ctrl)]
    r.update(source_stats(rr, lookup, 'baseline: random third player'))
    r['control_by_band'] = ctrl.group_by('band').len().sort('band').to_dicts()
    res['eval'] = r
    write_json(OUT / 'source_pairs.json', res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()