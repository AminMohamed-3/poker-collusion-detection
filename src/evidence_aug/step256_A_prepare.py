"""STEP256 A -- assemble the per-pair inputs for the Minimum-Bayes-Risk evidence decoder study.

Pure post-processing of saved development audit scores. No model is refitted and no evaluation row is read.

Builds, for every one of the 162 audit queries (development positive pairs outside all held folds):
  * every hand of the pair (the STEP235 score table already carries the whole history),
  * the pre-decoder stage-2 probability p of the top-20 rows (v1_REF - 10; -1e6 elsewhere),
  * the production T1 type flag for EVERY hand (production code, step104.type_attributes), not only the top 20,
  * hand time (hands.parquet started_at; phase == 'development' asserted on every hand touched),
  * the truth list from development_evidence.csv (relevance and the FULL list length for the AP denominator),
  * the STEP232 clean-negative event probabilities for every hand (whole-history activity mass).

It then reproduces the production typed decoder per query and checks it returns REF exactly.

Writes: experiments/step256_mbr_decoder_20260917/inputs256.parquet, prepare256.json
Development phase only.  Builds no submission.  Never uploads.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '4'
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
CV = POKER / 'Claude Version'
DATA = POKER / 'data'
sys.path.insert(0, str(CV / 'src'))
sys.dont_write_bytecode = True

OUT = CV / 'experiments' / 'step256_mbr_decoder_20260917'
S235 = CV / 'outputs_step235_20260917'
EVT = CV / 'experiments' / 'step232_event_detector_20260917' / 'hand_oof.parquet'
K = ['pair_id', 'hand_id']
CI = 'coordinated_isolation'
FAMS = ['directed_transfer', 'soft_play', CI]
FAMCOL = {f: 'p_' + f for f in FAMS}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def less_than_five(ps):
    """P(fewer than five of the preceding entries are listed), Poisson-binomial tail; production step104."""
    dist, out = np.zeros(6), np.empty(len(ps))
    dist[0] = 1.
    for i, p in enumerate(ps):
        out[i] = dist[:5].sum()
        new = dist * (1 - p)
        new[1:] += dist[:-1] * p
        new[5] += dist[5] * p
        dist = new
    return out


def ref_decode(p, t1, cr):
    """Production typed decoder on ONE pair's top-20 rows: q_i = p_i * P(<5 higher-priority listed)."""
    order = np.lexsort((cr, ~t1))
    q = np.empty(len(p))
    q[order] = less_than_five(p[order])
    return p * q + 1e-9 * p


def ap5(score, rel, n_listed):
    o = np.argsort(-score, kind='stable')[:5]
    r = rel[o]
    hits = np.cumsum(r)
    return float((hits / np.arange(1, len(o) + 1) * r).sum() / min(5, n_listed))


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    from evidence_aug import step104_C_typed_decoder_build as S104
    from evidence_aug.paths import check_protected
    prot = check_protected()

    a = pl.read_parquet(S235 / 'audit_scores_step235.parquet').select(K + ['v1_REF'])
    q = pl.read_parquet(S235 / 'per_query_step235.parquet').select('pair_id', 'half', 'family', 'REF')
    assert q.height == 162 and a['pair_id'].n_unique() == 162

    # ---- times and PHASE assertion on every hand touched --------------------------------------
    hands = (pl.scan_parquet(DATA / 'hands.parquet').select('hand_id', 'started_at', 'phase')
             .filter(pl.col('hand_id').is_in(a['hand_id'].unique().implode())).collect())
    assert hands.height == a['hand_id'].n_unique(), 'a scored hand is missing from hands.parquet'
    assert hands['phase'].unique().to_list() == ['development'], 'STEP256 touches only development hands'

    # ---- truth lists --------------------------------------------------------------------------
    ev = pl.read_csv(DATA / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
    n_listed = dict(ev.group_by('pair_id').len().iter_rows())

    # ---- production type flag for EVERY hand of every audit pair ------------------------------
    players = {r[0]: (r[1], r[2]) for r in pl.read_csv(DATA / 'development_labels.csv',
                                                       columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    tpath = OUT / 'all-hand-type-attributes.parquet'
    if tpath.exists():
        attrs = pl.read_parquet(tpath)
    else:
        attrs = S104.type_attributes(a.select(K), players)
        attrs.write_parquet(tpath)
    print('type attributes %d rows  %ds' % (attrs.height, round(time.perf_counter() - t0)), flush=True)
    assert attrs.height == a.height

    # ---- event probabilities over the whole history -------------------------------------------
    oof = pl.read_parquet(EVT).select(K + ['label', 'behavior_family', 'fold', 'table_id', 'p_any', 'p_fmax']
                                      + [FAMCOL[f] for f in FAMS])

    d = (a.join(q, on='pair_id', how='inner', validate='m:1')
         .join(attrs.select(K + ['T1']), on=K, how='left', validate='1:1')
         .join(hands.select('hand_id', 'started_at'), on='hand_id', how='left', validate='m:1')
         .join(ev, on=K, how='left', validate='1:1')
         .join(oof, on=K, how='left', validate='1:1', suffix='_ev'))
    assert d.height == a.height
    for c in ['T1', 'started_at', 'p_any', 'p_fmax', 'label', 'fold'] + [FAMCOL[f] for f in FAMS]:
        assert d[c].null_count() == 0, ('null in ' + c, d[c].null_count())
    assert (d['label'] == 1).all(), 'an audit hand is not on a labelled positive pair'
    assert (d['family'] == d['behavior_family']).all(), 'family disagrees between audit and event file'
    # one table and therefore one STEP232 fold per pair (so no pair was scored by a model that saw it)
    chk = d.group_by('pair_id').agg(pl.col('table_id').n_unique().alias('t'), pl.col('fold').n_unique().alias('f'))
    assert chk['t'].max() == 1 and chk['f'].max() == 1

    d = d.with_columns(t=pl.col('started_at').dt.epoch('us')).drop('started_at', 'behavior_family', 'label')
    d = d.with_columns(p_fam=pl.coalesce([pl.when(pl.col('family') == f).then(pl.col(FAMCOL[f])) for f in FAMS]))
    d = d.with_columns(rel=pl.col('evidence_rank').is_not_null().cast(pl.Float64),
                       top20=(pl.col('v1_REF') > -1e5),
                       T1eff=pl.col('T1') & (pl.col('family') != CI))   # production: type flag OFF for CI
    # deterministic whole-history order
    d = d.sort('pair_id', 't', 'hand_id')
    d = d.with_columns(hand_order=pl.int_range(pl.len()).over('pair_id'))

    # ---- REF reproduction ----------------------------------------------------------------------
    rows, ties_all, ties_top = [], 0, 0
    for (pid,), g in d.group_by(['pair_id'], maintain_order=True):
        ties_all += g.height - g['t'].n_unique()
        gt = g.filter(pl.col('top20'))
        ties_top += gt.height - gt['t'].n_unique()
        p = np.clip(gt['v1_REF'].to_numpy() - 10.0, 0, 1)
        t1 = gt['T1eff'].to_numpy()
        cr = np.arange(gt.height, dtype=float)          # gt is already in time order
        sc = ref_decode(p, t1, cr)
        rows.append({'pair_id': pid, 'ref_here': ap5(sc, gt['rel'].to_numpy(), n_listed[pid]),
                     'REF': g['REF'][0], 'family': g['family'][0], 'half': g['half'][0],
                     'n_top20': gt.height, 'n_hands': g.height, 'n_listed_total': n_listed[pid],
                     'n_listed_in_top20': int(gt['rel'].sum()), 'n_T1_top20': int(t1.sum()),
                     'p_sum_top20': float(p.sum()),
                     'evt_sum_outside': float(g.filter(~pl.col('top20'))['p_fam'].sum()),
                     'evt_sum_top20': float(gt['p_fam'].sum())})
    P = pl.DataFrame(rows)
    repro = float((P['ref_here'] - P['REF']).abs().max())
    print('REF reproduction max abs per-query diff', repro, flush=True)
    print('REF mean here %.6f  recorded %.6f' % (P['ref_here'].mean(), P['REF'].mean()), flush=True)
    assert repro == 0.0, 'the production decoder was not reproduced'

    # calibration of the raw event probability against the stage-2 probability on top-20 rows
    tt = d.filter(pl.col('top20'))
    ps = np.clip(tt['v1_REF'].to_numpy() - 10.0, 0, 1)
    pe = tt['p_fam'].to_numpy()
    res = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'inputs': {'audit_scores_step235.parquet': sha(S235 / 'audit_scores_step235.parquet'),
                      'per_query_step235.parquet': sha(S235 / 'per_query_step235.parquet'),
                      'hand_oof.parquet': sha(EVT),
                      'development_evidence.csv': sha(DATA / 'development_evidence.csv')},
           'phase_asserted': 'development', 'rows': d.height, 'queries': P.height,
           'REF_reproduction_max_abs_per_query': repro,
           'REF_mean_here': round(float(P['ref_here'].mean()), 6),
           'REF_mean_recorded': round(float(P['REF'].mean()), 6),
           'REF_by_family': {f: round(float(P.filter(pl.col('family') == f)['ref_here'].mean()), 4) for f in FAMS},
           'REF_by_half': [round(float(P.filter(pl.col('half') == h)['ref_here'].mean()), 6) for h in (0, 1)],
           'started_at_ties_within_pair_all_hands': ties_all, 'started_at_ties_within_top20': ties_top,
           'T1_rows_top20': int(tt['T1eff'].sum()), 'T1_rows_all': int(d['T1eff'].sum()),
           'list_len': {'mean': round(float(P['n_listed_total'].mean()), 3),
                        'short_lists_lt5': int((P['n_listed_total'] < 5).sum()),
                        'listed_inside_top20_mean': round(float(P['n_listed_in_top20'].mean()), 3)},
           'p_top20': {'mean': round(float(ps.mean()), 5), 'sum_per_pair_mean': round(float(P['p_sum_top20'].mean()), 3),
                       'q10': round(float(np.quantile(ps, .1)), 5), 'q90': round(float(np.quantile(ps, .9)), 5)},
           'event_p_fam': {'top20_mean': round(float(pe.mean()), 5),
                           'outside_mean': round(float(d.filter(~pl.col('top20'))['p_fam'].mean()), 5),
                           'outside_sum_per_pair_mean': round(float(P['evt_sum_outside'].mean()), 3),
                           'top20_sum_per_pair_mean': round(float(P['evt_sum_top20'].mean()), 3),
                           'spearman_vs_stage2_top20': round(float(np.corrcoef(
                               np.argsort(np.argsort(ps)), np.argsort(np.argsort(pe)))[0, 1]), 4)},
           'seconds': round(time.perf_counter() - t0, 1)}
    d.select(K + ['family', 'half', 'v1_REF', 'T1', 'T1eff', 'top20', 'rel', 'evidence_rank', 't',
                  'hand_order', 'p_fam', 'p_any', 'p_fmax', 'table_id']).write_parquet(OUT / 'inputs256.parquet')
    P.write_parquet(OUT / 'ref_per_query256.parquet')
    res['inputs256_sha256'] = sha(OUT / 'inputs256.parquet')
    assert check_protected() == prot
    (OUT / 'prepare256.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
