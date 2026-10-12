"""Step 76 [C] (author: claude).  Where the evidence MAP@5 loss actually sits.

Motivation (from the completed public table, 2026-09-12).  The probe algebra of 14.9 now resolves
every component of the best submission: P = .9603, E = .6241, B = .9756.  Perfect pair ranking is
worth only .7 x (1 - .9603) = +.0278 composite, so the user's +2 % target (+.0179) is at the very
edge of what the pair side can give even in principle and +3 % (+.0268) is beyond it.  Evidence
carries .2 x (1 - .6241) = +.0752 of headroom.  Independently, the leader's .93798 forces
E_leader >= (.93798 - .8)/.2 = .6899 even with PERFECT pairs and PERFECT behaviour -- so the leader's
advantage is necessarily on evidence, and is at least +.066 MAP@5.  Evidence is the lever.

But "evidence" is three different problems and the record has never separated them.  MAP@5 scores an
ORDERED list of 5 against a SET of truth hands, and our chain has two stages: stage 1 retrieves 20
candidates, stage 2 picks and orders 5.  A pair can lose MAP@5 three ways:

  RETRIEVAL   a listed hand is not in stage 1's top-20 at all         (94.71 % are, so this is small)
  SELECTION   it is in the top-20 but stage 2 leaves it out of the 5
  ORDER       it is in the 5 but ranked below an unlisted hand

The three want completely different fixes -- a better ranker, a better classifier, and a cheap
re-sort respectively -- and 18.14 already showed stage 1 and stage 2 are substitutes, so pointing the
next study at the wrong one wastes the run.  This step measures the ladder.

For each pair, with L = our ordered top-5, C = stage 1's top-20, R = the truth set:

  actual        ap5(L, R)
  order-oracle  |L n R| / min(5, |R|)                 -- our five, perfectly ordered
  select-oracle min(|C n R|, 5) / min(5, |R|)         -- the best five out of the top-20, ordered
  ceiling       1.0                                    -- every listed hand retrieved

  actual -> order-oracle   = the ORDER headroom
  order  -> select-oracle  = the SELECTION headroom
  select -> ceiling        = the RETRIEVAL headroom

Two realisable order rules are measured beside the oracle, because an oracle is not a lever:
`chrono` re-sorts our chosen five by time (14.11 rejected "earliest of top-K", which REPLACES the
score order over K candidates and so also changes the set -- re-sorting a set we have already chosen
is a different and much weaker intervention that has not been tested), and `reverse` is the
anti-oracle, included to show the dynamic range that ordering commands at all.

NO PRODUCTION BRANCH.  This is a diagnostic: it reports where the loss is and nothing is proposed for
submission from it.  Any lever it motivates gets its own pre-registered study with its own gate, and
that gate must name the incumbent it protects and check that a known-worse arm fails it (the standing
check from 18.15 / 18.17).

Training-only: the 162 audit pairs in 172 tables outside every held fold.  Held folds are not touched.
Reproduction control: the shipping arm must reproduce step 74's stored hybrid_full_chrono3 = 0.6159.

Output: experiments/label_process_20260911/audit-C-order-vs-set.json
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import check_protected, write_json
from evidence_aug.step5_evaluate import TIE_SEED, ap5, pool_bootstrap
from evidence_aug.step14_audit_truncation import OUT, TOP
from evidence_aug.step65_structural_evidence import log
from evidence_aug.step74_length_shift import (CI, build_d0, designs_of, fit_predict, frame_at)

STORED_HYBRID_FULL_CHRONO3 = 0.6159      # step 74, experiments/.../audit-length-shift.json
AUTHOR = 'claude'


def ranked_lists(d, pred, topk):
    """Reproduce per_pair_map's ordering exactly, but return the lists instead of the score.

    per_pair_map sorts by (score desc, tie asc) after sorting the frame by pair_id, hand_id, with the
    tie column drawn from TIE_SEED over the WHOLE frame.  The draw therefore depends on the frame's
    height and row order, so it is rebuilt here the same way rather than re-drawn per pair.
    """
    cands = (d['oof_rank'] <= topk).to_numpy()
    z = np.where(cands, pred + 10.0, d['oof'].to_numpy() - 1e6)
    key = (d.select('pair_id', 'hand_id', 'table_id')
           .with_columns(pl.Series('score', z), pl.Series('cand', cands))
           .sort('pair_id', 'hand_id'))
    key = key.with_columns(pl.Series('tie', np.random.default_rng(TIE_SEED).random(key.height)))
    top5, top20, chrono5 = {}, {}, {}
    order_col = d.sort('pair_id', 'hand_id').select('pair_id', 'hand_id', 'chrono_rank')
    key = key.join(order_col, on=['pair_id', 'hand_id'], validate='1:1')
    for (pid,), frame in key.group_by('pair_id'):
        srt = frame.sort(['score', 'tie'], descending=[True, False])
        top5[pid] = srt['hand_id'].head(5).to_list()
        top20[pid] = set(srt.filter(pl.col('cand'))['hand_id'].to_list())
        chrono5[pid] = srt.head(5).sort('chrono_rank')['hand_id'].to_list()
    return top5, top20, chrono5


def ladder(top5, top20, chrono5, truths):
    """per-pair: actual / chrono / reverse / order-oracle / select-oracle / ceiling"""
    out = {}
    for pid, L in top5.items():
        R = truths[pid]
        den = min(5, len(R))
        hit5 = len([h for h in L if h in R])
        hit20 = min(len(top20[pid] & R), 5)
        out[pid] = {
            'actual': ap5(L, R),
            'chrono': ap5(chrono5[pid], R),
            'reverse': ap5(list(reversed(L)), R),
            'order_oracle': hit5 / den,
            'select_oracle': hit20 / den,
            'ceiling': 1.0,
            'n_truth': len(R), 'hit5': hit5, 'hit20': hit20}
    return out


def mean_of(rows, k, keys=None):
    return float(np.mean([v[k] for p, v in rows.items() if keys is None or p in keys]))


def main():
    t0 = time.time()
    check_protected()
    d0, struct, rcols = build_d0()
    designs = designs_of(struct, rcols)
    d = frame_at(d0, 1.0)
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())

    print('=' * 100)
    print('STEP 76 [C].  IS THE EVIDENCE LOSS RETRIEVAL, SELECTION, OR ORDER?')
    print('=' * 100)

    # the shipping hybrid: r382 for transfer and soft play, ref for isolation
    parts = {}
    for dname in ('r382_chrono3', 'ref_chrono3'):
        t1 = time.time()
        pred = fit_predict(d, d, designs[dname], TOP)
        parts[dname] = ranked_lists(d, pred, TOP)
        log(f'fit {dname}', json.dumps({'sec': round(time.time() - t1)}))

    a5, a20, ac = parts['r382_chrono3']
    b5, b20, bc = parts['ref_chrono3']
    top5 = {p: (b5[p] if labels[p] == CI else a5[p]) for p in a5}
    top20 = {p: (b20[p] if labels[p] == CI else a20[p]) for p in a5}
    chrono5 = {p: (bc[p] if labels[p] == CI else ac[p]) for p in a5}

    rows = ladder(top5, top20, chrono5, truths)
    got = mean_of(rows, 'actual')
    assert abs(round(got, 4) - STORED_HYBRID_FULL_CHRONO3) < 1e-9, \
        f'reproduction control FAILED: hybrid_full_chrono3 {got:.4f} vs stored {STORED_HYBRID_FULL_CHRONO3}'
    print(f'\n  reproduction control PASSED: hybrid_full_chrono3 = {got:.4f} == step 74 stored value\n')

    res = {'author': AUTHOR, 'step': 76, 'training_only': True, 'held_folds_touched': False,
           'production_branch': 'none: diagnostic only, nothing is proposed for submission from it',
           'reproduction_control': {'hybrid_full_chrono3': round(got, 4),
                                    'stored': STORED_HYBRID_FULL_CHRONO3, 'passed': True},
           'population': {'pairs': len(truths), 'hands': d.height,
                          'listed_hands': int((d['relevance'] == 1).sum())},
           'ladder': {}, 'ladder_by_family': {}, 'headroom': {}, 'paired_ci95': {},
           'truth_size': {}, 'hit_profile': {}}

    STEPS = ['actual', 'chrono', 'reverse', 'order_oracle', 'select_oracle', 'ceiling']
    print(f"  {'rung':<16}{'MAP@5':>9}   {'vs actual':>10}   " + '  '.join(f'{f[:9]:>9}' for f in FAMILIES))
    print('  ' + '-' * 86)
    for k in STEPS:
        v = mean_of(rows, k)
        res['ladder'][k] = round(v, 4)
        fam = {f: round(mean_of(rows, k, {p for p in rows if labels[p] == f}), 4) for f in FAMILIES}
        res['ladder_by_family'][k] = fam
        print(f"  {k:<16}{v:>9.4f}   {v - res['ladder']['actual']:>+10.4f}   "
              + '  '.join(f'{fam[f]:>9.4f}' for f in FAMILIES))

    L = res['ladder']
    res['headroom'] = {
        'ORDER     actual->order_oracle': round(L['order_oracle'] - L['actual'], 4),
        'SELECTION order_oracle->select_oracle': round(L['select_oracle'] - L['order_oracle'], 4),
        'RETRIEVAL select_oracle->ceiling': round(L['ceiling'] - L['select_oracle'], 4),
        'realisable_order_rule_chrono': round(L['chrono'] - L['actual'], 4),
        'order_dynamic_range_reverse_to_oracle': round(L['order_oracle'] - L['reverse'], 4)}
    for k in ('chrono', 'order_oracle', 'select_oracle'):
        res['paired_ci95'][f'{k}_minus_actual'] = [round(x, 4) for x in pool_bootstrap(
            {p: rows[p][k] - rows[p]['actual'] for p in rows}, ptab)]

    ns = [v['n_truth'] for v in rows.values()]
    res['truth_size'] = {'mean': round(float(np.mean(ns)), 2), 'median': float(np.median(ns)),
                         'distribution': {str(k): int(sum(1 for n in ns if n == k)) for k in sorted(set(ns))},
                         'pairs_with_more_than_5': int(sum(1 for n in ns if n > 5))}
    res['hit_profile'] = {
        'mean_relevant_in_our_5': round(float(np.mean([v['hit5'] for v in rows.values()])), 3),
        'mean_relevant_in_top20_capped5': round(float(np.mean([v['hit20'] for v in rows.values()])), 3),
        'pairs_with_0_relevant_in_our_5': int(sum(1 for v in rows.values() if v['hit5'] == 0)),
        'pairs_where_top20_holds_more_than_our_5': int(sum(1 for v in rows.values() if v['hit20'] > v['hit5']))}

    print('\n  HEADROOM DECOMPOSITION (MAP@5, and composite at the .2 evidence weight)')
    for k, v in res['headroom'].items():
        print(f'    {k:<42}{v:>+8.4f}   composite {0.2 * v:>+8.4f}')
    print('\n  paired 95 % CI vs actual (pool bootstrap over tables)')
    for k, v in res['paired_ci95'].items():
        print(f'    {k:<34}[{v[0]:+.4f}, {v[1]:+.4f}]')
    print(f"\n  truth-set sizes: {res['truth_size']['distribution']}"
          f"  (>5 listed: {res['truth_size']['pairs_with_more_than_5']} pairs)")
    print(f"  hit profile: {json.dumps(res['hit_profile'])}")

    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-C-order-vs-set.json', res)
    check_protected()
    print(f"\n  wrote {OUT / 'audit-C-order-vs-set.json'}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
