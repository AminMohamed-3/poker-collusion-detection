"""STEP267 amendment_2 - QMATCH rescale of a reranker score file.

registration267.json amendment_2: general-purpose reranker score files carry raw P(yes) that is
mostly < .01, which the registered mapping p = clip(mean_conf/100, .01, .99) (built for 0-100 LLM
confidences) flattens to a near-constant.  This script produces a SAME-FORMAT score file whose
mean_conf is a zero-fit, label-free QUANTILE MATCH of the reranker's own logit onto the MECHPLUS
own-family probability distribution over the same held-out uids ("fuse in rank form", OBS 19.213):

  1. s = a single score per row, taken from the row's own logit/p/mean_conf fields (never fit).
  2. u = the average-rank quantile of s within the score file (ties broken by uid, stable).
  3. p_qmatch = the u-quantile of the MECHPLUS own-family p (mechplus_p_DT/_SP/_CI/_fmax by the
     hand's own family, joined uid -> (pair_id, hand_id) -> hand_oof_step254.parquet) over the
     SAME rows (numpy quantile, linear interpolation), clipped to [.001, .999].

No label is read anywhere in this script (keys_heldout.parquet's label column is never selected);
the optional --report Spearman is computed between s and the MECHPLUS own-family p, which is also
label-free.  M4 (reranker alone through the production decoder) is rank-only, so it is identical
whether it is fed this file or the raw file - only the fused arms (M2/M3) can move.

    python step267_rescale_scores.py --scores in.jsonl --out out.jsonl
                                     [--primary agnostic|max3] [--report report.json]

Development-phase data only.  Reads keys_heldout.parquet and hand_oof_step254.parquet (never the
label column); writes only the two files named on the command line.
"""
from __future__ import annotations

import os
import sys

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
CV = POKER / 'Claude Version'
KEYS = CV / 'experiments/step264_llm_hand_judge_20260918/keys_heldout.parquet'
OOF = CV / 'experiments/step254_mechanical_tensor_20260917/hand_oof_step254.parquet'
REG267 = CV / 'experiments/step267_text_rerankers_20260918/registration267.json'

AGNOSTIC = 'agnostic'
FAM_KEYS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
FAM_COL = {'directed_transfer': 'mechplus_p_DT', 'soft_play': 'mechplus_p_SP',
           'coordinated_isolation': 'mechplus_p_CI'}
EPS = 1e-6
QMATCH_LO, QMATCH_HI = .001, .999
QLEVELS = [0.0, .05, .25, .5, .75, .95, 1.0]


# --------------------------------------------------------------------------- score-file reading
def read_rows(path):
    rows = []
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    uids = [r['uid'] for r in rows]
    assert len(set(uids)) == len(uids), 'duplicate uid in the score file: ' + path
    return rows, uids


def _logit(p):
    """logit of p, clipped to [EPS, 1-EPS] for numerical safety (registration's '1e-6' clip)."""
    pc = min(max(float(p), EPS), 1.0 - EPS)
    return float(np.log(pc / (1.0 - pc)))


def query_score(row, key):
    """(s, p_raw) for one named query of one row: logit[key] if present, else logit(p[key])
    clipped at 1e-6, else logit(mean_conf/100).  p_raw is the row's own probability for that same
    branch (derived from s by the inverse sigmoid only when no p/mean_conf value exists at all)."""
    logit_d = row.get('logit') or {}
    if logit_d.get(key) is not None:
        s = float(logit_d[key])
        p_d = row.get('p') or {}
        p_raw = float(p_d[key]) if p_d.get(key) is not None else 1.0 / (1.0 + np.exp(-s))
        return s, p_raw
    p_d = row.get('p') or {}
    if p_d.get(key) is not None:
        p_raw = float(p_d[key])
        return _logit(p_raw), p_raw
    mc = row.get('mean_conf')
    p_raw = float(mc) / 100.0 if mc is not None else .5
    return _logit(p_raw), p_raw


def primary_score(row, primary):
    """s and p_raw for the row's PRIMARY query.  --primary max3 = max of the three family logits
    (ties among the three broken by the fixed FAM_KEYS order); p_raw_primary then follows
    whichever family logit was the max, so it is the p that s was actually derived from."""
    if primary == AGNOSTIC:
        return query_score(row, AGNOSTIC)
    triples = [query_score(row, k) for k in FAM_KEYS]
    idx = int(np.argmax([t[0] for t in triples]))
    return triples[idx]


# --------------------------------------------------------------------------- ranks and quantiles
def average_rank_stable(values, tie_keys):
    """1-based average rank over `values`; ties resolved by a stable sort on `tie_keys` (uid)
    before averaging, per registration267 amendment_2 ('ties broken by uid order (stable)').
    Because tied elements all receive the SAME averaged rank, the tie-break choice does not
    change any output value - it only fixes a deterministic, reproducible sort order."""
    n = len(values)
    order = sorted(range(n), key=lambda i: (values[i], tie_keys[i]))
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    pos = 1
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (pos + (pos + (j - i))) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        pos += (j - i + 1)
        i = j + 1
    return ranks


def quantiles_of(a, levels=QLEVELS):
    a = np.asarray(a, dtype=np.float64)
    return {('q%g' % lv): round(float(np.quantile(a, lv)), 6) for lv in levels}


def spearman_label_free(s, own_p):
    """Spearman(s, MECHPLUS own-family p): both arguments are model outputs, never a label."""
    from scipy.stats import spearmanr
    return float(spearmanr(s, own_p).statistic)


# --------------------------------------------------------------------------- the MECHPLUS join
def mechplus_own_family(uids):
    """uid -> (pair_id, hand_id) via keys_heldout.parquet -> own-family p via hand_oof_step254.
    parquet.  The label column of keys_heldout is never selected.  Returns (own_p, family,
    n_matched_keys, n_matched_oof, keys_family) all aligned to `uids` order."""
    keys = pl.read_parquet(KEYS, columns=['uid', 'pair_id', 'hand_id', 'family'])
    assert 'label' not in keys.columns
    assert keys.select('uid').is_duplicated().sum() == 0, 'keys_heldout.parquet has duplicate uids'

    oof = pl.read_parquet(OOF, columns=['pair_id', 'hand_id', 'family', 'mechplus_p_DT',
                                        'mechplus_p_SP', 'mechplus_p_CI', 'mechplus_p_fmax'])
    assert oof.select(['pair_id', 'hand_id']).is_duplicated().sum() == 0, \
        'hand_oof_step254.parquet has duplicate (pair_id, hand_id)'

    scope = pl.DataFrame({'uid': uids, '_ord': list(range(len(uids)))})
    j1 = scope.join(keys, on='uid', how='left', validate='m:1', maintain_order='left')
    n_missing_keys = int(j1['pair_id'].null_count())
    assert n_missing_keys == 0, (
        '%d of %d uids in the score file are not in keys_heldout.parquet (first few: %s)'
        % (n_missing_keys, len(uids),
           j1.filter(pl.col('pair_id').is_null())['uid'].to_list()[:5]))

    j2 = j1.join(oof, on=['pair_id', 'hand_id'], how='left', validate='m:1', maintain_order='left',
                suffix='_oof')
    assert j2.height == len(uids), 'join changed row count - a key is not unique as assumed'
    n_missing_oof = int(j2['family_oof'].null_count())
    assert n_missing_oof == 0, (
        '%d of %d uids join to keys_heldout but not to hand_oof_step254.parquet (first few: %s)'
        % (n_missing_oof, len(uids),
           j2.filter(pl.col('family_oof').is_null())['uid'].to_list()[:5]))

    fam = j2['family_oof']
    known = {'directed_transfer', 'soft_play', 'coordinated_isolation', 'none', 'NONE'}
    bad = set(fam.drop_nulls().unique().to_list()) - known
    assert not bad, 'unexpected family value(s) in hand_oof_step254.parquet: ' + str(bad)

    j3 = j2.with_columns(
        pl.when(pl.col('family_oof') == 'directed_transfer').then(pl.col('mechplus_p_DT'))
          .when(pl.col('family_oof') == 'soft_play').then(pl.col('mechplus_p_SP'))
          .when(pl.col('family_oof') == 'coordinated_isolation').then(pl.col('mechplus_p_CI'))
          .otherwise(pl.col('mechplus_p_fmax'))                    # none / NONE / null -> fmax
          .alias('own_p'))
    own_p = j3['own_p'].to_numpy().astype(np.float64)
    assert not np.isnan(own_p).any(), 'own-family p has nulls after the family dispatch'

    n_family_mismatch = int((j3['family'] != j3['family_oof']).sum())
    return {'own_p': own_p, 'family': j3['family_oof'].to_list(),
            'n_matched_keys': len(uids) - n_missing_keys,
            'n_matched_oof': len(uids) - n_missing_oof,
            'keys_family_vs_oof_family_mismatches': n_family_mismatch}


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--primary', choices=[AGNOSTIC, 'max3'], default=AGNOSTIC)
    ap.add_argument('--report', default=None)
    a = ap.parse_args(argv)
    t0 = time.time()

    rows, uids = read_rows(a.scores)
    n = len(rows)

    join = mechplus_own_family(uids)
    own_p = join['own_p']

    s = np.empty(n, dtype=np.float64)
    p_raw = np.empty(n, dtype=np.float64)
    for i, r in enumerate(rows):
        s[i], p_raw[i] = primary_score(r, a.primary)

    ranks = average_rank_stable(s.tolist(), uids)
    u = (ranks - .5) / n
    assert np.all((u > 0) & (u < 1)), 'quantile positions must lie strictly inside (0, 1)'
    p_qmatch = np.clip(np.quantile(own_p, u), QMATCH_LO, QMATCH_HI)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as fh:
        for i, r in enumerate(rows):
            out_row = dict(r)
            out_row['mean_conf'] = round(100.0 * float(p_qmatch[i]), 4)
            out_row['mapping'] = 'qmatch'
            out_row['s_primary'] = round(float(s[i]), 6)
            out_row['p_raw_primary'] = round(float(p_raw[i]), 6)
            fh.write(json.dumps(out_row, ensure_ascii=False) + '\n')
    print('wrote %d rows to %s' % (n, out_path), flush=True)

    if a.report:
        rho = spearman_label_free(s, own_p)
        family_counts = {}
        for f in join['family']:
            family_counts[f] = family_counts.get(f, 0) + 1
        report = {
            'step': 'STEP267 amendment_2 QMATCH rescale (label-free, zero-fit)',
            'registration': str(REG267), 'amendment': 'amendment_2',
            'scores_in': str(a.scores), 'scores_out': str(out_path), 'primary': a.primary,
            'n_rows': n,
            'n_matched_keys_heldout': join['n_matched_keys'],
            'n_matched_hand_oof': join['n_matched_oof'],
            'family_counts': family_counts,
            'keys_family_vs_hand_oof_family_mismatches': join['keys_family_vs_oof_family_mismatches'],
            'label_column_read': False,
            'quantiles_s_primary': quantiles_of(s),
            'quantiles_p_raw_primary': quantiles_of(p_raw),
            'quantiles_p_qmatch': quantiles_of(p_qmatch),
            'spearman_s_vs_mechplus_own_family_label_free': round(rho, 6),
            'mapping': 'u = (avg_rank(s) - .5) / n over all scored rows; '
                       'p_qmatch = clip(quantile_u(mechplus own-family p over the same rows), '
                       '.001, .999); mean_conf_out = round(100 * p_qmatch, 4)',
            'wall_seconds': round(time.time() - t0, 1)}
        rp = Path(a.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(rp, 'w', encoding='utf-8'), indent=1)
        print('wrote', rp, flush=True)
        print('n=%d matched_keys=%d matched_oof=%d primary=%s spearman(s, mechplus own-family)=%.4f'
              % (n, join['n_matched_keys'], join['n_matched_oof'], a.primary, rho), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
