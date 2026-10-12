"""Step 64d: BLOCK EXTENSION on the prefix testbed. Candidates S / M / M2 only permute the frozen block (risk >= .5); the
recall tail (positives below .5 at evaluation-like length) can only be recovered by PROMOTING pairs from below the block.
This step measures, cross-fold on the open population of step 64c (labelled pairs + prefix risk >= .1 + re-weighted tail
sample), what promotion buys and costs for every candidate score s:
  frozen               the fold model's prefix risk (the base order);
  block re-order       block pairs re-ordered by s, everything below unchanged (the M2 construction);
  extension x m        the block plus the non-block pairs whose weighted rank under s is within m x K (K = block size),
                       all ordered by s; the rest keep the frozen order.
For each: open AP (weighted, positives vs everything else), and the certified composition of the promoted pairs
(labelled positive / negative, unlabelled by full-history risk: >= .9 undisclosed colluder, [.1, .9) mid, < .1 null).
Scores s: the length-matched views alone, and the cross-fold learned open blends of step 64c (base, + SHORT3 + ALL5, + hand
pools, + all new). Development labels: validation only. Output: experiments/viewdis_20260911/block_extension.json."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, write_json
import numpy as np
import polars as pl
from pu_pairs.build_allpairs_fold import fold_pools
from pu_pairs.step64c_hand_view import wap, lg

VD = CV / 'experiments' / 'viewdis_20260911'
MULT = (1.0, 1.25, 1.5, 2.0)


def log(*a):
    print(*a, flush=True)


def composition(rows, w):
    lab = rows['label'].to_numpy(); rf = rows['risk_full'].to_numpy()
    def cnt(mask): return round(float(w[mask].sum()), 1)
    unl = np.isnan(lab) if lab.dtype.kind == 'f' else np.array([x is None for x in lab])
    lab_f = np.nan_to_num(lab.astype(float), nan=-1)
    return {'pos': cnt(lab_f == 1), 'neg': cnt(lab_f == 0), 'unl_full_ge.9': cnt(unl & (rf >= .9)), 'unl_full_.1_.9': cnt(unl & (rf >= .1) & (rf < .9)), 'unl_full_lt.1': cnt(unl & (rf < .1)), 'n_rows': int(len(lab))}


def evaluate(part, s, name):
    y = (part['label'] == 1).to_numpy().astype(float); w = part['weight'].to_numpy(); risk = part['risk_prefix'].to_numpy()
    blk = risk >= .5; K = float(w[blk].sum())
    res = {'K': round(K), 'frozen': round(wap(y, risk, w), 4)}
    # block re-order only: block pairs by s (shifted above everything), rest by frozen risk
    s_ro = np.where(blk, 10.0 + (s - s.min()) / (np.ptp(s) + 1e-9), risk)
    res['block_reorder'] = round(wap(y, s_ro, w), 4)
    order = np.argsort(-s, kind='stable'); cum = np.cumsum(w[order]); wrank = np.empty(len(s)); wrank[order] = cum
    for m in MULT:
        ext = ~blk & (wrank <= m * K)
        s_ext = np.where(blk | ext, 10.0 + (s - s.min()) / (np.ptp(s) + 1e-9), risk)
        res[f'ext_x{m}'] = {'open_AP': round(wap(y, s_ext, w), 4), 'promoted_weighted': round(float(w[ext].sum())), 'composition': composition(part.filter(pl.Series(ext)), w[ext]),
                            'tail_pos_promoted': int(((y == 1) & ext).sum()), 'tail_pos_total': int(((y == 1) & ~blk).sum())}
    return res


def main():
    t0 = time.time()
    op = pl.read_parquet(VD / 'tail_views_open_pred.parquet')
    tv = pl.read_parquet(VD / 'tail_views_prefix.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first', maintain_order=True).drop('fold_new')
    hv = pl.read_parquet(VD / 'hand_view_prefix.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first', maintain_order=True).drop('fold_new')
    mp = pl.read_parquet(VD / 'multiview_prefix.parquet').select('pair_id', 'r_RAW')
    d = op.join(tv, on='pair_id', how='left').join(hv, on='pair_id', how='left').join(mp, on='pair_id', how='left')
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2; tab = d['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab]); in2sh = np.array([t in f2 for t in tab]) & ~in1
    scores = {'SHORT3': lg(d['r_SHORT3'].to_numpy()), 'ALL5': lg(d['r_ALL5'].to_numpy()), 'FULL+.5RAW (S-like)': lg(d['risk_prefix'].to_numpy()) + .5 * lg(d['r_RAW'].to_numpy()) if 'r_RAW' in d.columns else None,
              'H_llr': d['H_llr'].to_numpy().astype(float), 'H_fammax': d['H_fammax'].to_numpy().astype(float)}
    for c in [c for c in d.columns if c.startswith('open_')]:
        scores[c] = d[c].to_numpy().astype(float)
    scores = {k: v for k, v in scores.items() if v is not None}
    res = {'rows': d.height, 'weighted_rows': round(float(d['weight'].sum())), 'scores': {}}
    for name, s in scores.items():
        s = np.nan_to_num(s, nan=-50.0)
        res['scores'][name] = {'pooled': evaluate(d, s, name), 'f1': evaluate(d.filter(pl.Series(in1)), s[in1], name), 'f2sh': evaluate(d.filter(pl.Series(in2sh)), s[in2sh], name)}
        log(name, json.dumps({k: (v if not isinstance(v, dict) else v.get('open_AP')) for k, v in res['scores'][name]['pooled'].items()}))
    res['seconds'] = round(time.time() - t0)
    write_json(VD / 'block_extension.json', res)


if __name__ == '__main__':
    main()
