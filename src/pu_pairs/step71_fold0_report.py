"""Step 71: fold-0 D report (P4-D, APPROVED 2026-09-13). Measurement only, no lever fitted.

Joins risk_full (frozen full-history score from the fold0-full build) onto the C-stage
views, applies the verbatim P4 classes (p4bcd_spec.md section 3):
  pos = labelled positives + unlabelled with full-length risk >= .9
  FP  = unlabelled with prefix risk >= .5 AND full-length risk < .1
  def A = pos vs FP; def C = labelled positives vs all other block pairs
Orders: frozen = risk_prefix; S-like = lg(r_FULL) + .5*lg(r_RAW) with R1_prefix rows
to bottom (-100, step67b pattern). Population: eligible rows (prefix shared_hands>=38)
in the prefix block (risk_prefix>=.5). 30 %-subsample SDs of (S-like minus frozen):
k=max(round(.3n),10), 2000 draws, seed 2901, draws with <3 positives skipped.

Persists fold0_views.parquet (+ copy beside the fold-1/2 multiview_prefix friends in
viewdis_20260911/) and fold0_D.json. Verdict: population READY or reason.
CPU only, utf-8.
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
import sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
import numpy as np
import polars as pl
from pu_pairs.build_allpairs_fold import EXP, sha
from pu_pairs.step60_multiview_blend import lg, ap

VD = CV / 'experiments' / 'viewdis_20260911'
MIN_SH = 38
THR = 0.5
SD_SEED = 2901
SD_ITERS = 2000


def log(*a):
    print(*a, flush=True)


def sub30_sd(y, a, b, seed=SD_SEED, iters=SD_ITERS):
    rng = np.random.default_rng(seed)
    n = len(y)
    k = max(int(round(0.3 * n)), 10)
    ds = []
    for _ in range(iters):
        idx = rng.choice(n, k, replace=False)
        if y[idx].sum() < 3:
            continue
        ds.append(ap(y[idx], b[idx]) - ap(y[idx], a[idx]))
    ds = np.array(ds)
    return {'iters_used': int(len(ds)),
            'mean_diff': round(float(ds.mean()), 5) if len(ds) else None,
            'sd': round(float(ds.std()), 5) if len(ds) else None,
            'share_negative': round(float((ds < 0).mean()), 4) if len(ds) else None}


def run_def(c, pos_mask, tag):
    y = pos_mask.to_numpy().astype(float)
    frozen = c['risk_prefix'].to_numpy()
    s_like = lg(c['r_FULL'].to_numpy()) + 0.5 * lg(c['r_RAW'].to_numpy())
    s_like = np.where(c['R1_prefix'].to_numpy(), s_like - 100, s_like)
    r = {'n': int(c.height), 'n_pos': int(y.sum()), 'n_neg': int(c.height - y.sum()),
         'AP_frozen': round(ap(y, frozen), 5), 'AP_S_like': round(ap(y, s_like), 5)}
    r['S_minus_frozen'] = round(r['AP_S_like'] - r['AP_frozen'], 5)
    r['sub30'] = sub30_sd(y, frozen, s_like)
    log(json.dumps({tag: r}))
    return r


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--out', type=str, default='fold0-prefix60')
    ap_.add_argument('--full-out', type=str, default='fold0-full')
    args = ap_.parse_args()
    out = EXP / args.out
    d = pl.read_parquet(out / 'fold0_views_noFull.parquet')
    full = pl.read_parquet(EXP / args.full_out / 'scores.parquet').select(
        'pair_id', pl.col('risk').alias('risk_full'),
        pl.col('shared_hands').alias('shared_full'))
    d = d.join(full, on='pair_id', how='left', validate='1:1')
    assert d['risk_full'].is_not_null().all(), 'missing full risk for some prefix pairs'
    d = d.with_columns(
        ((pl.col('label') == 1) | ((pl.col('label') < 0) & (pl.col('risk_full') >= 0.9))).alias('is_pos'),
        ((pl.col('label') < 0) & (pl.col('risk_prefix') >= THR) & (pl.col('risk_full') < 0.1)).alias('is_FP'))
    d.write_parquet(out / 'fold0_views.parquet')
    d.write_parquet(VD / 'fold0_views.parquet')
    res = {'step': 'P4D_fold0_report', 'pairs': d.height,
           'code_sha256': sha(ORIGINAL / 'Claude Version' / 'src' / 'pu_pairs' / 'step71_fold0_report.py'),
           'views_sha256': sha(out / 'fold0_views_noFull.parquet'),
           'full_scores_sha256': sha(EXP / args.full_out / 'scores.parquet')}
    elig = d.filter(pl.col('shared_hands') >= MIN_SH)
    res['eligible_prefix38'] = elig.height
    blk = elig.filter(pl.col('risk_prefix') >= THR)
    res['eligible_block'] = blk.height
    res['eligible_FP_count'] = int(blk['is_FP'].sum())
    res['eligible_pos_count'] = int(blk['is_pos'].sum())
    res['eligible_labelled_pos_in_block'] = int(((blk['label'] == 1)).sum())
    # def A: pos vs FP within the eligible block
    cA = blk.filter(pl.col('is_pos') | pl.col('is_FP'))
    res['defA_pos_vs_FP'] = run_def(cA, cA['is_pos'], 'defA') if cA.height else {'n': 0}
    # def C: labelled positives vs all other eligible block pairs
    res['defC_labelled_vs_rest'] = run_def(blk, blk['label'] == 1, 'defC') if blk.height else {'n': 0}
    fps = blk.filter(pl.col('is_FP')).select('pair_id', 'table_id', 'risk_prefix', 'risk_full',
                                             'r_FULL', 'r_RAW', 'shared_hands').sort('risk_prefix', descending=True)
    res['FP_shared_hands_median'] = float(fps['shared_hands'].median()) if fps.height else None
    a = res['defA_pos_vs_FP']
    c = res['defC_labelled_vs_rest']
    gate = (a.get('S_minus_frozen') is not None and a['S_minus_frozen'] > 0
            and c.get('S_minus_frozen') is not None and c['S_minus_frozen'] > 0)
    res['S_confirmed_independently'] = bool(gate)
    res['verdict'] = ('population READY' if res['eligible_FP_count'] > 0 and cA.height and blk.height
                      else 'NOT READY: no eligible FP/block population')
    write_json(out / 'fold0_D.json', res)
    write_json(VD / 'fold0_D.json', res)
    log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
