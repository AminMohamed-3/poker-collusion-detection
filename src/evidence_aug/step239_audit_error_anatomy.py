"""STEP239 -- anatomy of the production evidence errors on the 162-query audit (development phase only).

Inputs: outputs_step235_20260917/audit_scores_step235.parquet (decoded score v1_REF per pair x hand; -1e6 outside the
stage-1 top-20), experiments/step232_event_detector_20260917/hand_oof.parquet (clean-negative event probability),
development_evidence.csv.  No fitting.  Output: experiments/step239_error_anatomy_20260917/anatomy.json
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
OUT = CV / 'experiments' / 'step239_error_anatomy_20260917'
OUT.mkdir(parents=True, exist_ok=True)
SHARP, Q99 = 0.4662, 0.0852

a = pl.read_parquet(CV / 'outputs_step235_20260917' / 'audit_scores_step235.parquet').select('pair_id', 'hand_id', 'v1_REF')
q = pl.read_parquet(CV / 'outputs_step235_20260917' / 'per_query_step235.parquet').select('pair_id', 'family', 'REF')
TY = pl.read_parquet(CV / 'experiments' / 'typed_listing_decoder_C_20260913' / 'audit-type-attributes.parquet').select('pair_id', 'hand_id', 'T1')
o = pl.read_parquet(CV / 'experiments' / 'step232_event_detector_20260917' / 'hand_oof.parquet').select(
    'pair_id', 'hand_id', 'p_fmax', 'p_directed_transfer', 'p_soft_play', 'p_coordinated_isolation')
hands = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'started_at', 'phase'])
ev = pl.read_csv(ROOT / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
d = a.join(TY, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('T1').fill_null(False)).join(o, on=['pair_id', 'hand_id'], how='left').join(hands, on='hand_id', how='left').join(
    ev, on=['pair_id', 'hand_id'], how='left').join(q, on='pair_id', how='inner')
assert set(d['phase'].unique().to_list()) == {'development'} and d['p_fmax'].null_count() == 0
d = d.with_columns(t=pl.col('started_at').dt.epoch('s'), listed=pl.col('evidence_rank').is_not_null(),
                   in20=pl.col('v1_REF') > -1e5).drop('started_at')
res = {}
for fam in ['directed_transfer', 'soft_play', 'coordinated_isolation']:
    F = d.filter(pl.col('family') == fam)
    T = {'queries': 0, 'truth': 0, 'hit': 0}
    miss = {'not_in_top20': 0, 'invisible_in20': 0, 'grey_in20': 0, 'sharp_in20': 0}
    missb = {'block1': 0, 'block2': 0}
    truthb = {'block1': 0, 'block2': 0}
    pick = {'listed': 0, 'sharp_unlisted': 0, 'grey_unlisted': 0, 'honest_looking_unlisted': 0}
    sharp_pick_when = {'before_last_block1': 0, 'after_last_block1': 0}
    ap_by_nb1 = {}
    for pid, g in F.group_by('pair_id'):
        g = g.sort('t')
        L = g.filter(pl.col('listed')).sort('evidence_rank')
        lt = L['t'].to_list()
        n = len(lt)
        cut = next((k for k in range(1, n) if lt[k] < lt[k - 1]), n)
        b1 = set(L['hand_id'].to_list()[:cut])
        # v1_REF is the PRE-decoder probability + 10: decode it the production way before ranking
        G20 = g.filter(pl.col('in20'))
        pp = np.clip(G20['v1_REF'].to_numpy() - 10, 0, 1)
        tf = G20['T1'].to_numpy().astype(bool) & (fam != 'coordinated_isolation')
        order = np.lexsort((np.arange(G20.height), ~tf))
        dist = np.zeros(6); dist[0] = 1.; qq = np.empty(G20.height)
        for i in order:
            qq[i] = dist[:5].sum(); nw = dist * (1 - pp[i]); nw[1:] += dist[:-1] * pp[i]; nw[5] += dist[5] * pp[i]; dist = nw
        top5 = G20.with_columns(pl.Series('dec', pp * qq + 1e-9 * pp)).sort('dec', descending=True).head(5)
        picked = set(top5['hand_id'].to_list())
        T['queries'] += 1
        T['truth'] += n
        ap_by_nb1.setdefault(cut if n == 5 else -n, []).append(g['REF'][0])
        last_b1_t = max(lt[:cut])
        for r in L.iter_rows(named=True):
            blk = 'block1' if r['hand_id'] in b1 else 'block2'
            truthb[blk] += 1
            if r['hand_id'] in picked:
                T['hit'] += 1
                continue
            missb[blk] += 1
            if not r['in20']:
                miss['not_in_top20'] += 1
            elif r['p_fmax'] < Q99:
                miss['invisible_in20'] += 1
            elif r['p_fmax'] < SHARP:
                miss['grey_in20'] += 1
            else:
                miss['sharp_in20'] += 1
        for r in top5.iter_rows(named=True):
            if r['listed']:
                pick['listed'] += 1
            elif r['p_fmax'] > SHARP:
                pick['sharp_unlisted'] += 1
                sharp_pick_when['before_last_block1' if r['t'] < last_b1_t else 'after_last_block1'] += 1
            elif r['p_fmax'] > Q99:
                pick['grey_unlisted'] += 1
            else:
                pick['honest_looking_unlisted'] += 1
    res[fam] = {'totals': T, 'recall_at_5': T['hit'] / T['truth'], 'truth_by_block': truthb, 'missed_by_block': missb,
                'missed_truth_by_kind': miss, 'top5_picks_by_kind': pick, 'sharp_unlisted_picks_timing': sharp_pick_when,
                'mean_AP_by_block1_size (negative key = short list of that length)':
                    {str(k): [len(v), round(float(np.mean(v)), 4)] for k, v in sorted(ap_by_nb1.items())}}
(OUT / 'anatomy.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
print(json.dumps(res, indent=1))
