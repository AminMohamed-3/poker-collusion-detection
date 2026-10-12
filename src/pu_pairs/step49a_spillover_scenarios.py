"""Step 49a (label-free scenario analysis): what would moving the demoted set D just below the block do to PairAP?
Truth is unknown on evaluation, so sweep scenarios: F = number of false positives among the 909 block pairs, q = share of
D that is false positive, placement of the remaining F - q|D| false positives among the other block pairs (uniform or
weighted toward lower risk). Report AP before / after per cell; cells whose AP_before is consistent with the public P (.938)
are the ones that matter. Output: experiments/spillover_20260911/eval_scenarios.json."""
import os, sys, json
os.environ['POLARS_MAX_THREADS'] = '2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np, polars as pl

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'spillover_20260911'
RULES = {
    'R1 dominated & risk_noT<.1': lambda d: d['dominated'] & (d['risk_noT'].fill_nan(0) < .1),
    'R2 dominated & cop_T>=.6 & risk_noT<.1': lambda d: d['dominated'] & (d['cop_T'] >= .6) & (d['risk_noT'].fill_nan(0) < .1),
    'R1b dominated & risk_noT<.5': lambda d: d['dominated'] & (d['risk_noT'].fill_nan(0) < .5),
}


def ap_block(is_fp, n_pos_total):
    """AP contribution of the block ordering (positives outside the block contribute ~0 precision)."""
    tp = (~is_fp).astype(float)
    cum = np.cumsum(tp); ranks = np.arange(1, len(tp) + 1)
    return float((tp * cum / ranks).sum() / n_pos_total)


def main():
    d = pl.read_parquet(OUT / 'eval_rescoring.parquet').sort('risk_v13', descending=True)
    risk = d['risk_v13'].to_numpy(); n = d.height
    rng = np.random.default_rng(49)
    out = {'block': n}
    for rname, rule in RULES.items():
        inD = rule(d).to_numpy(); nD = int(inD.sum())
        # after: D moved just below the block, ordered by risk_noT descending
        order_after = np.concatenate([np.where(~inD)[0], np.where(inD)[0][np.argsort(-d['risk_noT'].fill_nan(0).to_numpy()[inD], kind='stable')]])
        cells = []
        for F in (150, 200, 300, 400, 500):
            for q in (.5, .7, .85, 1.0):
                nfpD = int(round(q * nD))
                if nfpD > nD or F - nfpD < 0 or F - nfpD > n - nD:
                    continue
                for placement in ('uniform', 'low_risk_weighted', 'steep_gamma3', 'steep_gamma6'):
                    res_b, res_a = [], []
                    others = np.where(~inD)[0]
                    gamma = {'uniform': 0, 'low_risk_weighted': 1, 'steep_gamma3': 3, 'steep_gamma6': 6}[placement]
                    w = ((1 - risk[others]) + 1e-3) ** gamma
                    w = w / w.sum()
                    for _ in range(300):
                        is_fp = np.zeros(n, dtype=bool)
                        is_fp[rng.choice(np.where(inD)[0], size=nfpD, replace=False)] = True
                        is_fp[rng.choice(others, size=F - nfpD, replace=False, p=w)] = True
                        n_pos_total = (n - F) / .97  # ~3 % of positives missed below the block
                        res_b.append(ap_block(is_fp, n_pos_total)); res_a.append(ap_block(is_fp[order_after], n_pos_total))
                    cells.append({'F_block_false_positives': F, 'q_fp_share_in_D': q, 'placement': placement, 'AP_before': round(float(np.mean(res_b)), 4), 'AP_after': round(float(np.mean(res_a)), 4),
                                  'delta_P': round(float(np.mean(res_a) - np.mean(res_b)), 4), 'delta_composite': round(.7 * float(np.mean(res_a) - np.mean(res_b)), 4)})
        consistent = [c for c in cells if .93 <= c['AP_before'] <= .946]
        out[rname] = {'n_D': nD, 'cells': cells, 'consistent_with_public_P': consistent,
                      'consistent_delta_composite_range': [min(c['delta_composite'] for c in consistent), max(c['delta_composite'] for c in consistent)] if consistent else None}
    write_json(OUT / 'eval_scenarios.json', out)
    for rname in RULES:
        print(rname, 'n_D', out[rname]['n_D'], 'consistent range (composite)', out[rname]['consistent_delta_composite_range'])
        for c in out[rname]['consistent_with_public_P']:
            print('  ', c)


if __name__ == '__main__':
    main()