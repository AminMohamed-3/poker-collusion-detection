"""Step 56 (label-free scenario analysis): an upper-band false-positive class, EVIDENCE CONCENTRATION.
Step 48 re-scored every block pair with its five listed evidence hands removed (plus random hands to the same count as
the noT variant; 'risk_evdrop_mean'). On development (step 48b, our own proxy evidence lists) the share of upper-band
(risk >= .9, not R1) pairs whose risk falls below .5 without their listed hands is 11/175 = 6.3 % for labelled positives
(7.7 % at <= 70 hands, 8.7 % at 70-100) and 20/83 = 24 % for unlabelled pairs (60 % at <= 70 hands). On evaluation the
same share is 124/503 = 25 % (34 % at <= 70 hands; [.9,.99) 54 %, [.99,1] 14 %). Unlike the RAW-low and R1 classes,
this class lives in the upper bands, where PairAP is actually decided, so its expected effect is large in both
directions. This step sweeps scenarios for three variants of an evidence-concentration tier stacked on candidate D:
  a: block & not R1 & not T1 & evdrop < .5                      (dev positives 6.3 %, unlabelled 24 %)
  b: a & n_hands <= 70                                           (dev positives 1/13, unlabelled 9/15)
  c: a & RAW < .5                                                (dev positives 2/175, unlabelled 7/83)
The sweep adds the recall deficit (share of genuine pairs below .5) as a scenario dimension, because it bounds how many
false positives the upper bands can hold at the public P = .938. Nothing is built here.
Output: experiments/viewdis_20260911/eval_scenarios_E.json."""
import os, sys, json
os.environ['POLARS_MAX_THREADS'] = '2'
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_aug.paths import CV, write_json

SP = CV / 'experiments' / 'spillover_20260911'
VD = CV / 'experiments' / 'viewdis_20260911'


def ap_block(is_fp, n_pos_total):
    tp = (~is_fp).astype(float)
    cum = np.cumsum(tp); ranks = np.arange(1, len(tp) + 1)
    return float((tp * cum / ranks).sum() / n_pos_total)


def band(c):
    return pl.when(pl.col(c) >= .99).then(pl.lit('[.99,1]')).when(pl.col(c) >= .9).then(pl.lit('[.9,.99)')).when(pl.col(c) >= .5).then(pl.lit('[.5,.9)')).otherwise(pl.lit('<.5'))


def main():
    res = pl.read_parquet(SP / 'eval_rescoring.parquet').select('pair_id', 'risk_v13', 'dominated', 'risk_noT', 'risk_evdrop_mean', 'risk_rand_mean', 'n_hands')
    ve = pl.read_parquet(VD / 'view_eval.parquet').select('pair_id', 'r_RAW', 'R1')
    d = res.join(ve, on='pair_id', validate='1:1').with_columns(band('risk_v13').alias('band')).sort('risk_v13', descending=True)
    d = d.with_columns((pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < .1)).alias('T2'))
    d = d.with_columns((~pl.col('T2') & (pl.col('r_RAW') < .05)).alias('T1'))
    base_free = ~pl.col('T2') & ~pl.col('T1')
    variants = {'a': base_free & (pl.col('risk_evdrop_mean') < .5),
                'b': base_free & (pl.col('risk_evdrop_mean') < .5) & (pl.col('n_hands') <= 70),
                'c': base_free & (pl.col('risk_evdrop_mean') < .5) & (pl.col('r_RAW') < .5)}
    for k, v in variants.items():
        d = d.with_columns(v.alias(f'T3{k}'))
    risk = d['risk_v13'].to_numpy(); raw = d['r_RAW'].to_numpy(); evd = d['risk_evdrop_mean'].to_numpy()
    in1 = d['T1'].to_numpy(); in2 = d['T2'].to_numpy()
    o1 = np.where(in1)[0][np.argsort(-raw[in1], kind='stable')]
    o2 = np.where(in2)[0][np.argsort(-d['risk_noT'].fill_nan(0).to_numpy()[in2], kind='stable')]
    n = d.height
    out = {'n_block': n, 'n_T1': int(in1.sum()), 'n_T2': int(in2.sum())}
    # development validation counts for the record (labels used only here)
    dv = pl.read_parquet(SP / 'dev_rescoring.parquet').unique(subset=['pair_id'], keep='first', maintain_order=True)
    lab = pl.read_parquet(VD / 'view_dev.parquet').select('pair_id', pl.col('R1').alias('R1v'), 'r_RAW')
    dd = dv.join(lab, on='pair_id', how='inner').with_columns(band('risk_full').alias('band'))
    up = dd.filter((pl.col('risk_full') >= .9) & ~pl.col('R1v') & ~(pl.col('r_RAW') < .05))
    dev_rules = {'a': pl.col('risk_evdrop_mean') < .5, 'b': (pl.col('risk_evdrop_mean') < .5) & (pl.col('n_hands') <= 70), 'c': (pl.col('risk_evdrop_mean') < .5) & (pl.col('r_RAW') < .5)}
    out['dev_upper_not_R1_not_T1'] = {k: {'positives': f"{int(up.filter(r & (pl.col('label') == 1)).height)}/{int(up.filter(pl.col('label') == 1).height)}",
                                          'unlabelled': f"{int(up.filter(r & (pl.col('label') == -1)).height)}/{int(up.filter(pl.col('label') == -1).height)}"} for k, r in dev_rules.items()}
    rng = np.random.default_rng(56)
    cells = []
    for k in variants:
        in3 = d[f'T3{k}'].to_numpy()
        o3 = np.where(in3)[0][np.argsort(-evd[in3], kind='stable')]
        rest = np.where(~in1 & ~in2 & ~in3)[0]
        rest_D = np.where(~in1 & ~in2)[0]
        order_C = np.concatenate([np.where(~in2)[0], o2]); order_D = np.concatenate([rest_D, o1, o2]); order_E = np.concatenate([rest, o3, o1, o2])
        # E' = tier 3 alone on C (no RAW-low tier), to read the two tiers separately
        order_E1 = np.concatenate([np.where(~in2 & ~in3)[0], o3, o2])
        idx3 = np.where(in3)[0]; idx1 = np.where(in1)[0]; idx2 = np.where(in2)[0]
        out[f'n_T3{k}'] = int(in3.sum()); out[f'T3{k}_by_band'] = d.filter(pl.col(f'T3{k}')).group_by('band').len().sort('band').to_dicts()
        for miss in (.0, .03, .06):
            for beta in (0, 3):
                for F in (100, 150, 200, 250, 300, 350, 400, 450, 500):
                    for q2, q1 in ((.9, .85), (1.0, .95)):
                        for q3 in (.3, .5, .65, .8, .9):
                            f1, f2, f3 = int(round(q1 * len(idx1))), int(round(q2 * len(idx2))), int(round(q3 * len(idx3)))
                            r = F - f1 - f2 - f3
                            if r < 0 or r > len(rest):
                                continue
                            for gamma in (2, 4):
                                w = ((1 - risk[rest]) + 1e-3) ** gamma * ((1 - raw[rest]) + 1e-3) ** beta; w = w / w.sum()
                                acc = {'C': [], 'D': [], 'E': [], 'E1': []}; b = []
                                for _ in range(150):
                                    is_fp = np.zeros(n, dtype=bool)
                                    is_fp[rng.choice(idx1, size=f1, replace=False)] = True
                                    is_fp[rng.choice(idx2, size=f2, replace=False)] = True
                                    is_fp[rng.choice(idx3, size=f3, replace=False)] = True
                                    is_fp[rng.choice(rest, size=r, replace=False, p=w)] = True
                                    npos = (n - F) / (1 - miss)
                                    b.append(ap_block(is_fp, npos))
                                    for name, o in (('C', order_C), ('D', order_D), ('E', order_E), ('E1', order_E1)):
                                        acc[name].append(ap_block(is_fp[o], npos))
                                b = float(np.mean(b))
                                cell = {'variant': k, 'miss': miss, 'beta': beta, 'F': F, 'q2': q2, 'q1': q1, 'q3': q3, 'gamma': gamma, 'AP_before': round(b, 4)}
                                for name in acc:
                                    a = float(np.mean(acc[name])); cell[f'd{name}'] = round(.7 * (a - b), 4)
                                cell['dE_minus_D'] = round(cell['dE'] - cell['dD'], 4); cell['dE1_minus_C'] = round(cell['dE1'] - cell['dC'], 4)
                                cells.append(cell)
    cons = [c for c in cells if .93 <= c['AP_before'] <= .946]
    def rng_(key, sub): return [min(c[key] for c in sub), max(c[key] for c in sub)] if sub else None
    summary = {}
    for k in variants:
        ck = [c for c in cons if c['variant'] == k]
        summary[k] = {'consistent_cells': len(ck), 'dC': rng_('dC', ck), 'dD': rng_('dD', ck), 'dE': rng_('dE', ck), 'dE_minus_D': rng_('dE_minus_D', ck), 'dE1_minus_C': rng_('dE1_minus_C', ck),
                      'by_q3': {str(q): {'n': len([c for c in ck if c['q3'] == q]), 'dE_minus_D': rng_('dE_minus_D', [c for c in ck if c['q3'] == q]), 'dE': rng_('dE', [c for c in ck if c['q3'] == q])} for q in (.3, .5, .65, .8, .9)},
                      'by_miss': {str(m): {'n': len([c for c in ck if c['miss'] == m]), 'F': sorted({c['F'] for c in ck if c['miss'] == m}), 'dC': rng_('dC', [c for c in ck if c['miss'] == m]), 'dE_minus_D': rng_('dE_minus_D', [c for c in ck if c['miss'] == m])} for m in (.0, .03, .06)}}
    out['summary'] = summary; out['consistent_cells'] = len(cons); out['cells'] = cells
    write_json(VD / 'eval_scenarios_E.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, indent=1))


if __name__ == '__main__':
    main()
