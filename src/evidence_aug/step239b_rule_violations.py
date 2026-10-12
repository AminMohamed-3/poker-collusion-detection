"""STEP239b -- sharp unlisted hands that the listing rule says cannot exist (development phase only, all 372 positives).

For a two-block list (a time descent inside the list): every first-type event of the pair is listed, and every
second-type event earlier than the last listed second-type hand is listed.  So an unlisted hand EARLIER than the last
listed block-2 hand is, by the rule, not a listable event.  Count how many of those look like sharp events anyway
(p_fmax above the honest 99.9 % quantile) and profile them against listed block-1 / block-2 hands.
Output: experiments/step239_error_anatomy_20260917/rule_violations.json
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
OUT = CV / 'experiments' / 'step239_error_anatomy_20260917'
SHARP = 0.4662
o = pl.read_parquet(CV / 'experiments' / 'step232_event_detector_20260917' / 'hand_oof.parquet')
hands = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'started_at', 'phase'])
ev = pl.read_csv(ROOT / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
raw = pl.read_parquet(str(ROOT / 'outputs_v2' / 'hands_development' / 'batch_*.parquet'))
d = o.filter(pl.col('label') == 1).join(hands, on='hand_id', how='left').join(ev, on=['pair_id', 'hand_id'], how='left')
assert set(d['phase'].unique().to_list()) == {'development'}
d = d.with_columns(t=pl.col('started_at').dt.epoch('s')).drop('started_at')
rows = []
for (pid,), g in d.group_by(['pair_id']):
    g = g.sort('t')
    L = g.filter(pl.col('listed') == 1).sort('evidence_rank')
    lt = L['t'].to_list(); n = len(lt)
    cut = next((k for k in range(1, n) if lt[k] < lt[k - 1]), n)
    two = cut < n
    fam = g['behavior_family'][0]
    lid = L['hand_id'].to_list()
    for h, b in zip(lid, ['b1'] * cut + ['b2'] * (n - cut)):
        rows.append((pid, h, fam, b, two))
    if two:
        lim = max(lt[cut:])
        V = g.filter((pl.col('listed') == 0) & (pl.col('t') < lim))
        for h, p in zip(V['hand_id'].to_list(), V['p_fmax'].to_list()):
            rows.append((pid, h, fam, 'impossible_sharp' if p > SHARP else 'impossible_dull', two))
        A = g.filter((pl.col('listed') == 0) & (pl.col('t') > lim) & (pl.col('p_fmax') > SHARP))
        for h in A['hand_id'].to_list():
            rows.append((pid, h, fam, 'later_sharp', two))
K = pl.DataFrame(rows, schema=['pair_id', 'hand_id', 'family', 'kind', 'two_block'], orient='row')
res = {}
for fam in ['directed_transfer', 'soft_play', 'coordinated_isolation']:
    k = K.filter((pl.col('family') == fam) & pl.col('two_block'))
    npairs = k['pair_id'].n_unique()
    c = dict(k.group_by('kind').len().iter_rows())
    imp = c.get('impossible_sharp', 0) + c.get('impossible_dull', 0)
    res[fam] = {'two_block_pairs': npairs, 'counts': c,
                'impossible_sharp_per_pair': c.get('impossible_sharp', 0) / max(npairs, 1),
                'impossible_sharp_rate': c.get('impossible_sharp', 0) / max(imp, 1),
                'honest_rate': 0.001}
(OUT / 'rule_violations.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
print(json.dumps(res, indent=1))
# profile: which raw features separate impossible_sharp from listed b1 / b2 (directed_transfer and soft_play)
feat = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id') and raw[c].dtype.is_numeric()]
J = K.join(raw.select(['pair_id', 'hand_id'] + feat), on=['pair_id', 'hand_id'], how='left')
J.write_parquet(OUT / 'rule_violation_rows.parquet')
for fam in ['directed_transfer', 'soft_play']:
    k = J.filter((pl.col('family') == fam) & pl.col('two_block'))
    A = k.filter(pl.col('kind') == 'impossible_sharp'); B1 = k.filter(pl.col('kind') == 'b1'); B2 = k.filter(pl.col('kind') == 'b2')
    out = []
    for c in feat:
        a, b1, b2 = A[c].to_numpy().astype(float), B1[c].to_numpy().astype(float), B2[c].to_numpy().astype(float)
        m = lambda x: float(np.nanmean(x)) if len(x) else float('nan')
        sd = np.nanstd(np.concatenate([a, b1, b2])) + 1e-9
        out.append((c, m(a), m(b1), m(b2), (m(a) - m(b1)) / sd, (m(a) - m(b2)) / sd))
    out.sort(key=lambda r: -min(abs(r[4]), abs(r[5])))
    print('\n', fam, 'n impossible_sharp', A.height, 'b1', B1.height, 'b2', B2.height)
    print('%-44s %9s %9s %9s %7s %7s' % ('feature (differs from BOTH blocks)', 'imp_sharp', 'block1', 'block2', 'd_b1', 'd_b2'))
    for r in out[:28]:
        print('%-44s %9.3f %9.3f %9.3f %7.2f %7.2f' % r)
