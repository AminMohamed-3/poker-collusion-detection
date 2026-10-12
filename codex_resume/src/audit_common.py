"""Shared data/metrics for new, training-only evidence screens.

Original and Claude modules are imported without editing them. The first-stage
cache is reused selection data, not a fully nested estimate of generalization.
"""
import os
for key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
CV = ROOT / 'Claude Version'
sys.path.insert(0, str(CV / 'src'))
import numpy as np
import polars as pl
from evidence_aug.paths import check_protected, sha
from evidence_aug.fold import load_fold, fold_pools
from evidence_aug.frame import FAMILIES
from evidence_aug.step14_audit_truncation import OUT as AUDIT, CLF
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step5_evaluate import half

PREV = HERE / 'experiments/symmetry_censor_20260912'


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_current():
    saved = pl.read_parquet(PREV / 'scores.parquet').sort('pair_id', 'hand_id')
    assert sha(PREV / 'scores.parquet') == json.loads((PREV / 'predictions-frozen.json').read_text(encoding='utf-8'))['scores_sha256']
    audit = pl.read_parquet(AUDIT / 'audit-oof.parquet').sort('pair_id', 'hand_id')
    assert saved.select('pair_id', 'hand_id').equals(audit.select('pair_id', 'hand_id'))
    outer = set().union(*(set(fold_pools(f)) for f in (0, 1, 2)))
    assert not set(audit['table_id']) & outer and audit['pair_id'].n_unique() == 162
    structf = pl.read_parquet(AUDIT / 'audit-structural-features.parquet')
    struct = [c for c in structf.columns if c not in ('pair_id', 'hand_id')]
    data, features, _, _ = load_fold(0)
    data = data.join(audit.select('pair_id', 'hand_id'), on=['pair_id', 'hand_id'], how='semi')
    rcols = [f for f in features if not f.startswith('family_') and f not in struct]
    d = audit.drop('oof', 'oof_rank').join(structf, on=['pair_id', 'hand_id'], validate='1:1').join(data.select(['pair_id', 'hand_id'] + rcols), on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    d = d.with_columns(pl.Series('oof', saved['stage1_only'].to_numpy()), pl.col(rcols).fill_null(-999.)).with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
    shared = struct + ['chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank']
    return d, shared, rcols, saved


def rankmean(pid, a, b):
    f = pl.DataFrame({'pair_id': pid, 'a': a, 'b': b})
    return f.select((-(pl.col('a').rank('average', descending=True).over('pair_id') + pl.col('b').rank('average', descending=True).over('pair_id')) / 2).alias('s'))['s'].to_numpy()


def evaluate(d, pred):
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    key = d.select('pair_id', 'hand_id', 'table_id')
    maps = {}
    res = {'audit_only': True, 'sets': {}, 'comparisons': {}, 'passed': []}
    for name, p in pred.items():
        summary, maps[name] = summarize(key.with_columns(pl.Series('score', p)), truths, fams)
        res['sets'][name] = summary
    for name in [x for x in maps if x != 'incumbent']:
        a, b = maps['incumbent'], maps[name]
        c = boot(a, b, ptab, pools)
        c['by_family'] = {f: float(np.mean([b[p] - a[p] for p in a if fams[p] == f])) for f in FAMILIES}
        c['halves'] = {str(h): float(np.mean([b[p] - a[p] for p in a if half(ptab[p]) == h])) for h in (0, 1)}
        c['gate_pass'] = c['mean_delta'] >= .01 and c['ci95'][0] > 0 and min(c['by_family'].values()) >= -.01 and min(c['halves'].values()) >= 0
        res['comparisons'][name] = c
        if c['gate_pass'] and name != 'stage1_only':
            res['passed'].append(name)
    return res
