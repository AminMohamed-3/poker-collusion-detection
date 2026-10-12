"""Step 161 screen (Claude): route 2, dev-only go/no-go before any registration. Do step-65 structural hand features, as pair aggregates,
add anything to the tail pair model's features?
Frame: the 1,860 labelled development pairs; base features = the v11 pair features as in the tail views' full-length frame (fold-1 policy ca
features; held pools not excluded, the same for both arms). Struct = per-pair mean and max of the step-65 hand features, made order-free
(A_/B_ fields as min/max over the two players). Views: full length, and 40% thinned hands (seed 7304, the tail views' short view), where
the struct aggregates use the same kept hands.
Model: LightGBM MODEL_PARAMS multiclass (ALL5 style), risk = 1 - P(none); out-of-fold over StratifiedGroupKFold(5, table groups) for 5 split
seeds. Reading: OOF AP of label 1 (base vs base+struct), overall and for the hardest positives (base OOF risk < .5).
Go/no-go fixed before running: GO only if base+struct - base AP >= +.003 on both views with >= 4 of 5 seeds positive on each; otherwise
route 2 closes with no registration. Development labels only; no evaluation data."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '3'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib, lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from pu_pairs.build_allpairs_fold import CONF, extra_aggregate
from evidence_aug.step65_structural_evidence import build as struct_build
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, MODEL_PARAMS, thin, aggregate, matrix  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

OUT = CV / 'experiments' / 'struct_pair_screen_20260915'
SEEDS = [16101, 16102, 16103, 16104, 16105]


def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); yy = y[o]; tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(1, int(yy.sum())))


def symmetric(h):
    cols = [c for c in h.columns if c not in ('pair_id', 'hand_id')]
    a = sorted(c[2:] for c in cols if c.startswith('A_') and 'B_' + c[2:] in cols)
    out = h.select('pair_id', 'hand_id', *[pl.col(c).cast(pl.Float64) for c in cols if not (c[:2] in ('A_', 'B_') and c[2:] in a)],
                   *[pl.min_horizontal('A_' + s, 'B_' + s).cast(pl.Float64).alias('min_' + s) for s in a],
                   *[pl.max_horizontal('A_' + s, 'B_' + s).cast(pl.Float64).alias('max_' + s) for s in a])
    return out


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted((ORIGINAL / 'outputs_v2' / 'hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    ca = pl.read_parquet(CONF / 'conditional-fold1-v1' / 'policy-screen' / 'hand-features.parquet')
    ca_features = [c for c in ca.columns if c.startswith('ca_')]
    combined = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1', maintain_order='left').with_columns(pl.col(ca_features).fill_null(0))
    sp = OUT / 'hand_struct_labelled.parquet'
    if sp.exists():
        hs = pl.read_parquet(sp)
    else:
        hs = symmetric(struct_build(raw.select('pair_id', 'hand_id'), labels.select('pair_id', 'player_1', 'player_2')))
        hs.write_parquet(sp)
    scols = [c for c in hs.columns if c not in ('pair_id', 'hand_id')]
    print('struct hands', hs.shape, round(time.time() - t0), flush=True)
    base_full = pl.read_parquet(ORIGINAL / 'outputs_v2' / 'pairs_development.parquet').join(tails(raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    full = base_full.join(extra_aggregate(combined, ca_features), on=['pair_id', 'table_id']).sort('pair_id')
    views = {'full': (full, hs)}
    rs = thin(raw, .4, 7304); cs = thin(combined, .4, 7304)
    v40 = aggregate(rs, names).join(tails(rs, events), on=['pair_id', 'table_id']).join(labels, on='pair_id')
    v40 = v40.join(extra_aggregate(cs, ca_features), on=['pair_id', 'table_id']).select(full.columns).sort('pair_id')
    views['thin40'] = (v40, hs.join(rs.select('pair_id', 'hand_id'), on=['pair_id', 'hand_id'], how='semi'))
    res = {'n_struct_hand_fields': len(scols), 'views': {}}
    for vn, (fr, h) in views.items():
        agg = h.group_by('pair_id').agg(*[pl.col(c).mean().alias('s_mean_' + c) for c in scols], *[pl.col(c).max().alias('s_max_' + c) for c in scols])
        fr = fr.join(agg, on='pair_id', how='left', validate='1:1').sort('pair_id')
        sf = [c for c in agg.columns if c != 'pair_id']
        assert fr.height == 1860 and fr.select(sf).null_count().sum_horizontal().item() == 0
        y = (fr['label'] == 1).to_numpy().astype(int); ymc = np.array([LABELS[f] for f in fr['behavior_family']]); g = fr['table_id'].to_numpy()
        XB, XS = matrix(fr, features), matrix(fr, features + sf)
        per = []
        for s in SEEDS:
            ob, os_ = np.zeros(len(y)), np.zeros(len(y))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(XB, ymc, g):
                ob[te] = 1 - lgb.LGBMClassifier(**MODEL_PARAMS).fit(XB[tr], ymc[tr]).predict_proba(XB[te])[:, 0]
                os_[te] = 1 - lgb.LGBMClassifier(**MODEL_PARAMS).fit(XS[tr], ymc[tr]).predict_proba(XS[te])[:, 0]
            hard = (y == 0) | (ob < .5)
            per.append({'seed': s, 'AP_base': ap(y, ob), 'AP_struct': ap(y, os_), 'dAP': ap(y, os_) - ap(y, ob),
                        'hard_pos': int((y[hard] == 1).sum()), 'dAP_hard': ap(y[hard], os_[hard]) - ap(y[hard], ob[hard])})
            print(vn, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in per[-1].items()}), round(time.time() - t0), flush=True)
        d = [p['dAP'] for p in per]
        res['views'][vn] = {'per_seed': per, 'mean_dAP': float(np.mean(d)), 'seeds_positive': int(sum(x > 0 for x in d)),
                            'mean_dAP_hard': float(np.mean([p['dAP_hard'] for p in per])), 'n_struct_pair_fields': len(sf)}
    res['GO'] = bool(all(v['mean_dAP'] >= .003 and v['seeds_positive'] >= 4 for v in res['views'].values()))
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(OUT / 'screen161.json', 'w', encoding='utf-8'), indent=1)
    print('GO', res['GO'], {k: (round(v['mean_dAP'], 4), v['seeds_positive'], round(v['mean_dAP_hard'], 4)) for k, v in res['views'].items()}, flush=True)


if __name__ == '__main__':
    main()
