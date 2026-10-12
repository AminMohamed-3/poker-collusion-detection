"""Step 151f (Claude): merge evaluation draws into the SX key. Checks: all draws present for every evaluation pair, finite;
draw k = 0 equals eval_views_S5.parquet exactly on the five view columns. Label-free.
Usage: step151f_C_SX_merge K (number of draws, 8 for STEP151) -> experiments/tail_keys_20260914/eval_views_SX{K}.parquet (+ eval_views_SX.parquet for K = 8)."""
import sys, json, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl
from evidence_aug.paths import CV, sha, write_json

TK = CV / 'experiments' / 'tail_keys_20260914'
VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main():
    K = int(sys.argv[1])
    d = pl.concat([pl.read_parquet(TK / f) for f in sorted(os.listdir(TK)) if f.startswith('sx_eval_k') and f.endswith('.parquet')])
    d = d.filter(pl.col('k') < K)
    assert sorted(d['k'].unique().to_list()) == list(range(K)), d['k'].unique().to_list()
    n = d['pair_id'].n_unique(); assert n == 112540 and d.height == n * K, (n, d.height)
    assert all(bool(d[f'r_{v}'].is_finite().all()) for v in VT)
    s5 = pl.read_parquet(TK / 'eval_views_S5.parquet')
    k0 = d.filter(pl.col('k') == 0).join(s5, on='pair_id', suffix='_s5', validate='1:1')
    k0diff = {v: float((k0[f'r_{v}'] - k0[f'r_{v}_s5']).abs().max()) for v in VT}
    assert max(k0diff.values()) == 0.0, k0diff
    L = d.with_columns([pl.Series(f'L_{v}', lg(d[f'r_{v}'])) for v in VT]).group_by('pair_id').agg([pl.col(f'L_{v}').mean() for v in VT])
    out = L.with_columns(pl.mean_horizontal([f'L_{v}' for v in VT]).alias('k_SX')).join(s5.select('pair_id', 'r_SHORT3', 'r_ALL5'), on='pair_id', validate='1:1').sort('pair_id')
    path = TK / f'eval_views_SX{K}.parquet'; out.write_parquet(path)
    res = {'draws': K, 'rows': out.height, 'k0_vs_S5_max_abs_diff': k0diff, 'output_sha256': sha(path), 'code_sha256': sha(os.path.abspath(__file__))}
    if K == 8:
        out.write_parquet(TK / 'eval_views_SX.parquet'); res['eval_views_SX_sha256'] = sha(TK / 'eval_views_SX.parquet')
    write_json(TK / f'eval_views_SX{K}.json', res); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
