"""Step 151b (Claude): re-order the risk values of evaluation pairs below .5 by a tail key on the public .90321 file (STEP150 output).
Key S5 = mean clipped logit of SHORT3, ALL5, LIN30, LIN300, FAMMAX (or SX with --key SX after the part-B rule passes).
No fits, no labels. Block rows, predicted_behavior and evidence stay byte-identical. Registration: outputs_v1_followups_20260914/prediction_step151.json.
Never uploads."""
import sys, json, time, argparse
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
from evidence_aug.paths import sha, check_protected
from validation import validate_submission

OUT = POKER / 'Claude Version/outputs_v1_followups_20260914'
PREREG, PREREG_SHA = OUT / 'prediction_step151.json', 'e4868079522a7807905424b0067e0f9be054e8a5a44ffe038dbd2496543f5c90'
BASE = OUT / 'submission_v1_ci_untyped_tail_short3all5.csv'
TK = POKER / 'Claude Version/experiments/tail_keys_20260914'
VIEWS = {'S5': TK / 'eval_views_S5.parquet', 'SX': TK / 'eval_views_SX.parquet', 'SX32': TK / 'eval_views_SX32.parquet'}
PREREG153, PREREG153_SHA = OUT / 'prediction_step152.json', '3184725ea875f508d82473aad865e33e20884ea80f10181c6c3a4fd112c2b317'  # STEP153 (file named 152)
CSV = {'S5': OUT / 'submission_v1_ci_untyped_tail_S5.csv', 'SX': OUT / 'submission_v1_ci_untyped_tail_SX.csv', 'SX32': OUT / 'submission_v1_ci_untyped_tail_SX32.csv'}
S5_COLS = ['r_SHORT3', 'r_ALL5', 'r_LIN30', 'r_LIN300', 'r_FAMMAX']


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--key', default='S5', choices=['S5', 'SX', 'SX32']); args = ap.parse_args()
    t0 = time.perf_counter()
    assert sha(PREREG) == PREREG_SHA and sha(PREREG153) == PREREG153_SHA
    for k, v in json.loads(PREREG.read_text(encoding='utf-8'))['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], k
    prot = check_protected()
    csv, receipt = CSV[args.key], OUT / f'receipt_step151_{args.key}.json'
    assert not csv.exists()

    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row')
    base = base.with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    tail = base.filter(pl.col('r') < .5)
    n_block = base.height - tail.height
    assert tail.height == 111631 and n_block == 909, (tail.height, n_block)

    views = pl.read_parquet(VIEWS[args.key])
    t = tail.join(views, on='pair_id', how='left', validate='1:1', maintain_order='left')
    kcols = S5_COLS if args.key == 'S5' else ['k_SX']
    assert all(t[c].null_count() == 0 for c in kcols + ['r_SHORT3', 'r_ALL5'])
    # lineage: the base tail is the STEP150 P order (ties only inside equal-value groups)
    P = lg(t['r_SHORT3']) + lg(t['r_ALL5'])
    t = t.with_columns(pl.Series('P', P))
    got = t.sort(['r', 'pair_id'], descending=[True, False])
    grp = got.group_by('risk_score').len().filter(pl.col('len') > 1)
    by_p = t.sort(['P', 'pair_id'], descending=[True, False])
    lineage_mismatch = int(sum(a != b for a, b in zip(got['pair_id'].to_list(), by_p['pair_id'].to_list())))
    assert lineage_mismatch <= 2 * int(grp['len'].sum()), lineage_mismatch

    key = np.mean([lg(t[c]) for c in S5_COLS], axis=0) if args.key == 'S5' else t['k_SX'].to_numpy()
    t = t.with_columns(pl.Series('key', key))
    key_ties = t.height - t['key'].n_unique()
    values = t.sort(['r', 'pair_id'], descending=[True, False])['risk_score']
    order = t.sort(['key', 'pair_id'], descending=[True, False]).select('pair_id', 'key')
    assign = order.with_columns(pl.Series('new_risk', values.to_list()))
    moved = t.select('row', 'pair_id', 'risk_score', 'r', 'key').join(assign.select('pair_id', 'new_risk'), on='pair_id', validate='1:1', maintain_order='left')

    result = base.join(moved.select('pair_id', 'new_risk'), on='pair_id', how='left', maintain_order='left')
    result = result.with_columns(pl.coalesce('new_risk', 'risk_score').alias('risk_score')).sort('row').drop('row', 'r', 'new_risk')
    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    result = result.select(cols)
    base_cols = base.sort('row').select(cols)
    assert result.drop('risk_score').equals(base_cols.drop('risk_score'))
    assert sorted(result['risk_score'].to_list()) == sorted(base_cols['risk_score'].to_list())
    result.write_csv(csv)

    a, b = BASE.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
    assert len(a) == len(b)
    diff = [k for k in range(len(a)) if a[k] != b[k]]
    tail_ids = set(tail['pair_id'].to_list())
    assert {a[k].split(b',')[0].decode() for k in diff} <= tail_ids
    for k in diff:
        x, y = a[k].split(b','), b[k].split(b',')
        assert x[0] == y[0] and x[2:] == y[2:], 'a non-risk cell changed'

    back = pl.read_csv(csv, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    bt = back.join(tail.select('pair_id'), on='pair_id', how='semi')
    assert bt.height == tail.height and float(bt['risk_score'].max()) < .5
    assert back.join(tail.select('pair_id'), on='pair_id', how='anti').sort('pair_id').equals(
        base.select('pair_id', pl.col('r').alias('risk_score')).join(tail.select('pair_id'), on='pair_id', how='anti').sort('pair_id'))
    got2 = bt.join(order, on='pair_id').sort(['risk_score', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    want = order['pair_id'].to_list()
    mismatch = sum(g != w for g, w in zip(got2, want))
    equal_value_groups = int((bt.group_by('risk_score').len()['len'] > 1).sum())

    from scipy.stats import spearmanr
    rank_old = by_p['pair_id'].to_list()
    overlap = {str(k): round(len(set(rank_old[:k]) & set(want[:k])) / k, 4) for k in (100, 500, 1000, 2000, 5000)}
    val = validate_submission(csv, POKER / 'data/evaluation_pairs.csv', POKER / 'data/hands.parquet', POKER / 'data/seats.parquet', POKER / 'data/sample_submission.csv')
    assert check_protected() == prot
    rec = {'registration_sha256': PREREG_SHA, 'key': args.key, 'views': str(VIEWS[args.key]), 'views_sha256': sha(VIEWS[args.key]),
           'base_sha256': sha(BASE), 'output': str(csv), 'output_sha256': sha(csv),
           'tail_rows': tail.height, 'block_rows': n_block, 'base_tail_order_vs_P_mismatch_rows': lineage_mismatch, 'base_equal_value_group_rows': int(grp['len'].sum()),
           'key_ties_in_tail': key_ties, 'registration_step153_sha256': PREREG153_SHA, 'rows_changed_vs_base': len(diff), 'only_tail_risk_cells_changed': True, 'value_multiset_unchanged': True,
           'order_mismatch_vs_key': mismatch, 'equal_value_groups_in_tail': equal_value_groups,
           'tail_spearman_P_vs_key': round(float(spearmanr(t['P'], t['key'])[0]), 4), 'tail_topk_overlap_P_vs_key': overlap,
           'validation': json.loads(json.dumps(val, default=str)), 'protected_unchanged': True, 'seconds': round(time.perf_counter() - t0, 1)}
    receipt.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: rec[k] for k in rec if k != 'validation'}, indent=1))
    print('validation', str(val)[:400])


if __name__ == '__main__':
    main()
