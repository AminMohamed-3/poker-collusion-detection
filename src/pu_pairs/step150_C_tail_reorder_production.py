"""Step 150 (Claude): re-order the risk values of evaluation pairs below .5 by lg(SHORT3)+lg(ALL5) (STEP131 arm P) on the public .90173 file.
No fits, no labels. Block rows, predicted_behavior and evidence stay byte-identical. Registration: outputs_v1_followups_20260914/prediction_step150.json.
Never uploads."""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
from evidence_aug.paths import sha, check_protected
from validation import validate_submission

OUT = POKER / 'Claude Version/outputs_v1_followups_20260914'
PREREG, PREREG_SHA = OUT / 'prediction_step150.json', 'f212ecd48beb4b51875aafd8b3703918fc105efd8a6b859dd508f90bcdef6004'
BASE = OUT / 'submission_v1_ci_untyped.csv'
VIEWS = POKER / 'Claude Version/experiments/viewdis_20260911/tail_views_eval_frame.parquet'
V13 = POKER / 'outputs_v13_coverage/submission_conditional_coverage.csv'
CSV = OUT / 'submission_v1_ci_untyped_tail_short3all5.csv'
RECEIPT = OUT / 'receipt_step150.json'


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main():
    t0 = time.perf_counter()
    assert sha(PREREG) == PREREG_SHA
    for k, v in json.loads(PREREG.read_text(encoding='utf-8'))['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], k
    prot = check_protected()
    assert not CSV.exists()

    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row')
    base = base.with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    tail = base.filter(pl.col('r') < .5)
    n_block = base.height - tail.height
    assert tail.height == 111631 and n_block == 909, (tail.height, n_block)

    v13 = pl.read_csv(V13, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    lin = tail.select('pair_id', 'r').join(v13, on='pair_id', how='left', validate='1:1')
    assert lin['risk_score'].null_count() == 0 and bool((lin['r'] == lin['risk_score']).all()), 'tail is not v13 risk'

    views = pl.read_parquet(VIEWS)
    t = tail.join(views, on='pair_id', how='left', validate='1:1', maintain_order='left')
    assert t['r_SHORT3'].null_count() == 0 and t['r_ALL5'].null_count() == 0
    t = t.with_columns(pl.Series('key', lg(t['r_SHORT3']) + lg(t['r_ALL5'])))
    key_ties = t.height - t['key'].n_unique()

    values = t.sort(['r', 'pair_id'], descending=[True, False])['risk_score']  # base strings, highest first
    order = t.sort(['key', 'pair_id'], descending=[True, False]).select('pair_id', 'key')
    assign = order.with_columns(pl.Series('new_risk', values.to_list()))
    moved = t.select('row', 'pair_id', 'risk_score', 'r', 'key').join(assign.select('pair_id', 'new_risk'), on='pair_id', validate='1:1', maintain_order='left')

    result = base.join(moved.select('pair_id', 'new_risk'), on='pair_id', how='left', maintain_order='left')
    result = result.with_columns(pl.coalesce('new_risk', 'risk_score').alias('risk_score')).sort('row').drop('row', 'r', 'new_risk')
    cols = [c for c in pl.read_csv(BASE, n_rows=1, infer_schema=False).columns]
    result = result.select(cols)
    base_cols = base.sort('row').select(cols)
    assert result.drop('risk_score').equals(base_cols.drop('risk_score'))
    assert sorted(result['risk_score'].to_list()) == sorted(base_cols['risk_score'].to_list())
    result.write_csv(CSV)

    a, b = BASE.read_bytes().split(b'\n'), CSV.read_bytes().split(b'\n')
    assert len(a) == len(b)
    diff = [k for k in range(len(a)) if a[k] != b[k]]
    tail_ids = set(tail['pair_id'].to_list())
    assert {a[k].split(b',')[0].decode() for k in diff} <= tail_ids
    for k in diff:
        x, y = a[k].split(b','), b[k].split(b',')
        assert x[0] == y[0] and x[2:] == y[2:], 'a non-risk cell changed'

    back = pl.read_csv(CSV, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    bt = back.join(tail.select('pair_id'), on='pair_id', how='semi')
    assert bt.height == tail.height and float(bt['risk_score'].max()) < .5
    assert back.join(tail.select('pair_id'), on='pair_id', how='anti').sort('pair_id').equals(
        base.select('pair_id', pl.col('r').alias('risk_score')).join(tail.select('pair_id'), on='pair_id', how='anti').sort('pair_id'))
    got = bt.join(order, on='pair_id').sort(['risk_score', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    want = order['pair_id'].to_list()
    mismatch = sum(g != w for g, w in zip(got, want))
    equal_value_groups = int((bt.group_by('risk_score').len()['len'] > 1).sum())

    from scipy.stats import spearmanr
    rank_old = t.sort(['r', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    overlap = {str(k): round(len(set(rank_old[:k]) & set(want[:k])) / k, 4) for k in (100, 500, 1000, 2000, 5000)}
    promoted = moved.filter(pl.col('new_risk').cast(pl.Float64) >= .1).join(t.filter(pl.col('r') < .1).select('pair_id'), on='pair_id', how='semi').height
    demoted = moved.filter(pl.col('new_risk').cast(pl.Float64) < .1).join(t.filter(pl.col('r') >= .1).select('pair_id'), on='pair_id', how='semi').height

    val = validate_submission(CSV, POKER / 'data/evaluation_pairs.csv', POKER / 'data/hands.parquet', POKER / 'data/seats.parquet', POKER / 'data/sample_submission.csv')
    assert check_protected() == prot
    rec = {'registration_sha256': PREREG_SHA, 'base_sha256': sha(BASE), 'output': str(CSV), 'output_sha256': sha(CSV),
           'tail_rows': tail.height, 'block_rows': n_block, 'tail_equals_v13_risk': True, 'key_ties_in_tail': key_ties,
           'rows_changed_vs_base': len(diff), 'only_tail_risk_cells_changed': True, 'value_multiset_unchanged': True,
           'order_mismatch_vs_key': mismatch, 'equal_value_groups_in_tail': equal_value_groups,
           'tail_spearman_old_vs_key': round(float(spearmanr(t['r'], t['key'])[0]), 4), 'tail_topk_overlap_old_vs_new': overlap,
           'pairs_moved_from_below_.1_to_at_least_.1': promoted, 'pairs_moved_from_at_least_.1_to_below_.1': demoted,
           'validation': json.loads(json.dumps(val, default=str)), 'protected_unchanged': True, 'seconds': round(time.perf_counter() - t0, 1)}
    RECEIPT.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: rec[k] for k in rec if k != 'validation'}, indent=1))
    print('validation', str(val)[:400])


if __name__ == '__main__':
    main()
