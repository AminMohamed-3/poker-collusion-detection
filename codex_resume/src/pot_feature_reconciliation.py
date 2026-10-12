"""Determine whether audit/production pot-name collision changes the reference."""
from audit_common import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT = HERE / 'experiments/pot_feature_reconciliation_20260912'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'receipt.json').exists()
    d, shared, extra, saved = load_current()
    raw, _, _, _ = load_fold(0)
    d = d.join(raw.select('pair_id', 'hand_id', pl.col('pot_bb').alias('raw_pot_bb')), on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    a, b = d['pot_bb'].to_numpy(), d['raw_pot_bb'].to_numpy()
    mapping = pl.DataFrame({'structural_pot_bb': a, 'raw_pot_bb': b}).unique().sort('structural_pot_bb')
    mapping.write_parquet(OUT / 'pot_mapping.parquet')
    hypotheses = {'identity': float(np.max(np.abs(a-b))), 'log1p': float(np.max(np.abs(np.log1p(a)-b))), 'clip100': float(np.max(np.abs(np.minimum(a,100)-b)))}
    pred = saved['incumbent'].to_numpy().copy()
    comparisons = []
    for fam in FAMILIES:
        ix = np.flatnonzero(((d['behavior_family'] == fam) & (d['oof_rank'] <= 20)).to_numpy())
        q = d[ix]
        cols = shared + (extra if fam != 'coordinated_isolation' else [])
        x = q.select(cols).to_numpy().astype(float)
        x[:, cols.index('pot_bb')] = q['raw_pot_bb'].to_numpy()
        y, pools = q['relevance'].to_numpy(), q['table_id'].to_numpy()
        for fold, (tr, te) in enumerate(GroupKFold(5).split(x, y, pools)):
            m = lgb.LGBMClassifier(**CLF, random_state=11).fit(x[tr], y[tr])
            p = m.predict_proba(x[te])[:,1] + 10
            pred[ix[te]] = p
            comparisons.append({'family': fam, 'fold': fold, 'max_abs_diff': float(np.max(np.abs(p - saved['incumbent'].to_numpy()[ix[te]])))})
    result = evaluate(d, {'incumbent': saved['incumbent'].to_numpy(), 'raw_pot_reference': pred})
    pl.DataFrame({'pair_id':d['pair_id'], 'hand_id':d['hand_id'], 'table_id':d['table_id'], 'raw_pot_reference':pred}).write_parquet(OUT / 'reference_scores.parquet')
    receipt = {'hypothesis_max_abs_diff': hypotheses, 'ranges': {'structural': [float(a.min()),float(a.max())], 'raw': [float(b.min()),float(b.max())]},
               'raw_reference_exact': bool(np.array_equal(pred,saved['incumbent'].to_numpy())), 'fold_controls': comparisons, 'metrics': result,
               'outer_evidence_evaluation': False, 'confirmation_aborted_before_any_stage2_model': True}
    write(OUT/'receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
