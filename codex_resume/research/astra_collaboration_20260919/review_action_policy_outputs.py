"""Read-only independent reconstruction; no fits or model selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
POKER = HERE.parents[2]
STUDY = POKER / 'goal_top5_20260919/foundation_pair/action_policy_llr'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    destination = HERE / 'action_policy_output_review.json'
    assert not destination.exists()
    receipt = json.loads((STUDY / 'RUN_RECEIPT.json').read_text())
    manifest = json.loads((STUDY / 'MANIFEST.json').read_text())
    assert receipt['manifest_sha256'] == sha(STUDY / 'MANIFEST.json')
    for path, expected in manifest['sources'].items():
        assert sha(Path(path)) == expected, path
    for path, expected in manifest['artifacts'].items():
        assert sha(STUDY / path) == expected, path
    reports = []
    for fold in receipt['folds']:
        outer = fold['outer']
        action_path = STUDY / f'action_predictions_outer{outer}.parquet'
        pair_path = STUDY / f'predictions_outer{outer}.parquet'
        assert sha(action_path) == fold['action_prediction_sha256']
        assert sha(pair_path) == fold['prediction_sha256']
        actions = pl.read_parquet(action_path)
        pairs = pl.read_parquet(pair_path).sort('pair_id')
        assert actions.select('pair_id','direction','hand_id','action_no').unique().height == actions.height
        held_tables = set(actions['table_id'])
        for fit in fold['fits']:
            path = STUDY / f'train_membership_outer{outer}_{fit["name"]}.parquet'
            assert sha(path) == fit['membership_sha256']
            member = pl.read_parquet(path)
            assert not held_tables.intersection(member['table_id'])
            weights = member['fit_weight'].to_numpy()
            assert np.isfinite(weights).all() and (weights > 0).all()
            assert np.isclose(weights.mean(), 1)
            mass = member.group_by('pair_id','direction').agg(pl.sum('fit_weight'))['fit_weight'].to_numpy()
            assert np.allclose(mass, mass[0])
        errors = {}
        for mode in ['real','shuffle']:
            probabilities = []
            for role in ['target','normal']:
                p = actions.select([f'{mode}_{role}_p{k}' for k in range(4)]).to_numpy()
                assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
                probabilities.append(p[np.arange(actions.height), actions['action_class'].to_numpy()])
            llr = np.log(np.maximum(probabilities[0], 1e-12)) - np.log(np.maximum(probabilities[1], 1e-12))
            rebuilt = actions.select('pair_id','direction').with_columns(pl.Series('value',llr)).group_by('pair_id','direction').agg(pl.mean('value')).group_by('pair_id').agg(pl.mean('value'),pl.len().alias('directions')).sort('pair_id')
            assert (rebuilt['directions'] == 2).all()
            assert rebuilt['pair_id'].to_list() == pairs['pair_id'].to_list()
            delta = float(np.max(np.abs(rebuilt['value'].to_numpy() - pairs[f'{mode}_llr'].to_numpy())))
            assert delta < 1e-12
            errors[mode] = delta
        reports.append({'outer':outer,'action_rows':actions.height,'pairs':pairs.height,'aggregation_max_errors':errors,'membership_disjoint':True})
    destination.write_text(json.dumps({'folds':reports,'new_fits':0,'quality_labels_read':False,'scope':'Prediction aggregation, saved probabilities and membership; no predictive-utility claim.'},indent=2)+'\n')
    print(destination)

if __name__ == '__main__':
    main()
