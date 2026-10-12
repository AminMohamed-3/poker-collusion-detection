import site
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib, json
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
POKER = HERE.parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ap(y, s, key):
    y = np.asarray(y, dtype=bool); s = np.asarray(s, dtype=np.float64)
    first = np.argsort(key, kind='stable'); order = first[np.argsort(-s[first], kind='stable')]
    yy = y[order]
    return float((np.cumsum(yy) / (np.arange(len(yy)) + 1))[yy].mean()) if yy.any() else 0.0
def contrast(frame, col):
    return float(frame.filter(pl.col('label') == 1)[col].mean() - frame.filter(pl.col('label') == 0)[col].mean())

def main():
    assert not (HERE / 'RESULTS.json').exists()
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    assert all(sha(Path(k)) == v for k, v in manifest['sources'].items())
    assert all(sha(HERE / k) == v for k, v in manifest['artifacts'].items())
    receipt = json.loads((HERE / 'RUN_RECEIPT.json').read_text())
    assert receipt['manifest_sha256'] == sha(HERE / 'MANIFEST.json') and sha(HERE/'REUSE_PROOF.json')==receipt['reuse_proof_sha256']
    parts = []
    for fold in receipt['folds']:
        path = HERE / f'predictions_outer{fold["outer"]}.parquet'
        assert sha(path) == fold['prediction_sha256'] and sha(HERE/f'models_outer{fold["outer"]}.joblib')==fold['model_sha256'] and sha(HERE/f'action_predictions_outer{fold["outer"]}.parquet')==fold['action_prediction_sha256']
        for fit in fold['fits']:assert sha(HERE/f'train_membership_outer{fold["outer"]}_{fit["name"]}.parquet')==fit['membership_sha256']
        parts.append(pl.read_parquet(path).with_columns(pl.lit(fold['outer']).alias('outer_fold')))
    data = pl.concat(parts)
    assert data.height == 1860 and data['pair_id'].n_unique() == 1860
    assert data.select(pl.all().exclude('pair_id', 'table_id').is_finite().all()).row(0) == (True, True, True)
    labels = pl.read_csv(POKER / 'data/development_labels.csv').select('pair_id', 'label', 'behavior_family')
    data = data.join(labels, on='pair_id', validate='1:1')
    real = contrast(data, 'real_llr'); shuffled = contrast(data, 'shuffle_llr'); primary = real - shuffled
    folds = {str(f): contrast(data.filter(pl.col('outer_fold') == f), 'real_llr') - contrast(data.filter(pl.col('outer_fold') == f), 'shuffle_llr') for f in [0, 1]}
    tables = sorted(data['table_id'].unique().to_list()); ta=data['table_id'].to_numpy(); by=[np.flatnonzero(ta==t) for t in tables]
    rng = np.random.default_rng(35935); boot = []
    for _ in range(1000):
        ix=np.concatenate([by[i] for i in rng.integers(0,len(by),len(by))]);sample=data[ix]
        boot.append(contrast(sample, 'real_llr') - contrast(sample, 'shuffle_llr'))
    syn=pl.DataFrame({'label':[1,0,1,0],'real_llr':[.4,.1,.2,.0],'shuffle_llr':[.2,.1,.1,.0]});ix=np.array([0,1,0,1,2,3]);literal=pl.concat([syn[:2],syn[:2],syn[2:]]);assert abs((contrast(syn[ix],'real_llr')-contrast(syn[ix],'shuffle_llr'))-(contrast(literal,'real_llr')-contrast(literal,'shuffle_llr')))<1e-15
    y = data['label'].to_numpy(); keys = data['pair_id'].to_numpy()
    result = {'real_target_minus_normal': real, 'shuffle_target_minus_normal': shuffled,
        'primary_real_minus_shuffle': primary, 'primary_by_fold': folds,
        'bootstrap95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        'real_llr_pair_ap_diagnostic': ap(y, data['real_llr'].to_numpy(), keys),
        'gate': primary > 0 and all(v > 0 for v in folds.values())}
    (HERE / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__': main()
