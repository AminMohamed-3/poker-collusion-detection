import site
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib, json, time, joblib
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
POKER = HERE.parents[2]
PRIV = HERE.parent / 'private_card_likelihood'
KEY = ['pair_id', 'direction', 'hand_id', 'action_no']
SEED = 35935

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def policy_weights(frame):
    counts = frame.group_by('pair_id', 'direction').len().rename({'len': '_pd_n'})
    out = frame.join(counts, on=['pair_id', 'direction'], validate='m:1')
    w = 1.0 / out['_pd_n'].to_numpy()
    w /= w.mean()
    totals = out.select('pair_id', 'direction').with_columns(pl.Series('_w', w)).group_by('pair_id', 'direction').agg(pl.sum('_w'))['_w'].to_numpy()
    assert np.allclose(totals, totals[0], atol=1e-12)
    return out, w

def policy_model():
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=15, min_child_samples=50,
        learning_rate=.1, random_state=SEED, n_jobs=2, verbosity=-1)
def physical_weights(frame):
    if '_w' in frame.columns: frame=frame.drop('_w')
    c=frame.group_by('hand_id','action_no','actor_id').len().rename({'len':'_n'})
    return frame.join(c,on=['hand_id','action_no','actor_id'],validate='m:1').with_columns((1/pl.col('_n')).alias('_w')).drop('_n')

def rotated_labels(pair_frame):
    rows = []
    unchanged_tables = 0
    for table, group in pair_frame.group_by('table_id', maintain_order=True):
        group = group.sort(pl.col('pair_id').map_elements(
            lambda x: hashlib.sha256(f'{SEED}|{x}'.encode()).hexdigest(), return_dtype=pl.String))
        labels = group['label'].to_numpy()
        rotated = np.roll(labels, 1)
        unchanged_tables += int(np.array_equal(labels, rotated))
        rows.append(group.select('pair_id', 'table_id').with_columns(pl.Series('shuffle_label', rotated)))
    return pl.concat(rows), unchanged_tables

def pair_score(frame, pt, pn):
    y = frame['action_class'].to_numpy(); ix = np.arange(frame.height)
    gain = np.log(np.maximum(pt[ix, y], 1e-12)) - np.log(np.maximum(pn[ix, y], 1e-12))
    out = frame.select('pair_id', 'direction').with_columns(pl.Series('llr', gain))
    out = out.group_by('pair_id', 'direction').agg(pl.mean('llr'))
    out = out.group_by('pair_id').agg(pl.mean('llr').alias('llr'), pl.col('direction').n_unique().alias('_n'))
    assert out['_n'].min() == 2
    return out.drop('_n')

def main():
    started = time.monotonic(); deadline = started + 900
    manifest_path = HERE / 'MANIFEST.json'
    release = HERE / 'RELEASE.json'
    assert release.exists() and json.loads(release.read_text())['manifest_sha256'] == sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert all(sha(Path(k)) == v for k, v in manifest['sources'].items())
    assert all(sha(HERE / k) == v for k, v in manifest['artifacts'].items())
    assert not (HERE / 'RUN_RECEIPT.json').exists() and not (HERE/'REUSE_PROOF.json').exists() and not any(HERE.glob('predictions_outer*.parquet')) and not any(HERE.glob('action_predictions_outer*.parquet')) and not any(HERE.glob('train_membership_outer*.parquet')) and not any(HERE.glob('models_outer*.joblib'))

    private_manifest = json.loads((PRIV / 'MANIFEST.json').read_text())
    inputs = private_manifest['inputs']
    action_paths = [POKER / k for k in private_manifest['sources']
                    if k.startswith('reports/conditional-action-experiment/action-cache/')]
    actions = pl.read_parquet(action_paths)
    tables = pl.read_parquet(PRIV / 'table_membership.parquet')
    row_membership = pl.read_parquet(PRIV/'row_membership.parquet').select(*KEY,'actor_id')
    capped = pl.read_parquet(PRIV / 'capped_training_keys.parquet').select(KEY)
    labels = pl.read_csv(POKER / 'data/development_labels.csv').select('pair_id', 'label')
    actions = actions.join(row_membership,on=KEY,validate='1:1').join(tables,on='table_id',validate='m:1')
    base_actions = actions
    actions = actions.join(labels, on='pair_id', validate='m:1')
    pair_labels = actions.select('pair_id', 'table_id', 'label').unique()
    receipts = []; reuse_proof=[]

    for outer in [0, 1]:
        package_path = PRIV / f'models_outer{outer}.joblib'
        package = joblib.load(package_path)
        assert package['inputs'] == inputs
        train = actions.filter(pl.col('outer_fold') != outer).join(capped, on=KEY, how='semi').with_row_index('_i')
        held = actions.filter(pl.col('outer_fold') == outer)
        assert train.height and held.height and not (set(train['table_id']) & set(held['table_id']))
        lineage_train=physical_weights(base_actions.filter(pl.col('outer_fold')!=outer).join(capped,on=KEY,how='semi'));lineage_held=base_actions.filter(pl.col('outer_fold')==outer)
        membership=package['membership'];assert membership['train_tables']==sorted(lineage_train['table_id'].unique().to_list()) and membership['held_tables']==sorted(lineage_held['table_id'].unique().to_list())
        assert membership['train_key_sha256']==hashlib.sha256(lineage_train.select(KEY).write_csv().encode()).hexdigest() and membership['held_key_sha256']==hashlib.sha256(lineage_held.select(KEY).write_csv().encode()).hexdigest()

        prediction = np.empty(train.height, dtype=np.float64)
        inner_col = f'inner_fold_outer{outer}'
        inner_receipt = []
        for inner in [0, 1]:
            scored = train.filter(pl.col(inner_col) == inner)
            fitted_tables = set(train.filter(pl.col(inner_col) != inner)['table_id'])
            assert not fitted_tables.intersection(scored['table_id'])
            prediction[scored['_i'].to_numpy()] = package['strength_inner'][inner].predict(scored.select(inputs).to_numpy())
            im=membership['inner'][inner];lscore=lineage_train.filter(pl.col(inner_col)==inner);lfit=physical_weights(lineage_train.filter(pl.col(inner_col)!=inner));assert im['score_rows']==lscore.height and im['train_rows']==lfit.height and im['score_key_sha256']==hashlib.sha256(lscore.select(KEY).write_csv().encode()).hexdigest() and im['train_key_sha256']==hashlib.sha256(lfit.select(KEY).write_csv().encode()).hexdigest()
            inner_receipt.append({'inner': inner, 'score_rows': scored.height,
                                  'score_tables': sorted(scored['table_id'].unique().to_list())})
        train_surprise = train['partner_strength'].to_numpy() - prediction
        held_surprise = held['partner_strength'].to_numpy() - package['strength_full'].predict(held.select(inputs).to_numpy())
        saved=pl.read_parquet(PRIV/f'predictions_outer{outer}.parquet').select(*KEY,'private_surprise');check=held.select(*KEY).with_columns(pl.Series('private_surprise',held_surprise)).join(saved,on=KEY,validate='1:1',suffix='_saved');assert check.height==held.height and np.array_equal(check['private_surprise'].to_numpy(),check['private_surprise_saved'].to_numpy())
        reuse_proof.append({'outer':outer,'train_key_sha256':membership['train_key_sha256'],'held_key_sha256':membership['held_key_sha256'],'held_surprise_exact':True})
        train = train.with_columns(pl.Series('private_surprise', train_surprise))
        held = held.with_columns(pl.Series('private_surprise', held_surprise))

        pair_train = pair_labels.filter(pl.col('table_id').is_in(train['table_id'].unique()))
        shuffled, unchanged = rotated_labels(pair_train)
        count_check=pair_train.group_by('table_id').agg(pl.sum('label').alias('a')).join(shuffled.group_by('table_id').agg(pl.sum('shuffle_label').alias('b')),on='table_id');assert count_check.filter(pl.col('a')!=pl.col('b')).is_empty()
        train = train.join(shuffled.select('pair_id', 'shuffle_label'), on='pair_id', validate='m:1')
        Xheld = held.select(inputs + ['private_surprise']).to_numpy()
        probabilities = {}
        model_bundle = {}
        fit_receipts = []
        for name, label_col, value in [('real_target', 'label', 1), ('real_normal', 'label', 0),
                                        ('shuffle_target', 'shuffle_label', 1), ('shuffle_normal', 'shuffle_label', 0)]:
            subset, weights = policy_weights(train.filter(pl.col(label_col) == value))
            assert subset.height and set(subset['action_class']) == {0, 1, 2, 3} and time.monotonic() < deadline
            model = policy_model().fit(subset.select(inputs + ['private_surprise']).to_numpy(),
                                       subset['action_class'].to_numpy(), sample_weight=weights)
            assert np.array_equal(model.classes_, np.arange(4))
            probabilities[name] = model.predict_proba(Xheld)
            q=probabilities[name];assert q.shape==(held.height,4) and np.isfinite(q).all() and (q>=0).all() and (q<=1).all() and np.allclose(q.sum(1),1,atol=1e-6)
            model_bundle[name] = model
            membership_path=HERE/f'train_membership_outer{outer}_{name}.parquet';subset.select(*KEY,'table_id').with_columns(pl.Series('fit_weight',weights)).write_parquet(membership_path)
            fit_receipts.append({'name': name, 'rows': subset.height,
                                 'pair_directions': subset.select('pair_id', 'direction').unique().height,
                                 'weight_mean': float(weights.mean()),'membership_sha256':sha(membership_path)})

        real = pair_score(held, probabilities['real_target'], probabilities['real_normal']).rename({'llr': 'real_llr'})
        control = pair_score(held, probabilities['shuffle_target'], probabilities['shuffle_normal']).rename({'llr': 'shuffle_llr'})
        output = held.select('pair_id', 'table_id').unique().join(real, on='pair_id', validate='1:1').join(control, on='pair_id', validate='1:1')
        action_output=held.select(*KEY,'table_id','action_class')
        for name,q in probabilities.items(): action_output=action_output.with_columns(*[pl.Series(f'{name}_p{k}',q[:,k]) for k in range(4)])
        action_path=HERE/f'action_predictions_outer{outer}.parquet';action_output.write_parquet(action_path)
        pred_path = HERE / f'predictions_outer{outer}.parquet'; output.write_parquet(pred_path)
        model_path = HERE / f'models_outer{outer}.joblib'; joblib.dump(model_bundle, model_path)
        replay=joblib.load(model_path)['real_target'].predict_proba(Xheld);assert np.array_equal(replay,probabilities['real_target'])
        receipts.append({'outer': outer, 'prediction_sha256': sha(pred_path), 'model_sha256': sha(model_path),
                         'action_prediction_sha256':sha(action_path),'unchanged_rotation_tables': unchanged, 'inner': inner_receipt, 'fits': fit_receipts,'real_target_replay_exact':True})

    assert time.monotonic() <= deadline
    (HERE/'REUSE_PROOF.json').write_text(json.dumps({'folds':reuse_proof},indent=2)+'\n')
    (HERE / 'RUN_RECEIPT.json').write_text(json.dumps({'manifest_sha256': sha(manifest_path),'reuse_proof_sha256':sha(HERE/'REUSE_PROOF.json'),
        'seconds': time.monotonic() - started, 'folds': receipts}, indent=2) + '\n')

if __name__ == '__main__': main()
