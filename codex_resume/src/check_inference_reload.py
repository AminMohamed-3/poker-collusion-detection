"""Truth-free replay of completed fold0 using serialized ensemble models."""
from ensemble_inference import *
from evidence_aug.fold import fold_roles
from evidence_aug.step10_chronology_fit import add_chronology
import time

SOURCE = HERE / 'experiments/method_ensemble_confirmation_v3_20260912'
OUT = HERE / 'experiments/production_inference_parity_20260912'


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'receipt.json').exists()
    saved = pl.read_parquet(SOURCE / 'scores/fold0-equal_methods.parquet').sort('pair_id', 'hand_id')
    s1 = pl.read_parquet(SOURCE / 'scores/fold0-stage1_only.parquet')
    data, features, _, _ = load_fold(0)
    data = data.join(saved.select('pair_id').unique(), on='pair_id', how='semi')
    data = add_chronology(data).join(s1.select('pair_id', 'hand_id', pl.col('score').alias('oof')), on=['pair_id', 'hand_id'], validate='1:1')
    structure = pl.read_parquet(CV / 'experiments/family_stage2_20260911/dev-structural-features.parquet')
    data = data.join(structure, on=['pair_id', 'hand_id'], validate='1:1').with_columns(pl.col('pot_bb_right').alias('pot_bb')).join(fold_roles(0), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
    data = data.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
    assert data.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id'))
    predicted = data['oof'].to_numpy().copy()
    for fam in FAMILIES:
        modeldir = SOURCE / 'models' / f'fold0-{fam}'
        recipe = json.loads((modeldir / 'recipe.json').read_text(encoding='utf-8'))
        ix = np.flatnonzero(((data['predicted_behavior'] == fam) & (data['oof_rank'] <= 20)).to_numpy())
        q = data[ix].with_columns(pl.col(recipe['columns']).fill_null(-999.)).drop('behavior_family')
        _, ep = predict_saved(q, modeldir)
        predicted[ix] = ep + 1000
        tail = ((data['predicted_behavior'] == fam) & (data['oof_rank'] > 20)).to_numpy()
        predicted[tail] = data['oof'].to_numpy()[tail] - 1e6
    diff = float(np.max(np.abs(predicted - saved['score'].to_numpy())))
    assert diff == 0., diff
    receipt = {'source_score_sha256': sha(SOURCE / 'scores/fold0-equal_methods.parquet'),
               'inference_code_sha256': sha(HERE / 'src/ensemble_inference.py'), 'queries': data['pair_id'].n_unique(),
               'rows': data.height, 'max_abs_prediction_difference': diff, 'held_evidence_truth_read': False,
               'seconds': time.perf_counter() - start}
    write(OUT / 'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
