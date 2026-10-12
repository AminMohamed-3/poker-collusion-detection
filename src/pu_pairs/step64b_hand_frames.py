"""Step 64b: per-hand PREFIX frames for an EXTENDED testbed population (folds 1-2), for the hand-level view of step 64c.
Step 61 cached per-hand frames (fold policy ca features + raw per-hand features, first 60 % of each table's hands) for the
prefix BLOCK only (prefix risk >= .5). The recall-tail question needs the same frames for pairs BELOW the block:
  population(fold) = labelled held pairs (development_labels, all of them) UNION prefix risk >= .1 UNION a label-free hash
  sample of TAIL_SAMPLE pairs with prefix risk < .1 (the null background the tail positives must beat).
Rows already in the step 61 cache are reused; the rest are rebuilt table by table with the fold policy (step 61 recipe).
Outputs: experiments/viewdis_20260911/{prefix_fold{k}_ext_raw.parquet, prefix_fold{k}_ext_ca.parquet, prefix_fold{k}_ext_population.parquet}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
import numpy as np
import polars as pl
import joblib
from pu_pairs.build_allpairs_fold import CONF, STYLE, EXP, fold_policy, base_features, v2_features, conditional_features
from pu_pairs.step61_prefix_rescoring import prefix_hands, VD

TAIL_SAMPLE = 1200
SAMPLE_SEED = 6401


def log(*a):
    print(*a, flush=True)


def population(fold):
    scores = pl.read_parquet(EXP / f'fold{fold}-prefix60' / 'scores.parquet').drop('label', 'behavior_family')
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    d = scores.join(labels, on='pair_id', how='left')
    lab = d.filter(pl.col('label').is_not_null())
    mid = d.filter(pl.col('label').is_null() & (pl.col('risk') >= .1))
    tail = d.filter(pl.col('label').is_null() & (pl.col('risk') < .1)).with_columns(pl.col('pair_id').hash(seed=SAMPLE_SEED).alias('_h')).sort('_h').head(TAIL_SAMPLE).drop('_h')
    pop = pl.concat([lab.with_columns(pl.lit('labelled').alias('why')), mid.with_columns(pl.lit('unlabelled_ge.1').alias('why')), tail.with_columns(pl.lit('tail_sample').alias('why'))]).sort('pair_id')
    assert pop['pair_id'].n_unique() == pop.height
    return pop


def build(fold, pop, t0):
    fr, fc = VD / f'prefix_fold{fold}_ext_raw.parquet', VD / f'prefix_fold{fold}_ext_ca.parquet'
    if fr.exists() and fc.exists():
        raw, ca = pl.read_parquet(fr), pl.read_parquet(fc)
        if set(raw['pair_id']) == set(pop['pair_id']):
            log(f'fold{fold}: cache hit'); return raw, ca
    raw0, ca0 = pl.read_parquet(VD / f'prefix_fold{fold}_raw.parquet'), pl.read_parquet(VD / f'prefix_fold{fold}_ca.parquet')
    have = set(raw0['pair_id']); todo = pop.filter(~pl.col('pair_id').is_in(have))
    log(f'fold{fold}: population {pop.height}, cached {pop.height - todo.height}, to build {todo.height}')
    policy = joblib.load(fold_policy(fold))
    caf = [c for c in pl.read_parquet(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', n_rows=1).columns if c.startswith('ca_')]
    counts = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(STYLE).iter_rows()}
    tables = sorted(set(todo['table_id']))
    hands_all = prefix_hands(tables)
    raws, cas = [raw0.filter(pl.col('pair_id').is_in(pop['pair_id']))], [ca0.filter(pl.col('pair_id').is_in(pop['pair_id']))]
    for i, t in enumerate(tables):
        h = hands_all.filter(pl.col('table_id') == t)
        p = todo.filter(pl.col('table_id') == t).select('pair_id', 'player_1', 'player_2', 'table_id')
        raw = base_features(h, p); raw = v2_features(raw, p)
        keys = raw.select('pair_id', 'hand_id', 'table_id').join(p.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        ca, _, _ = conditional_features(keys, h.select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind'), policy, counts, caf)
        raws.append(raw); cas.append(ca); gc.collect()
        if i % 10 == 0:
            log(json.dumps({'fold': fold, 'table': i, 'of': len(tables), 'pairs_in_table': p.height, 'seconds': round(time.time() - t0)}))
    raw = pl.concat(raws, how='vertical_relaxed').sort('pair_id', 'hand_id'); ca = pl.concat(cas, how='vertical_relaxed').sort('pair_id', 'hand_id')
    assert raw.select('pair_id', 'hand_id').equals(ca.select('pair_id', 'hand_id'))
    raw.write_parquet(fr); ca.write_parquet(fc)
    return raw, ca


def main():
    t0 = time.time(); res = {}
    for fold in (1, 2):
        pop = population(fold); pop.write_parquet(VD / f'prefix_fold{fold}_ext_population.parquet')
        raw, ca = build(fold, pop, t0)
        res[f'fold{fold}'] = {'population': pop.height, 'by_why': dict(pop.group_by('why').len().iter_rows()), 'pos': int((pop['label'] == 1).sum()),
                              'pos_below_.5': int(((pop['label'] == 1) & (pop['risk'] < .5)).sum()), 'hand_rows': raw.height, 'pairs_with_rows': raw['pair_id'].n_unique(), 'seconds': round(time.time() - t0)}
        log(json.dumps(res[f'fold{fold}']))
    write_json(VD / 'prefix_ext_frames.json', res)


if __name__ == '__main__':
    main()
