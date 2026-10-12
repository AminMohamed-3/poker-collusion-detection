"""Step 95 (Claude, 2026-09-13): pre-registered semi-supervised evidence student pilot.

Registration: experiments/semi_supervised_student_pilot_C_20260913/prediction-student-pilot.json
Modes, in order: selftest, controls, regen, phaseA, phaseB, exit.
Every output is write-once and inside the experiment directory; Codex Resume inputs are read-only.
"""
import os
for _key in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '2'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import argparse
import ast
import ctypes
import gc
import hashlib
import io
import json
import sys
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

BASE = Path('C:/Data/Projects/Money Making/work/poker')
RESUME = BASE / 'Codex Resume'
CLAUDE = BASE / 'Claude Version'
OUT = CLAUDE / 'experiments/semi_supervised_student_pilot_C_20260913'
REG = OUT / 'prediction-student-pilot.json'
REG_SHA = '9c6ac61167883cd7d8b59bf77c4fec8fceebf35a1c545bb79565117183a66315'
OLDPAIR = RESUME / 'experiments/semi_supervised_pair_yield_20260913'
OLD = RESUME / 'experiments/semi_supervised_evidence_yield_20260913'
FP = CLAUDE / 'experiments/semi_supervised_full_pair_yield_C_20260913'
FE = CLAUDE / 'experiments/semi_supervised_full_evidence_yield_C_20260913'
CACHE = RESUME / 'research/semi_supervised_full_features_20260913'
PREFLIGHT = RESUME / 'experiments/tabpfn3_relative_20260912/preflight.json'
TRUTH = BASE / 'data/development_evidence.csv'
STEP5 = CLAUDE / 'src/evidence_aug/step5_evaluate.py'
ARMS = ['original_only', 'repeated_real', 'pseudo']
COMPARATORS = ['original_only', 'repeated_real']
CHUNK = 10000
THRESHOLD = .95
TOLERANCE = 1e-10
BOOT_SEED, BOOT_REPS = 95, 2000
REPEAT_PREFIX = 'claude-ssl-repeat-v1'

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(BASE / 'src'))
import robust_pair_experiment as rpe  # noqa: E402  (its import sets thread variables to 2)
assert Path(rpe.__file__).resolve() == (BASE / 'src/robust_pair_experiment.py').resolve()
sys.path.insert(0, str(CLAUDE / 'src'))
from evidence_aug.paths import check_protected  # noqa: E402
MODEL_PARAMS, LABELS, pair_matrix = rpe.MODEL_PARAMS, rpe.LABELS, rpe.matrix
T0 = time.perf_counter()


class _Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def memory():
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    mb = 1024 * 1024
    return {'peak_working_set_mb': round(counters.PeakWorkingSetSize / mb, 1),
            'working_set_mb': round(counters.WorkingSetSize / mb, 1),
            'peak_pagefile_mb': round(counters.PeakPagefileUsage / mb, 1)}


def now():
    return datetime.now(timezone.utc).isoformat()


def elapsed():
    return round(time.perf_counter() - T0, 2)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, set):
        return [plain(v) for v in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def log(**values):
    print(json.dumps(plain({'seconds': elapsed(), **values, **memory()})), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def target(path):
    path = Path(path).resolve()
    assert OUT.resolve() in path.parents, f'refusing to write outside the experiment directory: {path}'
    assert not path.exists(), f'refusing to overwrite {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = target(path)
    path.write_text(json.dumps(plain(value), indent=2) + '\n', encoding='utf-8')
    return sha(path)


def write_parquet(path, frame):
    path = target(path)
    frame.write_parquet(path)
    return sha(path)


def save_booster(path, booster):
    path = target(path)
    booster.save_model(str(path))
    return lgb.Booster(model_file=str(path)), sha(path)


def load_registration():
    assert sha(REG) == REG_SHA, 'registration sha256 changed'
    reg = read(REG)
    inputs = reg['inputs_sha256']
    assert len(inputs) == 67, len(inputs)
    changed = sorted(p for p, s in inputs.items() if sha(p) != s)
    assert not changed, f'registered inputs changed: {changed}'
    return reg


def load_ap5():
    """Take ap5 verbatim from step5_evaluate without importing that module."""
    tree = ast.parse(STEP5.read_text(encoding='utf-8'))
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ap5']
    assert len(node) == 1
    namespace = {}
    exec(compile(ast.Module(body=node, type_ignores=[]), str(STEP5), 'exec'), namespace)
    return namespace['ap5']


def held_pools(part, fold):
    return {pool for pool, value in part['mapping'].items() if value == fold}


def ev_matrix(frame, columns):
    return np.nan_to_num(frame.select(columns).to_numpy().astype(np.float32), nan=0, posinf=0, neginf=0)


def verify_batches(static):
    receipts = []
    for index in range(85):
        path = CACHE / f'batch_{index:03}.json'
        receipt = read(path)
        assert receipt['exact_raw_parity'] is True and receipt['prior_sample_raw_parity'] is True
        assert sha(CACHE / receipt['hand_file']) == receipt['hand_sha256'], f'batch {index:03} changed'
        assert not set(receipt['pools']) & static['excluded']
        assert all(static['batch_of'][pool] == index for pool in receipt['pools'])
        receipts.append({'batch': index, 'receipt_sha256': sha(path), 'hand_sha256': receipt['hand_sha256']})
    return receipts


def load_hands(static, ids):
    """Hand rows for the requested pairs only, one batch at a time; never the whole cache."""
    ids = ids.select('pair_id').unique().sort('pair_id')
    need = static['meta'].join(ids, on='pair_id', how='semi')
    assert need.height == ids.height, 'pair ids missing from the full-cache metadata'
    assert not set(need['table_id']) & static['excluded']
    pieces = []
    for index in sorted({static['batch_of'][pool] for pool in need['table_id']}):
        scan = pl.scan_parquet(CACHE / f'batch_{index:03}-hands.parquet')
        pieces.append(scan.join(ids.lazy(), on='pair_id', how='semi').collect())
    hands = pl.concat(pieces).sort('pair_id', 'hand_id')
    assert hands.select('pair_id', 'hand_id').is_duplicated().sum() == 0
    assert set(hands['pair_id']) == set(ids['pair_id'])
    assert not set(hands['table_id']) & static['excluded']
    owner = hands.select('pair_id', 'table_id').unique().join(
        need.select('pair_id', pl.col('table_id').alias('meta_table')), on='pair_id')
    assert owner.height == ids.height and owner.filter(pl.col('table_id') != pl.col('meta_table')).height == 0
    return hands


def hand_pool_map(hand_ids):
    """Every pool in which the given physical hand ids occur (two string columns per batch)."""
    ids = pl.DataFrame({'hand_id': sorted(set(hand_ids))}, schema={'hand_id': pl.String})
    pieces = [pl.scan_parquet(CACHE / f'batch_{index:03}-hands.parquet').select('hand_id', 'table_id')
              .join(ids.lazy(), on='hand_id', how='semi').unique().collect() for index in range(85)]
    found = pl.concat(pieces).unique().sort('hand_id', 'table_id')
    assert set(found['hand_id']) == set(ids['hand_id'])
    return found


def held_hand_overlap(pool_map, hand_ids, held):
    used = pool_map.join(pl.DataFrame({'hand_id': sorted(set(hand_ids))}, schema={'hand_id': pl.String}), on='hand_id', how='semi')
    return used.filter(pl.col('table_id').is_in(sorted(held))).height


def players_of(static, pair_ids):
    rows = static['meta'].join(pl.DataFrame({'pair_id': sorted(set(pair_ids))}, schema={'pair_id': pl.String}),
                               on='pair_id', how='semi')
    return set(rows['player_1']) | set(rows['player_2'])


def held_players(static, held):
    rows = static['meta'].filter(pl.col('table_id').is_in(sorted(held)))
    return set(rows['player_1']) | set(rows['player_2'])


def load_static(reg):
    pre = read(PREFLIGHT)
    parts = pre['partitions']
    salts = list(reg['facts']['held_pools_per_fold'])
    assert [p['salt_index'] for p in parts] == [0, 1, 2] and [p['salt'] for p in parts] == salts
    pair_plan, evidence_plan = read(OLDPAIR / 'plan.json'), read(OLD / 'plan.json')
    assert pair_plan['salt']['mapping'] == parts[0]['mapping']
    assert pair_plan['reference_pools'] == pre['reference_pools']
    assert pair_plan['excluded_old_outer_pools'] == pre['excluded_old_outer_pools']
    assert pair_plan['pair_parameters'] == MODEL_PARAMS
    excluded, reference = set(pre['excluded_old_outer_pools']), set(pre['reference_pools'])
    assert len(excluded) == 228 and len(reference) == 65
    meta = pl.read_parquet(CACHE / 'selected-pairs.parquet').sort('pair_id')
    pools = set(meta['table_id'])
    assert meta.height == 66521 and meta['pair_id'].n_unique() == meta.height
    assert len(pools) == 169 and not pools & excluded
    for part in parts:
        mapped = set(part['mapping'])
        assert len(mapped) == 104 and not mapped & reference and mapped | reference == pools
        assert sorted(set(part['mapping'].values())) == [0, 1, 2, 3, 4]
    positives = meta.filter(pl.col('label') == 1).sort('pair_id')
    assert positives.height == reg['facts']['known_positive_queries'] == 162
    assert set(positives['table_id']) <= set(parts[0]['mapping'])
    assert dict(positives.group_by('behavior_family').len().rows()) == reg['facts']['positive_family_counts']
    assert meta.filter(pl.col('label') >= 0).height == reg['facts']['oldpair_known_rows_full']
    for part in parts:
        for fold in range(5):
            held = held_pools(part, fold)
            count = positives.filter(pl.col('table_id').is_in(sorted(held))).height
            assert len(held) == reg['facts']['held_pools_per_fold'][part['salt']][fold]
            assert count == reg['facts']['held_positive_queries_per_fold'][part['salt']][fold] == part['folds'][fold]['held_queries']
    fp_plan, fp_frames = read(FP / 'plan.json'), read(FP / 'frames-frozen.json')
    old_frames = read(OLDPAIR / 'frames-frozen.json')
    features = old_frames['features']
    assert len(features) == 765 and fp_frames['features'] == features
    digest = hashlib.sha256(json.dumps(features, separators=(',', ':')).encode('utf-8')).hexdigest()
    assert digest == fp_plan['old_frame_features_sha256']
    assert list(old_frames['outputs']) == ['full', 'v70', 'known_v40']
    views = evidence_plan['feature_views']
    hand_columns = list(pl.read_parquet_schema(CACHE / 'batch_000-hands.parquet'))
    assert hand_columns[:3] == ['pair_id', 'hand_id', 'table_id'] and views['rawv2_163'] == hand_columns[3:]
    assert list(views) == ['base107', 'rawv2_163']
    assert views['base107'] == [c for c in hand_columns[3:] if not c.startswith('v2_')]
    assert evidence_plan['boosting_rounds'] == 150
    feature_plan = read(CACHE / 'feature-plan.json')
    batch_of = {}
    for index, batch in enumerate(feature_plan['batch_pools']):
        for pool in batch:
            assert pool not in batch_of
            batch_of[pool] = index
    assert len(feature_plan['batch_pools']) == 85 and set(batch_of) == pools
    return {'pre': pre, 'parts': parts, 'meta': meta, 'positives': positives, 'pools': pools,
            'reference': reference, 'excluded': excluded, 'pair_features': features,
            'pair_parameters': pair_plan['pair_parameters'], 'views': views,
            'teacher_parameters': evidence_plan['teacher_parameters'], 'rounds': evidence_plan['boosting_rounds'],
            'batch_of': batch_of, 'oldpair_outputs': list(old_frames['outputs'])}


# ----------------------------------------------------------------------------------------------
# Pair teacher: Codex sample pair screen fit(), with the full-corpus query scored in chunks.
# ----------------------------------------------------------------------------------------------
def selected(scores):
    """FP screen selected() == Codex fit() candidate rule."""
    return scores.filter(
        (pl.col('risk_full') >= THRESHOLD) & (pl.col('risk_v70') >= THRESHOLD)
        & (pl.col('family_full') == pl.col('family_v70'))
    ).sort(['table_id', 'selection_score', 'pair_id'], descending=[False, True, False]).group_by(
        'table_id', maintain_order=True).head(2).sort('pair_id')


def risk_frame(keys, full_prob, short_prob):
    assert full_prob.shape == short_prob.shape and full_prob.shape[1] == 4
    risk = 1 - full_prob[:, 0]
    risk70 = 1 - short_prob[:, 0]
    family = np.argmax(full_prob[:, 1:], axis=1) + 1
    family70 = np.argmax(short_prob[:, 1:], axis=1) + 1
    return keys.select('pair_id', 'table_id').with_columns(
        pl.Series('risk_full', risk), pl.Series('risk_v70', risk70), pl.Series('family_full', family),
        pl.Series('family_v70', family70), pl.Series('selection_score', np.minimum(risk, risk70)))


def read_oldpair_frames(static):
    frames = {name: pl.read_parquet(OLDPAIR / f'{name}.parquet') for name in static['oldpair_outputs']}
    for frame in frames.values():
        assert not set(frame['table_id']) & static['excluded']
    return frames


def fit_pair_teacher(static, frames, held, dest):
    all_pools = set(frames['full']['table_id'])
    assert all_pools == static['pools']
    train_pools = all_pools - held
    train = pl.concat([x.filter(pl.col('table_id').is_in(sorted(train_pools)) & (pl.col('label') >= 0))
                       for x in frames.values()]).sort('pair_id')
    held_rows = frames['full'].filter(pl.col('table_id').is_in(sorted(held)))
    hp = set(held_rows['player_1']) | set(held_rows['player_2'])
    tp = set(train['player_1']) | set(train['player_2'])
    assert not hp & tp and not held & set(train['table_id'])
    assert not hp & players_of(static, train['pair_id']) and not held_players(static, held) & tp
    y = np.array([LABELS[f] for f in train['behavior_family']], dtype=np.int32)
    assert set(y) == {0, 1, 2, 3}
    X = pair_matrix(train, static['pair_features'])
    started = time.perf_counter()
    model = lgb.LGBMClassifier(**static['pair_parameters'])
    model.fit(X, y, sample_weight=np.full(train.height, 1 / 3))
    reloaded, model_sha = save_booster(dest / 'pair-teacher.txt', model.booster_)
    receipt = {'train_views_rows': train.height, 'train_known_pairs': train['pair_id'].n_unique(),
               'train_pools': len(train_pools), 'held_pools': len(held), 'held_pool_player_overlap': 0,
               'held_pool_rows_in_train': 0, 'fit_seconds': round(time.perf_counter() - started, 2),
               'model_sha256': model_sha}
    return model, reloaded, receipt


def score_unknown(static, fitted, helds):
    """Label -1 rows of training pools from FP full/v70, 10,000 rows at a time, all folds per chunk."""
    columns = ['pair_id', 'table_id', 'label', *static['pair_features']]
    store = {view: {fold: {'keys': [], 'prob': []} for fold in fitted} for view in ['full', 'v70']}
    for view in ['full', 'v70']:
        path = FP / f'{view}.parquet'
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        assert rows == 66521
        for offset in range(0, rows, CHUNK):
            chunk = pl.scan_parquet(path).select(columns).slice(offset, CHUNK).collect()
            assert chunk.height <= CHUNK and not set(chunk['table_id']) & static['excluded']
            for fold, (model, reloaded) in fitted.items():
                query = chunk.filter(~pl.col('table_id').is_in(sorted(helds[fold])) & (pl.col('label') < 0))
                if query.height == 0:
                    continue
                Q = pair_matrix(query, static['pair_features'])
                probability = model.predict_proba(Q)
                assert np.array_equal(probability, reloaded.predict(Q))
                store[view][fold]['keys'].append(query.select('pair_id', 'table_id'))
                store[view][fold]['prob'].append(probability)
            del chunk
            gc.collect()
        log(event='scored_unknown_view', view=view)
    outputs = {}
    for fold in fitted:
        keys = pl.concat(store['full'][fold]['keys'])
        keys70 = pl.concat(store['v70'][fold]['keys'])
        assert keys['pair_id'].equals(keys70['pair_id']) and keys['pair_id'].is_sorted()
        assert keys['pair_id'].n_unique() == keys.height
        outputs[fold] = risk_frame(keys, np.vstack(store['full'][fold]['prob']), np.vstack(store['v70'][fold]['prob']))
    return outputs


def query_boundary(static, held, scores):
    """FP full_boundary_check on the scored query rows."""
    query_rows = static['meta'].join(scores.select('pair_id'), on='pair_id', how='semi')
    assert not held & set(query_rows['table_id'])
    assert not held_players(static, held) & (set(query_rows['player_1']) | set(query_rows['player_2']))
    return {'query_unknown_pairs': scores.height, 'query_pools': query_rows['table_id'].n_unique(),
            'held_query_player_overlap': 0, 'held_pool_query_rows': 0}


def read_known_pair_frames(static):
    """Known (label >= 0) pair rows of FP full/v70, labels dropped, read 10,000 rows at a time."""
    columns = ['pair_id', 'table_id', 'label', *static['pair_features']]
    frames = {}
    for view in ['full', 'v70']:
        path, pieces = FP / f'{view}.parquet', []
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        for offset in range(0, rows, CHUNK):
            chunk = pl.scan_parquet(path).select(columns).slice(offset, CHUNK).collect()
            pieces.append(chunk.filter(pl.col('label') >= 0).drop('label'))
            del chunk
        frames[view] = pl.concat(pieces).sort('pair_id')
        gc.collect()
    assert frames['full']['pair_id'].equals(frames['v70']['pair_id']) and frames['full'].height == 787
    assert set(frames['full']['table_id']) <= static['pools']
    return frames


def held_known_predictions(static, booster, held, known):
    full = known['full'].filter(pl.col('table_id').is_in(sorted(held))).sort('pair_id')
    short = known['v70'].join(full.select('pair_id'), on='pair_id', how='semi').sort('pair_id')
    assert full['pair_id'].equals(short['pair_id'])
    return risk_frame(full, booster.predict(pair_matrix(full, static['pair_features'])),
                      booster.predict(pair_matrix(short, static['pair_features'])))


# ----------------------------------------------------------------------------------------------
# Evidence teachers and agreement: Codex sample evidence screen fit() and FE score().
# ----------------------------------------------------------------------------------------------
def training_rows(static, hands, held):
    known = hands.join(static['positives'].select('pair_id'), on='pair_id', how='semi')
    train = known.filter(~pl.col('table_id').is_in(sorted(held))).sort('pair_id', 'hand_id')
    held_ids = set(static['positives'].filter(pl.col('table_id').is_in(sorted(held)))['pair_id'])
    trainids = train.select('pair_id').unique()
    assert not set(trainids['pair_id']) & held_ids
    truth = pl.scan_csv(TRUTH).select('pair_id', 'hand_id').join(trainids.lazy(), on='pair_id', how='semi').with_columns(
        pl.lit(1.).alias('listed')).collect()
    assert truth.height == truth.unique().height and not set(truth['pair_id']) & held_ids
    keys = train.select('pair_id', 'hand_id')
    train = train.join(truth, on=['pair_id', 'hand_id'], how='left').with_columns(
        pl.col('listed').fill_null(0.), (1 / pl.len().over('pair_id')).alias('row_weight'))
    assert train.select('pair_id', 'hand_id').equals(keys), 'left join changed row order'
    assert set(train['pair_id']) == set(static['positives']['pair_id']) - held_ids
    info = {'train_known_positive_queries': trainids.height, 'training_hand_rows': train.height,
            'truth_rows_for_training_ids': truth.height, 'listed_training_rows': int(train['listed'].sum()),
            'held_pool_truth_rows_read': 0}
    return train, info


def fit_evidence_teachers(static, train, unknown, dest):
    y, w = train['listed'].to_numpy(), train['row_weight'].to_numpy()
    params = static['teacher_parameters']
    probs, receipts = [], []
    for name, features in static['views'].items():
        X, Q = ev_matrix(train, features), ev_matrix(unknown, features)
        model = lgb.train(params, lgb.Dataset(X, label=y, weight=w, feature_name=features), num_boost_round=static['rounds'])
        q = model.predict(Q) if Q.shape[0] else np.zeros(0)
        reloaded, model_sha = save_booster(dest / f'{name}.txt', model)
        assert np.array_equal(q, reloaded.predict(Q) if Q.shape[0] else np.zeros(0))
        probs.append(q)
        receipts.append({'name': name, 'features': len(features), 'sha256': model_sha, 'exact_reload': True})
    z = unknown.select('pair_id', 'hand_id', 'table_id').with_columns(
        pl.Series('p_base', probs[0], dtype=pl.Float64), pl.Series('p_v2', probs[1], dtype=pl.Float64),
        pl.Series('p_mean', (probs[0] + probs[1]) / 2, dtype=pl.Float64))
    return z, receipts


AGREEMENT_SCHEMA = {'pair_id': pl.String, 'table_id': pl.String, 'top5_overlap': pl.Int64,
                    'top_mean_probability': pl.Float64, 'accepted': pl.Boolean}


def agreement(z, chosen_ids, train_known_queries):
    rows, targets = [], []
    for group in z.partition_by('pair_id', maintain_order=True):
        a = group.sort(['p_base', 'hand_id'], descending=[True, False]).head(5)
        b = group.sort(['p_v2', 'hand_id'], descending=[True, False]).head(5)
        overlap = len(set(a['hand_id']) & set(b['hand_id']))
        mx = float(group['p_mean'].max())
        accept = overlap >= 4 and mx >= .5
        pid, pool = group['pair_id'][0], group['table_id'][0]
        rows.append({'pair_id': pid, 'table_id': pool, 'top5_overlap': overlap, 'top_mean_probability': mx, 'accepted': accept})
        if accept and pid in chosen_ids:
            q = np.clip(group['p_mean'].to_numpy(), .05, .95)
            q *= min(1., 5 / q.sum())
            assert q.sum() <= 5 + 1e-10
            targets.append(group.with_columns(pl.Series('soft_listed_target', q), pl.lit(.15 / group.height).alias('pseudo_row_weight')))
    all_diag = pl.DataFrame(rows, schema=AGREEMENT_SCHEMA)
    diag = all_diag.join(pl.DataFrame({'pair_id': sorted(chosen_ids)}, schema={'pair_id': pl.String}),
                         on='pair_id', how='semi').sort('pair_id')
    target = None
    if targets:
        target = pl.concat(targets)
        n = target['pair_id'].n_unique()
        factor = min(1., .15 * train_known_queries / (.15 * n))
        target = target.with_columns(pl.col('pseudo_row_weight') * factor)
        assert float(target.group_by('table_id').agg(pl.col('pseudo_row_weight').sum())['pseudo_row_weight'].max()) <= .30 + 1e-9
        assert target['pseudo_row_weight'].sum() <= .15 * train_known_queries + 1e-9
    return all_diag, diag, target


FAMILY_NAMES = {index: name for name, index in LABELS.items()}


def family_counts(frame, column='family_full'):
    return {FAMILY_NAMES[int(k)]: int(n) for k, n in frame.group_by(column).len().sort(column).rows()}


def fold_receipt(part, salt_index, fold, pair_receipt, chosen, info, diag, soft, z, models, hashes):
    accepted = diag.filter(pl.col('accepted'))

    def largest(key):
        if soft is None:
            return 0.0
        return float(soft.group_by(key).agg(pl.col('pseudo_row_weight').sum())['pseudo_row_weight'].max())

    return {'salt_index': salt_index, 'salt': part['salt'], 'fold': fold, 'pair_teacher': pair_receipt,
            'selected_pairs': chosen.height, 'selected_pools': chosen['table_id'].n_unique(),
            'selected_family_counts': family_counts(chosen), **info,
            'candidate_pairs': diag.height, 'scored_pairs': z['pair_id'].n_unique(),
            'top5_overlap_distribution': {int(k): int(n) for k, n in diag.group_by('top5_overlap').len().sort('top5_overlap').rows()},
            'agreement_ge4_pairs': diag.filter(pl.col('top5_overlap') >= 4).height,
            'top_mean_probability_ge05_pairs': diag.filter(pl.col('top_mean_probability') >= .5).height,
            'accepted_pairs': accepted.height, 'accepted_pools': accepted['table_id'].n_unique(),
            'accepted_pair_ids': accepted['pair_id'].to_list(),
            'accepted_predicted_families': family_counts(chosen.join(accepted.select('pair_id'), on='pair_id', how='semi')),
            'pseudo_rows': 0 if soft is None else soft.height,
            'total_pseudo_weight': 0.0 if soft is None else float(soft['pseudo_row_weight'].sum()),
            'max_pseudo_pair_weight': largest('pair_id'), 'max_pseudo_pool_weight': largest('table_id'),
            'evidence_models': models, 'held_pool_rows': 0, 'held_player_overlap': 0, 'held_hand_id_overlap': 0,
            **hashes, 'yield_gate_applied': False, 'held_quality_metrics_read': 0}


def regenerate_salt(static, salt_index, root, extra_ids=None):
    """Pair teachers, chunked selection, evidence teachers, agreement and soft targets for one salt."""
    part = static['parts'][salt_index]
    helds = {fold: held_pools(part, fold) for fold in range(5)}
    base = root / f'salt{salt_index}'
    frames = read_oldpair_frames(static)
    fitted, pair = {}, {}
    for fold in range(5):
        model, reloaded, receipt = fit_pair_teacher(static, frames, helds[fold], base / f'fold{fold}')
        fitted[fold], pair[fold] = (model, reloaded), receipt
        log(event='pair_teacher_fitted', salt=salt_index, fold=fold, rows=receipt['train_views_rows'])
    del frames
    gc.collect()
    scores = score_unknown(static, fitted, helds)
    del fitted
    gc.collect()
    chosen = {}
    for fold in range(5):
        dest = base / f'fold{fold}'
        pair[fold].update(query_boundary(static, helds[fold], scores[fold]))
        chosen[fold] = selected(scores[fold])
        pair[fold]['scores_sha256'] = write_parquet(dest / 'unknown-scores.parquet', scores[fold])
        pair[fold]['selected_sha256'] = write_parquet(dest / 'selected.parquet', chosen[fold])
    del scores
    gc.collect()
    ids = [static['positives'].select('pair_id')] + [chosen[fold].select('pair_id') for fold in range(5)]
    if extra_ids is not None:
        ids.extend(extra_ids[f].select('pair_id') for f in range(5))
    hands = load_hands(static, pl.concat(ids).unique())
    pool_map = hand_pool_map(hands['hand_id'])
    log(event='hands_loaded', salt=salt_index, rows=hands.height, pairs=hands['pair_id'].n_unique())
    receipts = []
    for fold in range(5):
        started, dest, held = time.perf_counter(), base / f'fold{fold}', helds[fold]
        train, info = training_rows(static, hands, held)
        candidates = chosen[fold].select('pair_id')
        if extra_ids is not None:
            candidates = pl.concat([candidates, extra_ids[fold].select('pair_id')]).unique()
        unknown = hands.join(candidates, on='pair_id', how='semi').sort('pair_id', 'hand_id')
        assert not held & set(unknown['table_id']) and not held & set(train['table_id'])
        hp = held_players(static, held)
        assert not hp & players_of(static, train['pair_id']) and not hp & players_of(static, unknown['pair_id'])
        assert held_hand_overlap(pool_map, list(train['hand_id']) + list(unknown['hand_id']), held) == 0
        z, models = fit_evidence_teachers(static, train, unknown, dest)
        _, diag, soft = agreement(z, set(chosen[fold]['pair_id']), info['train_known_positive_queries'])
        hashes = {'teacher_probabilities_sha256': write_parquet(dest / 'teacher-probabilities.parquet', z),
                  'agreement_sha256': write_parquet(dest / 'agreement.parquet', diag),
                  'soft_targets_sha256': None if soft is None else write_parquet(dest / 'soft-targets.parquet', soft)}
        receipt = fold_receipt(part, salt_index, fold, pair[fold], chosen[fold], info, diag, soft, z, models, hashes)
        receipt['seconds'] = round(time.perf_counter() - started, 2)
        write_json(dest / 'yield-receipt.json', receipt)
        receipts.append(receipt)
        log(event='evidence_fold_done', salt=salt_index, fold=fold, accepted=receipt['accepted_pairs'],
            pseudo_weight=receipt['total_pseudo_weight'])
    del hands, pool_map
    gc.collect()
    return receipts


# ----------------------------------------------------------------------------------------------
# Controls C1-C3 on salt a.
# ----------------------------------------------------------------------------------------------
def max_abs(a, b, columns):
    if a.height == 0 or not columns:
        return 0.0
    return max(float(np.max(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in columns)


def compare_frames(mine, saved, keys, probability_columns):
    out = {'rows_mine': mine.height, 'rows_saved': saved.height, 'columns_equal': mine.columns == saved.columns}
    out['keys_equal'] = bool(out['columns_equal'] and mine.height == saved.height
                             and mine.select(keys).equals(saved.select(keys)))
    if not out['keys_equal']:
        out.update(exact=False, max_abs_difference=None, other_columns_equal=False)
        return out
    others = [c for c in mine.columns if c not in keys and c not in probability_columns]
    out['exact'] = bool(mine.equals(saved) and all(np.array_equal(mine[c].to_numpy(), saved[c].to_numpy())
                                                    for c in probability_columns))
    out['max_abs_difference'] = max_abs(mine, saved, probability_columns)
    out['other_columns_equal'] = bool(mine.select(others).equals(saved.select(others)))
    return out


def within(result):
    return bool(result['keys_equal'] and result['max_abs_difference'] is not None
                and result['max_abs_difference'] <= TOLERANCE)


MISSING = {'exact': False, 'keys_equal': False, 'max_abs_difference': None, 'other_columns_equal': False}


def control_fold(reg, fold, receipt):
    base = OUT / f'controls/salt0/fold{fold}'
    risk = ['risk_full', 'risk_v70', 'selection_score']
    scores = compare_frames(pl.read_parquet(base / 'unknown-scores.parquet'),
                            pl.read_parquet(FP / f'fold{fold}/unknown-scores.parquet'), ['pair_id', 'table_id'], risk)
    mine_sel, saved_sel = pl.read_parquet(base / 'selected.parquet'), pl.read_parquet(FP / f'fold{fold}/selected.parquet')
    selection = compare_frames(mine_sel, saved_sel, ['table_id', 'pair_id'], risk)
    selection['pair_set_equal'] = set(mine_sel['pair_id']) == set(saved_sel['pair_id'])
    codex = read(OLDPAIR / f'fold{fold}/fit.json')
    pt = receipt['pair_teacher']
    c1 = {'scores': scores, 'selection': selection,
          'model_file_identical_to_codex': pt['model_sha256'] == codex['model_sha256'],
          'train_rows_equal_codex': pt['train_views_rows'] == codex['train_views_rows'] and pt['train_known_pairs'] == codex['train_known_pairs']}
    c1['status'] = 'PASS' if scores['exact'] and selection['exact'] else (
        'PASS_WITH_TOLERANCE' if within(scores) and selection['pair_set_equal'] else 'STOP')
    probs = compare_frames(pl.read_parquet(base / 'teacher-probabilities.parquet'),
                           pl.read_parquet(FE / f'fold{fold}/teacher-probabilities-with-parity.parquet'),
                           ['pair_id', 'hand_id', 'table_id'], ['p_base', 'p_v2', 'p_mean'])
    agree = compare_frames(pl.read_parquet(base / 'agreement.parquet'), pl.read_parquet(FE / f'fold{fold}/agreement.parquet'),
                           ['pair_id', 'table_id'], ['top_mean_probability'])
    saved_soft = FE / f'fold{fold}/soft-targets.parquet'
    assert saved_soft.exists()
    soft = MISSING if not (base / 'soft-targets.parquet').exists() else compare_frames(
        pl.read_parquet(base / 'soft-targets.parquet'), pl.read_parquet(saved_soft), ['pair_id', 'hand_id', 'table_id'],
        ['p_base', 'p_v2', 'p_mean', 'soft_listed_target', 'pseudo_row_weight'])
    old = read(OLD / f'fold{fold}/fit.json')
    flags = bool(agree['keys_equal'] and agree['other_columns_equal'])
    c2 = {'probabilities': probs, 'agreement': agree, 'acceptance_flags_identical': flags, 'soft_targets': soft,
          'model_files_identical_to_codex': {m['name']: m['sha256'] == n['sha256'] for m, n in zip(receipt['evidence_models'], old['models'])},
          'training_hand_rows_equal_codex': receipt['training_hand_rows'] == old['training_hand_rows'],
          'train_known_positive_queries_equal_codex': receipt['train_known_positive_queries'] == old['train_known_positive_queries']}
    c2['status'] = 'PASS' if probs['exact'] and agree['exact'] and soft['exact'] else (
        'PASS_WITH_TOLERANCE' if within(probs) and flags and within(soft) else 'STOP')
    fact = reg['facts']['salt_a_pseudo_corpus'][fold]
    counts = {'pseudo_pairs': receipt['accepted_pairs'] == fact['pseudo_pairs'],
              'pseudo_pools': receipt['accepted_pools'] == fact['pseudo_pools'], 'rows': receipt['pseudo_rows'] == fact['rows'],
              'total_pseudo_weight': abs(receipt['total_pseudo_weight'] - fact['total_pseudo_weight']) <= 1e-9}
    c3 = {'selection_exact': selection['exact'], 'selection_pair_set_equal': selection['pair_set_equal'],
          'agreement_exact': agree['exact'], 'soft_targets_exact': soft['exact'], 'registered_counts_match': counts}
    c3['status'] = 'PASS' if selection['exact'] and agree['exact'] and soft['exact'] and all(counts.values()) else (
        'PASS_WITH_TOLERANCE' if 'STOP' not in (c1['status'], c2['status']) and selection['pair_set_equal']
        and within(agree) and flags and within(soft) and all(counts.values()) else 'STOP')
    return {'fold': fold, 'C1': c1, 'C2': c2, 'C3': c3}


def worst(statuses):
    return 'STOP' if 'STOP' in statuses else ('PASS_WITH_TOLERANCE' if 'PASS_WITH_TOLERANCE' in statuses else 'PASS')


def mode_controls():
    reg = load_registration()
    protected_before = check_protected()
    static = load_static(reg)
    assert not (OUT / 'stop-record.json').exists() and not (OUT / 'controls').exists()
    batches = verify_batches(static)
    log(event='entry_verified', mode='controls', registered_inputs=67, protected=len(protected_before))
    extra = {fold: pl.read_parquet(OLDPAIR / f'fold{fold}/selected.parquet').select('pair_id') for fold in range(5)}
    receipts = regenerate_salt(static, 0, OUT / 'controls', extra_ids=extra)
    folds = [control_fold(reg, fold, receipts[fold]) for fold in range(5)]
    status = {c: worst([f[c]['status'] for f in folds]) for c in ['C1', 'C2', 'C3']}
    overall = worst(list(status.values()))
    protected_after = check_protected()
    assert protected_after == protected_before, 'protected files changed during controls'
    result = {'created_utc': now(), 'registration_sha256': REG_SHA, 'script_sha256': sha(Path(__file__)),
              'status': overall, 'per_control': status,
              'folds': folds, 'yield_receipts': receipts, 'batches': batches,
              'protected_before': protected_before, 'protected_after': protected_after, 'C7_controls_phase': True,
              'student_fits': 0, 'quality_reads': 0, 'seconds': elapsed(), 'memory': memory()}
    write_json(OUT / 'controls/controls-result.json', result)
    if overall == 'STOP':
        write_json(OUT / 'stop-record.json', {
            'created_utc': now(), 'status': 'REGENERATION_NOT_FAITHFUL',
            'decision_rule': reg['decision_rules']['stop_at_C1_to_C3'], 'per_control': status,
            'evidence': folds, 'controls_result_sha256': sha(OUT / 'controls/controls-result.json'),
            'student_fits': 0, 'quality_reads': 0, 'script_sha256': sha(Path(__file__)),
            'deviations': [d for d in DEVIATIONS if d['item'] in STOP_DEVIATIONS]})
    log(event='controls_done', status=overall, per_control=status)


def mode_regen():
    reg = load_registration()
    protected_before = check_protected()
    static = load_static(reg)
    controls_path = OUT / 'controls/controls-result.json'
    assert controls_path.exists() and not (OUT / 'stop-record.json').exists() and not (OUT / 'regen').exists()
    controls = read(controls_path)
    assert controls['status'] in ('PASS', 'PASS_WITH_TOLERANCE') and controls['protected_before'] == protected_before
    batches = verify_batches(static)
    log(event='entry_verified', mode='regen')
    yields = {}
    for salt_index in [1, 2]:
        yields[salt_index] = regenerate_salt(static, salt_index, OUT / 'regen')
        gc.collect()
    protected_after = check_protected()
    assert protected_after == protected_before, 'protected files changed during regen'
    files = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted((OUT / 'regen').rglob('*')) if p.is_file()}
    keep = ['salt', 'fold', 'selected_pairs', 'selected_pools', 'accepted_pairs', 'accepted_pools',
            'total_pseudo_weight', 'pseudo_rows', 'accepted_predicted_families', 'top5_overlap_distribution']
    distinct = {static['parts'][s]['salt']: len({p for r in rs for p in r['accepted_pair_ids']}) for s, rs in yields.items()}
    write_json(OUT / 'regen/regen-frozen.json', {
        'created_utc': now(), 'registration_sha256': REG_SHA, 'script_sha256': sha(Path(__file__)),
        'controls_result_sha256': sha(controls_path),
        'files_sha256': files, 'yield_per_cell': [{k: r[k] for k in keep} for rs in yields.values() for r in rs],
        'distinct_accepted_pairs_per_salt': distinct, 'yield_gate_applied': False, 'batches': batches,
        'protected_before': protected_before, 'protected_after': protected_after, 'C7_regen_phase': True,
        'student_fits': 0, 'quality_reads': 0, 'seconds': elapsed(), 'memory': memory()})
    log(event='regen_done', distinct_accepted_pairs=distinct)


# ----------------------------------------------------------------------------------------------
# Phase A helpers: repeat matching, student fits, boundaries and weights.
# ----------------------------------------------------------------------------------------------
MATCH_SCHEMA = {'pseudo_pair_id': pl.String, 'pseudo_table_id': pl.String, 'pseudo_total_weight': pl.Float64,
                'copy_pair_id': pl.String, 'copy_table_id': pl.String, 'same_pool': pl.Boolean,
                'reused': pl.Boolean, 'hash': pl.String}


def repeat_matching(salt_index, fold, pseudo_pairs, train_queries):
    """pseudo_pairs: pair_id, table_id, weight. train_queries: pair_id, table_id of training known positives."""
    def digest(pseudo, candidate):
        return hashlib.sha256(f'{REPEAT_PREFIX}:{salt_index}:{fold}:{pseudo}:{candidate}'.encode('utf-8')).hexdigest()
    queries = sorted(train_queries.select('pair_id', 'table_id').rows())
    pool_of = dict(queries)
    used, matches = set(), []
    for pid, pool, weight in sorted(pseudo_pairs.select('pair_id', 'table_id', 'weight').rows()):
        same = [q for q, t in queries if t == pool and q not in used]
        rest = [q for q, _ in queries if q not in used]
        pool_list = same or rest or [q for q, _ in queries]
        choice = min(pool_list, key=lambda c: digest(pid, c))
        matches.append({'pseudo_pair_id': pid, 'pseudo_table_id': pool, 'pseudo_total_weight': weight,
                        'copy_pair_id': choice, 'copy_table_id': pool_of[choice], 'same_pool': bool(same),
                        'reused': not same and not rest, 'hash': digest(pid, choice)})
        used.add(choice)
    return pl.DataFrame(matches, schema=MATCH_SCHEMA)


def repeated_rows(train, matches):
    pieces = []
    for m in matches.iter_rows(named=True):
        rows = train.filter(pl.col('pair_id') == m['copy_pair_id'])
        assert rows.height > 0
        pieces.append(rows.with_columns(pl.lit(m['pseudo_total_weight'] / rows.height).alias('row_weight')))
    if not pieces:
        return train.clear()
    return pl.concat(pieces).sort('pair_id', 'hand_id', maintain_order=True)


def fit_student(static, parts, dest, name):
    """parts: [(frame, label column, weight column)] in the registered row order."""
    features = static['views']['rawv2_163']
    X = np.vstack([ev_matrix(frame, features) for frame, _, _ in parts])
    y = np.concatenate([frame[label].to_numpy().astype(np.float64) for frame, label, _ in parts])
    w = np.concatenate([frame[weight].to_numpy().astype(np.float64) for frame, _, weight in parts])
    model = lgb.train(static['teacher_parameters'], lgb.Dataset(X, label=y, weight=w, feature_name=features),
                      num_boost_round=static['rounds'])
    reloaded, model_sha = save_booster(dest / f'student-{name}.txt', model)
    return model, reloaded, {'rows': int(X.shape[0]), 'total_weight': float(w.sum()), 'model_sha256': model_sha}


def predict_exact(model, reloaded, frame, features):
    Q = ev_matrix(frame, features)
    p = model.predict(Q)
    assert np.array_equal(p, reloaded.predict(Q))
    return p


def row_boundary(static, pool_map, held, role, frame):
    if frame.height == 0:
        return {'role': role, 'rows': 0}
    hp = held_players(static, held)
    assert not held & set(frame['table_id']), role
    assert not set(frame['table_id']) & static['excluded'], role
    assert not hp & players_of(static, frame['pair_id']), role
    assert held_hand_overlap(pool_map, frame['hand_id'], held) == 0, role
    return {'role': role, 'rows': frame.height, 'held_pool_rows': 0, 'held_player_overlap': 0, 'held_hand_id_overlap': 0}


def known_pair_hand_ids(static):
    """(pair_id, hand_id, table_id) of every known pair (label >= 0); id columns only, one batch at a time."""
    ids = static['meta'].filter(pl.col('label') >= 0).select('pair_id')
    pieces = [pl.scan_parquet(CACHE / f'batch_{index:03}-hands.parquet').select('pair_id', 'hand_id', 'table_id')
              .join(ids.lazy(), on='pair_id', how='semi').collect() for index in range(85)]
    found = pl.concat(pieces)
    assert set(found['pair_id']) == set(ids['pair_id']) and not set(found['table_id']) & static['excluded']
    return found


def pair_teacher_boundary(static, shared, held):
    """Training rows of the cell pair teacher (OLDPAIR full, v70, known_v40; label >= 0; training pools)."""
    train = pl.concat([x.filter(~pl.col('table_id').is_in(sorted(held)) & (pl.col('label') >= 0))
                       for x in shared['light'].values()])
    hp = held_players(static, held)
    assert not held & set(train['table_id']) and not hp & (set(train['player_1']) | set(train['player_2']))
    hands = shared['known_hands'].join(train.select('pair_id').unique(), on='pair_id', how='semi')
    assert set(hands['pair_id']) == set(train['pair_id'])
    assert held_hand_overlap(shared['pool_map'], hands['hand_id'], held) == 0
    return {'role': 'pair_teacher_training_rows', 'rows': train.height, 'pairs': train['pair_id'].n_unique(),
            'hand_ids_checked': hands.height, 'held_pool_rows': 0, 'held_player_overlap': 0, 'held_hand_id_overlap': 0}


def cell_weights(soft, repeated, matches, train_queries):
    out = {'train_known_positive_queries': train_queries, 'global_cap': .15 * train_queries}
    if soft is None or soft.height == 0:
        assert repeated.height == 0
        return {**out, 'pseudo_pairs': 0, 'total_pseudo_weight': 0.0, 'repeated_total_weight': 0.0, 'passes': True}
    pair = soft.group_by('pair_id').agg(pl.col('pseudo_row_weight').sum())
    pool = soft.group_by('table_id').agg(pl.col('pseudo_row_weight').sum())
    copies = repeated.group_by('pair_id').agg(pl.col('row_weight').sum())
    out.update(pseudo_pairs=pair.height, max_pair_weight=float(pair['pseudo_row_weight'].max()),
               max_pool_weight=float(pool['pseudo_row_weight'].max()), total_pseudo_weight=float(soft['pseudo_row_weight'].sum()),
               repeated_total_weight=float(repeated['row_weight'].sum()))
    matched = matches.join(pair, left_on='pseudo_pair_id', right_on='pair_id')
    assert matched.height == matches.height == pair.height
    assert (matched['pseudo_total_weight'] - matched['pseudo_row_weight']).abs().max() <= 1e-12
    if not matches['reused'].any():
        joined = matches.join(copies, left_on='copy_pair_id', right_on='pair_id')
        assert (joined['pseudo_total_weight'] - joined['row_weight']).abs().max() <= 1e-9
    assert out['max_pair_weight'] <= .15 + 1e-9 and out['max_pool_weight'] <= .30 + 1e-9
    assert out['total_pseudo_weight'] <= out['global_cap'] + 1e-9
    assert abs(out['total_pseudo_weight'] - out['repeated_total_weight']) <= 1e-9
    return {**out, 'passes': True}


def cell_corpus(salt_index, fold):
    if salt_index == 0:
        return {'soft': FE / f'fold{fold}/soft-targets.parquet', 'selected': FP / f'fold{fold}/selected.parquet',
                'agreement': FE / f'fold{fold}/agreement.parquet', 'teacher': OLD / f'fold{fold}/rawv2_163.txt',
                'pair_teacher': OLDPAIR / f'fold{fold}/pair-teacher.txt'}
    base = OUT / f'regen/salt{salt_index}/fold{fold}'
    return {'soft': base / 'soft-targets.parquet', 'selected': base / 'selected.parquet', 'agreement': base / 'agreement.parquet',
            'teacher': base / 'rawv2_163.txt', 'pair_teacher': base / 'pair-teacher.txt'}


def verify_upstream(protected_before):
    controls_path, regen_path = OUT / 'controls/controls-result.json', OUT / 'regen/regen-frozen.json'
    assert not (OUT / 'stop-record.json').exists()
    controls, regen = read(controls_path), read(regen_path)
    assert controls['status'] in ('PASS', 'PASS_WITH_TOLERANCE') and controls['protected_before'] == protected_before
    assert regen['controls_result_sha256'] == sha(controls_path)
    changed = sorted(p for p, s in regen['files_sha256'].items() if sha(OUT / p) != s)
    assert not changed, f'regenerated corpus changed: {changed}'
    return controls, regen, {'controls_result_sha256': sha(controls_path), 'regen_frozen_sha256': sha(regen_path)}


def mode_phaseA():
    reg = load_registration()
    protected_before = check_protected()
    static = load_static(reg)
    assert not (OUT / 'phaseA').exists() and not (OUT / 'predictions-frozen.json').exists()
    _, _, upstream = verify_upstream(protected_before)
    batches = verify_batches(static)
    log(event='entry_verified', mode='phaseA')
    corpus = {(s, f): cell_corpus(s, f) for s in range(3) for f in range(5)}
    softs = {cell: (pl.read_parquet(c['soft']).sort('pair_id', 'hand_id') if c['soft'].exists() else None)
             for cell, c in corpus.items()}
    ids = [static['positives'].select('pair_id')] + [s.select('pair_id') for s in softs.values() if s is not None]
    shared = {'hands': load_hands(static, pl.concat(ids).unique()), 'known_hands': known_pair_hand_ids(static)}
    shared['pool_map'] = hand_pool_map(pl.concat([shared['hands']['hand_id'], shared['known_hands']['hand_id']]))
    shared['light'] = {n: pl.read_parquet(OLDPAIR / f'{n}.parquet', columns=['pair_id', 'table_id', 'player_1', 'player_2', 'label'])
                       for n in static['oldpair_outputs']}
    shared['known'] = read_known_pair_frames(static)
    log(event='phaseA_inputs_loaded', hand_rows=shared['hands'].height)
    files, cells = {}, []
    for (salt_index, fold), c in corpus.items():
        cell, cell_files = run_cell(reg, static, shared, salt_index, fold, c, softs[(salt_index, fold)])
        files.update(cell_files)
        cells.append(cell)
        log(event='cell_done', salt=salt_index, fold=fold, pseudo_pairs=cell['pseudo_pairs'], C4=cell['C4_original_equals_teacher'])
    protected_after = check_protected()
    assert protected_after == protected_before, 'protected files changed during phase A'
    keep = ['salt', 'fold', 'held_queries', 'pseudo_pairs', 'pseudo_pools', 'pseudo_predicted_families',
            'unmatched_repeat_pools', 'repeat_reuse', 'C4_original_equals_teacher']
    write_json(OUT / 'predictions-frozen.json', {
        'created_utc': now(), 'registration_sha256': REG_SHA, 'script_sha256': sha(Path(__file__)),
        **upstream, 'files_sha256': files,
        'score_files': sorted(p for p in files if p.endswith('held-hand-scores.parquet')),
        'cells': [{k: x[k] for k in keep} for x in cells],
        'C4_all_exact': all(x['C4_original_equals_teacher'] for x in cells), 'C5_all_boundaries': True,
        'C6_all_weights': all(x['C6_weights']['passes'] for x in cells), 'excluded_rows_loaded': 0,
        'held_pool_truth_rows_read': 0, 'batches': batches, 'protected_before': protected_before,
        'protected_after': protected_after, 'C7_phaseA': True, 'quality_reads': 0,
        'seconds': elapsed(), 'memory': memory()})
    log(event='phaseA_done')


def run_cell(reg, static, shared, salt_index, fold, c, soft):
    started, part = time.perf_counter(), static['parts'][salt_index]
    held, rel = held_pools(part, fold), f'phaseA/salt{salt_index}/fold{fold}'
    dest, hands, features = OUT / rel, shared['hands'], static['views']['rawv2_163']
    train, info = training_rows(static, hands, held)
    queries = static['positives'].filter(pl.col('table_id').is_in(sorted(held)))
    heldq = hands.join(queries.select('pair_id'), on='pair_id', how='semi').sort('pair_id', 'hand_id')
    assert heldq['pair_id'].n_unique() == queries.height == reg['facts']['held_positive_queries_per_fold'][part['salt']][fold]
    if soft is None:
        pseudo = train.clear()
        pairs = pl.DataFrame(schema={'pair_id': pl.String, 'table_id': pl.String, 'weight': pl.Float64})
    else:
        pseudo = soft.select('pair_id', 'hand_id', 'table_id', 'soft_listed_target', 'pseudo_row_weight').join(
            hands.drop('table_id'), on=['pair_id', 'hand_id'], how='inner').sort('pair_id', 'hand_id')
        assert pseudo.height == soft.height
        pairs = soft.group_by('pair_id', 'table_id').agg(pl.col('pseudo_row_weight').sum().alias('weight')).sort('pair_id')
    matches = repeat_matching(salt_index, fold, pairs, train.select('pair_id', 'table_id').unique())
    repeated = repeated_rows(train, matches)
    weights = cell_weights(soft, repeated, matches, info['train_known_positive_queries'])
    boundaries = [pair_teacher_boundary(static, shared, held),
                  row_boundary(static, shared['pool_map'], held, 'evidence_teacher_and_real_rows', train),
                  row_boundary(static, shared['pool_map'], held, 'pseudo_rows', pseudo),
                  row_boundary(static, shared['pool_map'], held, 'repeated_rows', repeated)]
    real = (train, 'listed', 'row_weight')
    arms = {'original_only': [real], 'repeated_real': [real, (repeated, 'listed', 'row_weight')],
            'pseudo': [real] + ([] if soft is None else [(pseudo, 'soft_listed_target', 'pseudo_row_weight')])}
    scores, fits, files = heldq.select('pair_id', 'hand_id', 'table_id'), {}, {}
    for arm, rows in arms.items():
        model, reloaded, fits[arm] = fit_student(static, rows, dest, arm)
        files[f'{rel}/student-{arm}.txt'] = fits[arm]['model_sha256']
        scores = scores.with_columns(pl.Series(arm, predict_exact(model, reloaded, heldq, features)))
    teacher = lgb.Booster(model_file=str(c['teacher']))
    scores = scores.with_columns(pl.Series('teacher_rawv2_163', teacher.predict(ev_matrix(heldq, features))))
    c4 = bool(np.array_equal(scores['original_only'].to_numpy(), scores['teacher_rawv2_163'].to_numpy()))
    assert c4, f'C4 failed in salt {salt_index} fold {fold}'
    held_known = held_known_predictions(static, lgb.Booster(model_file=str(c['pair_teacher'])), held, shared['known'])
    files[f'{rel}/held-hand-scores.parquet'] = write_parquet(dest / 'held-hand-scores.parquet', scores)
    files[f'{rel}/held-known-pair-teacher.parquet'] = write_parquet(dest / 'held-known-pair-teacher.parquet', held_known)
    files[f'{rel}/repeat-matches.parquet'] = write_parquet(dest / 'repeat-matches.parquet', matches)
    chosen = pl.read_parquet(c['selected'])
    cell = {'salt_index': salt_index, 'salt': part['salt'], 'fold': fold, 'held_pools': len(held),
            'held_queries': queries.height, 'held_query_hands': heldq.height, **info,
            'pseudo_pairs': pairs.height, 'pseudo_pools': pairs['table_id'].n_unique(),
            'pseudo_predicted_families': family_counts(chosen.join(pairs.select('pair_id'), on='pair_id', how='semi')),
            'pseudo_rows': pseudo.height, 'repeated_rows': repeated.height,
            'unmatched_repeat_pools': matches.filter(~pl.col('same_pool'))['pseudo_table_id'].to_list(),
            'repeat_reuse': bool(matches['reused'].any()) if matches.height else False,
            'arms_identical_no_pseudo': soft is None, 'fits': fits, 'C4_original_equals_teacher': c4,
            'C5_boundaries': boundaries, 'C6_weights': weights, 'corpus': {k: str(v) for k, v in c.items()},
            'corpus_sha256': {k: sha(v) for k, v in c.items() if v.exists()},
            'held_known_pairs_scored': held_known.height, 'held_pool_truth_rows_read': 0,
            'seconds': round(time.perf_counter() - started, 2)}
    files[f'{rel}/cell.json'] = write_json(dest / 'cell.json', cell)
    return cell, files


# ----------------------------------------------------------------------------------------------
# Phase B metrics (also exercised by selftest on synthetic data).
# ----------------------------------------------------------------------------------------------
def score_queries(scores, truths, ap5):
    rows = []
    for group in scores.partition_by('pair_id', maintain_order=True):
        pid = group['pair_id'][0]
        truth = truths[pid]
        row = {'pair_id': pid, 'table_id': group['table_id'][0], 'n_hands': group.height, 'n_listed': len(truth),
               'n_listed_in_hands': len(truth & set(group['hand_id']))}
        for arm in ARMS:
            ranked = group.sort([arm, 'hand_id'], descending=[True, False])['hand_id'].to_list()
            row[f'ap_{arm}'] = ap5(ranked[:5], truth)
            row[f'hits20_{arm}'] = len(set(ranked[:20]) & truth)
        rows.append(row)
    return rows


def half_of(table_id):
    return int(hashlib.sha256(table_id.encode()).hexdigest(), 16) % 2


def compute_metrics(pq):
    per_salt = {}
    for salt_index in sorted(pq['salt_index'].unique().to_list()):
        x = pq.filter(pl.col('salt_index') == salt_index)
        assert x.height == 162 and x['pair_id'].n_unique() == 162
        per_salt[salt_index] = {'map5': {arm: float(x[f'ap_{arm}'].mean()) for arm in ARMS},
                                'recall20': {arm: float(x[f'hits20_{arm}'].sum() / x['n_listed'].sum()) for arm in ARMS}}
    assert sorted(per_salt) == [0, 1, 2]
    pooled = pq.group_by('pair_id').agg(
        pl.col('table_id').first(), pl.col('table_id').n_unique().alias('tables'), pl.col('behavior_family').first(),
        pl.len().alias('salts'), *[pl.col(f'ap_{arm}').mean() for arm in ARMS]).sort('pair_id')
    assert pooled.height == 162 and (pooled['salts'] == 3).all() and (pooled['tables'] == 1).all()
    pools = sorted(set(pooled['table_id']))
    index = np.array([pools.index(t) for t in pooled['table_id']])
    ns = np.bincount(index, minlength=len(pools)).astype(np.float64)
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(pools), (BOOT_REPS, len(pools)))
    halves = np.array([half_of(t) for t in pooled['table_id']])
    families = pooled['behavior_family'].to_numpy()
    comparisons = {}
    for comparator in COMPARATORS:
        d = pooled['ap_pseudo'].to_numpy() - pooled[f'ap_{comparator}'].to_numpy()
        sums = np.bincount(index, weights=d, minlength=len(pools))
        boot = sums[ix].sum(axis=1) / ns[ix].sum(axis=1)
        low, high = (float(v) for v in np.quantile(boot, [.025, .975]))
        salt_delta = {s: per_salt[s]['map5']['pseudo'] - per_salt[s]['map5'][comparator] for s in per_salt}
        family = {f: float(d[families == f].mean()) for f in sorted(set(families))}
        half = {h: float(d[halves == h].mean()) for h in (0, 1)}
        conditions = {'every_salt_delta_gt_0': all(v > 0 for v in salt_delta.values()),
                      'bootstrap_lower_gt_0': low > 0,
                      'every_family_delta_ge_minus_0.010': all(v >= -0.010 for v in family.values()),
                      'both_halves_delta_ge_0': all(v >= 0 for v in half.values())}
        comparisons[comparator] = {'salt_deltas': salt_delta, 'pooled_delta': float(d.mean()),
                                   'bootstrap_ci95': [low, high], 'bootstrap_mean': float(boot.mean()),
                                   'family_deltas': family, 'family_query_counts': {f: int((families == f).sum()) for f in family},
                                   'half_deltas': half, 'half_query_counts': {h: int((halves == h).sum()) for h in (0, 1)},
                                   'conditions': conditions, 'all_conditions': all(conditions.values())}
    feasibility = all(comparisons[c]['all_conditions'] for c in COMPARATORS)
    small = feasibility and all(comparisons[c]['pooled_delta'] >= .005 for c in COMPARATORS)
    verdict = 'SUPPORTED_SMALL_LEAD' if small else ('SUPPORTED_FEASIBILITY_LEAD' if feasibility else 'NOT_SUPPORTED')
    return {'per_salt': per_salt, 'pooled_map5': {arm: float(pooled[f'ap_{arm}'].mean()) for arm in ARMS},
            'bootstrap': {'seed': BOOT_SEED, 'resamples': BOOT_REPS, 'pools': len(pools), 'shared_by_comparators': True},
            'comparisons': comparisons, 'SUPPORTED_FEASIBILITY_LEAD': feasibility,
            'pooled_delta_ge_0.005_both': all(comparisons[c]['pooled_delta'] >= .005 for c in COMPARATORS),
            'verdict': verdict}


DEVIATIONS = [
    {'item': 'C1 comparison columns', 'verdict_impact': 'none',
     'reading': 'C1 compares risk_full and risk_v70 (1 - class-0 probability) and selection_score with np.array_equal; FP unknown-scores.parquet stores these, not the four-class vectors.'},
    {'item': 'C2 key set', 'verdict_impact': 'none',
     'reading': 'Salt-a controls score FP-selected plus same-fold OLDPAIR-selected pairs, the FE teacher-probabilities-with-parity key set.'},
    {'item': 'hand source', 'verdict_impact': 'none; C2 and C4 check it',
     'reading': 'Teacher, student, pseudo and held-query hand rows come from the full-feature cache (receipts assert exact raw parity with the sample cache the Codex teachers used); salt b/c pseudo pairs exist only there.'},
    {'item': 'held hand-id boundary', 'verdict_impact': 'none',
     'reading': 'C5 scans (hand_id, table_id) of all 85 batches for the hand ids of each role and asserts none occurs in a held pool.'},
    {'item': 'pair-rule diagnostic', 'verdict_impact': 'none (non-gating)',
     'reading': 'Phase A stores pair-teacher risks on held known pairs (FP full/v70 label >= 0 rows) without labels; phase B applies the frozen selection rule and joins pair labels.'},
    {'item': 'recall@20', 'verdict_impact': 'none (non-gating)',
     'reading': 'Pooled fraction of listed hands ranked in the top 20 (score descending, hand_id ascending), the step65f/step71 convention.'},
    {'item': 'weight slack', 'verdict_impact': 'none',
     'reading': 'C6 caps use the reference-code float slack of 1e-9; repeated copies are also checked pair by pair.'},
    {'item': 'bootstrap draw', 'verdict_impact': 'none',
     'reading': 'default_rng(95).integers(0, pools, (2000, pools)) drawn once and shared by both comparators, the step5_evaluate.pool_bootstrap form.'},
    {'item': 'dry run before the official run', 'verdict_impact': 'none; no real evidence truth or held quality was read',
     'reading': 'Before any official output, drafts of this script ran in the session scratchpad, outside the project: a controls-only rehearsal (salt a, C1-C3 PASS bit-exact, no student fit) and one end-to-end dry run; after small edits (this text, a per-query n_listed_in_hands column, dry-run comparison fields) the final script ran end to end there again. Each end-to-end dry run did controls, regen and phase A on the real inputs, then phase B and exit on a synthetic random truth file with permuted pair labels; no real evidence truth or held quality metric was read. exit-check.json compares official model and parquet hashes with the final dry run.'},
    {'item': 'C5 pair-teacher hand ids', 'verdict_impact': 'none',
     'reading': 'Pair-teacher training rows are pair aggregates; C5 checks the hand ids of their known pairs in the full cache. Pair-teacher query rows (unknown pairs, used only for selection) are checked for held pools and players; their hand ids rely on the certified zero cross-pool hand overlap of the full cache (FP frames-frozen.json).'},
]
STOP_DEVIATIONS = {'C1 comparison columns', 'C2 key set', 'hand source', 'dry run before the official run'}


def cell_diagnostics(static, salt_index, fold, held):
    base = OUT / f'phaseA/salt{salt_index}/fold{fold}'
    cell, corpus = read(base / 'cell.json'), cell_corpus(salt_index, fold)
    agree = pl.read_parquet(corpus['agreement'])
    pairs = pl.read_parquet(base / 'held-known-pair-teacher.parquet')
    chosen = selected(pairs).join(static['meta'].select('pair_id', 'label'), on='pair_id', how='left')
    positives = static['positives'].filter(pl.col('table_id').is_in(sorted(held))).height
    hits = chosen.filter(pl.col('label') == 1).height
    return {'salt': static['parts'][salt_index]['salt'], 'fold': fold, 'pseudo_pairs': cell['pseudo_pairs'],
            'pseudo_pools': cell['pseudo_pools'], 'pseudo_predicted_families': cell['pseudo_predicted_families'],
            'unmatched_repeat_pools': cell['unmatched_repeat_pools'],
            'candidate_pairs': agree.height,
            'top5_overlap_distribution': {int(k): int(n) for k, n in agree.group_by('top5_overlap').len().sort('top5_overlap').rows()},
            'pair_rule_on_held_known_pairs': {'held_known_pairs': pairs.height, 'selected': chosen.height,
                                              'share_label_1': hits / chosen.height if chosen.height else None,
                                              'held_positives': positives, 'recall_of_held_positives': hits / positives}}


def mode_phaseB():
    audit_path, per_query_path = OUT / 'audit-C-ssl-student-pilot.json', OUT / 'per-query-ap.parquet'
    assert not audit_path.exists() and not per_query_path.exists(), 'C8: results exist; phase B runs once'
    reg = load_registration()
    protected_before = check_protected()
    static = load_static(reg)
    frozen_path = OUT / 'predictions-frozen.json'
    frozen = read(frozen_path)
    changed = sorted(p for p, s in frozen['files_sha256'].items() if sha(OUT / p) != s)
    assert not changed, f'C8: frozen predictions changed: {changed}'
    assert len(frozen['score_files']) == 15 and frozen['C4_all_exact'] and frozen['C6_all_weights']
    controls, regen, upstream = verify_upstream(protected_before)
    assert upstream == {k: frozen[k] for k in upstream}
    attempts = OUT / 'phaseB-attempts'
    attempt = (len(list(attempts.glob('attempt*.json'))) + 1) if attempts.exists() else 1
    write_json(attempts / f'attempt{attempt}.json', {'started_utc': now(), 'predictions_frozen_sha256': sha(frozen_path)})
    ap5, ids = load_ap5(), static['positives'].select('pair_id')
    truth = pl.scan_csv(TRUTH).select('pair_id', 'hand_id').join(ids.lazy(), on='pair_id', how='semi').collect()
    assert truth.height == truth.unique().height
    truths = {}
    for pid, hand in truth.rows():
        truths.setdefault(pid, set()).add(hand)
    assert set(truths) == set(ids['pair_id'])
    family = dict(static['positives'].select('pair_id', 'behavior_family').rows())
    rows, diagnostics = [], []
    for salt_index in range(3):
        for fold in range(5):
            held = held_pools(static['parts'][salt_index], fold)
            scores = pl.read_parquet(OUT / f'phaseA/salt{salt_index}/fold{fold}/held-hand-scores.parquet')
            assert set(scores['table_id']) <= held and scores['pair_id'].is_sorted()
            for r in score_queries(scores, truths, ap5):
                rows.append({'salt_index': salt_index, 'fold': fold, 'behavior_family': family[r['pair_id']], **r})
            diagnostics.append(cell_diagnostics(static, salt_index, fold, held))
    pq = pl.DataFrame(rows)
    metrics = compute_metrics(pq)
    buffer = io.BytesIO()
    pq.write_parquet(buffer)
    payload = buffer.getvalue()
    protected_after = check_protected()
    assert protected_after == protected_before, 'protected files changed during phase B'
    audit = build_audit(reg, metrics, diagnostics, controls, regen, frozen, attempt,
                        hashlib.sha256(payload).hexdigest(), protected_before, protected_after)
    audit_bytes = (json.dumps(plain(audit), indent=2) + '\n').encode('utf-8')
    assert write_bytes(per_query_path, payload) == audit['per_query_ap_sha256']
    write_bytes(audit_path, audit_bytes)
    log(event='phaseB_done', verdict=metrics['verdict'])


def write_bytes(path, payload):
    path = target(path)
    path.write_bytes(payload)
    return sha(path)


def build_audit(reg, metrics, diagnostics, controls, regen, frozen, attempt, per_query_sha, before, after):
    deviations = list(DEVIATIONS)
    script = sha(Path(__file__))
    phases = {'controls': controls.get('script_sha256'), 'regen': regen.get('script_sha256'),
              'phaseA': frozen.get('script_sha256'), 'phaseB': script}
    if controls['status'] == 'PASS_WITH_TOLERANCE':
        deviations.append({'item': 'controls tolerance', 'verdict_impact': 'none within 1e-10',
                           'reading': 'At least one of C1-C3 passed by the registered tolerance rule, not bit-exactly; see controls.'})
    if attempt > 1:
        deviations.append({'item': 'phase B attempts', 'verdict_impact': 'disclosed',
                           'reading': f'Phase B was invoked {attempt} times; earlier attempts wrote no results.'})
    summary = [{'fold': f['fold'], 'C1': f['C1']['status'], 'C1_scores_exact': f['C1']['scores']['exact'],
                'C1_max_abs_difference': f['C1']['scores']['max_abs_difference'],
                'C1_selection_exact': f['C1']['selection']['exact'],
                'C1_model_file_identical_to_codex': f['C1']['model_file_identical_to_codex'],
                'C2': f['C2']['status'], 'C2_probabilities_exact': f['C2']['probabilities']['exact'],
                'C2_max_abs_difference': f['C2']['probabilities']['max_abs_difference'],
                'C2_acceptance_flags_identical': f['C2']['acceptance_flags_identical'],
                'C2_soft_targets_exact': f['C2']['soft_targets']['exact'],
                'C2_model_files_identical_to_codex': f['C2']['model_files_identical_to_codex'],
                'C3': f['C3']['status'], 'C3_registered_counts_match': f['C3']['registered_counts_match']}
               for f in controls['folds']]
    rule = reg['decision_rules']['NOT_SUPPORTED' if metrics['verdict'] == 'NOT_SUPPORTED' else 'SUPPORTED']
    audit = {'created_utc': now(), 'step': 95, 'registration': str(REG), 'registration_sha256': REG_SHA,
             'question': reg['question'], 'verdict': metrics['verdict'], 'decision_rule_applied': rule,
             'metrics': metrics, 'gate_registered': reg['gate'],
             'controls': {'status': controls['status'], 'per_control': controls['per_control'], 'folds': summary},
             'C4_original_equals_teacher_all_cells': frozen['C4_all_exact'],
             'C5_boundaries_all_cells': frozen['C5_all_boundaries'], 'C6_weights_all_cells': frozen['C6_all_weights'],
             'C7_protected_identical': {'controls': controls['protected_before'] == controls['protected_after'],
                                        'regen': regen['protected_before'] == regen['protected_after'],
                                        'phaseA': frozen['protected_before'] == frozen['protected_after'], 'phaseB': before == after},
             'C8_single_metrics_invocation': {'attempt': attempt, 'frozen_hashes_verified': len(frozen['files_sha256'])},
             'yield_salt_a_regenerated': [{k: r[k] for k in ['fold', 'selected_pairs', 'selected_pools', 'accepted_pairs',
                                                               'accepted_pools', 'total_pseudo_weight', 'pseudo_rows']}
                                          for r in controls['yield_receipts']],
             'yield_salt_b_c': regen['yield_per_cell'], 'distinct_accepted_pairs_per_salt': regen['distinct_accepted_pairs_per_salt'],
             'phaseA_cells': frozen['cells'],
             'diagnostics': {'cells': diagnostics,
                             'recall20_per_salt_per_arm': {s: v['recall20'] for s, v in metrics['per_salt'].items()}},
             'prediction_registered': reg['prediction'], 'deviations': deviations,
             'runtime_seconds': {'controls': controls['seconds'], 'regen': regen['seconds'], 'phaseA': frozen['seconds'], 'phaseB': elapsed()},
             'peak_working_set_mb': {'controls': controls['memory']['peak_working_set_mb'], 'regen': regen['memory']['peak_working_set_mb'],
                                     'phaseA': frozen['memory']['peak_working_set_mb'], 'phaseB': memory()['peak_working_set_mb']},
             'per_query_ap_sha256': per_query_sha, 'predictions_frozen_sha256': sha(OUT / 'predictions-frozen.json'),
             'protected_before': before, 'protected_after': after, 'script_path': str(Path(__file__).resolve()),
             'script_sha256': script, 'script_sha256_per_phase': phases,
             'script_sha256_consistent_across_phases': len(set(phases.values())) == 1,
             'truth_file': str(TRUTH), 'truth_sha256': sha(TRUTH), 'upload': False}
    return audit


def compare_dry_run(root):
    """sha256 of official model and parquet files against the scratch dry run at the same relative paths."""
    dry_exit = json.loads((root / 'exit-check.json').read_text(encoding='utf-8'))
    dry_audit = json.loads((root / 'audit-C-ssl-student-pilot.json').read_text(encoding='utf-8'))
    out = {'root': str(root), 'dry_run_script_sha256': dry_exit['script_sha256'],
           'same_script_as_official': dry_exit['script_sha256'] == sha(Path(__file__)),
           'dry_run_truth_file': dry_audit['truth_file'], 'compared': 0, 'identical': 0, 'different': [], 'missing_in_dry_run': []}
    for sub in ['controls', 'regen', 'phaseA']:
        if not (OUT / sub).exists():
            continue
        for path in sorted((OUT / sub).rglob('*')):
            if path.suffix not in ('.txt', '.parquet'):
                continue
            rel = path.relative_to(OUT).as_posix()
            if not (root / rel).exists():
                out['missing_in_dry_run'].append(rel)
                continue
            out['compared'] += 1
            if sha(path) == sha(root / rel):
                out['identical'] += 1
            else:
                out['different'].append(rel)
    return out


def mode_exit():
    reg = load_registration()
    protected = check_protected()
    controls_path = OUT / 'controls/controls-result.json'
    controls = read(controls_path)
    assert protected == controls['protected_before'] == controls['protected_after'], 'protected files changed'
    report = {'created_utc': now(), 'registration_sha256_verified': REG_SHA,
              'registered_inputs_verified': len(reg['inputs_sha256']), 'protected': protected,
              'protected_identical_to_entry': True, 'controls_result_sha256': sha(controls_path)}
    stop = OUT / 'stop-record.json'
    if stop.exists():
        report['stop_record_sha256'] = sha(stop)
    else:
        for name in ['regen/regen-frozen.json', 'predictions-frozen.json']:
            frozen = read(OUT / name)
            changed = sorted(p for p, s in frozen['files_sha256'].items() if sha(OUT / p) != s)
            assert not changed, f'{name}: {changed}'
            report[name] = {'sha256': sha(OUT / name), 'files_verified': len(frozen['files_sha256'])}
        audit_path = OUT / 'audit-C-ssl-student-pilot.json'
        audit = read(audit_path)
        assert sha(OUT / 'per-query-ap.parquet') == audit['per_query_ap_sha256'] and audit['registration_sha256'] == REG_SHA
        report.update(audit_sha256=sha(audit_path), verdict=audit['verdict'],
                      script_matches_audit=audit['script_sha256'] == sha(Path(__file__)))
    report['script_sha256'] = sha(Path(__file__))
    if os.environ.get('STEP95_DRY_RUN_ROOT'):
        report['dry_run_comparison'] = compare_dry_run(Path(os.environ['STEP95_DRY_RUN_ROOT']))
    write_json(OUT / 'exit-check.json', report)
    log(event='exit_done', verdict=report.get('verdict'), stop=stop.exists())


def synthetic_queries(rng, boost):
    pools = [f'P{i:03}' for i in range(104)]
    pool_of = {f'Q{i:03}': pools[i % 104] for i in range(162)}
    family = {q: ['coordinated_isolation', 'directed_transfer', 'soft_play'][i % 3] for i, q in enumerate(pool_of)}
    truths = {q: {f'{q}H{j:02}' for j in rng.choice(30, 5, replace=False)} for q in pool_of}
    ap5, rows = load_ap5(), []
    for salt_index in range(3):
        keys = [(q, f'{q}H{j:02}', pool_of[q]) for q in pool_of for j in range(30)]
        frame = pl.DataFrame(keys, schema=['pair_id', 'hand_id', 'table_id'], orient='row')
        base = rng.random(frame.height)
        listed = np.array([h in truths[q] for q, h, _ in keys], dtype=float)
        frame = frame.with_columns(pl.Series('original_only', base), pl.Series('repeated_real', base),
                                   pl.Series('pseudo', base + boost * listed))
        for r in score_queries(frame, truths, ap5):
            rows.append({'salt_index': salt_index, 'fold': 0, 'behavior_family': family[r['pair_id']], **r})
    return pl.DataFrame(rows)


def mode_selftest():
    ap5 = load_ap5()
    assert ap5(['a', 'b', 'c', 'd', 'e'], {'a'}) == 1.0 and abs(ap5(['x', 'a'], {'a', 'b'}) - .25) < 1e-12
    assert ap5(['x'], set()) == 0.0
    rng = np.random.default_rng(1)
    same = compute_metrics(synthetic_queries(rng, 0.0))
    assert same['verdict'] == 'NOT_SUPPORTED' and all(abs(c['pooled_delta']) < 1e-15 for c in same['comparisons'].values())
    lead = compute_metrics(synthetic_queries(rng, 2.0))
    assert lead['verdict'] == 'SUPPORTED_SMALL_LEAD', lead['comparisons']
    queries = pl.DataFrame({'pair_id': ['k1', 'k2', 'k3'], 'table_id': ['A', 'A', 'B']})
    pseudo = pl.DataFrame({'pair_id': ['u2', 'u1', 'u3'], 'table_id': ['A', 'A', 'C'], 'weight': [.15, .1, .15]})
    m1, m2 = repeat_matching(1, 2, pseudo, queries), repeat_matching(1, 2, pseudo, queries)
    assert m1.equals(m2) and m1['pseudo_pair_id'].to_list() == ['u1', 'u2', 'u3']
    assert m1['same_pool'].to_list() == [True, True, False] and m1['copy_pair_id'].to_list()[2] == 'k3'
    assert not m1['reused'].any() and set(m1['copy_pair_id']) == {'k1', 'k2', 'k3'}
    m3 = repeat_matching(0, 0, pseudo, queries.head(1))
    assert m3['reused'].to_list() == [False, True, True]
    z = pl.DataFrame({'pair_id': ['p'] * 6 + ['q'] * 6, 'hand_id': [f'h{i}' for i in range(12)], 'table_id': ['A'] * 12,
                      'p_base': [.9, .8, .7, .6, .5, .1, .2, .2, .2, .2, .2, .2], 'p_v2': [.9, .8, .7, .6, .1, .5, .5, .1, .1, .1, .1, .1]})
    z = z.with_columns(((pl.col('p_base') + pl.col('p_v2')) / 2).alias('p_mean'))
    _, diag, soft = agreement(z, {'p', 'q'}, 100)
    assert diag['accepted'].to_list() == [True, False] and diag['top5_overlap'].to_list() == [4, 5]
    assert soft['pair_id'].unique().to_list() == ['p'] and abs(soft['pseudo_row_weight'].sum() - .15) < 1e-12
    print(json.dumps({'selftest': 'ok', 'lead_pooled_delta': lead['comparisons']['original_only']['pooled_delta'],
                      'lead_ci': lead['comparisons']['original_only']['bootstrap_ci95'], 'writes': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['selftest', 'controls', 'regen', 'phaseA', 'phaseB', 'exit'])
    mode = parser.parse_args().mode
    {'selftest': mode_selftest, 'controls': mode_controls, 'regen': mode_regen, 'phaseA': mode_phaseA,
     'phaseB': mode_phaseB, 'exit': mode_exit}[mode]()
