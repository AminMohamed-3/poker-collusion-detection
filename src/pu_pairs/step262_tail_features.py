"""STEP262 step 2: MECHPLUS (and 'plus' control) per-hand event probabilities for every tail-harness pair.

registration262.json 'inputs'.  For each surface prefix40/60/80 and each development table, the
850-column MECHPLUS per-hand matrix is rebuilt in the step254_eval order on THAT SURFACE'S hands
(the three surfaces are built separately: base_features' per-player baseline residuals and the
step254_prep A/B role are functions of the hand universe, so a prefix-80 build may NOT be masked
down to prefix-40), scored by the STEP257 fold model that held the table out, and only
(pair_id, hand_id, p_DT, p_SP, p_CI, p_fmax) is kept.

  163  build_allpairs_fold.base_features -> v2_features on the surface's hands (the code that produced
       experiments/pu_pairs_20260911/fold{f}-{surface}/scores.parquet).  Gate: the per-pair row count
       must equal the stored shared_hands of that surface.
  25   evidence_aug/step237_features.pairhand_features            (step258_prefix_features.build_237)
  662  evidence_aug/step254_features.py path + step254_prep chronology rebuilt from the surface's hands
                                                                  (step258_prefix_features.build_mech)

Development phase asserted on every hand.  New files only.

Usage:
  python step262_tail_features.py universe
  python step262_tail_features.py shard SURFACE SHARD NSHARDS
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '1')
import sys, gc, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
DATA = ROOT + 'data/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl, lightgbm as lgb                       # noqa: E402  (polars first: fixes the thread pool)
from pu_pairs import step258_prefix_features as PF                      # noqa: E402  (sets env to 4; polars is already up)
from pu_pairs.build_allpairs_fold import base_features, v2_features     # noqa: E402

OUT = C + 'experiments/step262_mech_tail_view_20260918/'
S257 = C + 'experiments/step257_eval_mech_scores_20260918/'
S258 = C + 'experiments/step258_mech_block_view_20260918/'
X = C + 'experiments/pu_pairs_20260911/'
SURF = ('prefix40', 'prefix60', 'prefix80')
FRAC = {'prefix40': .4, 'prefix60': .6, 'prefix80': .8}
FAMS = (('DT', 'directed_transfer'), ('SP', 'soft_play'), ('CI', 'coordinated_isolation'))


def log(*a):
    print(*a, flush=True)


def prefix_hands(tables, frac):
    """step40_prefix_audit / step61 / step131 rule, with the surface's fraction."""
    hd = pl.read_parquet(DATA + 'hands.parquet').filter(
        (pl.col('phase') == 'development') & pl.col('table_id').is_in(tables))
    hd = hd.sort('table_id', 'started_at').with_columns(
        (pl.int_range(pl.len()).over('table_id') / pl.len().over('table_id')).alias('_pos'))
    return hd.filter(pl.col('_pos') < frac).drop('_pos')


# ------------------------------------------------------------------ universe
def build_universe():
    """The D rows of the harness at each surface (step262_repro_tail.build_D, before the key join)."""
    parts = []
    for p in SURF:
        d = pl.concat([pl.read_parquet(X + f'fold{f}-{p}/scores.parquet',
                                       columns=['pair_id', 'table_id', 'shared_hands', 'risk', 'label'])
                       .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)])
        d = d.unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38)
        parts.append(d.with_columns(pl.lit(p).alias('surface')))
    u = pl.concat(parts)
    members = pl.concat([pl.read_csv(X + f'fold{f}/pairs.csv').select('pair_id', 'player_1', 'player_2')
                         for f in (1, 2)]).unique(subset=['pair_id'])
    u = u.join(members, on='pair_id', how='left')
    assert u['player_1'].null_count() == 0
    os.makedirs(OUT, exist_ok=True)
    u.write_parquet(OUT + 'universe262.parquet')
    rec = {'rows': int(u.height), 'pairs': int(u['pair_id'].n_unique()),
           'tables': int(u['table_id'].n_unique()),
           'per_surface': {p: int(u.filter(pl.col('surface') == p).height) for p in SURF},
           'pair_hands_per_surface': {p: int(u.filter(pl.col('surface') == p)['shared_hands'].sum()) for p in SURF}}
    rec['pair_hands_total'] = sum(rec['pair_hands_per_surface'].values())
    json.dump(rec, open(OUT + 'universe262.json', 'w', encoding='utf-8'), indent=1)
    log(json.dumps(rec, indent=1))


# ------------------------------------------------------------------ one shard
def load_boosters():
    fo = json.load(open(S257 + 'models/feature_order.json', encoding='utf-8'))['all']
    fp = json.load(open(S258 + 'models/feature_order_plus.json', encoding='utf-8'))['feature_order']
    assert fo[:len(fp)] == fp, 'the plus order is not the prefix of the mechplus order'
    B = {('mechplus', t, k): lgb.Booster(model_file=S257 + f'models/mechplus_{t}_fold{k}.txt')
         for t, _ in FAMS for k in range(5)}
    B.update({('plus', t, k): lgb.Booster(model_file=S258 + f'models/plus_{t}_fold{k}.txt')
              for t, _ in FAMS for k in range(5)})
    return B, fo, fp


def shard(surface, ishard, nshards):
    t0 = time.time()
    frac = FRAC[surface]
    u = pl.read_parquet(OUT + 'universe262.parquet').filter(pl.col('surface') == surface)
    tables = sorted(u['table_id'].unique().to_list())
    mine = [t for i, t in enumerate(tables) if i % nshards == ishard]
    sdir = OUT + 'scores/'
    os.makedirs(sdir, exist_ok=True)
    dest = sdir + f'scores_{surface}_s{nshards:03d}_{ishard:03d}.parquet'
    assert not os.path.exists(dest), dest
    B, FO, FP = load_boosters()
    ft = json.load(open(S257 + 'models/fold_tables.json', encoding='utf-8'))['table_to_fold']
    allpairs = pl.concat([pl.read_csv(X + f'fold{f}/pairs.csv').select('pair_id', 'player_1', 'player_2', 'table_id')
                          for f in (1, 2)]).unique(subset=['pair_id'])
    want = {c: True for c in FO}
    parts, gates = [], {'shared_hands_mismatches': 0, 'tables': 0, 'rows': 0}
    for it, tab in enumerate(mine):
        h = prefix_hands([tab], frac)
        assert (h['phase'] == 'development').all()
        p_all = allpairs.filter(pl.col('table_id') == tab).select('pair_id', 'player_1', 'player_2', 'table_id')
        raw = base_features(h, p_all)
        p_all = p_all.join(raw.select('pair_id').unique(), on='pair_id', how='semi')
        raw = v2_features(raw, p_all)
        keep = u.filter(pl.col('table_id') == tab).select('pair_id', 'shared_hands')
        raw = raw.join(keep.select('pair_id'), on='pair_id', how='semi')
        cnt = raw.group_by('pair_id').len().join(keep, on='pair_id', how='inner')
        gates['shared_hands_mismatches'] += int((cnt['len'] != cnt['shared_hands']).sum())
        assert cnt.height == keep.height, (cnt.height, keep.height)
        feats163 = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]
        assert feats163 == FO[:163], 'the 163 production column order moved'
        keys = raw.select('pair_id', 'hand_id', 'table_id').join(
            p_all.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        chrono = PF.build_chrono(keys, h.drop('phase'))
        mech, MECH = PF.build_mech(keys, chrono)
        f237 = PF.build_237(keys)
        F237 = [c for c in f237.columns if c not in ('pair_id', 'hand_id')]
        assert FO[163:163 + len(F237)] == F237 and FO[163 + len(F237):] == MECH, 'column order moved'
        full = (raw.select('pair_id', 'hand_id', *feats163)
                .join(f237, on=['pair_id', 'hand_id'], how='left')
                .join(mech, on=['pair_id', 'hand_id'], how='left'))
        assert full.height == keys.height
        assert full.select(F237 + MECH).null_count().to_numpy().sum() == 0, 'join left nulls'
        del raw, mech, f237, chrono
        gc.collect()
        n = full.height
        Xm = np.empty((n, len(FO)), np.float32)
        for i, c in enumerate(FO):
            Xm[:, i] = full[c].to_numpy()
        k = int(ft[tab])
        cols = {'pair_id': full['pair_id'], 'hand_id': full['hand_id']}
        for arm, nc in (('mechplus', len(FO)), ('plus', len(FP))):
            pr = {t: B[(arm, t, k)].predict(Xm[:, :nc], num_threads=1) for t, _ in FAMS}
            fm = np.max(np.stack([pr[t] for t, _ in FAMS]), axis=0)
            for t, _ in FAMS:
                cols[f'{arm}_p_{t}'] = pl.Series(pr[t].astype(np.float32))
            cols[f'{arm}_p_fmax'] = pl.Series(fm.astype(np.float32))
        parts.append(pl.DataFrame(cols).with_columns(pl.lit(tab).alias('table_id'), pl.lit(k).alias('mfold')))
        gates['tables'] += 1
        gates['rows'] += n
        del Xm, full
        gc.collect()
        if it % 5 == 0 or it == len(mine) - 1:
            log(json.dumps({'surface': surface, 'shard': ishard, 'table': it + 1, 'of': len(mine),
                            'rows': gates['rows'], 'seconds': round(time.time() - t0)}))
    out = pl.concat(parts)
    out.write_parquet(dest)
    gates['seconds'] = round(time.time() - t0, 1)
    gates['dest'] = dest
    json.dump(gates, open(sdir + f'log_{surface}_s{nshards:03d}_{ishard:03d}.json', 'w', encoding='utf-8'), indent=1)
    log('DONE', json.dumps(gates))


def main():
    if sys.argv[1] == 'universe':
        build_universe()
    else:
        shard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))


if __name__ == '__main__':
    main()
