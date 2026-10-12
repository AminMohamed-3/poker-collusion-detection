"""Step 25: are the simulator's agents individually parametrised beyond four style rates?

Builds a per-player play profile from ALL of that player's development-phase actions (public, label-free, ~1k actions
per player): action-class rates by street x facing-a-bet (32, shrunk), preflop VPIP/PFR by preflop-strength tercile (6),
raise/bet sizing (2), all-in rate, showdown rate, and continuation-bet style rate. Joins the ACTOR's profile onto Codex's
action-cache rows and refits the reference policy recipe with the extra inputs; compares held-fold log-loss and the
listed-vs-unlisted evidence signal exactly as step 24 did. Training-only; no evidence labels enter any model.
Output: Claude Version/experiments/policy_20260911/step25-fold{F}.json and player-profiles.parquet
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
sys.path.insert(0, str(ORIGINAL / 'src'))
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import log_loss
from advanced_features import CARDS, preflop_strength  # noqa: E402
from policy.step24_policy_capacity import CAE, OUT, hand_scores, rank_metrics  # noqa: E402

STREET = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}


def build_profiles():
    path = OUT / 'player-profiles.parquet'
    if path.exists():
        return pl.read_parquet(path)
    t = time.perf_counter()
    hands = pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').filter(pl.col('phase') == 'development').select('hand_id', 'big_blind')
    acts = pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet').join(hands, on='hand_id').with_columns(
        pl.col('street').replace_strict(STREET, return_dtype=pl.Int8).alias('s'),
        (pl.col('to_call') > 0).cast(pl.Int8).alias('facing'),
        pl.when(pl.col('action') == 'fold').then(0).when(pl.col('action') == 'check').then(1)
          .when((pl.col('action') == 'call') | ((pl.col('action') == 'all_in') & (pl.col('amount') <= pl.col('to_call')))).then(2).otherwise(3).cast(pl.Int8).alias('k'),
        (pl.col('action') == 'all_in').cast(pl.Float32).alias('allin'),
        (pl.col('amount') / (pl.col('pot_before') + 1e-9)).alias('size_ratio')).collect()
    # global rates for shrinkage
    g = acts.group_by('s', 'facing').agg(*[(pl.col('k') == c).mean().alias(f'g{c}') for c in range(4)])
    cell = acts.group_by('player_id', 's', 'facing').agg(pl.len().alias('n'), *[(pl.col('k') == c).sum().alias(f'c{c}') for c in range(4)]).join(g, on=['s', 'facing'])
    m = 20.0
    cell = cell.with_columns(*[((pl.col(f'c{c}') + m * pl.col(f'g{c}')) / (pl.col('n') + m)).alias(f'r{c}') for c in range(4)])
    wide = cell.select('player_id', 's', 'facing', 'r0', 'r1', 'r2', 'r3').pivot(on=['s', 'facing'], index='player_id', values=['r0', 'r1', 'r2', 'r3'])
    # missing cells (player never in that situation) -> global rate
    gmap = {(int(r['s']), int(r['facing'])): [r[f'g{c}'] for c in range(4)] for r in g.iter_rows(named=True)}
    fills = {}
    for col in wide.columns:
        if col == 'player_id': continue
        # column names like r0_{"s","facing"}_{0,1}  -> parse
        rc = int(col[1]); rest = col.split('_', 1)[1]
        digits = [int(x) for x in ''.join(ch if ch.isdigit() else ' ' for ch in rest).split()]
        fills[col] = gmap[(digits[0], digits[1])][rc]
    wide = wide.with_columns(*[pl.col(c).fill_null(v).cast(pl.Float32).alias(f'pp_{i:02d}') for i, (c, v) in enumerate(fills.items())]).select('player_id', *[f'pp_{i:02d}' for i in range(len(fills))])
    # preflop strength tercile VPIP / PFR
    seats = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2', 'went_to_showdown').join(hands, on='hand_id').collect()
    codes = {k: v for k, v in CARDS.items()}
    c1 = seats['hole_card_1'].replace_strict(codes, return_dtype=pl.Int16).to_numpy(); c2 = seats['hole_card_2'].replace_strict(codes, return_dtype=pl.Int16).to_numpy()
    lo = np.minimum(c1, c2); hi = np.maximum(c1, c2)
    key = lo.astype(np.int64) * 64 + hi
    uniq, inv = np.unique(key, return_inverse=True)
    strength = np.array([preflop_strength(int(k // 64), int(k % 64)) for k in uniq], dtype=np.float32)[inv]
    seats = seats.with_columns(pl.Series('pf', strength))
    q = np.quantile(strength, [1 / 3, 2 / 3])
    seats = seats.with_columns(pl.when(pl.col('pf') < q[0]).then(0).when(pl.col('pf') < q[1]).then(1).otherwise(2).cast(pl.Int8).alias('terc'))
    pre = acts.filter(pl.col('s') == 0).group_by('hand_id', 'player_id').agg((pl.col('k') >= 2).any().cast(pl.Float32).alias('vpip'), (pl.col('k') == 3).any().cast(pl.Float32).alias('pfr'))
    pre = seats.select('hand_id', 'player_id', 'terc').join(pre, on=['hand_id', 'player_id'], how='left').with_columns(pl.col('vpip').fill_null(0.0), pl.col('pfr').fill_null(0.0))
    gp = pre.group_by('terc').agg(pl.col('vpip').mean().alias('gv'), pl.col('pfr').mean().alias('gp'))
    pt = pre.group_by('player_id', 'terc').agg(pl.len().alias('n'), pl.col('vpip').sum().alias('v'), pl.col('pfr').sum().alias('p')).join(gp, on='terc').with_columns(
        ((pl.col('v') + m * pl.col('gv')) / (pl.col('n') + m)).alias('vpip'), ((pl.col('p') + m * pl.col('gp')) / (pl.col('n') + m)).alias('pfr'))
    ptw = pt.select('player_id', 'terc', 'vpip', 'pfr').pivot(on='terc', index='player_id', values=['vpip', 'pfr'])
    ptw = ptw.select('player_id', *[pl.col(c).cast(pl.Float32).alias(f'pt_{i:02d}') for i, c in enumerate([c for c in ptw.columns if c != 'player_id'])])
    # sizing, all-in, showdown
    sz = acts.filter(pl.col('k') == 3).group_by('player_id').agg(pl.col('size_ratio').clip(0, 5).mean().cast(pl.Float32).alias('ps_size_mean'), pl.col('size_ratio').clip(0, 5).std().cast(pl.Float32).alias('ps_size_std'))
    al = acts.group_by('player_id').agg(pl.col('allin').mean().cast(pl.Float32).alias('ps_allin'), pl.len().cast(pl.Float32).alias('ps_actions'))
    sd = seats.group_by('player_id').agg(pl.col('went_to_showdown').cast(pl.Float32).mean().alias('ps_showdown'))
    prof = wide.join(ptw, on='player_id', how='left').join(sz, on='player_id', how='left').join(al, on='player_id', how='left').join(sd, on='player_id', how='left')
    prof = prof.fill_null(strategy='mean') if False else prof.with_columns(*[pl.col(c).fill_null(pl.col(c).mean()) for c in prof.columns if c != 'player_id'])
    prof.write_parquet(path)
    print('profiles', prof.shape, round(time.perf_counter() - t, 1), 's', flush=True)
    return prof


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, default=1); args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    prof = build_profiles()
    pcols = [c for c in prof.columns if c != 'player_id']
    plan = json.loads((CAE / 'policy-plan.json').read_text()); inputs = plan['inputs']
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    evidence = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    held = set(pl.read_csv(ORIGINAL / 'reports' / 'tabicl-confirmation' / f'fold{args.fold}' / 'held-pairs.csv')['table_id'])
    data = pl.read_parquet(sorted((CAE / 'action-cache').glob('*.parquet'))).join(labels.select('pair_id', 'player_1', 'player_2', 'label', 'behavior_family'), on='pair_id')
    data = data.with_columns(pl.when(pl.col('direction') == 0).then(pl.col('player_1')).otherwise(pl.col('player_2')).alias('actor'),
                             pl.when(pl.col('direction') == 0).then(pl.col('player_2')).otherwise(pl.col('player_1')).alias('partner'))
    data = data.join(prof.rename({**{c: 'own_' + c for c in pcols}, 'player_id': 'actor'}), on='actor', how='left')
    data = data.join(prof.rename({**{c: 'par_' + c for c in pcols}, 'player_id': 'partner'}), on='partner', how='left')
    own = ['own_' + c for c in pcols]; par = ['par_' + c for c in pcols]
    assert data.select(pl.all_horizontal([pl.col(c).is_not_null() for c in own + par]).all()).item()
    train_all = data.filter((pl.col('label') == 0) & ~pl.col('table_id').is_in(sorted(held)))
    rng = np.random.default_rng(55193)
    ref = train_all.with_columns(pl.Series('r', rng.random(train_all.height))).sort('pair_id', 'r').group_by('pair_id', maintain_order=True).head(128)
    if ref.height > 100000: ref = ref.sample(n=100000, seed=55193)
    held_neg = data.filter((pl.col('label') == 0) & pl.col('table_id').is_in(sorted(held)))
    held_pos = data.filter((pl.col('label') == 1) & pl.col('table_id').is_in(sorted(held)))
    res = {'fold': args.fold, 'profile_columns': len(pcols), 'rows': {'reference_train': ref.height, 'all_train': train_all.height, 'held_neg': held_neg.height, 'held_pos': held_pos.height}}
    base = dict(plan['parameters'], n_jobs=2)
    specs = {'reference': (ref, inputs, base),
             'ref+own_profile': (ref, inputs + own, base),
             'ref+own+partner_profile': (ref, inputs + own + par, base),
             'all_rows+own_profile_300x31': (train_all, inputs + own, dict(base, n_estimators=300, num_leaves=31, min_child_samples=40, colsample_bytree=.8))}
    for name, (tr, cols, params) in specs.items():
        t = time.perf_counter()
        clf = lgb.LGBMClassifier(**params); clf.fit(tr.select(cols).to_numpy().astype(np.float32), tr['action_class'].to_numpy())
        pn = clf.predict_proba(held_neg.select(cols).to_numpy().astype(np.float32)); pp = clf.predict_proba(held_pos.select(cols).to_numpy().astype(np.float32))
        yn = held_neg['action_class'].to_numpy()
        r = {'seconds': round(time.perf_counter() - t, 1), 'inputs': len(cols), 'held_neg_logloss': float(log_loss(yn, pn, labels=[0, 1, 2, 3])), 'held_neg_accuracy': float((pn.argmax(1) == yn).mean()),
             'held_neg_logloss_by_class': {str(k): float(-np.log(np.maximum(pn[yn == k, k], 1e-5)).mean()) for k in range(4)}}
        hs = hand_scores(held_pos, pp)
        r['evidence_ranking'] = rank_metrics(hs, evidence)
        imp = np.asarray(clf.feature_importances_, float)
        r['profile_importance_share'] = float(imp[[i for i, c in enumerate(cols) if c.startswith(('own_pp', 'own_pt', 'own_ps', 'par_'))]].sum() / imp.sum())
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != 'evidence_ranking'}), flush=True)
        for col in ('surprise_max', 'surprise_sum', 'surprise_facing_max', 'raise_weak_max'):
            print('   ', col, {k: round(v, 3) if isinstance(v, float) else v for k, v in r['evidence_ranking'][col].items()})
    write_json(OUT / f'step25-fold{args.fold}.json', res)


if __name__ == '__main__':
    main()
