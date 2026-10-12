"""STEP219 SCREEN (not an audit): which production rcols blocks help coordinated_isolation stage 2, measured on the
development CI pairs the 162-query audit never uses.

The CI stage-2 list excludes all 378 rcols. STEP218 found that one rcols block (the v2 isolation events) lifts CI
by +.0137 on the audit. To look for more without fishing on the audit, this screen uses only the CI pairs OUTSIDE
the audit (development labels; top-20 rows of the production development frame):
  list  = production CI columns + S204.NEW + CI6 (STEP205) + CIv8 (STEP218), i.e. the current reference;
  block = every rcols column sharing one base name (suffixes _sum/_gap/_min/_max stripped), CIv8 excluded;
  model = production stage-2 LightGBM (ac.CLF, random_state 11), 5-fold GroupKFold by table, two fold seeds
          (table order shuffled with seeds 2191 and 2192);
  score = per-pair AP@5 over the pair's top-20 rows (normalised by min(5, relevant rows in the top 20)),
          pairs without a relevant top-20 row skipped; delta = block arm minus list arm, mean over pairs.
Selection rule, fixed here: blocks with delta > 0 under BOTH fold seeds, ranked by the mean delta; the top two go
to a registered audit (STEP220) as separate CI arms. Nothing else is selected on this screen.
Usage: python step219_C_ci_rcols_screen.py
Writes outputs_step219_20260917/screen219.json. Never touches evaluation data.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step219_20260917'
KEYS = ['pair_id', 'hand_id']
CIF = 'coordinated_isolation'
SEEDS = (2191, 2192)
G = {}


def _init(Z, y, pid, tab, blocks, base_n, clf):
    G.update(Z=Z, y=y, pid=pid, tab=tab, blocks=blocks, base_n=base_n, clf=clf)


def ap5(y, s):
    o = np.argsort(-s, kind='mergesort')[:5]
    hits = y[o]
    n = min(5, int(y.sum()))
    if n == 0:
        return None
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / n)


def folds_for(seed):
    from sklearn.model_selection import GroupKFold
    tabs = np.unique(G['tab'])
    perm = {t: i for i, t in enumerate(np.random.default_rng(seed).permutation(tabs))}
    grp = np.array([perm[t] for t in G['tab']])
    return list(GroupKFold(n_splits=5).split(np.zeros(len(grp)), groups=grp))


def evaluate(job):
    import lightgbm as lgb
    name, seed = job
    Z, y, pid = G['Z'], G['y'], G['pid']
    cols = list(range(G['base_n'])) + (G['blocks'][name] if name else [])
    pred = np.full(len(y), np.nan)
    for tr, te in folds_for(seed):
        m = lgb.LGBMClassifier(**G['clf'], random_state=11).fit(Z[tr][:, cols], y[tr])
        pred[te] = m.predict_proba(Z[te][:, cols])[:, 1]
    out = {}
    for p in np.unique(pid):
        r = pid == p
        v = ap5(y[r], pred[r])
        if v is not None:
            out[str(p)] = v
    return name, seed, out


def main():
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step167_C_lp_production as S167
    from evidence_aug import step204_C_ci_causal_production as S204
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'screen219.json'
    assert not dest.exists()
    reg218 = json.loads((CV / 'outputs_step218_20260917/registration218.json').read_text(encoding='utf-8'))
    ref_ci_extra = list(reg218['lists']['civ8'][CIF])
    S108, _ = S167.modules()
    CSP, TOP = S108.CSP, S108.TOP
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    rcols, struct = S167.audit_columns()
    data, _, _, _, _ = CSP.development_frame()
    players = pl.read_csv(CSP.ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    D = S165.load()
    audit_pairs = set(D['Q']['pair_id'].unique().to_list())
    base_ci = list(bundle['cols'][CIF])
    rename = {'oof': 's1', 'oof_rank': 's1_rank'}   # production names of the stage-1 score
    assert sorted(base_ci) == sorted(rename.get(c, c) for c in D['shared'] + D['cs_cols'] + D['pf_cols']), \
        'production CI list differs from the audit base'
    top = data.filter((pl.col('s1_rank') <= TOP) & (pl.col('behavior_family') == CIF) & ~pl.col('pair_id').is_in(list(audit_pairs)))
    requested = top.select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, players)
    pref, _ = S108.S98.prefix_for_keys(requested)
    top = top.join(state, on=KEYS, how='left', validate='1:1').join(pref, on=KEYS, how='left', validate='1:1')
    for rel, cols in reg218['feature_files'].items():
        want = [c for c in cols if c in set(ref_ci_extra)]
        if want:
            top = top.join(pl.read_parquet(CV / rel).select(KEYS + want), on=KEYS, how='left', validate='1:1')
    top = top.sort(KEYS)
    list_cols = base_ci + ref_ci_extra
    assert len(set(list_cols)) == len(list_cols)
    assert top.select(list_cols).null_count().sum_horizontal().item() == 0, 'unmatched screen rows'
    ph = (pl.scan_parquet(POKER / 'data/hands.parquet').select('hand_id', 'phase').collect()
          .join(top.select('hand_id').unique(), on='hand_id'))
    assert ph['phase'].unique().to_list() == ['development']
    cand = [c for c in rcols if c not in set(list_cols)]
    groups = {}
    for c in cand:
        groups.setdefault(re.sub(r'_(sum|gap|min|max)$', '', c), []).append(c)
    Zb = top.select(list_cols).to_numpy().astype(np.float64)
    Zc = top.select(cand).fill_null(-999.).to_numpy().astype(np.float64)
    Zc[np.isnan(Zc)] = -999.
    Z = np.column_stack([Zb, Zc])
    pos = {c: len(list_cols) + i for i, c in enumerate(cand)}
    blocks = {g: [pos[c] for c in cs] for g, cs in groups.items()}
    y = top['relevance'].to_numpy().astype(int)
    pid = top['pair_id'].to_numpy()
    tab = top['table_id'].to_numpy()
    print(f'screen rows {len(y)}, pairs {len(np.unique(pid))}, relevant {y.sum()}, list {len(list_cols)}, '
          f'candidate blocks {len(blocks)} ({len(cand)} columns)  {round(time.perf_counter() - t0)}s', flush=True)
    jobs = [(None, s) for s in SEEDS] + [(g, s) for g in blocks for s in SEEDS]
    with Pool(int(os.environ.get('SCREEN_WORKERS', '6')), initializer=_init,
              initargs=(Z, y, pid, tab, blocks, len(list_cols), dict(D['ac'].CLF))) as pool:
        res = pool.map(evaluate, jobs, chunksize=4)
    base = {s: out for g, s, out in res if g is None}
    rows = []
    for g in blocks:
        d = {}
        for gg, s, out in res:
            if gg == g:
                d[s] = float(np.mean([out[p] - base[s][p] for p in base[s]]))
        rows.append({'block': g, 'columns': groups[g], 'delta_by_seed': d, 'mean': float(np.mean(list(d.values()))),
                     'both_positive': all(v > 0 for v in d.values())})
    rows.sort(key=lambda r: -r['mean'])
    chosen = [r['block'] for r in rows if r['both_positive']][:2]
    rec = {'step': 'STEP219', 'kind': 'screen on non-audit development CI pairs', 'rows': int(len(y)),
           'pairs_scored': {str(s): len(base[s]) for s in SEEDS}, 'base_map_by_seed': {str(s): float(np.mean(list(base[s].values()))) for s in SEEDS},
           'list_columns': len(list_cols), 'n_blocks': len(blocks), 'chosen_for_audit': chosen,
           'top20': rows[:20], 'bottom5': rows[-5:], 'all': [{k: r[k] for k in ('block', 'mean', 'both_positive')} for r in rows],
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'seconds': round(time.perf_counter() - t0, 1)}
    dest.write_text(json.dumps(rec, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: rec[k] for k in ('rows', 'pairs_scored', 'base_map_by_seed', 'n_blocks', 'chosen_for_audit', 'seconds')}), flush=True)
    for r in rows[:12]:
        print(r['block'], round(r['mean'], 4), {k: round(v, 4) for k, v in r['delta_by_seed'].items()}, flush=True)


if __name__ == '__main__':
    main()
