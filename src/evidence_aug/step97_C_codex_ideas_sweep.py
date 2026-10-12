"""step 97 [C]. Codex evidence ideas one by one on the shipped recipe B0, then greedy stacking.

author: claude (agent)

REGISTRATION: experiments/codex_ideas_sweep_C_20260913/prediction-codex-ideas-sweep.json, frozen before
any fit. This file never writes it. Its sha256 and every inputs_sha256 entry are verified at entry and exit.

SURFACE: Codex Resume/src/audit_common.load_current(): 162 training-only queries, 3,240 top-20 rows, outer
folds 0-2 excluded by its own assert. Per-family GroupKFold(5) on table_id, LightGBM CLF, random_state=11,
2 threads. Codex modules are imported with bytecode writing disabled; nothing is written under Codex Resume.
Codex code is reused directly where it is a function (audit_common.load_current and evaluate,
symmetry_censor_audit.swapped, subset_likelihood_audit.make_objective); every other arm follows its audit
script line for line.

MODES
  replicate  scratch only: C0-C3 (incumbent, B0 and the 13 arms on the plain incumbent against the saved
             Codex scores). No metric is computed.
  dry        scratch only: the whole pipeline with a stand-in evaluator built from prediction names alone.
             It never reads labels or scores, so no metric is computed (code-path test).
  official   the single official run into the experiment directory; refuses to overwrite.
"""
import sys
sys.dont_write_bytecode = True
import os
for _k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse, ctypes, hashlib, json, platform, time, traceback
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Data/Projects/Money Making/work/poker')
CV = ROOT / 'Claude Version'
CODEX = ROOT / 'Codex Resume'
sys.path.insert(0, str(CV / 'src'))
sys.path.insert(0, str(CODEX / 'src'))
import numpy as np
import polars as pl
import lightgbm as lgb
import sklearn
from sklearn.model_selection import GroupKFold
import audit_common as ac
from symmetry_censor_audit import swapped
from subset_likelihood_audit import make_objective
from evidence_aug.paths import check_protected, sha
from evidence_aug.fold import fold_pools

CLF = ac.CLF
FAMILIES = ac.FAMILIES
CI = 'coordinated_isolation'
RES = CODEX / 'research'
EXPS = CODEX / 'experiments'
EXP_DIR = CV / 'experiments' / 'codex_ideas_sweep_C_20260913'
PREREG = EXP_DIR / 'prediction-codex-ideas-sweep.json'
PREREG_SHA = '33903378d56bed3b38b8d77b8a3d709fa07b2b5883fc19092f28bfa9442cd2da'
SCRATCH = Path('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2902d2e3-d89a-4488-96d3-ba0785bfb759/scratchpad/s97')
AUDIT_NAME = 'audit-C-codex-ideas-sweep.json'
PROV_NAME = 'provenance-codex-ideas-sweep.json'
SCREEN = {'mean_delta_min': .005, 'halves_min': 0., 'family_min': -.010}
ACCEPT = {'mean_delta_min': .003, 'halves_min': 0., 'family_min': -.010}
KEYS = ['pair_id', 'hand_id']
ARM = {}       # the 13 registered arms, filled from the prereg at entry
FIT_LOG = []   # every recipe receipt from every stage (C3, C8)


def utc():
    return datetime.now(timezone.utc).isoformat()


def before(a, b):
    return datetime.fromisoformat(a) <= datetime.fromisoformat(b)


def dump(path, obj, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'refusing to overwrite {path}')
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    path.write_text(text + '\n', encoding='utf-8')


class _PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
        'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
        'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def memory():
    """Peak working set and peak private commit of this process (K32GetProcessMemoryInfo)."""
    try:
        k32 = ctypes.WinDLL('kernel32')
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        assert k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return {'peak_working_set_mb': round(pmc.PeakWorkingSetSize / 2 ** 20, 1),
                'peak_private_commit_mb': round(pmc.PeakPagefileUsage / 2 ** 20, 1)}
    except Exception as exc:
        return {'error': repr(exc)}


def check_prereg():
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    bad = [rel for rel, digest in reg['inputs_sha256'].items() if sha(ROOT / rel) != digest]
    digest = sha(PREREG)
    return reg, {'prereg_sha256': digest, 'prereg_sha256_ok': digest == PREREG_SHA, 'inputs_n': len(reg['inputs_sha256']),
                 'inputs_mismatched': bad, 'pass': digest == PREREG_SHA and not bad}


def codex_snapshot():
    """Entries of Codex Resume, its experiments, src and src/__pycache__ folders and the 11 study experiment folders
    read here, with size and mtime for files. Codex Resume/research is never listed."""
    studies = ['symmetry_censor', 'contestable_state', 'preflop_equity_precision', 'marginal_payout', 'retrospective_strength',
               'ordered_surprise', 'outsider_cards', 'real_history_encoder', 'within_hand_sequence', 'nori_wager', 'subset_likelihood']
    snap = {}
    for folder in [CODEX, EXPS, CODEX / 'src', CODEX / 'src' / '__pycache__'] + [EXPS / f'{s}_20260912' for s in studies]:
        for p in sorted(folder.iterdir()):
            snap[str(p)] = 'dir' if p.is_dir() else [p.stat().st_size, p.stat().st_mtime_ns]
    return snap


def snapshot_diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


FAKE_SINGLE = {'prefix_strength': .020, 'terminal_strength': .015, 'nori': .012, 'equity4096': .010,
               'within_hand_ordered': .009, 'subset_nll': .008, 'symmetric_censor_half': .007, 'layered_payout': .006}


def fake_evaluate(pred, stage):
    """DRY RUN ONLY. Synthetic numbers picked from prediction names; labels and scores are never read."""
    res = {'FAKE_DRY_RUN': True, 'comparisons': {}, 'passed': [],
           'sets': {n: {'map5': .5, 'by_family': {f: .5 for f in FAMILIES}} for n in pred}}
    for n in pred:
        if n == 'incumbent':
            continue
        m = FAKE_SINGLE.get(n, -.004)
        if n == 'stage1_only':
            m = -.05
        if stage == 'greedy' and n == 'within_hand_ordered':
            m = -.001
        res['comparisons'][n] = {'mean_delta': m, 'ci95': [m - .01, m + .01], 'pairs_better': 0, 'pairs_worse': 0, 'n': 162,
                                 'by_family': {f: m for f in FAMILIES}, 'halves': {'0': m, '1': m}, 'gate_pass': False}
    return res


class Evaluator:
    """Every metric call goes through here. It refuses to run before the single-arm predictions are frozen (C5)."""

    def __init__(self, d, fake):
        self.d, self.fake, self.open, self.calls, self.first_utc = d, fake, False, 0, None

    def __call__(self, pred, stage):
        assert self.open, 'C5: evaluation requested before the single-arm predictions were frozen'
        self.calls += 1
        self.first_utc = self.first_utc or utc()
        return fake_evaluate(pred, stage) if self.fake else ac.evaluate(self.d, pred)


STATIC = {  # block arm: (feature parquet, frozen JSON listing its columns, key), as in each Codex audit script
    'layered_payout': ('marginal_payout_20260912/full-layered.parquet', 'marginal_payout_20260912/full-freeze.json', 'features'),
    'terminal_strength': ('retrospective_strength_20260912/terminal.parquet', 'retrospective_strength_20260912/freeze.json', 'features'),
    'prefix_strength': ('retrospective_strength_20260912/prefix.parquet', 'retrospective_strength_20260912/freeze.json', 'features'),
    'ordered_surprise': ('ordered_surprise_20260912/ordered.parquet', 'ordered_surprise_20260912/feature-freeze.json', 'features'),
    'outsider_linked': ('outsider_decisions_20260912/linked.parquet', 'outsider_decisions_20260912/feature-freeze.json', 'features'),
    'outsider_card_control': ('outsider_decisions_20260912/card_control.parquet', 'outsider_decisions_20260912/feature-freeze.json', 'features'),
    'history_mean': ('real_history_encoder_20260912/history_mean.parquet', 'real_history_encoder_20260912/corpus-receipt.json', 'simple_features'),
}
FOLD = {  # fold-specific blocks: detector fold k uses the fold-k features for its training and held rows
    'within_hand_ordered': ('within_hand_sequence_20260912/ordered-{k}-features.parquet', 'within_hand_sequence_20260912/features-receipt.json', 'columns'),
    'nori': ('nori_wager_20260912/nori-{k}-features.parquet', 'nori_wager_20260912/features-receipt.json', 'columns'),
}
KINDS = {'equity4096': 'replacement', 'history_random_encoder': 'block', 'symmetric_censor_half': 'training',
         'subset_nll': 'objective', **{a: 'block' for a in STATIC}, **{a: 'fold_block' for a in FOLD}}


def load_context():
    t0 = time.perf_counter()
    d, shared, rcols, saved = ac.load_current()
    keys = d.select(KEYS)
    top = (d['oof_rank'] <= 20).to_numpy()
    expected = d.filter(pl.col('oof_rank') <= 20).select(KEYS)
    outer = set().union(*(set(fold_pools(f)) for f in (0, 1, 2)))
    overlap = len(set(d['table_id']) & outer)
    assert overlap == 0 and d['pair_id'].n_unique() == 162 and int(top.sum()) == 3240
    reads = {}

    def rjson(rel):
        p = RES / rel
        reads[str(p)] = sha(p)
        return json.loads(p.read_text(encoding='utf-8'))

    def align(table, cols):
        f = keys.join(table, on=KEYS, how='left', validate='1:1').sort(KEYS)
        assert f.select(KEYS).equals(keys)
        return f.select(cols).to_numpy()

    # B0: the 29 cs_ fields after the base columns, as contestable_state_audit.py
    cs = pl.read_parquet(RES / 'contestable_state_20260912/features.parquet').sort(KEYS)
    cs_cols = [c for c in cs.columns if c not in KEYS]
    assert len(cs_cols) == 29 and all(c.startswith('cs_') for c in cs_cols)
    assert expected.equals(cs.select(KEYS)) and not set(cs_cols) & set(d.columns)
    # sec 12: the 24 eq_ values replaced in place, as preflop_equity_precision_audit.py
    eq = pl.read_parquet(RES / 'preflop_equity_precision_20260912/features4096.parquet').sort(KEYS)
    eq_fields = [c for c in eq.columns if c not in KEYS]
    assert len(eq_fields) == 24 and all(c.startswith('eq_') for c in eq_fields) and expected.equals(eq.select(KEYS))
    cached = pl.read_parquet(RES / 'preflop_equity_precision_20260912/baseline256-cached.parquet').sort(KEYS)
    assert cached.select(KEYS).equals(expected)
    assert np.array_equal(cached.select(eq_fields).to_numpy(), d.filter(pl.col('oof_rank') <= 20).select(eq_fields).to_numpy())
    high = {c: c + '__4096' for c in eq_fields}
    dx = d.join(cs, on=KEYS, how='left', validate='1:1').join(eq.rename(high), on=KEYS, how='left', validate='1:1').sort(KEYS)
    assert dx.select(KEYS).equals(keys)
    blocks, bcols = {}, {}
    for arm, (rel, jrel, jkey) in STATIC.items():
        t = pl.read_parquet(RES / rel).sort(KEYS)
        cols = rjson(jrel)[jkey]
        assert t.columns == KEYS + cols and t.select(KEYS).equals(expected) and not set(cols) & set(d.columns)
        blocks[arm], bcols[arm] = align(t, cols), cols
    # sec 19 random frozen encoder keyed by target-keys.parquet, as real_history_audit.py
    tk = pl.read_parquet(RES / 'real_history_encoder_20260912/target-keys.parquet').sort(KEYS)
    rnd = np.load(RES / 'real_history_encoder_20260912/random_encoder.npy', allow_pickle=False)
    assert tk.equals(expected) and rnd.shape == (3240, 34) and np.isfinite(rnd).all()
    ecols = [f'rh_encoder_{i}' for i in range(34)]
    assert not set(ecols) & set(d.columns)
    blocks['history_random_encoder'] = align(tk.with_columns(*[pl.Series(n, rnd[:, i]) for i, n in enumerate(ecols)]), ecols)
    bcols['history_random_encoder'] = ecols
    for arm, (pattern, jrel, jkey) in FOLD.items():
        cols = rjson(jrel)[jkey]
        assert len(cols) == 26 and not set(cols) & set(d.columns)
        for k in range(5):
            t = pl.read_parquet(RES / pattern.format(k=k)).sort(KEYS)
            assert t.columns == KEYS + cols and t.select(KEYS).equals(expected)
            blocks[(arm, k)] = align(t, cols)
        bcols[arm] = cols
    wh_detector = rjson('within_hand_sequence_20260912/corpus-freeze.json')['detector_design']
    wh_reference = rjson('within_hand_sequence_20260912/training-freeze.json')['design']
    nori_corpus = rjson('nori_wager_20260912/corpus-freeze.json')
    meta = d.group_by('pair_id').agg(pl.col('relevance').sum().alias('listed_count'),
                                     pl.col('chrono_rank').filter(pl.col('relevance') == 1).max().alias('last_listed'))
    dm = d.select(KEYS + ['relevance', 'chrono_rank']).join(meta, on='pair_id', validate='m:1').sort(KEYS)
    assert dm.select(KEYS).equals(keys)
    fam = {}
    for f in FAMILIES:
        ix = np.flatnonzero(((d['behavior_family'] == f) & (d['oof_rank'] <= 20)).to_numpy())
        q, mq = dx[ix], dm[ix]
        base = shared + (rcols if f != CI else [])
        x = q.select(base).to_numpy().astype(float)
        y = q['relevance'].to_numpy().astype(int)
        g = q['table_id'].to_numpy()
        assert np.isfinite(x).all()
        # late-unlisted censor rows, as symmetry_censor_audit.py
        cens = ((mq['listed_count'] >= 5) & (mq['relevance'] == 0) & (mq['chrono_rank'] > mq['last_listed'])).to_numpy()
        assert cens.dtype == bool
        folds = list(GroupKFold(n_splits=5).split(x, y, g))
        for k, (tr, te) in enumerate(folds):
            assert not set(g[tr]) & set(g[te])
            for detector in (wh_detector, nori_corpus['detector_design']):
                row = next(item for item in detector if item['family'] == f and item['fold'] == k)
                assert set(g[tr]) == set(row['training_pools']) and set(g[te]) == set(row['held_pools'])
            assert wh_reference[k]['fold'] == k and not set(g[te]) & set(wh_reference[k]['training_pools'])
            ref = next(item for item in nori_corpus['design'] if item['fold'] == k)
            refpools = set(ref['train_pools'])
            assert not refpools & set(g[te]) and not refpools & set(ref['held_pools'])
            assert not refpools & set(nori_corpus['outer_pools']) and not refpools & set(nori_corpus['encoder_exclusions'][str(k)])
        fam[f] = {'ix': ix, 'q': q, 'base': base, 'y': y, 'g': g, 'pid': q['pair_id'].to_numpy(), 'cens': cens, 'folds': folds}
    groups = {'B0 (incumbent and cs_ fields)': set(shared) | set(rcols) | set(cs_cols)}
    for arm, cols in bcols.items():
        groups.setdefault(ARM[arm]['idea_group'], set()).update(cols)
    names = list(groups)
    c7 = {'pairwise_disjoint': all(not groups[a] & groups[b] for i, a in enumerate(names) for b in names[i + 1:]),
          'field_names_by_group': {k: len(v) for k, v in groups.items()},
          'note': 'sec12 substitutes eq_ values under existing names and the TRAIN arms add no field'}
    info = {'rows': d.height, 'top20_rows': int(top.sum()), 'queries': d['pair_id'].n_unique(), 'outer_pools_checked': len(outer),
            'audit_tables_in_outer_pools': overlap, 'shared_columns': len(shared), 'rcols': len(rcols), 'cs_fields': len(cs_cols),
            'top20_rows_by_family': {f: len(fam[f]['ix']) for f in FAMILIES},
            'censored_training_candidates_by_family': {f: int(fam[f]['cens'].sum()) for f in FAMILIES},
            'exchange_fields_by_family': {f: sum(c.startswith(('A_', 'B_')) for c in fam[f]['base']) for f in FAMILIES},
            'fields_by_block_arm': {a: len(c) for a, c in bcols.items()}, 'extra_json_reads_sha256': reads,
            'clf': dict(CLF, random_state=11), 'load_seconds': round(time.perf_counter() - t0, 1)}
    return {'d': d, 'keys': keys, 'top': top, 'saved': saved, 'cs_cols': cs_cols, 'high': high, 'blocks': blocks,
            'bcols': bcols, 'fam': fam, 'C7': c7, 'info': info}


def recipe(b0=True):
    return {'b0': b0, 'eq': False, 'blocks': (), 'train': None, 'members': ()}


def add(r, arm):
    info = ARM[arm]
    assert all(ARM[m]['idea_group'] != info['idea_group'] for m in r['members']), 'C7: two arms of one idea_group in a recipe'
    r = dict(r, members=r['members'] + (arm,))
    if info['kind'] == 'replacement':
        r['eq'] = True
    elif info['kind'] in ('block', 'fold_block'):
        r['blocks'] = r['blocks'] + (arm,)
    else:
        assert info['kind'] in ('training', 'objective') and r['train'] is None
        r['train'] = arm
    return r


def build(members, b0=True):
    r = recipe(b0)
    for m in members:
        r = add(r, m)
    return r


def label(r):
    return ('B0' if r['b0'] else 'incumbent') + ''.join('+' + m for m in r['members'])


def design(ctx, r, f):
    """Base columns (sec 12 substitutes the eq_ values in place), then the cs_ fields for B0, then feature blocks in
    the order they were added. Training and objective changes act on whatever matrix this gives."""
    F = ctx['fam'][f]
    tail = ctx['cs_cols'] if r['b0'] else []
    x = F['q'].select([ctx['high'].get(c, c) if r['eq'] else c for c in F['base']] + tail).to_numpy().astype(float)
    cols = F['base'] + tail + [c for b in r['blocks'] for c in ctx['bcols'][b]]
    assert len(set(cols)) == len(cols), 'C7: duplicate feature names'

    def matrix(k):
        parts = [x] + [ctx['blocks'][(b, k) if ARM[b]['kind'] == 'fold_block' else b][F['ix']] for b in r['blocks']]
        z = np.column_stack(parts) if len(parts) > 1 else x
        assert z.shape == (len(F['ix']), len(cols)) and np.isfinite(z).all()
        return z
    return cols, matrix


def fit_fold(r, z, cols, F, tr, te):
    """One detector fit on the training rows, scored on the held rows. C3 replays it from booster_.model_to_string()."""
    y, pid = F['y'], F['pid']
    if r['train'] is None:
        m = lgb.LGBMClassifier(**CLF, random_state=11).fit(z[tr], y[tr])
        p = m.predict_proba(z[te])[:, 1]
        s = m.booster_.model_to_string()
        ok = np.array_equal(p, lgb.Booster(model_str=s).predict(z[te], num_threads=2))
    elif r['train'] == 'symmetric_censor_half':
        # as symmetry_censor_audit.py; appended fields carry no A_/B_ prefix, so only incumbent A/B pairs are exchanged
        assert not any(c.startswith(('A_', 'B_')) for c in cols[len(F['base']):])
        zs = swapped(z, cols)
        assert np.array_equal(swapped(zs, cols), z, equal_nan=True)
        w = np.where(F['cens'][tr], .5, 1.)
        for pair in np.unique(pid[tr]):
            q = pid[tr] == pair
            w[q] *= q.sum() / w[q].sum()
        xx = np.concatenate([z[tr], zs[tr]])
        yy = np.tile(y[tr], 2)
        w = np.tile(w / 2, 2)
        params = dict(CLF)
        params['min_child_samples'] = 2 * CLF['min_child_samples']
        m = lgb.LGBMClassifier(**params, random_state=11).fit(xx, yy, sample_weight=w)
        pa = m.predict_proba(z[te])[:, 1]
        pb = m.predict_proba(zs[te])[:, 1]
        p = (pa + pb) / 2
        s = m.booster_.model_to_string()
        b = lgb.Booster(model_str=s)
        ok = np.array_equal(pa, b.predict(z[te], num_threads=2)) and np.array_equal(pb, b.predict(zs[te], num_threads=2))
    else:
        assert r['train'] == 'subset_nll'
        objective, calls, _ = make_objective(pid[tr], y[tr])  # as subset_likelihood_audit.py
        m = lgb.LGBMRegressor(**CLF, random_state=11, objective=objective).fit(z[tr], y[tr])
        assert calls[0] == CLF['n_estimators']
        p = m.predict(z[te])
        assert np.isfinite(p).all() and np.min(p) > -1000
        s = m.booster_.model_to_string()
        ok = np.array_equal(p, lgb.Booster(model_str=s).predict(z[te], num_threads=2))
    return p, hashlib.sha256(s.encode('utf-8')).hexdigest(), bool(ok)


def run_recipe(ctx, r):
    t0 = time.perf_counter()
    d, top = ctx['d'], ctx['top']
    pred = d['oof'].to_numpy().copy() - 1e6
    fills = np.zeros(d.height, dtype=int)
    fits = []
    for f in FAMILIES:
        F = ctx['fam'][f]
        cols, matrix = design(ctx, r, f)
        for k, (tr, te) in enumerate(F['folds']):
            p, model_sha, ok = fit_fold(r, matrix(k), cols, F, tr, te)
            pred[F['ix'][te]] = p + 10
            fills[F['ix'][te]] += 1
            fits.append((f, len(cols), ok, model_sha))
    rec = {'recipe': label(r), 'fits': len(fits), 'replay_exact_all': all(x[2] for x in fits),
           'top20_filled_once': bool(np.all(fills[top] == 1) and np.all(fills[~top] == 0)),
           'tail_is_oof_minus_1e6': bool(np.array_equal(pred[~top], d['oof'].to_numpy()[~top] - 1e6)),
           'n_features_by_family': {x[0]: x[1] for x in fits}, 'model_string_sha256': [x[3] for x in fits],
           'seconds': round(time.perf_counter() - t0, 1)}
    FIT_LOG.append(rec)
    print(f"  fit {rec['recipe']}: {rec['seconds']}s replay_exact={rec['replay_exact_all']} filled_once={rec['top20_filled_once']}", flush=True)
    return pred, rec


def compare(ctx, mine, theirs):
    top = ctx['top']
    a, b = mine[top], theirs[top]
    out = {'bit_exact_top20': bool(np.array_equal(a, b)), 'max_abs_diff_top20': float(np.max(np.abs(a - b))),
           'all_19325_rows_equal': bool(np.array_equal(mine, theirs))}
    if not out['bit_exact_top20']:
        f = ctx['d'].select('pair_id').filter(pl.Series(top)).with_columns(pl.Series('a', a), pl.Series('b', b))
        f = f.with_columns(pl.col('a').rank('average').over('pair_id').alias('ra'), pl.col('b').rank('average').over('pair_id').alias('rb'))
        out['rank_equivalent_within_pair'] = bool((f['ra'] == f['rb']).all())
    out['status'] = 'bit_exact' if out['bit_exact_top20'] else (
        'rank_equivalent_below_1e-12' if out['max_abs_diff_top20'] < 1e-12 and out['rank_equivalent_within_pair'] else 'excluded')
    return out


def saved_scores(ctx, study):
    t = pl.read_parquet(EXPS / f'{study}_20260912' / 'scores.parquet').sort(KEYS)
    assert t.select(KEYS).equals(ctx['keys'])
    return t


def replication(ctx):
    """C1, C2 and C3 against the saved Codex scores. No metric is computed here."""
    rep = {'C1': {}, 'C2': {}}
    inc, rec = run_recipe(ctx, recipe(b0=False))
    rep['C1']['incumbent_vs_symmetry_censor_incumbent'] = dict(compare(ctx, inc, ctx['saved']['incumbent'].to_numpy()), fit=rec)
    b0, rec = run_recipe(ctx, recipe())
    saved_b0 = saved_scores(ctx, 'contestable_state')['contestable_state'].to_numpy()
    rep['C1']['B0_vs_contestable_state'] = dict(compare(ctx, b0, saved_b0), fit=rec)
    ci = ctx['fam'][CI]['ix']
    for arm, info in ARM.items():
        p, rec = run_recipe(ctx, add(recipe(b0=False), arm))
        c = compare(ctx, p, saved_scores(ctx, info['codex_study'])[info['codex_saved_arm']].to_numpy())
        c.update(codex_study=info['codex_study'], codex_saved_arm=info['codex_saved_arm'], fit=rec)
        if info['kind'] == 'replacement':
            c['ci_keeps_incumbent_scores'] = bool(np.array_equal(p[ci], inc[ci]))
        rep['C2'][arm] = c
        print('  C2', arm, c['status'], c['max_abs_diff_top20'], flush=True)
    rep['C1_pass'] = all(v['bit_exact_top20'] for v in rep['C1'].values())
    rep['C2_excluded'] = [a for a, v in rep['C2'].items() if v['status'] == 'excluded']
    rep['C2_all_bit_exact'] = all(v['status'] == 'bit_exact' for v in rep['C2'].values())
    return inc, b0, rep


def passes(s, rule):
    return bool(s['mean_delta'] >= rule['mean_delta_min'] and min(s['halves'].values()) >= rule['halves_min']
                and min(s['by_family'].values()) >= rule['family_min'])


def summary(res, name):
    c = res['comparisons'][name]
    worst = min(c['by_family'], key=c['by_family'].get)
    return {'map5': res['sets'][name]['map5'], 'reference_map5': res['sets']['incumbent']['map5'], 'mean_delta': c['mean_delta'],
            'ci95': c['ci95'], 'halves': c['halves'], 'by_family': c['by_family'], 'worst_family': [worst, c['by_family'][worst]],
            'pairs_better': c['pairs_better'], 'pairs_worse': c['pairs_worse'], 'n': c['n']}


def write_scores(ctx, path, cols):
    """Write keys plus score columns, read the file back, compare exactly and return its sha256."""
    assert not path.exists(), f'refusing to overwrite {path}'
    ctx['d'].select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(k, v) for k, v in cols.items()]).write_parquet(path)
    back = pl.read_parquet(path)
    assert back.select(KEYS).equals(ctx['keys']) and all(np.array_equal(back[k].to_numpy(), v) for k, v in cols.items())
    return sha(path)


def singles(ctx, reg, arms, inc, b0, out, EV, res):
    preds, res['fits'] = {}, {}
    ci = ctx['fam'][CI]['ix']
    for arm in arms:
        preds[arm], res['fits'][arm] = run_recipe(ctx, add(recipe(), arm))
        if ARM[arm]['kind'] == 'replacement':
            res['fits'][arm]['ci_keeps_B0_scores'] = bool(np.array_equal(preds[arm][ci], b0[ci]))
    cols = {'incumbent': inc, 'B0': b0, 'stage1_only': ctx['d']['oof'].to_numpy(), **preds}
    digest = write_scores(ctx, out / 'scores-singles.parquet', cols)
    res['frozen'] = {'scores_file': 'scores-singles.parquet', 'scores_sha256': digest, 'columns': list(cols),
                     'evaluator_calls_before_freeze': EV.calls, 'written_utc': utc()}
    dump(out / 'singles-frozen.json', res['frozen'])
    assert json.loads((out / 'singles-frozen.json').read_text(encoding='utf-8'))['scores_sha256'] == sha(out / 'scores-singles.parquet')
    EV.open = True
    s = summary(EV({'incumbent': inc, 'B0': b0}, 'sanity'), 'B0')
    recorded = reg['recorded_codex_deltas_vs_609180']['contestable_state:contestable_state']
    res['sanity_B0_vs_incumbent'] = dict(s, recorded_codex_delta=recorded, matches_recorded_at_6dp=round(s['mean_delta'], 6) == recorded)
    res['C4_stage1_only_vs_B0'] = summary(EV({'incumbent': b0, 'stage1_only': cols['stage1_only']}, 'C4'), 'stage1_only')
    band = reg['prediction']['single_delta_vs_B0_point_low80_high80']
    res['arms'] = {}
    for arm in arms:
        s = summary(EV({'incumbent': b0, arm: preds[arm]}, 'single'), arm)
        s.update(improver=passes(s, SCREEN), idea_group=ARM[arm]['idea_group'], production_class=ARM[arm]['production_class'],
                 recorded_delta_vs_609180=ARM[arm]['recorded_delta_vs_609180'], predicted_point_low80_high80=band[arm],
                 inside_predicted_80=bool(band[arm][1] <= s['mean_delta'] <= band[arm][2]))
        res['arms'][arm] = s
        print('  single', arm, round(s['mean_delta'], 6), 'improver' if s['improver'] else '-', flush=True)
    res['improvers'] = [a for a in arms if res['arms'][a]['improver']]
    return preds


def greedy(ctx, sres, preds, b0, out, EV, res):
    arms = sres['arms']
    order = sorted(sres['improvers'], key=lambda a: (-arms[a]['mean_delta'], a))
    S, S_pred, tried = recipe(), b0, set()
    res.update(order=order, log=[], frozen={'rule': 'each step writes S and S + X scores and their sha256 here before (S + X) vs S is evaluated', 'steps': []})
    dump(out / 'greedy-frozen.json', res['frozen'])
    ci = ctx['fam'][CI]['ix']
    for arm in order:
        info = ARM[arm]
        entry = {'candidate': arm, 'idea_group': info['idea_group'], 'production_class': info['production_class'],
                 'single_mean_delta_vs_B0': arms[arm]['mean_delta']}
        if info['production_class'] == 'C':
            res['log'].append(dict(entry, action='not eligible: production class C never enters the production stack'))
            continue
        if info['idea_group'] in tried:
            res['log'].append(dict(entry, action='not tried: a higher-ranked variant of this idea_group was already tried'))
            continue
        tried.add(info['idea_group'])
        SX = add(S, arm)
        p, rec = run_recipe(ctx, SX)
        if not S['members']:
            rec['identical_to_single_fit'] = bool(np.array_equal(p, preds[arm]))
        if info['kind'] == 'replacement':
            rec['ci_keeps_S_scores'] = bool(np.array_equal(p[ci], S_pred[ci]))
        step = len(res['frozen']['steps']) + 1
        name = f'scores-greedy-step{step}-{arm}.parquet'
        digest = write_scores(ctx, out / name, {'S': S_pred, 'S_plus_X': p})
        res['frozen']['steps'].append({'step': step, 'candidate': arm, 'S': label(S), 'S_plus_X': label(SX), 'scores_file': name,
                                       'scores_sha256': digest, 'frozen_utc': utc()})
        dump(out / 'greedy-frozen.json', res['frozen'], overwrite=True)
        assert json.loads((out / 'greedy-frozen.json').read_text(encoding='utf-8'))['steps'][-1]['scores_sha256'] == sha(out / name)
        started = utc()
        s = summary(EV({'incumbent': S_pred, arm: p}, 'greedy'), arm)
        accepted = passes(s, ACCEPT)
        res['log'].append(dict(entry, action='tried', step=step, S=label(S), S_plus_X=label(SX), result=s, accepted=accepted,
                               evaluation_started_utc=started, fit=rec))
        print('  greedy step', step, arm, round(s['mean_delta'], 6), 'ACCEPT' if accepted else 'reject', flush=True)
        if accepted:
            S, S_pred = SX, p
    res['S_star'] = list(S['members'])
    return S, S_pred


def descriptive(ctx, sres, preds, S, S_pred, inc, b0, out, EV, res):
    """Descriptive only: nothing here changes S_star."""
    arms = sres['arms']

    def rank(a):
        return -arms[a]['mean_delta'], a
    cols, best = {'S_star': S_pred}, {}
    for a in sorted([a for a in sres['improvers'] if ARM[a]['production_class'] != 'C'], key=rank):
        best.setdefault(ARM[a]['idea_group'], a)
    union = sorted(best.values(), key=rank)
    res['union_members'] = union
    if len(union) >= 2:
        cols['union'], res['union_fit'] = run_recipe(ctx, build(union))
    elif len(union) == 1:
        cols['union'] = preds[union[0]]
        res['union_fit'] = 'one member: identical recipe to its single arm, whose frozen prediction is reused'
    for m in S['members']:
        rest = [x for x in S['members'] if x != m]
        if rest:
            cols['S_star_without_' + m], res['S_star_without_' + m + '_fit'] = run_recipe(ctx, build(rest))
        else:
            cols['S_star_without_' + m] = b0
    research = sorted([a for a in sres['improvers'] if ARM[a]['production_class'] == 'C'], key=rank)
    res['research_class_C_improvers'] = research
    if research:
        cols['research_stack'], res['research_stack_fit'] = run_recipe(ctx, build(list(S['members']) + research))
    digest = write_scores(ctx, out / 'scores-descriptive.parquet', cols)
    res['frozen'] = {'scores_file': 'scores-descriptive.parquet', 'columns': list(cols), 'scores_sha256': digest, 'frozen_utc': utc()}
    dump(out / 'descriptive-frozen.json', res['frozen'])
    res['evaluations_started_utc'] = utc()
    if 'union' in cols:
        res['union_vs_B0'] = summary(EV({'incumbent': b0, 'union': cols['union']}, 'descriptive'), 'union')
    res['leave_one_out_S_star_minus_S_star_without'] = {}
    for m in S['members']:
        pair = {'incumbent': cols['S_star_without_' + m], 'S_star': S_pred}
        res['leave_one_out_S_star_minus_S_star_without'][m] = summary(EV(pair, 'descriptive'), 'S_star')
    res['S_star_vs_incumbent'] = summary(EV({'incumbent': inc, 'S_star': S_pred}, 'descriptive'), 'S_star')
    if research:
        pair = {'incumbent': S_pred, 'research_stack': cols['research_stack']}
        res['research_stack_vs_S_star'] = summary(EV(pair, 'descriptive'), 'research_stack')


def controls(audit, protected0, protected1, ctx, EV):
    c = {'C0': {'pass': protected0 == protected1, 'at_entry': protected0, 'at_exit': protected1}}
    rep = audit.get('replication')
    if rep:
        fields = ('status', 'bit_exact_top20', 'max_abs_diff_top20', 'all_19325_rows_equal')
        c['C1'] = {'pass': rep['C1_pass'], **{k: {x: v[x] for x in fields} for k, v in rep['C1'].items()}}
        c['C2'] = {'pass': rep['C2_all_bit_exact'], 'excluded': rep['C2_excluded'], 'status': {a: v['status'] for a, v in rep['C2'].items()},
                   'max_abs_diff_top20': {a: v['max_abs_diff_top20'] for a, v in rep['C2'].items()}}
    c['C3'] = {'pass': bool(FIT_LOG) and all(r['replay_exact_all'] for r in FIT_LOG), 'recipes': len(FIT_LOG),
               'boosters_replayed': sum(r['fits'] for r in FIT_LOG), 'failures': [r['recipe'] for r in FIT_LOG if not r['replay_exact_all']]}
    s = audit.get('singles') or {}
    if 'C4_stage1_only_vs_B0' in s:
        ci95 = s['C4_stage1_only_vs_B0']['ci95']
        c['C4'] = {'pass': bool(ci95[1] < 0), 'ci95': ci95, 'mean_delta': s['C4_stage1_only_vs_B0']['mean_delta']}
    if 'frozen' in s:
        out, g = Path(audit['output_dir']), audit.get('greedy') or {}
        started = {e['step']: e['evaluation_started_utc'] for e in g.get('log', []) if e.get('action') == 'tried'}
        steps = (g.get('frozen') or {}).get('steps', [])
        files_ok = sha(out / s['frozen']['scores_file']) == s['frozen']['scores_sha256'] and all(
            sha(out / st['scores_file']) == st['scores_sha256'] for st in steps)
        ok = (s['frozen']['evaluator_calls_before_freeze'] == 0 and EV is not None and EV.first_utc is not None
              and before(s['frozen']['written_utc'], EV.first_utc) and files_ok
              and all(st['step'] in started and before(st['frozen_utc'], started[st['step']]) for st in steps))
        dsc = audit.get('descriptive_only') or {}
        c['C5'] = {'pass': bool(ok), 'singles_frozen_utc': s['frozen']['written_utc'], 'first_evaluation_started_utc': EV.first_utc if EV else None,
                   'singles_scores_sha256': s['frozen']['scores_sha256'], 'greedy_steps_frozen_before_their_evaluation': len(steps),
                   'frozen_score_files_unchanged_at_exit': bool(files_ok), 'evaluations_total': EV.calls if EV else 0,
                   'descriptive_frozen_before_its_evaluations (not registered)': bool('frozen' in dsc and before(dsc['frozen']['frozen_utc'], dsc['evaluations_started_utc']))}
    if ctx:
        i = ctx['info']
        c['C6'] = {'pass': i['audit_tables_in_outer_pools'] == 0, 'outer_pools_checked': i['outer_pools_checked'],
                   'audit_tables_in_outer_pools': i['audit_tables_in_outer_pools'],
                   'reads': 'load_current training-only audit; Codex training-only feature files, receipts and saved audit scores; no evaluation, public or held-truth file'}
        c['C7'] = {'pass': ctx['C7']['pairwise_disjoint'], **ctx['C7'], 'variants_never_co_occur': 'asserted in every recipe build'}
    c['C8'] = {'pass': bool(FIT_LOG) and all(r['top20_filled_once'] and r['tail_is_oof_minus_1e6'] for r in FIT_LOG),
               'recipes_checked': len(FIT_LOG)}
    return c


def prediction_check(reg, audit):
    p, arms = reg['prediction'], audit['singles']['arms']
    k = sum(v['improver'] for v in arms.values())
    out = {'improvers': {'observed': k, 'point': p['improvers']['point'], 'range80': p['improvers']['range80'],
                         'inside80': p['improvers']['range80'][0] <= k <= p['improvers']['range80'][1]},
           'single_deltas_inside_their_80_band': sum(v['inside_predicted_80'] for v in arms.values()), 'singles': len(arms)}
    n = len(audit['S_star']['members'])
    md = audit['S_star']['minus_B0']['mean_delta']
    band = p['S_star_minus_B0_point_low80_high80']
    out['S_star_arms'] = {'observed': n, 'point': p['S_star_arms']['point'], 'range80': p['S_star_arms']['range80'],
                          'inside80': p['S_star_arms']['range80'][0] <= n <= p['S_star_arms']['range80'][1]}
    out['S_star_minus_B0'] = {'observed': md, 'point_low80_high80': band, 'inside80': band[1] <= md <= band[2]}
    out['S_star_empty'] = {'observed': n == 0, 'registered_probability': p['P_S_star_empty']}
    return out


DEVIATIONS = [
    'Greedy (silent point): the prereg allows at most one arm per idea_group in a stack but does not say whether a lower-ranked variant may be tried after its higher-ranked sibling was rejected. Most conservative reading used: each idea_group gets at most one greedy trial, its highest-ranked improver; a later variant of an already tried group is logged as not tried.',
    'Greedy step 1 refits B0 + X (the prereg says fit S + X); equality with the frozen single-arm fit is checked and recorded (identical_to_single_fit). Later steps refit S + X.',
    'Sec 12 replacement: coordinated_isolation has no eq_ input, so its matrix is unchanged and it is refit, where Codex copied the base scores; equality with the parent recipe scores is checked and recorded (ci_keeps_* fields).',
    'Symmetric training on B0 or a stack: Codex swapped() runs on the full column list after asserting that no appended field has an A_/B_ prefix, so only the incumbent A_/B_ fields (counted in data.exchange_fields_by_family) are exchanged and appended fields stay unchanged, as the prereg placement says.',
    'Combination order: base columns (sec 12 substitutes eq_ values in place), then the 29 cs_ fields, then feature blocks in the order they entered S. Training and objective changes act on that matrix.',
    'C3 replays every booster from booster_.model_to_string() in memory (Codex saved model files); no model file is written.',
    'Abort rule (silent point): in the official run nothing is evaluated if C1, C3 or C8 fails during replication; a C2 failure excludes that arm as registered.',
    'Descriptive union: best production-class improver per idea_group by single delta vs B0 (ties by name), added in that order; with one member its frozen single prediction is reused (identical recipe); with none it is reported empty.',
    'Descriptive leave-one-out is reported as S_star minus (S_star without the member); each reduced stack is refit, and B0 is used when the reduced stack is empty.',
    'S_star minus B0 (a registered prediction target) is evaluated after the greedy stage from scores frozen in the greedy step files; if S_star is empty it is B0 and the delta is 0 by construction (not evaluated).',
    'Descriptive predictions are also written and hashed (scores-descriptive.parquet, descriptive-frozen.json) before their evaluations; the prereg does not require this.',
    'Codex study-level gates, mechanism contrasts and selection rules are not applied; only the prereg screen (singles) and accept rule (greedy) decide, on unrounded values.',
    'Fold alignment of the fold-specific blocks is asserted against the Codex detector designs (within_hand_sequence and nori_wager corpus-freeze.json, within_hand_sequence training-freeze.json); these JSON files and the column-list JSON files are not in inputs_sha256, so their sha256 is recorded under data.extra_json_reads_sha256.',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['replicate', 'dry', 'official'])
    mode = parser.parse_args().mode
    t0 = time.perf_counter()
    if mode == 'official':
        present = sorted(p.name for p in EXP_DIR.iterdir())
        assert present == [PREREG.name], f'official run refuses to overwrite existing outputs: {present}'
        out = EXP_DIR
    else:
        out = SCRATCH / f"{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True)
    reg, inputs0 = check_prereg()
    assert inputs0['pass'], inputs0
    ARM.update({a['name']: a for a in reg['arms']})
    assert len(ARM) == 13 and set(ARM) == set(KINDS) and all(KINDS[a] == v['kind'] for a, v in ARM.items())
    snap0 = codex_snapshot()
    protected0 = check_protected()
    audit = {'step': 97, 'author': 'claude (agent)', 'mode': mode, 'FAKE_EVALUATOR_DRY_RUN': mode == 'dry', 'started_utc': utc(),
             'prereg': str(PREREG), 'prereg_sha256': PREREG_SHA, 'inputs_check_at_entry': inputs0, 'screen': SCREEN,
             'greedy_accept': ACCEPT, 'output_dir': str(out)}
    ctx = EV = None
    try:
        ctx = load_context()
        audit['data'] = ctx['info']
        inc, b0, audit['replication'] = replication(ctx)
        if mode != 'replicate':
            assert audit['replication']['C1_pass'], 'C1 failed: the base recipes do not reproduce'
            assert all(r['replay_exact_all'] and r['top20_filled_once'] and r['tail_is_oof_minus_1e6'] for r in FIT_LOG), 'C3 or C8 failed in replication'
            EV = Evaluator(ctx['d'], fake=mode == 'dry')
            arms = [a for a in ARM if a not in audit['replication']['C2_excluded']]
            audit['arms_excluded_by_C2'] = audit['replication']['C2_excluded']
            audit['singles'] = {}
            preds = singles(ctx, reg, arms, inc, b0, out, EV, audit['singles'])
            audit['greedy'] = {}
            S, S_pred = greedy(ctx, audit['singles'], preds, b0, out, EV, audit['greedy'])
            audit['S_star'] = {'members': list(S['members']), 'recipe': label(S)}
            if S['members']:
                audit['S_star']['minus_B0'] = summary(EV({'incumbent': b0, 'S_star': S_pred}, 'S_star'), 'S_star')
            else:
                audit['S_star']['minus_B0'] = {'empty': True, 'mean_delta': 0.0, 'note': 'S_star is B0'}
            audit['descriptive_only'] = {}
            descriptive(ctx, audit['singles'], preds, S, S_pred, inc, b0, out, EV, audit['descriptive_only'])
            audit['prediction_check'] = prediction_check(reg, audit)
        audit['status'] = 'complete'
    except Exception:
        audit['status'] = 'FAILED'
        audit['error'] = traceback.format_exc()
        print(audit['error'], flush=True)
    finally:
        protected1 = check_protected()
        audit['inputs_check_at_exit'] = check_prereg()[1]
        try:
            audit['controls'] = controls(audit, protected0, protected1, ctx, EV)
        except Exception:
            audit['controls'] = {'error': traceback.format_exc()}
        snap1 = codex_snapshot()
        changed = snapshot_diff(snap0, snap1)
        audit['codex_resume_readonly'] = {'entries_at_entry': len(snap0), 'entries_at_exit': len(snap1),
                                          'changed_entries': changed, 'unchanged': not changed}
        audit['deviations_and_judgement_calls'] = DEVIATIONS
        audit['seconds'] = round(time.perf_counter() - t0, 1)
        audit['memory'] = memory()
        audit['finished_utc'] = utc()
        dump(out / AUDIT_NAME, audit)
        me = Path(__file__).resolve()
        dump(out / PROV_NAME, {
            'step': 97, 'mode': mode, 'script': str(me), 'script_sha256': sha(me), 'python': platform.python_version(),
            'lightgbm': lgb.__version__, 'polars': pl.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
            'platform': platform.platform(), 'threads_env': {k: os.environ.get(k) for k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS')},
            'prereg_sha256': sha(PREREG), 'codex_code_used_sha256': {n: sha(CODEX / 'src' / n) for n in (
                'audit_common.py', 'symmetry_censor_audit.py', 'subset_likelihood_audit.py', 'subset_likelihood.py')},
            'outputs_sha256': {p.name: sha(p) for p in sorted(out.iterdir()) if p.name not in (PROV_NAME, PREREG.name)},
            'written_utc': utc()})
        brief = {k: (v.get('pass') if isinstance(v, dict) else v) for k, v in audit['controls'].items()}
        print('status', audit['status'], 'controls', brief, 'codex_unchanged', not changed, 'seconds', audit['seconds'], flush=True)
    assert protected0 == protected1, 'C0: protected files changed'


if __name__ == '__main__':
    main()
