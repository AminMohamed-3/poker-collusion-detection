"""STEP200: the DIRECT test of the censoring lane -- train on the inferred hidden labels.

STEP199 closed this lane on an indirect estimate (a depth control with n=11 and n=21, plus
arithmetic through STEP133's +.034-per-doubling). Amin asked for it tested to its full extent. This
puts the inferred hidden labels into stage-2 training and reads decoded MAP on the 162-query audit.

What is new against the three steps that already died here:
  STEP137  self-labels NULL -- production's own ranking is redundant with what production predicts
  STEP196  LLM annotator labels -- monotone damage, -.0083 to -.0227
  STEP199  added TARGETING (only capped pairs can be censored) and SIZING (about +0.84 per pair)
  HERE     adds an INDEPENDENT selector: the three induced rules pick evidence from gameplay alone
           and never see production's score. Their top-5 picks that are NOT listed are exactly what
           a censored rank-6 hand looks like. That is what STEP137 could not have.

Two controls can kill it, and are the point of the design:
  RANDOM_capped   -- if the rule-selected arm does no better than a random unlisted hand, SELECTION
                     carries nothing and any movement is generic label noise.
  RULE_allpairs   -- if adding to all pairs matches adding only to capped pairs, TARGETING carries
                     nothing and the censoring story adds nothing over blanket smoothing.

Labels enter STAGE-2 TRAINING ONLY; evaluation is ALWAYS against true relevance; one shared network
fit so only the training signal varies. That is STEP196's construction, reused not reinvented.

Audit only. No evaluation pair, no held fold, never uploads.

Usage: python step200_C_censored_labels_train.py REGSHA
"""
import os
import sys
sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
from evidence_aug import step165_C_e2e_audit as S165

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step200_20260917'
REG = OUT / 'registration200.json'
S193 = CV / 'outputs_step193_20260916'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
PACK = {'coordinated_isolation': 'ci_packets', 'directed_transfer': 'directed_transfer_packets',
        'soft_play': 'soft_play_packets'}


def load_rule(path):
    spec = importlib.util.spec_from_file_location(f'rule_{path.parent.name}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rank_hands


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    assert not (OUT / 'results200.json').exists(), 'preserve a completed read'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.step167_C_lp_production import modules
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    d, Q = D['d'], D['Q']
    qpos = {(p, h): i for i, (p, h) in enumerate(Q.select(K).iter_rows())}
    y_true = Q['relevance'].to_numpy().astype(int)
    nq = len(Q)

    # ---- reconstruct packet position -> hand_id, and VERIFY against the stored answer keys -----
    S108, _ = modules()
    CSP = S108.CSP
    lab = pl.read_csv(CSP.ROOT / 'data/development_labels.csv',
                      columns=['pair_id', 'player_1', 'player_2', 'behavior_family'])
    members = sorted({q for r in lab.iter_rows(named=True) for q in (r['player_1'], r['player_2'])})
    seats = (pl.scan_parquet(CSP.ROOT / 'data/seats.parquet')
             .filter(pl.col('player_id').is_in(members)).select('hand_id', 'player_id').collect())
    hands_of = {}
    for hid, q in seats.iter_rows():
        hands_of.setdefault(q, set()).add(hid)
    # detz is imported, not retyped: it is the exact function step193 used before sorting, and the
    # ordering reconstructed here has to match step193's byte for byte. This venv has no tzdata, so
    # iter_rows panics on a tz-aware Datetime (ZoneInfoNotFoundError('UTC')).
    from evidence_aug.step193_C_rule_induction_packets import detz
    hstart = dict(detz(pl.scan_parquet(CSP.ROOT / 'data/hands.parquet')
                       .select('hand_id', 'started_at').collect()).iter_rows())
    order_of = {}
    for r in lab.iter_rows(named=True):
        shared = hands_of.get(r['player_1'], set()) & hands_of.get(r['player_2'], set())
        order_of[r['pair_id']] = sorted(shared, key=lambda h: (hstart[h], h))
    fam_of = {r['pair_id']: r['behavior_family'] for r in lab.iter_rows(named=True)}
    for famname in PACK:
        key = json.loads((S193 / f'ANSWER_KEY_{famname}.json').read_text(encoding='utf-8'))
        for pid, ho in key['hand_order'].items():
            assert order_of[pid] == list(ho), f'hand_order reconstruction differs for {pid}'
    print(f'hand_order reconstruction verified against all 3 answer keys  '
          f'{round(time.perf_counter()-t0)}s', flush=True)

    # ---- the induced rules' UNLISTED picks ------------------------------------------------------
    rule_pick = {}
    for famname, pdir in PACK.items():
        fn = load_rule(S193 / pdir / 'rule.py')
        held = set(json.loads((S193 / f'ANSWER_KEY_{famname}.json').read_text(encoding='utf-8'))['held_out'])
        for f in sorted((S193 / pdir).glob('pair_*.json')):
            pid = f.stem[5:]
            pair = json.loads(f.read_text(encoding='utf-8'))
            picks = [int(x) for x in fn(pair)]
            ids = [order_of[pid][i] for i in picks if i < len(order_of[pid])]
            rows = [qpos[(pid, h)] for h in ids if (pid, h) in qpos]
            rule_pick[pid] = {'rows': [i for i in rows if y_true[i] == 0],
                              'held_during_induction': pid in held}
    print(f'rules run over {len(rule_pick)} pairs  {round(time.perf_counter()-t0)}s', flush=True)

    # ---- pair bookkeeping: which audit pairs are capped -----------------------------------------
    qpairs = Q['pair_id'].to_numpy()
    nlisted = {p: int(y_true[qpairs == p].sum()) for p in np.unique(qpairs)}
    capped = {p for p, n in nlisted.items() if n == 5}
    print(f'audit pairs {len(nlisted)}: capped {len(capped)}, uncapped {len(nlisted)-len(capped)}',
          flush=True)

    s1 = Q['oof'].to_numpy().astype(float)
    rng = np.random.default_rng(200)

    def pick_rows(source, which, n=1):
        """source: 'rule' | 's1' | 'random'.  which: 'capped' | 'all' | 'held'."""
        out = []
        for p in np.unique(qpairs):
            if which == 'capped' and p not in capped:
                continue
            if which == 'held' and not rule_pick.get(p, {}).get('held_during_induction'):
                continue
            idx = np.flatnonzero((qpairs == p) & (y_true == 0))
            if not len(idx):
                continue
            if source == 'rule':
                cand = [i for i in rule_pick.get(p, {}).get('rows', [])]
            elif source == 's1':
                cand = list(idx[np.argsort(-s1[idx])])
            else:
                cand = list(rng.permutation(idx))
            out.extend(cand[:n])
        a = np.array(sorted(set(int(x) for x in out)), dtype=int)
        assert (y_true[a] == 0).all(), 'an added row was already positive'
        return a

    ADD = {'RULE_capped_w025': (pick_rows('rule', 'capped'), 0.25),
           'RULE_capped_w050': (pick_rows('rule', 'capped'), 0.50),
           'RULE_capped_w100': (pick_rows('rule', 'capped'), 1.00),
           'RULE_capped_N2_w025': (pick_rows('rule', 'capped', 2), 0.25),
           'S1_capped_w025': (pick_rows('s1', 'capped'), 0.25),
           'RANDOM_capped_w025': (pick_rows('random', 'capped'), 0.25),
           'RULE_allpairs_w025': (pick_rows('rule', 'all'), 0.25),
           'RULE_heldpairs_w025': (pick_rows('rule', 'held'), 0.25)}
    for k_, (a, w) in ADD.items():
        print(f'  {k_:24s} adds {len(a):4d} positives at weight {w}', flush=True)

    # ---- one shared network fit ------------------------------------------------------------------
    X, C = S165.matrices(D)
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'],
                 d['relevance'].to_numpy(), pairs, tables)
    jobs = []
    for k_, (tr, te) in enumerate(outer):
        for s in S165.SEEDS_INNER_OUTER:
            jobs.append((('outer', k_), 'joint', s, rowmask(tr), rowmask(te)))
        for j, (itr, ite) in enumerate(GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
            for s in S165.SEEDS_INNER:
                jobs.append((('inner', k_, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite])))
    print(f'X {X.shape}  {len(jobs)} network fits (shared by all arms)', flush=True)
    with Pool(14, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(S165.fit_job, jobs, chunksize=1)
    print(f'network refits done {round(time.perf_counter()-t0)}s', flush=True)

    AF_te, AF_in = {k_: [] for k_ in range(5)}, {k_: {} for k_ in range(5)}
    for jtag, m, s, rws, tq, p, S, trace, sec in out:
        if jtag[0] == 'outer':
            AF_te[jtag[1]].append((tq, S165.aligned_features(D, rws, tq, S, p)))
        else:
            AF_in[jtag[1]].setdefault(jtag[2], []).append((tq, S165.aligned_features(D, rws, tq, S, p)))
    folds_q, EA, EB = [], {}, np.full((nq, len(S165.NN_COLS)), np.nan)
    for k_, (tr, te) in enumerate(outer):
        trq = np.flatnonzero(np.isin(qpairs, upairs[tr]))
        teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
        ea = np.full((nq, len(S165.NN_COLS)), np.nan)
        for j, lst in AF_in[k_].items():
            ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
        eb = np.full((nq, len(S165.NN_COLS)), np.nan)
        eb[AF_te[k_][0][0]] = np.mean([a for _, a in AF_te[k_]], axis=0)
        folds_q.append((trq, teq))
        EA[k_] = ea[trq]
        EB[teq] = eb[teq]
    assert np.isfinite(EB).all()

    ac = D['ac']
    colsf = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    Zf = {f: Q.select(colsf[f]).to_numpy().astype(float) for f in FAMS}

    def run(add, w):
        y = y_true.copy()
        sw = np.ones(nq)
        if add is not None and len(add):
            y[add] = 1
            sw[add] = w
        p = D['SS'].copy()
        for f in FAMS:
            z = Zf[f]
            for k_, (tr, te) in enumerate(folds_q):
                a = tr[isf[f][tr]]
                m = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                    np.column_stack([z[a], EA[k_][isf[f][tr]]]), y[a], sample_weight=sw[a])
                b = te[isf[f][te]]
                if len(b):
                    p[D['QIX'][b]] = m.predict_proba(np.column_stack([z[b], EB[b]]))[:, 1] + 10
        return p

    scores = {'control': run(None, None)}
    for name, (a, w) in ADD.items():
        scores[name] = run(a, w)
    print(f'all {len(scores)} arms scored {round(time.perf_counter()-t0)}s', flush=True)

    dec = D['dec']
    truths = {k_[0]: set(g['hand_id']) for k_, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    kf = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabq[q]) for q in qids])
    R = lambda x: round(float(x), 6)
    M = {}
    for arm, s in scores.items():
        _, m = summarize(kf.with_columns(pl.Series('score', dec(s))), truths, fams)
        M[arm] = np.array([m[q] for q in qids])

    res = {'registration_sha256': reg_sha, 'level': 'LP original pool (u=0), 162-query audit',
           'map5_control': R(M['control'].mean()), 'n_queries': len(qids),
           'added_counts': {k_: int(len(a)) for k_, (a, w) in ADD.items()}, 'arms': {}}
    for arm in ADD:
        delta = M[arm] - M['control']
        res['arms'][arm] = {
            'map5': R(M[arm].mean()), 'decoded_delta': R(delta.mean()),
            'half0': R(delta[hv == 0].mean()), 'half1': R(delta[hv == 1].mean()),
            'positive_on_both_halves': bool(delta[hv == 0].mean() > 0 and delta[hv == 1].mean() > 0),
            'per_family': {f: R(delta[[fams[q] == f for q in qids]].mean()) for f in FAMS}}
    rule, rnd = res['arms']['RULE_capped_w025'], res['arms']['RANDOM_capped_w025']
    allp = res['arms']['RULE_allpairs_w025']
    res['THE_CONTROLS'] = {
        'rule_minus_random': R(rule['decoded_delta'] - rnd['decoded_delta']),
        'selection_carries_information': bool(rule['decoded_delta'] - rnd['decoded_delta'] > 0.003),
        'capped_minus_allpairs': R(rule['decoded_delta'] - allp['decoded_delta']),
        'targeting_carries_information': bool(abs(rule['decoded_delta'] - allp['decoded_delta']) > 0.003),
        'monotone_in_weight': [res['arms'][a]['decoded_delta'] for a in
                               ('RULE_capped_w025', 'RULE_capped_w050', 'RULE_capped_w100')]}
    res['gates'] = {'control_reproduces_0690': {
        'here': R(M['control'].mean()), 'tolerance': 0.002,
        'passed': bool(abs(M['control'].mean() - 0.690) <= 0.002)}}
    res['bar'] = {'corrected_build_bar_decoded': 0.0073,
                  'clears_bar': bool(rule['decoded_delta'] >= 0.0073)}
    c = res['THE_CONTROLS']
    res['verdict'] = ('LANE_STANDS' if (rule['positive_on_both_halves'] and
                                        c['selection_carries_information'] and
                                        res['bar']['clears_bar'])
                      else 'REAL_BUT_UNDER_BAR' if (rule['positive_on_both_halves'] and
                                                    c['selection_carries_information'])
                      else 'LANE_CLOSED')
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results200.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
