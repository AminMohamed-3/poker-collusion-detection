"""STEP194: causally anchored preflop fold features for the stage-1 structural block.

Production's pf_others_fold (step65_structural_evidence.hand_features) counts outsider folds after
`opener_ix` -- the FIRST AGGRESSOR OF ANY KIND, including an outsider. When an outsider opens and a
pair member three-bets, that counts the folds to the outsider's open, not the folds to the pair.
The two anchors disagree on 43.2 % of hands.

STEP193's isolated agent, given only gameplay, independently landed on the other anchor: folds
after the PAIR's first aggression, with folds BEFORE it as a separate penalty. Measured on that
step's 69 training packets (coordinated_isolation, 13,616 hands, 345 listed):

    pf_others_fold        (production anchor)   AUC .8687
    folds AFTER pair aggression                 AUC .9272
    folds BEFORE pair aggression                AUC .9534

Registration outputs_step194_20260916/registration194.json sha 1f78c23c. Codex Resume is READ-ONLY
and is not touched: the columns are computed here and joined onto the frame, then appended to the
`struct` list that S167.xmatrix consumes.

Usage: python step194_C_causal_fold_features.py PHASE REGSHA      (PHASE = build | screen)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step194_20260916'
REG = OUT / 'registration194.json'
FEAT = OUT / 'causal_fold_features.parquet'
NEW = ['pf_others_fold_after_pair_agg', 'pf_others_fold_before_pair_agg',
       'pf_pair_aggresses_preflop', 'pf_others_call_after_pair_agg']


def agg(action, amount, to_call):
    return action in ('raise', 'bet') or (action == 'all_in' and (amount or 0) > (to_call or 0))


def causal_features(acts, p1, p2):
    """acts: ordered list of (street, player_id, action, amount, to_call) for ONE hand.

    Anchored on the pair's first preflop aggression rather than on whoever opened. When the pair
    never aggresses preflop there is no anchor, so every outsider fold is counted as 'before' and
    the after-count is zero -- the same convention the induced rule used.
    """
    pf = [a for a in acts if a[0] == 'preflop']
    pair = (p1, p2)
    ix = None
    for i, a in enumerate(pf):
        if a[1] in pair and agg(a[2], a[3], a[4]):
            ix = i
            break
    if ix is None:
        before = sum(1 for a in pf if a[1] not in pair and a[2] == 'fold')
        return {'pf_others_fold_after_pair_agg': 0.0,
                'pf_others_fold_before_pair_agg': float(before),
                'pf_pair_aggresses_preflop': 0.0,
                'pf_others_call_after_pair_agg': 0.0}
    after = sum(1 for a in pf[ix + 1:] if a[1] not in pair and a[2] == 'fold')
    before = sum(1 for a in pf[:ix] if a[1] not in pair and a[2] == 'fold')
    calls = sum(1 for a in pf[ix + 1:]
                if a[1] not in pair and a[2] in ('call', 'all_in') and not agg(a[2], a[3], a[4]))
    return {'pf_others_fold_after_pair_agg': float(after),
            'pf_others_fold_before_pair_agg': float(before),
            'pf_pair_aggresses_preflop': 1.0,
            'pf_others_call_after_pair_agg': float(calls)}


def phase_build():
    from evidence_aug.step167_C_lp_production import modules
    S108, _ = modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    t0 = time.perf_counter()
    frame, _, _, _, _ = CSP.development_frame()
    keys = frame.select(KEYS).unique()
    lab = pl.read_csv(CSP.ROOT / 'data/development_labels.csv',
                      columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    hids = keys['hand_id'].unique().to_list()
    print(f'{keys.height} (pair, hand) rows over {len(hids)} distinct hands', flush=True)

    acts = (pl.scan_parquet(CSP.ROOT / 'data/actions.parquet')
            .filter(pl.col('hand_id').is_in(hids) & (pl.col('street') == 'preflop'))
            .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call')
            .sort('hand_id', 'action_no').collect())
    print(f'loaded {acts.height} preflop actions  {round(time.perf_counter()-t0)}s', flush=True)
    arows = {}
    for hid, _, st, pid, a, amt, tc in acts.iter_rows():
        arows.setdefault(hid, []).append((st, pid, a, amt, tc))

    rows = []
    for pid, hid in keys.iter_rows():
        f = causal_features(arows.get(hid, []), *pp[pid])
        f['pair_id'], f['hand_id'] = pid, hid
        rows.append(f)
    feat = pl.DataFrame(rows).select(KEYS + NEW)
    assert feat.height == keys.height
    assert feat.select(NEW).to_numpy().dtype.kind == 'f'
    assert np.isfinite(feat.select(NEW).to_numpy()).all(), 'non-finite feature'
    OUT.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(FEAT)
    print(f'wrote {FEAT.name} {feat.shape}  {round(time.perf_counter()-t0)}s', flush=True)
    print(feat.select(NEW).describe())


def phase_screen(reg_sha):
    """Refit the stage-1 out-of-fold features with struct extended, and read the AUC."""
    from multiprocessing import Pool
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step167_C_lp_production as S167
    import hashlib
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert reg['new_columns'] == NEW

    S108, _ = S167.modules()
    CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
    INC = json.loads((CV / 'outputs_step167_20260915/work/fit.json').read_text(encoding='utf-8'))
    rcols, struct = S167.audit_columns()
    for c in NEW:
        assert c not in struct and c not in rcols, f'{c} already exists upstream'

    t0 = time.perf_counter()
    data, _, _, _, _ = CSP.development_frame()
    players = pl.read_csv(CSP.ROOT / 'data/development_labels.csv',
                          columns=['pair_id', 'player_1', 'player_2'])
    requested = data.filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, players)
    data = data.join(state, on=KEYS, how='left', validate='1:1')
    pref, _ = S108.S98.prefix_for_keys(requested)
    data = data.join(pref, on=KEYS, how='left', validate='1:1', maintain_order='left')
    data = data.sort(KEYS)

    before_keys = data.select(KEYS)
    feat = pl.read_parquet(FEAT)
    data = data.join(feat, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert data.select(KEYS).equals(before_keys), 'the join changed row order or count'
    assert data.select(NEW).null_count().to_numpy().sum() == 0, 'unmatched rows after join'
    print(f'GATE join preserves keys: True   rows {data.height}', flush=True)

    results = {}
    for arm, cols in (('WITHOUT_new_columns', struct), ('WITH_new_columns', struct + NEW)):
        X, fam = S167.xmatrix(data, rcols, cols, 'behavior_family')
        QIX = np.flatnonzero((data['s1_rank'] <= TOP).to_numpy())
        C = S167.cmatrix(data[QIX], S108)
        rel = data['relevance'].to_numpy()
        pairs, tables = data['pair_id'].to_numpy(), data['table_id'].to_numpy()
        upairs, pfirst = np.unique(pairs, return_index=True)
        ptab = tables[pfirst]
        rowmask = lambda sel: np.isin(pairs, upairs[sel])
        folds = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
        jobs = [(('oof', k), 'joint', s, rowmask(tr), rowmask(te))
                for k, (tr, te) in enumerate(folds) for s in S167.SEEDS_OOF]
        init = (X.astype(np.float32), C.astype(np.float32), QIX, fam, rel, pairs, tables)
        with Pool(12, initializer=S165.pool_init, initargs=init) as pool:
            full = pool.map(S167.fit_full, S167.SEEDS_FULL, chunksize=1)
            oof = pool.map(S165.fit_job, jobs, chunksize=1)
        Dd = {'d': data.select('pair_id'), 'fam': fam, 'QIX': QIX, 'Q': data[QIX].select('pair_id')}
        acc = np.zeros((len(QIX), len(S165.NN_COLS)))
        cnt = np.zeros(len(QIX))
        for tag, m, s, rows, tq, p, S, trace, sec in oof:
            acc[tq] += S165.aligned_features(Dd, rows, tq, S, p)
            cnt[tq] += 1
        E = acc / cnt[:, None]
        auc = float(roc_auc_score(rel[QIX], E[:, 5]))
        results[arm] = {'X_shape': list(X.shape), 'oof_nn_p2_auc_top20': auc,
                        'best_epochs': [tr[2] for _, _, _, _, tr in full]}
        print(f'{arm:22} X {X.shape}  AUC {auc:.10f}  {round(time.perf_counter()-t0)}s', flush=True)

    base = INC['oof_nn_p2_auc_top20']
    ctrl = results['WITHOUT_new_columns']['oof_nn_p2_auc_top20']
    test = results['WITH_new_columns']['oof_nn_p2_auc_top20']
    tol = 1e-3
    print(f'\nGATE control reproduces STEP167 within the registered {tol:g}: '
          f'|{ctrl - base:+.3e}| -> {abs(ctrl - base) <= tol}')
    print(f'STEP167 recorded      {base:.10f}')
    print(f'control (struct only) {ctrl:.10f}')
    print(f'WITH new columns      {test:.10f}')
    print(f'DELTA vs control      {test - ctrl:+.6f}   (registered prediction +0.002, '
          f'band [-0.001, +0.006])')
    results['summary'] = {'registration_sha256': reg_sha, 'step167_auc': base,
                          'control_auc': ctrl, 'with_new_auc': test,
                          'delta_vs_control': test - ctrl,
                          'control_gate_tolerance': tol,
                          'control_gate_passed': bool(abs(ctrl - base) <= tol),
                          'seconds': round(time.perf_counter() - t0, 1)}
    (OUT / 'results194_screen.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('wrote results194_screen.json', flush=True)


def main():
    phase = sys.argv[1]
    if phase == 'build':
        phase_build()
    elif phase == 'screen':
        phase_screen(sys.argv[2])
    else:
        raise SystemExit('PHASE must be build or screen')


if __name__ == '__main__':
    main()
