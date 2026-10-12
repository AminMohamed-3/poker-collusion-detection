"""STEP258 step 5: does a MECH view add to the production M3x block blend?

Arms (registration258.json, written before any of these fits):
  M3x                     incumbent: FS_M3, l2 = 10, def-A training rows, 32-draw SHORT3/ALL5 views
  M3x+MECH6               PRIMARY   + the six MECHPLUS aggregates over the pair's prefix hands
  M3x+MECH6+MECH6noT      SECONDARY + the same six again over the prefix hands without the third player T
  M3x+PERM6               CONTROL 1 (within-pair shuffled mechanical block)
  M3x+PLUS6               CONTROL 2 (163 + 25 only, old information)

Cross-fold protocol, definitions and metric: step177 verbatim (fold-1-only tables scored by the
fold-2-only fit and vice versa; training rows = def-A certified; def C is a guard).
Transform: cnt_q999, cnt_q99 and noisy_or enter as log1p, the three top-k logits raw; the blend's own
logreg standardises everything.  Writes results258.json.  Never uploads.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import ap, logreg, predict
from pu_pairs.step62_multiview_candidate import DEFS
from pu_pairs.step258_repro_m3x import groups, crossfold

OUT = C + 'experiments/step258_mech_block_view_20260918/'
DEST = OUT + 'results258.json'
AGG = ['cnt_q999', 'cnt_q99', 'top1_logit', 'top3_logit', 'top5_logit', 'noisy_or']
LOG1P = {'cnt_q999', 'cnt_q99', 'noisy_or'}
L2 = 10.0


def log(*a):
    print(*a, flush=True)


def block6(agg, prefix):
    """the six aggregates of one arm as standardisable columns (log1p on the three count-like ones)."""
    cols = []
    for nm in AGG:
        v = agg[f'{prefix}_{nm}'].to_numpy().astype(float)
        cols.append(np.log1p(np.clip(v, 0, None)) if nm in LOG1P else v)
    return np.column_stack(cols)


def main():
    t0 = time.time()
    assert not os.path.exists(DEST), 'preserve a completed run'
    d = pl.read_parquet(OUT + 'testbed32.parquet')
    agg = pl.read_parquet(OUT + 'aggregates258.parquet')
    assert agg['pair_id'].to_list() == d['pair_id'].to_list(), 'aggregate row order differs'
    repro = json.load(open(OUT + 'repro258.json', encoding='utf-8'))
    assert repro['G_fold_match_maxdiff'] == 0.0 and repro['G_fit_M3x_reproduces_receipt_step156a']
    assert all(repro['G_ap_M3x_reproduces_block_screen'].values())

    in1, in2, insh = groups(d)
    G = {'f1': in1, 'f2sh': in2 | insh}
    cert = {D: d['cls'].is_in(DEFS[D][0] + DEFS[D][1]).to_numpy() for D in ('A', 'C')}
    ypos = {D: d['cls'].is_in(DEFS[D][0]).to_numpy().astype(float) for D in ('A', 'C')}
    X0 = X_of(d, FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT')

    have = {p for p in ('mechplus', 'mechplusnoT', 'perm', 'plus')
            if f'{p}_{AGG[0]}' in agg.columns}
    log('aggregate blocks present:', sorted(have))
    arms = {'M3x': (X0, FS_M3)}
    if 'mechplus' in have:
        arms['M3x+MECH6'] = (np.column_stack([X0, block6(agg, 'mechplus')]), FS_M3 + ['MECH6_' + a for a in AGG])
    if 'mechplusnoT' in have:
        arms['M3x+MECH6+MECH6noT'] = (
            np.column_stack([X0, block6(agg, 'mechplus'), block6(agg, 'mechplusnoT')]),
            FS_M3 + ['MECH6_' + a for a in AGG] + ['noT6_' + a for a in AGG])
    if 'perm' in have:
        arms['M3x+PERM6'] = (np.column_stack([X0, block6(agg, 'perm')]), FS_M3 + ['PERM6_' + a for a in AGG])
    if 'plus' in have:
        arms['M3x+PLUS6'] = (np.column_stack([X0, block6(agg, 'plus')]), FS_M3 + ['PLUS6_' + a for a in AGG])

    # ---------- subgroup masks (all evaluated inside the def-A certified rows)
    risk = d['risk_prefix'].to_numpy()
    R1 = d['R1_prefix'].to_numpy().astype(bool)
    sh = d['shared_hands'].to_numpy()
    lab_pos = d['pair_id'].to_list()
    ev = (pl.read_csv(ROOT + 'data/development_evidence.csv').select('pair_id', 'hand_id')
          .group_by('pair_id').len().rename({'len': 'n_listed'}))
    nl = dict(zip(ev['pair_id'].to_list(), ev['n_listed'].to_list()))
    n_listed = np.array([nl.get(p, 0) for p in lab_pos])
    lowdose_pos = (ypos['A'] == 1) & np.isin(n_listed, (3, 4)) & (d['label'].to_numpy() == 1)
    cnt999 = agg['mechplus_cnt_q999'].to_numpy() if 'mechplus' in have else np.zeros(d.height)
    subsets = {
        'band[.99,1]': risk >= .99,
        'band[.9,.99)': (risk >= .9) & (risk < .99),
        'band[.5,.9)': (risk >= .5) & (risk < .9),
        'lowdose_pos_vs_all_negA': lowdose_pos | (ypos['A'] == 0),
        'R1_pairs': R1,
        'short_pairs_sh<=50': sh <= 50,
    }
    meta = {'n_pairs': int(d.height), 'n_certA': int(cert['A'].sum()), 'n_posA': int(ypos['A'][cert['A']].sum()),
            'n_lowdose_pos_certA': int((lowdose_pos & cert['A']).sum()),
            'n_R1_certA': int((R1 & cert['A']).sum()), 'n_R1_pos_certA': int((R1 & cert['A'] & (ypos['A'] == 1)).sum()),
            'n_short_certA': int(((sh <= 50) & cert['A']).sum()),
            'subset_sizes': {k: {'n': int((m & cert['A']).sum()), 'n_pos': int(ypos['A'][m & cert['A']].sum())}
                             for k, m in subsets.items()}}
    log(json.dumps(meta, indent=1))

    res, preds = {}, {}
    for name, (X, fs) in arms.items():
        pred, m1, m2 = crossfold(X, ypos['A'], cert['A'], in1, in2, insh)
        preds[name] = pred
        R = {}
        for D in ('A', 'C'):
            R[f'{D}_pooled'] = ap(ypos[D][cert[D]], pred[cert[D]])
            for g, m in G.items():
                R[f'{D}_{g}'] = ap(ypos[D][cert[D] & m], pred[cert[D] & m])
        for k, m in subsets.items():
            sel = cert['A'] & m
            R[f'sub_{k}'] = ap(ypos['A'][sel], pred[sel]) if 0 < ypos['A'][sel].sum() < sel.sum() else None
        R['weights_f1fit'] = {v: round(float(w), 4) for v, w in zip(fs, m1[0])}
        R['weights_f2fit'] = {v: round(float(w), 4) for v, w in zip(fs, m2[0])}
        res[name] = R
        log(f'{name:22s} A_pooled {R["A_pooled"]:.6f} A_f1 {R["A_f1"]:.6f} A_f2sh {R["A_f2sh"]:.6f} '
            f'C_pooled {R["C_pooled"]:.6f}')

    inc = res['M3x']
    delta = {}
    for name in arms:
        if name == 'M3x':
            continue
        delta[name] = {k: round(float(res[name][k] - inc[k]), 6)
                       for k in inc if not k.startswith('weights_')
                       and res[name][k] is not None and inc[k] is not None}

    # ---------- rank moves vs the incumbent (positive = promoted toward the top of the block)
    def rank_from_top(s):
        r = np.empty(len(s), float)
        r[np.argsort(-s, kind='stable')] = np.arange(len(s))
        return r
    r_inc = rank_from_top(preds['M3x'])
    moves = {}
    for name in arms:
        if name == 'M3x':
            continue
        mv = r_inc - rank_from_top(preds[name])          # > 0 : moved up
        moves[name] = {
            'mean_rank_move_R1': round(float(mv[R1].mean()), 2),
            'mean_rank_move_notR1': round(float(mv[~R1].mean()), 2),
            'mean_rank_move_cnt999_ge2': round(float(mv[cnt999 >= 2].mean()), 2) if (cnt999 >= 2).any() else None,
            'mean_rank_move_cnt999_eq0': round(float(mv[cnt999 == 0].mean()), 2) if (cnt999 == 0).any() else None,
            'n_cnt999_ge2': int((cnt999 >= 2).sum()), 'n_cnt999_eq0': int((cnt999 == 0).sum()),
            'mean_abs_rank_move': round(float(np.abs(mv).mean()), 2),
            'spearman_vs_M3x': round(float(np.corrcoef(r_inc, rank_from_top(preds[name]))[0, 1]), 5)}

    # ---------- gate G258
    def clauses(name):
        dl = delta[name]
        pv = delta.get('M3x+PERM6', {}).get('A_pooled', None)
        c = {'c1_pooled_ge_.003': bool(dl['A_pooled'] >= .003),
             'c2_both_directions_gt0': bool(dl['A_f1'] > 0 and dl['A_f2sh'] > 0),
             'c3_defC_pooled_ge0': bool(dl['C_pooled'] >= 0),
             'c4_ge_2x_PERM6': (None if pv is None else bool(dl['A_pooled'] >= 2 * pv))}
        c['PASS'] = bool(all(v for v in c.values() if v is not None)) and c['c4_ge_2x_PERM6'] is not None
        return c
    gate = {n: clauses(n) for n in ('M3x+MECH6', 'M3x+MECH6+MECH6noT') if n in delta}
    sec = 'M3x+MECH6+MECH6noT'
    pri = 'M3x+MECH6'
    replace = None
    if sec in gate and pri in delta:
        replace = bool(gate[sec]['PASS']
                       and (delta[sec]['A_pooled'] - delta[pri]['A_pooled']) >= .002
                       and delta[sec]['A_f1'] >= delta[pri]['A_f1']
                       and delta[sec]['A_f2sh'] >= delta[pri]['A_f2sh'])
    build_arm = None
    if gate.get(sec, {}).get('PASS') and replace:
        build_arm = sec
    elif gate.get(pri, {}).get('PASS'):
        build_arm = pri

    out = {'step': 258, 'registration': 'registration258.json',
           'reproduction': {k: repro[k] for k in ('G_fit_M3_reproduces_multiview_M3_fit', 'G_fold_match_maxdiff',
                                                  'G_fit_M3x_reproduces_receipt_step156a',
                                                  'G_ap_M3x_reproduces_block_screen',
                                                  'incumbent_M3x_crossfold_defA_trained')},
           'parity': json.load(open(OUT + 'aggregates258_receipt.json', encoding='utf-8'))['parity_full149'],
           'T_gate': json.load(open(OUT + 'aggregates258_receipt.json', encoding='utf-8'))['T_gate'],
           'meta': meta,
           'arms': {n: {k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in r.items()}
                    for n, r in res.items()},
           'delta_vs_M3x': delta, 'rank_moves_vs_M3x': moves,
           'gate_G258': gate, 'secondary_replaces_primary': replace, 'build_arm': build_arm,
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(DEST, 'w', encoding='utf-8'), indent=1)
    log(json.dumps({'delta_vs_M3x': delta, 'gate_G258': gate, 'build_arm': build_arm}, indent=1))
    log('wrote', DEST)


if __name__ == '__main__':
    main()
