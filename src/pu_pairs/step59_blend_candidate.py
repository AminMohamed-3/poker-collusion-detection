"""Step 59: candidate S = C + a SMOOTH RAW-informed reorder of the non-R1 block (label-free on evaluation; development
labels used only on the 60 %-prefix testbed of step 58 for validation and for choosing the blend weight).
Step 58 found, on the prefix testbed (folds 1-2 held pools at evaluation-like history length, 335 certified
length-induced false positives in the block against 146 labelled + 75 undisclosed positives), that the RAW view is
informative throughout the block, not only below .05: ordering the block by logit(FULL) + w * logit(RAW) raises the
testbed block AP by +.014 to +.020 (w = .5) against +.001 to +.004 for the RAW < .05 tier of candidate D.
Part 1 (development): AP of the variants on the testbed under three positive / negative definitions, table bootstrap.
Part 2 (evaluation, label-free): scenario sweep of the exact candidate order (R1 at the bottom as in C; the other block
pairs by the blend) against C and D, FP placement RAW-blind (beta 0) and RAW-informed (beta 3, 6).
Part 3 (--build): the candidate is a rank permutation of the non-R1 block: the sorted multiset of the base's non-R1 block
risk values is re-assigned in blend order, so the value distribution and the block membership are unchanged; R1 pairs
receive [.49, .4949] by risk_noT exactly as in C; every other row is byte-identical to the .87625 base. Prediction
written before the CSV; validation, protected hashes and a receipt follow.
Outputs: experiments/viewdis_20260911/{blend_testbed.json, eval_scenarios_S.json}; outputs_blend_20260911/{prediction_blend.json,
submission_blend_w{w}_plus_R1.csv, receipt_blend.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, argparse
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected, sha
import numpy as np
import polars as pl

VD = CV / 'experiments' / 'viewdis_20260911'
SP = CV / 'experiments' / 'spillover_20260911'
OUT = CV / 'outputs_blend_20260911'
BASE = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
CFILE = CV / 'outputs_spillover_20260911' / 'submission_spillover_demotion_R1.csv'


def lg(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def ap(y, s):
    o = np.argsort(-s, kind='stable'); y = y[o]; tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / max(1, y.sum()))


def ap_block(is_fp, n_pos_total):
    tp = (~is_fp).astype(float); cum = np.cumsum(tp); ranks = np.arange(1, len(tp) + 1)
    return float((tp * cum / ranks).sum() / n_pos_total)


def blend_scores(risk, raw, w, below):
    """logit(FULL) + w * logit(RAW); pairs with risk >= below keep their own order above everything else."""
    s = lg(risk) + w * lg(raw)
    if below < 1.0:
        top = risk >= below
        s = s.copy(); s[top] = 1e6 + risk[top]
    return s


def band(c):
    return pl.when(pl.col(c) >= .99).then(pl.lit('[.99,1]')).when(pl.col(c) >= .9).then(pl.lit('[.9,.99)')).when(pl.col(c) >= .5).then(pl.lit('[.5,.9)')).otherwise(pl.lit('<.5'))


def testbed():
    d = pl.read_parquet(VD / 'prefix_testbed_views.parquet')
    vd = pl.read_parquet(VD / 'view_dev.parquet').select('pair_id', pl.col('R1').alias('R1_full'))
    d = d.join(vd, on='pair_id', how='left').filter(pl.col('risk_prefix') >= .5)
    d = d.with_columns(pl.when(pl.col('label') == 1).then(pl.lit('pos')).when(pl.col('label') == 0).then(pl.lit('neg'))
                       .when((pl.col('risk_full') >= .9) & ~pl.col('R1_full').fill_null(False)).then(pl.lit('undisclosed'))
                       .when(pl.col('risk_full') >= .9).then(pl.lit('undisclosed_R1'))
                       .when(pl.col('risk_full') >= .5).then(pl.lit('ambiguous')).when(pl.col('risk_full') >= .1).then(pl.lit('mid')).otherwise(pl.lit('FP')).alias('cls'))
    defs = {'A: pos+undisclosed vs FP+undisclosed_R1': (['pos', 'undisclosed'], ['FP', 'undisclosed_R1']),
            'B: pos+undisclosed vs all other block pairs': (['pos', 'undisclosed'], ['FP', 'undisclosed_R1', 'ambiguous', 'mid', 'neg']),
            'C: labelled pos vs all other block pairs': (['pos'], ['FP', 'undisclosed_R1', 'ambiguous', 'mid', 'neg'])}
    variants = {'frozen': lambda r, w_: r, 'tier RAW<.05': None}
    for w in (.25, .5, 1.0):
        variants[f'blend w={w} all'] = (w, 1.0); variants[f'blend w={w} below .99'] = (w, .99)
    res = {'class_counts': {str(k): int(v) for k, v in d.group_by('cls').len().iter_rows()}}
    rng = np.random.default_rng(59)
    for dname, (posset, negset) in defs.items():
        c = d.filter(pl.col('cls').is_in(posset + negset))
        y = c['cls'].is_in(posset).to_numpy().astype(float); risk = c['risk_prefix'].to_numpy(); raw = c['r_RAW'].to_numpy(); tabs = c['table_id'].to_numpy()
        scores = {'frozen': risk}
        t = risk.copy(); m = raw < .05
        if m.sum():
            t[m] = .49 + .0099 * (raw[m] - raw[m].min()) / max(1e-9, raw[m].max() - raw[m].min())
        scores['tier RAW<.05'] = t
        for name, v in variants.items():
            if isinstance(v, tuple):
                scores[name] = blend_scores(risk, raw, v[0], v[1])
        r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum()), 'AP': {k: round(ap(y, s), 4) for k, s in scores.items()}}
        ut = np.unique(tabs); idxs = {tb: np.where(tabs == tb)[0] for tb in ut}
        boots = {k: [] for k in scores if k != 'frozen'}
        for _ in range(400):
            pick = rng.choice(ut, size=len(ut), replace=True); idx = np.concatenate([idxs[tb] for tb in pick])
            b0 = ap(y[idx], risk[idx])
            for k in boots:
                boots[k].append(ap(y[idx], scores[k][idx]) - b0)
        r['delta_vs_frozen_CI95'] = {k: [round(float(np.quantile(v, .025)), 4), round(float(np.quantile(v, .975)), 4)] for k, v in boots.items()}
        r['delta_vs_frozen_share_positive'] = {k: round(float(np.mean(np.array(v) > 0)), 3) for k, v in boots.items()}
        res[dname] = r
    return res


def scenarios(w, below, draws=150):
    res = pl.read_parquet(SP / 'eval_rescoring.parquet').select('pair_id', 'risk_v13', 'dominated', 'risk_noT')
    ve = pl.read_parquet(VD / 'view_eval.parquet').select('pair_id', 'r_RAW', 'R1')
    d = res.join(ve, on='pair_id', validate='1:1').with_columns(band('risk_v13').alias('band')).sort('risk_v13', descending=True)
    d = d.with_columns((pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < .1)).alias('T2'))
    assert (d['T2'] == d['R1']).all()
    d = d.with_columns((~pl.col('T2') & (pl.col('r_RAW') < .05)).alias('T1'))
    risk = d['risk_v13'].to_numpy(); raw = d['r_RAW'].to_numpy(); in1 = d['T1'].to_numpy(); in2 = d['T2'].to_numpy(); n = d.height
    o1 = np.where(in1)[0][np.argsort(-raw[in1], kind='stable')]
    o2 = np.where(in2)[0][np.argsort(-d['risk_noT'].fill_nan(0).to_numpy()[in2], kind='stable')]
    rest_C = np.where(~in2)[0]; rest_D = np.where(~in1 & ~in2)[0]
    bl = blend_scores(risk, raw, w, below)
    order_C = np.concatenate([rest_C, o2]); order_D = np.concatenate([rest_D, o1, o2])
    order_S = np.concatenate([rest_C[np.argsort(-bl[rest_C], kind='stable')], o2])
    # movement summary of S relative to the base order within the block
    rank_base = np.empty(n, int); rank_base[np.arange(n)] = np.arange(n)
    rank_S = np.empty(n, int); rank_S[order_S] = np.arange(n)
    mv = rank_S - rank_base
    out = {'w': w, 'below': below, 'n_block': n, 'n_R1': int(in2.sum()), 'n_T1': int(in1.sum()),
           'rank_move_nonR1': {'mean_abs': round(float(np.abs(mv[~in2]).mean()), 1), 'q90_abs': int(np.quantile(np.abs(mv[~in2]), .9)), 'max_abs': int(np.abs(mv[~in2]).max())},
           'moves_by_band': {}}
    for b in ('[.99,1]', '[.9,.99)', '[.5,.9)'):
        m = (d['band'] == b).to_numpy() & ~in2
        out['moves_by_band'][b] = {'n': int(m.sum()), 'down>50': int((mv[m] > 50).sum()), 'up>50': int((mv[m] < -50).sum()), 'mean_move': round(float(mv[m].mean()), 1)}
    rng = np.random.default_rng(59); cells = []
    idx1 = np.where(in1)[0]; idx2 = np.where(in2)[0]
    for miss in (.0, .03, .06):
        for beta in (0, 3, 6):
            for F in (300, 350, 400, 450, 500):
                for q2, q1 in ((.9, .85), (1.0, .95)):
                    f1, f2 = int(round(q1 * len(idx1))), int(round(q2 * len(idx2))); r = F - f1 - f2
                    if r < 0 or r > len(rest_D):
                        continue
                    for gamma in (2, 4):
                        wgt = ((1 - risk[rest_D]) + 1e-3) ** gamma * ((1 - raw[rest_D]) + 1e-3) ** beta; wgt = wgt / wgt.sum()
                        acc = {'C': [], 'D': [], 'S': []}; b0 = []
                        for _ in range(draws):
                            is_fp = np.zeros(n, dtype=bool)
                            is_fp[rng.choice(idx1, size=f1, replace=False)] = True
                            is_fp[rng.choice(idx2, size=f2, replace=False)] = True
                            is_fp[rng.choice(rest_D, size=r, replace=False, p=wgt)] = True
                            npos = (n - F) / (1 - miss)
                            b0.append(ap_block(is_fp, npos))
                            for name, o in (('C', order_C), ('D', order_D), ('S', order_S)):
                                acc[name].append(ap_block(is_fp[o], npos))
                        b0 = float(np.mean(b0))
                        cell = {'miss': miss, 'beta': beta, 'F': F, 'q2': q2, 'q1': q1, 'gamma': gamma, 'AP_before': round(b0, 4)}
                        for name in acc:
                            cell[f'd{name}'] = round(.7 * (float(np.mean(acc[name])) - b0), 4)
                        cell['dS_minus_C'] = round(cell['dS'] - cell['dC'], 4); cell['dS_minus_D'] = round(cell['dS'] - cell['dD'], 4)
                        cells.append(cell)
    cons = [c for c in cells if .93 <= c['AP_before'] <= .946]
    def rng_(key, sub): return [min(c[key] for c in sub), max(c[key] for c in sub)] if sub else None
    out['consistent_cells'] = len(cons)
    out['summary'] = {'dC': rng_('dC', cons), 'dD': rng_('dD', cons), 'dS': rng_('dS', cons), 'dS_minus_C': rng_('dS_minus_C', cons), 'dS_minus_D': rng_('dS_minus_D', cons),
                      'by_beta': {str(b): {'n': len([c for c in cons if c['beta'] == b]), 'dS': rng_('dS', [c for c in cons if c['beta'] == b]), 'dS_minus_C': rng_('dS_minus_C', [c for c in cons if c['beta'] == b]), 'dS_minus_D': rng_('dS_minus_D', [c for c in cons if c['beta'] == b])} for b in (0, 3, 6)},
                      'by_miss': {str(m): {'n': len([c for c in cons if c['miss'] == m]), 'dS': rng_('dS', [c for c in cons if c['miss'] == m]), 'dS_minus_C': rng_('dS_minus_C', [c for c in cons if c['miss'] == m])} for m in (.0, .03, .06)}}
    out['cells'] = cells
    return out, d, bl, order_S


def build(w, below, d, bl, interval, testbed_res, scen):
    OUT.mkdir(exist_ok=True)
    base = pl.read_csv(BASE)
    assert base.columns == ['pair_id', 'risk_score', 'predicted_behavior', 'evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4', 'evidence_hand_5']
    bmap = dict(zip(base['pair_id'].to_list(), base['risk_score'].to_list()))
    ids = d['pair_id'].to_list(); in2 = d['T2'].to_numpy()
    base_risk = np.array([bmap[i] for i in ids]); assert np.allclose(base_risk, d['risk_v13'].to_numpy(), atol=1e-9)
    non = np.where(~in2)[0]
    order = non[np.argsort(-bl[non], kind='stable')]
    values = np.sort(base_risk[non])[::-1]
    mapping = {ids[i]: float(v) for i, v in zip(order, values)}
    T2 = d.filter(pl.col('T2')).sort('risk_noT', descending=True, nulls_last=True)
    def spread(x, lo, hi):
        x = np.asarray(x, dtype=float); return lo + (hi - lo) * (x - x.min()) / max(x.max() - x.min(), 1e-9)
    m2 = dict(zip(T2['pair_id'].to_list(), spread(T2['risk_noT'].fill_nan(0).to_numpy(), .49, .4949).tolist()))
    mapping.update(m2); assert len(mapping) == d.height
    DEST = OUT / f'submission_blend_w{w}_plus_R1.csv'
    pred = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'candidate': str(DEST.relative_to(CV)),
            'rule': f'S = C (R1: dominated & risk_noT < .1, {T2.height} pairs -> [.49,.4949] by risk_noT) + rank permutation of the other {len(non)} block pairs by logit(FULL) + {w} * logit(RAW)' + ('' if below >= 1 else f' (pairs with risk >= {below} keep their order on top)'),
            'construction': 'Base = the measured .87625 submission. The sorted multiset of the non-R1 block risk values is re-assigned in blend order (value distribution and block membership unchanged); R1 exactly as candidate C; every other row byte-identical.',
            'hypothesis': ('OBS 14.36: on the 60 %-prefix testbed (evaluation-like length, 335 certified length-induced false positives vs 221 positives in the block) the RAW view is informative '
                           'throughout the block: the blend raises the testbed block AP by +.014 to +.020 against +.001 to +.004 for the RAW < .05 tier; false positives have RAW median .09, '
                           'labelled positives .99, undisclosed positives .95.'),
            'label_free': 'Evaluation side uses only the frozen v13 risk, the RAW view (trained on the same development labels as production) and the R1 rule. No leaderboard feedback used.',
            'dev_validation_summary': testbed_res, 'blend_weight_selection': 'w chosen on the prefix testbed (development labels, model selection): w = .5 best under all three positive/negative definitions.',
            'prediction': {'interval_composite_delta': interval, 'scenario_file': 'experiments/viewdis_20260911/eval_scenarios_S.json', 'scenario_summary': scen['summary'], 'rank_moves': {'rank_move_nonR1': scen['rank_move_nonR1'], 'moves_by_band': scen['moves_by_band']}},
            'decision_rule': ('Score - .87625 measures C + blend jointly. > +.003: the false-positive population is RAW-informed as the testbed says; C is then uploaded to separate the two parts '
                              '(S - C > +.002 confirms the blend, the next step is w = 1 or FULL+RAW+CA). <= -.003: the block false positives are not RAW-low at evaluation; S and D close, C stays '
                              'pre-registered on its own. No re-tuning on outcomes.'),
            'not_uploaded_by_claude': True}
    write_json(OUT / 'prediction_blend.json', pred)
    cand = base.with_columns(pl.col('pair_id').replace_strict(mapping, default=None, return_dtype=pl.Float64).alias('new')).with_columns(
        pl.coalesce([pl.col('new'), pl.col('risk_score')]).alias('risk_score')).drop('new')
    cand.write_csv(DEST)
    chk = pl.read_csv(DEST)
    same_cols = all((chk[c] == base[c]).all() for c in base.columns if c != 'risk_score')
    changed = int((chk['risk_score'] != base['risk_score']).sum())
    C = pl.read_csv(CFILE)
    Dc = pl.read_csv(CV / 'outputs_rawlow_20260911' / 'submission_rawlow_tier_thr0.05_plus_R1.csv')
    cj = chk.join(C.select('pair_id', pl.col('risk_score').alias('risk_C')), on='pair_id').join(Dc.select('pair_id', pl.col('risk_score').alias('risk_D')), on='pair_id').filter(pl.col('pair_id').is_in(T2['pair_id'].to_list()))
    from scipy.stats import spearmanr
    r1_same_as_D = bool((cj['risk_score'] == cj['risk_D']).all()); r1_order_vs_C = float(spearmanr(cj['risk_score'].to_numpy(), cj['risk_C'].to_numpy()).correlation)
    blk_after = chk.filter(pl.col('risk_score') >= .5)
    multiset_ok = bool(np.allclose(np.sort(blk_after['risk_score'].to_numpy()), np.sort(base_risk[non])))
    outside_same = bool((chk.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy() == base.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy()).all())
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    prot = check_protected()
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST), 'sha256': sha(DEST), 'base_sha256': sha(BASE), 'C_sha256': sha(CFILE), 'rule': pred['rule'],
               'n_R1': T2.height, 'n_reordered_nonR1': len(non), 'rows_with_changed_risk': changed, 'R1_rows_identical_to_D': r1_same_as_D, 'R1_order_spearman_vs_C': r1_order_vs_C, 'note_R1_values': 'R1 gets [.49,.4949] (step-55 convention shared with D); C spread the same pairs over [.49,.4999]: same order, different spacing.', 'block_value_multiset_unchanged': multiset_ok,
               'rows_outside_block_identical_to_base': outside_same, 'non_risk_columns_identical_to_base': bool(same_cols), 'block_ge_.5_after': int(blk_after.height),
               'validation': vrep, 'protected_check': prot, 'prediction_written_first': True, 'do_not_upload_by_claude': True}
    write_json(OUT / 'receipt_blend.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('validation', 'protected_check')}, indent=1))
    print('validation:', json.dumps(vrep)[:400])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--w', type=float, default=.5); p.add_argument('--below', type=float, default=1.0)
    p.add_argument('--build', action='store_true'); p.add_argument('--interval', type=float, nargs=2, default=None)
    a = p.parse_args()
    tb = testbed(); write_json(VD / 'blend_testbed.json', tb); print(json.dumps(tb, indent=1))
    scen, d, bl, order_S = scenarios(a.w, a.below)
    write_json(VD / 'eval_scenarios_S.json', scen); print(json.dumps({k: v for k, v in scen.items() if k != 'cells'}, indent=1))
    if a.build:
        assert a.interval is not None
        summary = {k: v for k, v in tb.items()}
        for k in summary:
            if isinstance(summary[k], dict) and 'AP' in summary[k]:
                summary[k] = {'n_pos': summary[k]['n_pos'], 'n_neg': summary[k]['n_neg'], 'AP_frozen': summary[k]['AP']['frozen'], f'AP_blend_w{a.w}': summary[k]['AP'][f'blend w={a.w} ' + ('all' if a.below >= 1 else 'below .99')],
                              'AP_tier': summary[k]['AP']['tier RAW<.05'], 'CI_blend_minus_frozen': summary[k]['delta_vs_frozen_CI95'][f'blend w={a.w} ' + ('all' if a.below >= 1 else 'below .99')]}
        build(a.w, a.below, d, bl, a.interval, summary, scen)


if __name__ == '__main__':
    main()
