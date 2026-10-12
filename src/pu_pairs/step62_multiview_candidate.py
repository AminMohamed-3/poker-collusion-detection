"""Step 62: candidate M = C's R1 demotion + LEARNED multi-view blend of the evaluation block (pre-registered).
Step 60 (OBS 14.37): on the 60 %-prefix testbed a logistic blend of the eight view logits (FULL, RAW, CA, BASE, OPP, V2, MEAN,
TAILS) + log(shared_hands), l2 = 10, beats the fixed logit(FULL)+.5 logit(RAW) blend that candidate S carried (+.0143 public):
def A cross-fold .9614 vs .9547 (test fold 2 + shared) and .9710 vs .9634 (test fold 1). This step
 (1) decides on the testbed whether the reorder should stay inside the block (risk >= .5, as S) or extend to the [.1, .5) tail,
     comparing frozen / S-like fixed blend / learned block-only / learned global orders at floor .1 and .5 (development labels =
     model selection, allowed);
 (2) fits the final blend on the union of both folds' certified block rows (def A: pos + undisclosed vs FP + undisclosed_R1);
 (3) scores the evaluation block from multiview_eval.parquet (FULL = the frozen v13 risk, i.e. the submission values), runs the
     step-59 scenario sweep with orders C / S / M under RAW-informed and M-informed false-positive placement, and pre-registers
     an interval (cells consistent with AP_before in [.93, .946] AND with the measured S delta, dS in [.010, .019]);
 (4) builds the candidate exactly as S: R1 pairs (205) -> [.49, .4949] by risk_noT (identical to D and S), the other 704 block
     pairs get the SAME sorted multiset of base values re-assigned in M order; every other row byte-identical to the base.
Prediction JSON is written before the CSV; receipt after validation and the protected-file check.
Outputs: experiments/viewdis_20260911/{tail_decision_M.json, eval_scenarios_M.json, multiview_final_blend.json},
         outputs_multiview_20260911/{prediction_multiview.json, submission_multiview_learned_plus_R1.csv, receipt_multiview.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
import numpy as np
import polars as pl
from pu_pairs.build_allpairs_fold import sha, fold_pools
from pu_pairs.step60_multiview_blend import lg, ap, logreg, predict, NEW_VIEWS
from pu_pairs.step59_blend_candidate import ap_block, band, BASE, CFILE

VD = CV / 'experiments' / 'viewdis_20260911'
SP = CV / 'experiments' / 'spillover_20260911'
OUT = CV / 'outputs_multiview_20260911'
SFILE = CV / 'outputs_blend_20260911' / 'submission_blend_w0.5_plus_R1.csv'
VIEWS = ['FULL', 'RAW', 'CA'] + NEW_VIEWS
FEATS = VIEWS + ['sh']
L2 = 10.0
DEFS = {'A': (['pos', 'undisclosed'], ['FP', 'undisclosed_R1']), 'C': (['pos'], ['FP', 'undisclosed_R1', 'ambiguous', 'mid', 'neg'])}


def log(*a):
    print(*a, flush=True)


def X_of(c, full_col, sh_col='shared_hands'):
    cols = []
    for v in FEATS:
        if v == 'sh': cols.append(np.log(c[sh_col].to_numpy().astype(float)))
        elif v == 'FULL': cols.append(lg(c[full_col].to_numpy()))
        else: cols.append(lg(c[f'r_{v}'].to_numpy()))
    return np.column_stack(cols)


def tail_decision(d):
    """Testbed AP over the floor-.1 population under different reorder scopes; learned scores are cross-fold (never fitted on the scored table)."""
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
    res = {}
    for dn, (posset, negset) in DEFS.items():
        c = d.filter((pl.col('risk_prefix') >= .1) & pl.col('cls').is_in(posset + negset)).sort('risk_prefix', descending=True)
        y = c['cls'].is_in(posset).to_numpy().astype(float); tab = c['table_id'].to_numpy(); risk = c['risk_prefix'].to_numpy(); raw = c['r_RAW'].to_numpy()
        blk = risk >= .5
        in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
        X = X_of(c, 'risk_prefix')
        def crossfold(rows):
            """learned score for `rows` (bool mask defining the training scope), each row scored by the other fold's model."""
            pred = np.full(len(y), np.nan)
            m1 = logreg(X[in1 & rows], y[in1 & rows], L2); pred[(in2 | insh)] = predict(m1, X[(in2 | insh)])
            m2 = logreg(X[in2 & rows], y[in2 & rows], L2); pred[in1] = predict(m2, X[in1])
            return pred, (m1, m2)
        p_blk, mb = crossfold(blk); p_all, ma = crossfold(np.ones(len(y), bool))
        fixed = lg(risk) + .5 * lg(raw)
        def reorder(score, scope):
            """assign the sorted multiset of frozen risks within `scope` in `score` order; outside scope unchanged."""
            out = risk.copy(); idx = np.where(scope)[0]; o = idx[np.argsort(-score[idx], kind='stable')]; out[o] = np.sort(risk[idx])[::-1]; return out
        tail = (~blk)
        orders = {'frozen': risk, 'S-like: fixed blend, block only': reorder(fixed, blk),
                  'M-block: learned(block fit), block only': reorder(p_blk, blk),
                  'M-block + tail separately (block fit)': np.where(blk, reorder(p_blk, blk), reorder(p_blk, tail)),
                  'M-block + tail separately (global fit for tail)': np.where(blk, reorder(p_blk, blk), reorder(p_all, tail)),
                  'M-global: learned(global fit), [.1,1] one multiset': reorder(p_all, np.ones(len(y), bool)),
                  'fixed blend global [.1,1] one multiset': reorder(fixed, np.ones(len(y), bool))}
        r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum()), 'n_block': int(blk.sum()), 'n_tail': int(tail.sum()), 'pos_in_tail': int(y[tail].sum())}
        for k, s in orders.items():
            r[k] = {'AP_floor.1': round(ap(y, s), 4), 'AP_test_f2sh': round(ap(y[in2 | insh], s[in2 | insh]), 4), 'AP_test_f1': round(ap(y[in1], s[in1]), 4)}
        r['weights_block_fit'] = {'f1': {v: round(float(w), 2) for v, w in zip(FEATS, mb[0][0])}, 'f2': {v: round(float(w), 2) for v, w in zip(FEATS, mb[1][0])}}
        r['weights_global_fit'] = {'f1': {v: round(float(w), 2) for v, w in zip(FEATS, ma[0][0])}, 'f2': {v: round(float(w), 2) for v, w in zip(FEATS, ma[1][0])}}
        res[f'def {dn}'] = r
    return res


def final_fit(d):
    posset, negset = DEFS['A']
    c = d.filter((pl.col('risk_prefix') >= .5) & pl.col('cls').is_in(posset + negset))
    y = c['cls'].is_in(posset).to_numpy().astype(float); X = X_of(c, 'risk_prefix')
    m = logreg(X, y, L2)
    # in-sample AP (reported, not used) and per-fold refits for a stability check
    info = {'n_rows': int(len(y)), 'n_pos': int(y.sum()), 'l2': L2, 'features': FEATS, 'weights_standardised': {v: round(float(w), 3) for v, w in zip(FEATS, m[0])},
            'mu': {v: round(float(u), 3) for v, u in zip(FEATS, m[2])}, 'sd': {v: round(float(s), 3) for v, s in zip(FEATS, m[3])}, 'bias': round(float(m[1]), 3),
            'AP_in_sample': round(ap(y, predict(m, X)), 4), 'AP_frozen_same_rows': round(ap(y, c['risk_prefix'].to_numpy()), 4), 'AP_fixed_same_rows': round(ap(y, lg(c['risk_prefix'].to_numpy()) + .5 * lg(c['r_RAW'].to_numpy())), 4)}
    return m, info


def eval_frame():
    res = pl.read_parquet(SP / 'eval_rescoring.parquet').select('pair_id', 'risk_v13', 'dominated', 'risk_noT')
    mv = pl.read_parquet(VD / 'multiview_eval.parquet').select('pair_id', 'shared_hands', 'r_RAW', 'r_CA', *[f'r_{v}' for v in NEW_VIEWS], 'R1')
    d = res.join(mv, on='pair_id', validate='1:1').with_columns(band('risk_v13').alias('band')).sort('risk_v13', descending=True)
    d = d.with_columns((pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < .1)).alias('T2'))
    assert (d['T2'] == d['R1']).all() and d.height == 909
    return d


def scenarios(d, pM, draws=150):
    risk = d['risk_v13'].to_numpy(); raw = d['r_RAW'].to_numpy(); in2 = d['T2'].to_numpy(); n = d.height
    fixed = lg(risk) + .5 * lg(raw)
    o2 = np.where(in2)[0][np.argsort(-d['risk_noT'].fill_nan(0).to_numpy()[in2], kind='stable')]
    rest = np.where(~in2)[0]
    order_C = np.concatenate([rest, o2]); order_S = np.concatenate([rest[np.argsort(-fixed[rest], kind='stable')], o2]); order_M = np.concatenate([rest[np.argsort(-pM[rest], kind='stable')], o2])
    rank_S = np.empty(n, int); rank_S[order_S] = np.arange(n); rank_M = np.empty(n, int); rank_M[order_M] = np.arange(n)
    mv = rank_M - rank_S
    from scipy.stats import spearmanr
    out = {'n_block': n, 'n_R1': int(in2.sum()), 'spearman_M_vs_S_nonR1': round(float(spearmanr(pM[rest], fixed[rest]).correlation), 4), 'spearman_M_vs_frozen_nonR1': round(float(spearmanr(pM[rest], risk[rest]).correlation), 4),
           'rank_move_M_vs_S_nonR1': {'mean_abs': round(float(np.abs(mv[~in2]).mean()), 1), 'q90_abs': int(np.quantile(np.abs(mv[~in2]), .9)), 'max_abs': int(np.abs(mv[~in2]).max())}, 'moves_by_band': {}}
    for b in ('[.99,1]', '[.9,.99)', '[.5,.9)'):
        m = (d['band'] == b).to_numpy() & ~in2
        out['moves_by_band'][b] = {'n': int(m.sum()), 'down>50_vs_S': int((mv[m] > 50).sum()), 'up>50_vs_S': int((mv[m] < -50).sum())}
    # M-informed placement uses the PERCENTILE RANK of the learned score among block pairs (v1 used the logistic probability,
    # which saturates at 1 over the [.99,1] band and left the top-band false positives uninformed: dM - dS collapsed to ~0 by construction).
    def pct(x):
        r = np.empty(len(x)); r[np.argsort(x, kind='stable')] = np.arange(len(x)) / max(len(x) - 1, 1); return r
    pm = pct(pM)
    rng = np.random.default_rng(62); cells = []; idx2 = np.where(in2)[0]
    for placement in ('RAW', 'M'):
        for miss in (.0, .03, .06):
            for beta in (0, 3, 6):
                for F in (300, 350, 400, 450, 500):
                    for q2 in (.9, 1.0):
                        f2 = int(round(q2 * len(idx2))); r = F - f2
                        if r < 0 or r > len(rest): continue
                        for gamma in (2, 4):
                            inf = (1 - raw[rest]) if placement == 'RAW' else (1 - pm[rest])
                            wgt = ((1 - risk[rest]) + 1e-3) ** gamma * (inf + 1e-3) ** beta; wgt = wgt / wgt.sum()
                            acc = {'C': [], 'S': [], 'M': []}; b0 = []
                            for _ in range(draws):
                                is_fp = np.zeros(n, dtype=bool); is_fp[rng.choice(idx2, size=f2, replace=False)] = True; is_fp[rng.choice(rest, size=r, replace=False, p=wgt)] = True
                                npos = (n - F) / (1 - miss); b0.append(ap_block(is_fp, npos))
                                for name, o in (('C', order_C), ('S', order_S), ('M', order_M)):
                                    acc[name].append(ap_block(is_fp[o], npos))
                            b0 = float(np.mean(b0))
                            cell = {'placement': placement, 'miss': miss, 'beta': beta, 'F': F, 'q2': q2, 'gamma': gamma, 'AP_before': round(b0, 4)}
                            for name in acc: cell[f'd{name}'] = round(.7 * (float(np.mean(acc[name])) - b0), 4)
                            cell['dM_minus_S'] = round(cell['dM'] - cell['dS'], 4); cells.append(cell)
    def rng_(key, sub): return [min(c[key] for c in sub), max(c[key] for c in sub)] if sub else None
    cons = [c for c in cells if .93 <= c['AP_before'] <= .946]; match = [c for c in cons if .010 <= c['dS'] <= .019]
    out['consistent_cells'] = len(cons); out['cells_matching_measured_S'] = len(match)
    out['summary'] = {'consistent': {'dS': rng_('dS', cons), 'dM': rng_('dM', cons), 'dM_minus_S': rng_('dM_minus_S', cons)},
                      'matching_S': {'dS': rng_('dS', match), 'dM': rng_('dM', match), 'dM_minus_S': rng_('dM_minus_S', match)},
                      'matching_S_by_placement': {p: {'n': len([c for c in match if c['placement'] == p]), 'dM_minus_S': rng_('dM_minus_S', [c for c in match if c['placement'] == p])} for p in ('RAW', 'M')},
                      'matching_S_by_beta': {str(b): {'n': len([c for c in match if c['beta'] == b]), 'dM_minus_S': rng_('dM_minus_S', [c for c in match if c['beta'] == b])} for b in (0, 3, 6)}}
    out['cells'] = cells
    return out, order_M, rest


def build(d, pM, interval, fit_info, tail_res, scen):
    OUT.mkdir(exist_ok=True)
    base = pl.read_csv(BASE); bmap = dict(zip(base['pair_id'].to_list(), base['risk_score'].to_list()))
    ids = d['pair_id'].to_list(); in2 = d['T2'].to_numpy(); base_risk = np.array([bmap[i] for i in ids]); assert np.allclose(base_risk, d['risk_v13'].to_numpy(), atol=1e-9)
    non = np.where(~in2)[0]; order = non[np.argsort(-pM[non], kind='stable')]; values = np.sort(base_risk[non])[::-1]
    mapping = {ids[i]: float(v) for i, v in zip(order, values)}
    S = pl.read_csv(SFILE); smap = dict(zip(S['pair_id'].to_list(), S['risk_score'].to_list()))
    for i in np.where(in2)[0]:
        mapping[ids[i]] = float(smap[ids[i]])  # R1 rows exactly as S (= D)
    assert len(mapping) == d.height
    DEST = OUT / 'submission_multiview_learned_plus_R1.csv'
    pred = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'candidate': str(DEST.relative_to(CV)),
            'rule': f'M = C (R1: dominated & risk_noT < .1, {int(in2.sum())} pairs -> [.49,.4949] by risk_noT, rows identical to S and D) + rank permutation of the other {len(non)} block pairs by a logistic blend of the eight view logits (FULL = frozen v13 risk, RAW, CA, BASE, OPP, V2, MEAN, TAILS) + log(shared_hands), l2 = 10, fitted on the certified 60 %-prefix block of folds 1-2 (def A).',
            'construction': 'Base = the measured .87625 submission. The sorted multiset of the non-R1 block risk values is re-assigned in M order (value distribution and block membership unchanged); every other row byte-identical to the base. Differs from S (measured +.0143) only in the order of the 704 non-R1 block pairs.',
            'hypothesis': 'OBS 14.37: on the certified prefix block the learned blend beats the fixed FULL+.5 RAW blend cross-fold under both definitions (def A .9614 vs .9547 and .9710 vs .9634; def C .9656 vs .9554 and .9751 vs .9685). Length-induced false positives are recognised by several partially independent views (V2, OPP, BASE, MEAN add to RAW) and by shared_hands.',
            'label_free': 'Evaluation side uses only frozen risks, views trained on development labels (production recipe), shared_hands and the R1 rule. No leaderboard feedback used to fit anything; the measured S delta is used only to select scenario cells for the forecast.',
            'tail_decision': tail_res, 'final_fit': fit_info,
            'prediction': {'interval_composite_delta_vs_base': interval['vs_base'], 'interval_M_minus_S': interval['vs_S'], 'scenario_file': 'experiments/viewdis_20260911/eval_scenarios_M.json', 'scenario_summary': scen['summary'],
                           'spearman_M_vs_S_nonR1': scen['spearman_M_vs_S_nonR1'], 'rank_moves_vs_S': scen['rank_move_M_vs_S_nonR1'], 'moves_by_band': scen['moves_by_band']},
            'decision_rule': 'Score - .89055 (S) measures the learned blend against the fixed one. > +.002: the multi-view discriminator generalises; next candidates stack step-61 stability/noT features and the tail extension. In [-.002, +.002]: the fixed RAW blend already captured what the views know; stop refining the block order, move to recall (positives below .5) and evidence. < -.002: the learned weights over-fit the prefix (shared_hands weight suspected first); M closes, S stays the best. No re-tuning on outcomes.',
            'not_uploaded_by_claude': True}
    write_json(OUT / 'prediction_multiview.json', pred)
    cand = base.with_columns(pl.col('pair_id').replace_strict(mapping, default=None, return_dtype=pl.Float64).alias('new')).with_columns(pl.coalesce([pl.col('new'), pl.col('risk_score')]).alias('risk_score')).drop('new')
    cand.write_csv(DEST); chk = pl.read_csv(DEST)
    same_cols = all((chk[c] == base[c]).all() for c in base.columns if c != 'risk_score'); changed = int((chk['risk_score'] != base['risk_score']).sum()); changed_vs_S = int((chk['risk_score'] != S['risk_score']).sum())
    r1_ids = [ids[i] for i in np.where(in2)[0]]
    r1_same_as_S = bool((chk.filter(pl.col('pair_id').is_in(r1_ids)).sort('pair_id')['risk_score'] == S.filter(pl.col('pair_id').is_in(r1_ids)).sort('pair_id')['risk_score']).all())
    blk_after = chk.filter(pl.col('risk_score') >= .5); multiset_ok = bool(np.allclose(np.sort(blk_after['risk_score'].to_numpy()), np.sort(base_risk[non])))
    outside_same = bool((chk.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy() == base.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy()).all())
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    prot = check_protected()
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST), 'sha256': sha(DEST), 'base_sha256': sha(BASE), 'S_sha256': sha(SFILE), 'C_sha256': sha(CFILE), 'rule': pred['rule'],
               'n_R1': int(in2.sum()), 'n_reordered_nonR1': len(non), 'rows_with_changed_risk_vs_base': changed, 'rows_with_changed_risk_vs_S': changed_vs_S, 'R1_rows_identical_to_S': r1_same_as_S, 'block_value_multiset_unchanged': multiset_ok,
               'rows_outside_block_identical_to_base': outside_same, 'non_risk_columns_identical_to_base': bool(same_cols), 'block_ge_.5_after': int(blk_after.height), 'validation': vrep, 'protected_check': prot, 'prediction_written_first': True, 'do_not_upload_by_claude': True}
    write_json(OUT / 'receipt_multiview.json', receipt)
    log(json.dumps({k: v for k, v in receipt.items() if k not in ('validation', 'protected_check')}, indent=1)); log('validation:', json.dumps(vrep)[:400])


def main():
    t0 = time.time()
    d = pl.read_parquet(VD / 'multiview_prefix.parquet')
    tail_res = tail_decision(d); write_json(VD / 'tail_decision_M.json', tail_res)
    log(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith('weights')} for k, v in tail_res.items()}, indent=1))
    m, fit_info = final_fit(d); write_json(VD / 'multiview_final_blend.json', fit_info); log(json.dumps(fit_info, indent=1))
    ev = eval_frame(); pM = predict(m, X_of(ev, 'risk_v13'))
    scen, order_M, rest = scenarios(ev, pM); write_json(VD / 'eval_scenarios_M.json', scen)
    log(json.dumps({k: v for k, v in scen.items() if k != 'cells'}, indent=1))
    s = scen['summary']['matching_S'] if scen['cells_matching_measured_S'] else scen['summary']['consistent']
    # testbed transfer: S measured +.0143 public for a testbed block delta of +.0138 (def A); take the cross-fold M-block minus S-like deltas as the second source
    tb = [tail_res[dn]['M-block: learned(block fit), block only'][t] - tail_res[dn]['S-like: fixed blend, block only'][t] for dn in ('def A', 'def C') for t in ('AP_test_f2sh', 'AP_test_f1')]
    lo, hi = min(min(tb), s['dM_minus_S'][0]), max(max(tb), s['dM_minus_S'][1])
    interval = {'vs_S': [round(lo - .002, 3), round(hi + .002, 3)], 'vs_base': [round(.0143 + lo - .003, 3), round(.0143 + hi + .003, 3)],
                'testbed_M_minus_S_crossfold': [round(x, 4) for x in tb], 'scenario_M_minus_S_matching_cells': s['dM_minus_S'],
                'note': 'Union of (a) the cross-fold testbed deltas M-block minus S-like (transfer factor ~1: S measured +.0143 for a testbed +.0138) and (b) the scenario cells matching the measured S (AP_before in [.93,.946], dS in [.010,.019]) under RAW-probability and M-rank placements; widened by +-.002 (vs S) / +-.003 (vs base). Point expectation: testbed transfer, M - S about +.006.'}
    log('interval', interval)
    build(ev, pM, interval, fit_info, {k: {kk: vv for kk, vv in v.items() if not kk.startswith('weights')} for k, v in tail_res.items()}, scen)
    log('seconds', round(time.time() - t0))


if __name__ == '__main__':
    main()
