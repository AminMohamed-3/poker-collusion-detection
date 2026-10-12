"""Step 48b (development side; labels used ONLY to group the report and to validate rules, never to choose anything):
spill-over re-scoring test with known positives as controls, plus the AP effect of candidate demotion rules.

Development held pools (fold 1 and fold 2, OOF risk from experiments/pu_pairs_20260911/fold{k}/scores.parquet): the block
(risk >= .5) holds 233 / 246 pairs: 89 / 92 labelled positives, 144 / 154 unlabelled pairs, no labelled negative.
126 unlabelled and 32 labelled-positive block pairs are "dominated" (share a player with a higher-risk block pair).
Per-hand frames for the block pairs are rebuilt with the fold-frame code (base_features / v2_features / conditional_features
with the fold policy) and cached; parity with scores.parquet is asserted. Variants as in step 48 (noT / rand / evdrop /
noT_keepEv). Listed evidence: truth (development_evidence.csv) for labelled positives; for every pair a PROXY listing =
top-5 hands by the per-hand conditional feature that best flags truth-listed hands among positives (chosen here, reported).
Prediction if the spill-over hypothesis holds: dominated UNLABELLED pairs collapse under noT but not under rand / evdrop;
labelled positives (rings included) keep their risk. Rules R1-R3 are then applied to the fold scores to measure AP
(labelled positives vs everything else, and labelled-only) before / after and the number of labelled positives demoted.
Outputs: Claude Version/experiments/spillover_20260911/dev_rescoring.parquet, dev_rescoring.json, dev_fold{k}_{raw,ca}.parquet."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time, gc
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from pu_pairs.build_allpairs_fold import CONF, STYLE, EXP, fold_policy, base_features, v2_features, conditional_features
from pu_pairs.step48_spillover_rescoring import dominance, band, make_scorer, variants, add_flags, group_summary

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'spillover_20260911'
THR = .5


def build_frames(fold, block, t0):
    fr, fc = OUT / f'dev_fold{fold}_raw.parquet', OUT / f'dev_fold{fold}_ca.parquet'
    if fr.exists() and fc.exists():
        raw, ca = pl.read_parquet(fr), pl.read_parquet(fc)
        if set(raw['pair_id']) == set(block['pair_id']):
            return raw, ca
    policy = joblib.load(fold_policy(fold))
    caf = [c for c in pl.read_parquet(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', n_rows=1).columns if c.startswith('ca_')]
    counts = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(STYLE).iter_rows()}
    tables = sorted(set(block['table_id']))
    hands_all = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter((pl.col('phase') == 'development') & pl.col('table_id').is_in(tables))
    raws, cas = [], []
    for i, t in enumerate(tables):
        h = hands_all.filter(pl.col('table_id') == t)
        p = block.filter(pl.col('table_id') == t).select('pair_id', 'player_1', 'player_2', 'table_id')
        raw = base_features(h, p)
        raw = v2_features(raw, p)
        keys = raw.select('pair_id', 'hand_id', 'table_id').join(p.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        ca, _, _ = conditional_features(keys, h.select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind'), policy, counts, caf)
        raws.append(raw); cas.append(ca)
        gc.collect()
        if i % 20 == 0:
            print(json.dumps({'fold': fold, 'table': i, 'of': len(tables), 'seconds': round(time.time() - t0)}), flush=True)
    raw = pl.concat(raws, how='vertical_relaxed').sort('pair_id', 'hand_id')
    ca = pl.concat(cas, how='vertical_relaxed').sort('pair_id', 'hand_id')
    raw.write_parquet(fr); ca.write_parquet(fc)
    return raw, ca


def run_fold(fold, t0):
    scores = pl.read_parquet(EXP / f'fold{fold}' / 'scores.parquet')
    block = scores.filter(pl.col('risk') >= THR).sort('pair_id')
    ids = block['pair_id'].to_list()
    dom = dominance(block)
    raw, ca = build_frames(fold, block, t0)
    assert raw.select('pair_id', 'hand_id').equals(ca.select('pair_id', 'hand_id'))
    caf = [c for c in ca.columns if c.startswith('ca_')]
    bundle = joblib.load(CONF / 'results' / f'fold{fold}' / 'conditional-pair.joblib'); model = bundle['model']; features = list(bundle['features'])
    names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    events = [c for c in names if c.startswith('v2_')]
    score = make_scorer(model, features, names, events, caf)
    full, _ = score(raw, ca)
    oof = dict(block.select('pair_id', 'risk').iter_rows())
    diff = np.array([abs(full[p] - oof[p]) for p in ids])
    parity = {'max_abs_diff': float(diff.max()), 'n_gt_1e-6': int((diff > 1e-6).sum())}
    print(json.dumps({'fold': fold, 'parity': parity, 'block': len(ids), 'dominated': len(dom), 'pair_hands': raw.height, 'seconds': round(time.time() - t0)}), flush=True)

    hands = sorted(set(raw['hand_id']))
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet', columns=['hand_id', 'player_id']).filter(pl.col('hand_id').is_in(hands))
    dealt = defaultdict(set)
    for h_, p_ in seats.iter_rows():
        dealt[h_].add(p_)
    label = dict(block.select('pair_id', 'label').iter_rows())
    fam = dict(block.select('pair_id', 'behavior_family').iter_rows())
    tev = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').filter(pl.col('pair_id').is_in(ids)).sort('pair_id', 'evidence_rank')
    truth_ev = dict(tev.group_by('pair_id', maintain_order=True).agg(pl.col('hand_id')).iter_rows())
    # proxy listing: the per-hand conditional feature that best flags truth-listed hands among labelled positives (pooled AUC)
    pos_ids = [p for p in ids if label[p] == 1 and p in truth_ev]
    ph = ca.filter(pl.col('pair_id').is_in(pos_ids))
    listed = np.array([h_ in set(truth_ev[p]) for p, h_ in ph.select('pair_id', 'hand_id').iter_rows()])
    proxy_auc = {c: float(roc_auc_score(listed, ph[c].to_numpy())) for c in caf}
    proxy_auc = {c: max(a, 1 - a) for c, a in proxy_auc.items()}
    proxy_feat = max(proxy_auc, key=proxy_auc.get)
    sign = 1.0 if roc_auc_score(listed, ph[proxy_feat].to_numpy()) >= .5 else -1.0
    proxy_ev = {}
    for p, hs, vs in ca.select('pair_id', 'hand_id', proxy_feat).group_by('pair_id').agg(pl.col('hand_id'), pl.col(proxy_feat)).iter_rows():
        order = sorted(zip([-sign * v for v in vs], hs))[:5]
        proxy_ev[p] = [h_ for _, h_ in order]
    proxy_recall = float(np.mean([len(set(proxy_ev[p]) & set(truth_ev[p])) / min(5, len(truth_ev[p])) for p in pos_ids]))
    hands_of = dict(raw.group_by('pair_id').agg(pl.col('hand_id')).iter_rows())
    info, T_of = {}, {}
    for pid, p1, p2 in block.select('pair_id', 'player_1', 'player_2').iter_rows():
        counts_all = defaultdict(int)
        for h_ in hands_of[pid]:
            for q in dealt[h_] - {p1, p2}:
                counts_all[q] += 1
        top = max(counts_all.items(), key=lambda kv: (kv[1], kv[0]))
        if pid in dom:
            T, kind = dom[pid][2], 'dominator_partner'
        else:
            T, kind = top[0], 'max_copresent_all_hands'
        T_of[pid] = T
        te = truth_ev.get(pid, [])
        info[pid] = {'T': T, 'T_kind': kind, 'cop_T_truth_evidence': float(np.mean([T in dealt[h_] for h_ in te])) if te else None, 'n_truth_evidence': len(te),
                     'cop_T_proxy_evidence': float(np.mean([T in dealt[h_] for h_ in proxy_ev[pid]])),
                     'dominated': pid in dom, 'dom_risk': dom[pid][0] if pid in dom else None, 'dom_pair': dom[pid][1] if pid in dom else None,
                     'dom_label': label[dom[pid][1]] if pid in dom else None, 'label': label[pid], 'behavior_family': fam[pid]}
    varP = variants(raw, ca, score, ids, T_of, proxy_ev, dealt, seed_base=100 * fold)
    varT = variants(raw, ca, score, pos_ids, T_of, truth_ev, dealt, seed_base=100 * fold + 50) if pos_ids else {}
    rows = []
    for p in ids:
        r = {'fold': fold, 'pair_id': p, 'risk_oof': oof[p], **varP[p], **info[p]}
        if p in varT:
            r.update({'risk_evdrop_truth_mean': varT[p]['risk_evdrop_mean'], 'risk_noT_keepTruthEv': varT[p]['risk_noT_keepEv']})
        else:
            r.update({'risk_evdrop_truth_mean': None, 'risk_noT_keepTruthEv': None})
        rows.append(r)
    meta = {'parity': parity, 'proxy_feature': proxy_feat, 'proxy_sign': sign, 'proxy_auc_listed': round(proxy_auc[proxy_feat], 4), 'proxy_recall_at5_truth': round(proxy_recall, 3),
            'proxy_top5_features': sorted(proxy_auc.items(), key=lambda kv: -kv[1])[:5]}
    return pl.DataFrame(rows), meta, scores


def ap_rules(df, scores_by_fold):
    """Apply candidate rules to the fold scores (all pairs) and report AP before / after. Labels only for evaluation."""
    out = {}
    d = {(f, p): row for f, p, row in zip(df['fold'].to_list(), df['pair_id'].to_list(), df.to_dicts())}
    conds = {
        'R1 dominated & risk_noT<.1 -> risk_noT': lambda r: r['dominated'] and (r['risk_noT'] != r['risk_noT'] or r['risk_noT'] < .1),
        'R1b dominated & risk_noT<.5 -> risk_noT': lambda r: r['dominated'] and (r['risk_noT'] != r['risk_noT'] or r['risk_noT'] < .5),
        'R2 dominated & cop_T_proxy>=.6 & risk_noT<.1 -> risk_noT': lambda r: r['dominated'] and r['cop_T_proxy_evidence'] >= .6 and (r['risk_noT'] != r['risk_noT'] or r['risk_noT'] < .1),
        'R3 dominated & risk_noT<.1 & risk_evdrop>=.5 -> risk_noT': lambda r: r['dominated'] and (r['risk_noT'] != r['risk_noT'] or r['risk_noT'] < .1) and r['risk_evdrop_mean'] >= .5,
        'CONTROL non-dominated & risk_noT<.1 -> risk_noT (T = most co-present third)': lambda r: (not r['dominated']) and (r['risk_noT'] != r['risk_noT'] or r['risk_noT'] < .1),
    }
    for name, cond in conds.items():
        res = {'per_fold': {}, 'demoted_positives': []}
        ys, r0s, r1s, ylab = [], [], [], []
        n_dem = 0
        for fold, s in scores_by_fold.items():
            y = (s['label'] == 1).to_numpy(); lab = s['label'].to_numpy(); r0 = s['risk'].to_numpy(); r1 = r0.copy()
            pid = s['pair_id'].to_list()
            for i, p in enumerate(pid):
                row = d.get((fold, p))
                if row is not None and cond(row):
                    r1[i] = 0.0 if row['risk_noT'] != row['risk_noT'] else row['risk_noT']
                    n_dem += 1
                    if row['label'] == 1:
                        res['demoted_positives'].append({'fold': fold, 'pair_id': p, 'family': row['behavior_family'], 'risk': round(float(r0[i]), 4), 'risk_new': round(float(r1[i]), 4), 'dom_label': row['dom_label']})
            m = lab >= 0
            res['per_fold'][f'fold{fold}'] = {'AP_all_before': round(float(average_precision_score(y, r0)), 4), 'AP_all_after': round(float(average_precision_score(y, r1)), 4),
                                              'AP_labelled_before': round(float(average_precision_score(y[m], r0[m])), 4), 'AP_labelled_after': round(float(average_precision_score(y[m], r1[m])), 4)}
            ys.append(y); r0s.append(r0); r1s.append(r1); ylab.append(m)
        y, r0, r1, m = np.concatenate(ys), np.concatenate(r0s), np.concatenate(r1s), np.concatenate(ylab)
        res['pooled'] = {'AP_all_before': round(float(average_precision_score(y, r0)), 4), 'AP_all_after': round(float(average_precision_score(y, r1)), 4),
                         'AP_labelled_before': round(float(average_precision_score(y[m], r0[m])), 4), 'AP_labelled_after': round(float(average_precision_score(y[m], r1[m])), 4)}
        res['n_demoted'] = n_dem; res['n_demoted_positives'] = len(res['demoted_positives'])
        out[name] = res
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts, metas, scores_by_fold = [], {}, {}
    for fold in (1, 2):
        df, meta, scores = run_fold(fold, t0)
        parts.append(df); metas[f'fold{fold}'] = meta; scores_by_fold[fold] = scores
    df = add_flags(pl.concat(parts, how='vertical_relaxed'), 'risk_oof')
    df.write_parquet(OUT / 'dev_rescoring.parquet')
    summ = {'meta': metas, 'block': df.height, 'dominated': int(df['dominated'].sum()), 'seconds': round(time.time() - t0)}
    groups = {}
    for dmn in (True, False):
        for lab in (1, -1, 0):
            g = df.filter((pl.col('dominated') == dmn) & (pl.col('label') == lab))
            if g.height == 0:
                continue
            gs = group_summary(g)
            gs['cop_T_proxy_evidence_mean'] = round(float(g['cop_T_proxy_evidence'].mean()), 3)
            gs['share_cop_T_proxy_ge_3of5'] = round(float((g['cop_T_proxy_evidence'] >= .6).mean()), 3)
            if lab == 1:
                gs['cop_T_truth_evidence_mean'] = round(float(g['cop_T_truth_evidence'].drop_nulls().mean()), 3)
                gs['share_cop_T_truth_ge_3of5'] = round(float((g['cop_T_truth_evidence'].drop_nulls() >= .6).mean()), 3)
                gs['risk_evdrop_truth_median'] = round(float(g['risk_evdrop_truth_mean'].drop_nulls().median()), 4)
                gs['share_evdrop_truth_stays_ge_.5'] = round(float((g['risk_evdrop_truth_mean'].drop_nulls() >= .5).mean()), 3)
                gs['risk_noT_keepTruthEv_median'] = round(float(g['risk_noT_keepTruthEv'].drop_nulls().median()), 4)
            if dmn:
                gs['dominator_label_counts'] = g.group_by('dom_label').len().sort('dom_label').to_dicts()
                gs['families'] = g.group_by('behavior_family').len().sort('behavior_family').to_dicts()
            groups[f'dominated={dmn} label={lab}'] = gs
    summ['groups'] = groups
    by_band = {}
    for lab in (1, -1):
        for b in ('[.5,.9)', '[.9,.99)', '[.99,1]'):
            g = df.filter(pl.col('dominated') & (pl.col('label') == lab) & (pl.col('band') == b))
            if g.height:
                by_band[f'dominated label={lab} band={b}'] = {'n': g.height, 'share_collapse': round(float(g['collapse'].mean()), 3), 'share_collapse_vs_evdrop': round(float(g['collapse_vs_evdrop'].mean()), 3),
                                                               'share_drops_below_block': round(float(g['drops_below_block'].mean()), 3),
                                                               'risk_noT_median': round(float(g['risk_noT'].fill_nan(0).median()), 4), 'risk_rand_median': round(float(g['risk_rand_mean'].median()), 4)}
    summ['dominated_by_band'] = by_band
    by_cop = {}
    for lo, hi in ((0, .2), (.2, .6), (.6, 1.01)):
        for lab in (1, -1):
            g = df.filter(pl.col('dominated') & (pl.col('label') == lab) & (pl.col('cop_T_proxy_evidence') >= lo) & (pl.col('cop_T_proxy_evidence') < hi))
            if g.height:
                by_cop[f'dominated label={lab} cop_T_proxy in [{lo},{hi})'] = {'n': g.height, 'share_collapse': round(float(g['collapse'].mean()), 3),
                                                                             'risk_noT_median': round(float(g['risk_noT'].fill_nan(0).median()), 4), 'risk_rand_median': round(float(g['risk_rand_mean'].median()), 4)}
    summ['dominated_by_cop_T_proxy'] = by_cop
    summ['rules'] = ap_rules(df, scores_by_fold)
    write_json(OUT / 'dev_rescoring.json', summ)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()