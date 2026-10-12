"""Step 50: neighbour re-scoring of the block in BOTH directions (evaluation label-free; development labels only to validate).
Step 48 tested only the lower-risk member of each neighbouring couple of block pairs (the 'dominated' one). A spill-over
pair with a shorter history can outrank the genuine pair (fewer hands, so the inherited hands weigh more); then the genuine
pair is the dominated one, does not collapse, and the false positive is never tested. Here every ordered relation (P, N) of
block pairs sharing exactly one player is re-scored: risk of P without the hands in which N's other player T was dealt,
with a same-size random-drop control (2 seeds), plus the proxy top-5 overlap and family agreement of 14.31.
Rules evaluated (dev: labelled positives demoted must be ~0; eval: counts):
  R1     P dominated by N (highest-risk neighbour above P) and risk_P_noT(N) < .1                      [step 48 rule]
  R4any  exists N: risk_P_noT(N) < .1 and risk_N_noT(P) >= .5 (N robust to losing P's partner)
  R4rev  R4any restricted to N with risk_N < risk_P (the direction step 48 never tested)
  R4rev+ R4rev and (proxy overlap >= 1 or same predicted family)
  MUTUAL exists N: both collapse (< .1) without each other's partner (ambiguous, never demoted)
Outputs: experiments/spillover_20260911/neighbour_eval.parquet, neighbour_dev.parquet, neighbour_rules.json."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json, check_protected
import numpy as np
import polars as pl
import joblib
from sklearn.metrics import average_precision_score
from pu_pairs.build_allpairs_fold import CONF, EXP
from pu_pairs.step48_spillover_rescoring import dominance, make_scorer, MODEL
from pu_pairs.step48b_spillover_dev import build_frames

CV = ORIGINAL / 'Claude Version'
OUT = CV / 'experiments' / 'spillover_20260911'
THR = .5
PROXY = 'ca_fold_benefit_absmax'
SEEDS = [11, 12]


def relations(block):
    """ordered (P, N, T): block pairs sharing exactly one player; T = N's other player."""
    by_player = defaultdict(list)
    rows = {pid: (p1, p2, r) for pid, p1, p2, r in block.select('pair_id', 'player_1', 'player_2', 'risk').iter_rows()}
    for pid, (p1, p2, r) in rows.items():
        by_player[p1].append(pid); by_player[p2].append(pid)
    rel = []
    for pid, (p1, p2, r) in rows.items():
        for shared in (p1, p2):
            for n in by_player[shared]:
                if n == pid: continue
                q1, q2, rn = rows[n]
                if {q1, q2} == {p1, p2}: continue
                T = q2 if q1 == shared else q1
                rel.append((pid, n, T, r, rn))
    return rel, rows


def top5(ca):
    return dict(ca.sort(['pair_id', PROXY], descending=[False, True]).group_by('pair_id', maintain_order=True).agg(pl.col('hand_id').head(5)).iter_rows())


def rescore_relations(rel, raw, ca, score, dealt):
    """risk of P without T's hands for every (P, N, T); batched by neighbour rank; random-drop control with 2 seeds."""
    per_pair = defaultdict(list)
    for P, N, T, r, rn in rel:
        per_pair[P].append((rn, N, T))
    for P in per_pair:
        per_pair[P].sort(reverse=True)
    maxdeg = max(len(v) for v in per_pair.values())
    hand_pid = raw.select('pair_id', 'hand_id')
    pairs_arr = hand_pid['pair_id'].to_list(); hands_arr = hand_pid['hand_id'].to_list()
    rows_of = defaultdict(list)
    for i, p in enumerate(pairs_arr):
        rows_of[p].append(i)
    full, n_full = score(raw, ca)
    out = {}
    for k in range(maxdeg):
        ids_k = [P for P, v in per_pair.items() if len(v) > k]
        T_of = {P: per_pair[P][k][2] for P in ids_k}
        keep_pair = np.array([p in T_of for p in pairs_arr])
        isT = np.array([keep_pair[i] and (T_of[p] in dealt[h]) for i, (p, h) in enumerate(zip(pairs_arr, hands_arr))])
        sel = keep_pair & ~isT
        noT, n_noT = score(raw.filter(pl.Series(sel)), ca.filter(pl.Series(sel)))
        n_drop = {P: n_full[P] - n_noT.get(P, 0) for P in ids_k}
        rand_runs = []
        for seed in SEEDS:
            rs = np.random.default_rng(seed + 7 * k)
            keep = keep_pair.copy()
            for P in ids_k:
                d = n_drop[P]
                if d > 0:
                    keep[rs.choice(rows_of[P], size=d, replace=False)] = False
            rand_runs.append(score(raw.filter(pl.Series(keep)), ca.filter(pl.Series(keep)))[0])
        for P in ids_k:
            rn, N, T = per_pair[P][k]
            rr = [run.get(P, float('nan')) for run in rand_runs]
            out[(P, N)] = {'T': T, 'n_hands': n_full[P], 'n_dropped': n_drop[P], 'share_hands_with_T': n_drop[P] / n_full[P],
                           'risk_P_noT': noT.get(P, float('nan')), 'risk_P_rand': float(np.nanmean(rr)) if any(x == x for x in rr) else float('nan')}
    return out, full


def build_table(rel, rows, resc, t5, fam, label, phase, fold):
    recs = []
    for P, N, T, r, rn in rel:
        a = resc[(P, N)]; b = resc.get((N, P))
        recs.append({'phase': phase, 'fold': fold, 'P': P, 'N': N, 'T': T, 'risk_P': r, 'risk_N': rn, **a,
                     'risk_N_noTP': b['risk_P_noT'] if b else float('nan'), 'risk_N_rand': b['risk_P_rand'] if b else float('nan'),
                     'overlap5': len(set(t5.get(P, [])) & set(t5.get(N, []))), 'same_family': fam.get(P) == fam.get(N),
                     'label_P': label.get(P, -1), 'label_N': label.get(N, -1)})
    return pl.DataFrame(recs)


def pair_rules(tab, block):
    """Per-pair rule flags from the relation table. Returns DataFrame pair_id, risk, label, flags, new risk under each rule."""
    dom = dominance(block)
    risk = dict(block.select('pair_id', 'risk').iter_rows())
    label = dict(block.select('pair_id', 'label').iter_rows()) if 'label' in block.columns else {}
    rows = []
    g = {P: sub for P, sub in tab.group_by('P')}
    for pid in block['pair_id'].to_list():
        key = (pid,) if (pid,) in g else pid
        sub = g.get(key)
        flags = {'R1': False, 'R4any': False, 'R4rev': False, 'R4rev_plus': False, 'MUTUAL': False, 'n_neighbours': 0}
        new = {'R1': risk[pid], 'R4any': risk[pid], 'R4rev': risk[pid], 'R4rev_plus': risk[pid]}
        if sub is not None:
            flags['n_neighbours'] = sub.height
            for r in sub.iter_rows(named=True):
                noT = r['risk_P_noT'] if r['risk_P_noT'] == r['risk_P_noT'] else 0.0
                nN = r['risk_N_noTP'] if r['risk_N_noTP'] == r['risk_N_noTP'] else 0.0
                collapse_P = noT < .1
                robust_N = nN >= .5
                if pid in dom and r['N'] == dom[pid][1] and collapse_P:
                    flags['R1'] = True; new['R1'] = min(new['R1'], noT)
                if collapse_P and robust_N:
                    flags['R4any'] = True; new['R4any'] = min(new['R4any'], noT)
                    if r['risk_N'] < r['risk_P']:
                        flags['R4rev'] = True; new['R4rev'] = min(new['R4rev'], noT)
                        if r['overlap5'] >= 1 or r['same_family']:
                            flags['R4rev_plus'] = True; new['R4rev_plus'] = min(new['R4rev_plus'], noT)
                if collapse_P and nN < .1:
                    flags['MUTUAL'] = True
        rows.append({'pair_id': pid, 'risk': risk[pid], 'label': label.get(pid, -1), 'dominated': pid in dom, **flags, **{f'new_{k}': v for k, v in new.items()}})
    return pl.DataFrame(rows)


def main():
    t0 = time.time(); check_protected()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    # ---------------- evaluation (label-free)
    ev = pl.read_parquet(OUT / 'eval_rescoring.parquet').select('pair_id', pl.col('risk_v13').alias('risk'))
    ep = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2')
    block = ev.join(ep, on='pair_id')
    raw = pl.read_parquet(OUT / 'eval_block_raw.parquet'); ca = pl.read_parquet(OUT / 'eval_block_ca.parquet')
    caf = [c for c in ca.columns if c.startswith('ca_')]
    names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]; events = [c for c in names if c.startswith('v2_')]
    bundle = joblib.load(MODEL); score = make_scorer(bundle['model'], list(bundle['features']), names, events, caf)
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet', columns=['hand_id', 'player_id']).filter(pl.col('hand_id').is_in(sorted(set(raw['hand_id']))))
    dealt = defaultdict(set)
    for h, p in seats.iter_rows():
        dealt[h].add(p)
    rel, rows = relations(block)
    resc, full = rescore_relations(rel, raw, ca, score, dealt)
    sub = pl.read_csv(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv').select('pair_id', 'predicted_behavior')
    fam = dict(sub.iter_rows())
    tab_e = build_table(rel, rows, resc, top5(ca), fam, {}, 'eval', 0)
    tab_e.write_parquet(OUT / 'neighbour_eval.parquet')
    pr_e = pair_rules(tab_e, block)
    result['eval'] = {'block': block.height, 'ordered_relations': len(rel), 'pairs_with_neighbours': int((pr_e['n_neighbours'] > 0).sum()),
                      'counts': {k: int(pr_e[k].sum()) for k in ('R1', 'R4any', 'R4rev', 'R4rev_plus', 'MUTUAL')},
                      'R4rev_not_R1': int((pr_e['R4rev'] & ~pr_e['R1']).sum()), 'R4rev_plus_not_R1': int((pr_e['R4rev_plus'] & ~pr_e['R1']).sum()),
                      'R4any_not_R1': int((pr_e['R4any'] & ~pr_e['R1']).sum()),
                      'R4rev_by_band': pr_e.filter(pl.col('R4rev') & ~pl.col('R1')).with_columns(pl.when(pl.col('risk') >= .99).then(pl.lit('[.99,1]')).when(pl.col('risk') >= .9).then(pl.lit('[.9,.99)')).otherwise(pl.lit('[.5,.9)')).alias('band')).group_by('band').len().sort('band').to_dicts(),
                      'seconds': round(time.time() - t0)}
    pr_e.write_parquet(OUT / 'neighbour_eval_pairs.parquet')
    print(json.dumps(result['eval']), flush=True)
    # ---------------- development (labels only to validate)
    tabs, prs, scores_by_fold = [], [], {}
    for fold in (1, 2):
        scores = pl.read_parquet(EXP / f'fold{fold}' / 'scores.parquet'); scores_by_fold[fold] = scores
        blk = scores.filter(pl.col('risk') >= THR).sort('pair_id')
        raw, ca = build_frames(fold, blk, t0)
        caf = [c for c in ca.columns if c.startswith('ca_')]
        names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]; events = [c for c in names if c.startswith('v2_')]
        b = joblib.load(CONF / 'results' / f'fold{fold}' / 'conditional-pair.joblib'); sc = make_scorer(b['model'], list(b['features']), names, events, caf)
        seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet', columns=['hand_id', 'player_id']).filter(pl.col('hand_id').is_in(sorted(set(raw['hand_id']))))
        dealt = defaultdict(set)
        for h, p in seats.iter_rows():
            dealt[h].add(p)
        rel, rows = relations(blk)
        resc, full = rescore_relations(rel, raw, ca, sc, dealt)
        oof = dict(blk.select('pair_id', 'risk').iter_rows())
        par = max(abs(full[p] - oof[p]) for p in oof)
        fam = dict(blk.select('pair_id', 'predicted_class').iter_rows()); label = dict(blk.select('pair_id', 'label').iter_rows())
        t = build_table(rel, rows, resc, top5(ca), fam, label, 'dev', fold); tabs.append(t)
        pr = pair_rules(t, blk).with_columns(pl.lit(fold).alias('fold')); prs.append(pr)
        print(json.dumps({'fold': fold, 'parity': par, 'block': blk.height, 'relations': len(rel), 'seconds': round(time.time() - t0)}), flush=True)
    tab_d = pl.concat(tabs); tab_d.write_parquet(OUT / 'neighbour_dev.parquet')
    pr_d = pl.concat(prs); pr_d.write_parquet(OUT / 'neighbour_dev_pairs.parquet')
    dev = {'block': pr_d.height, 'labelled_positives_in_block': int((pr_d['label'] == 1).sum()), 'rules': {}}
    for rule in ('R1', 'R4any', 'R4rev', 'R4rev_plus', 'MUTUAL'):
        sel = pr_d.filter(pl.col(rule))
        info = {'n_flagged': sel.height, 'labelled_positives_flagged': int((sel['label'] == 1).sum()), 'unlabelled_flagged': int((sel['label'] == -1).sum()),
                'labelled_negatives_flagged': int((sel['label'] == 0).sum()), 'not_R1': int((~sel['R1']).sum()) if rule != 'R1' else 0,
                'positives': sel.filter(pl.col('label') == 1).select('fold', 'pair_id', 'risk', 'n_neighbours').to_dicts()[:20]}
        if rule != 'MUTUAL':
            ys, r0s, r1s = [], [], []
            newmap = dict(zip(zip(pr_d['fold'].to_list(), pr_d['pair_id'].to_list()), pr_d[f'new_{rule}'].to_list()))
            flagmap = dict(zip(zip(pr_d['fold'].to_list(), pr_d['pair_id'].to_list()), pr_d[rule].to_list()))
            per_fold = {}
            for fold, s in scores_by_fold.items():
                y = (s['label'] == 1).to_numpy(); r0 = s['risk'].to_numpy(); r1 = r0.copy()
                for i, p in enumerate(s['pair_id'].to_list()):
                    if flagmap.get((fold, p)):
                        r1[i] = newmap[(fold, p)]
                per_fold[f'fold{fold}'] = {'AP_all_before': round(float(average_precision_score(y, r0)), 4), 'AP_all_after': round(float(average_precision_score(y, r1)), 4)}
                ys.append(y); r0s.append(r0); r1s.append(r1)
            y, r0, r1 = np.concatenate(ys), np.concatenate(r0s), np.concatenate(r1s)
            info['AP_all_pooled'] = {'before': round(float(average_precision_score(y, r0)), 4), 'after': round(float(average_precision_score(y, r1)), 4)}; info['per_fold'] = per_fold
        dev['rules'][rule] = info
    # relation-level summary on dev by label of P
    for lab, name in ((1, 'P labelled positive'), (-1, 'P unlabelled')):
        s = tab_d.filter(pl.col('label_P') == lab)
        dev[f'relations {name}'] = {'n': s.height, 'share_P_collapses_lt_.1': round(float((s['risk_P_noT'].fill_nan(0) < .1).mean()), 3),
                                    'share_P_collapses_and_N_robust': round(float(((s['risk_P_noT'].fill_nan(0) < .1) & (s['risk_N_noTP'].fill_nan(0) >= .5)).mean()), 3),
                                    'share_reverse_direction': round(float((s['risk_N'] < s['risk_P']).mean()), 3),
                                    'share_P_collapses_reverse': round(float(((s['risk_P_noT'].fill_nan(0) < .1) & (s['risk_N'] < s['risk_P'])).mean()), 3),
                                    'overlap5_ge1_share': round(float((s['overlap5'] >= 1).mean()), 3), 'same_family_share': round(float(s['same_family'].mean()), 3)}
    result['dev'] = dev; result['seconds'] = round(time.time() - t0)
    write_json(OUT / 'neighbour_rules.json', result)
    print(json.dumps(result['dev'], indent=1))


if __name__ == '__main__':
    main()