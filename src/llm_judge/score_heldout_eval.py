"""STEP264: LOCAL evaluation of the LLM judge, run after heldout_scores.jsonl comes back.

Never runs on the GPU box: it needs keys_heldout.parquet, which maps the opaque uid back to
(pair_id, hand_id) and therefore never leaves this machine.

Input   heldout_scores.jsonl, one row per held-out prompt:
          {"uid": "...", "mean_conf": 0-100, "k": 4,
           "confidences": [..], "verdict_counts": {"NONE": 3, "SOFT_PLAY": 1}}   (last two optional)

Reports, always against mechplus_p_fmax on exactly the SAME rows:
  * POS vs (NEG_RANDOM + NEG_HARD): AP, AUC, TPR@FPR 1e-3 and 1e-2, overall and per family
  * within-pair: POS vs NEG_WITHIN, per-pair AUC averaged over the short-list pairs, plus pooled
  * a 2-feature logistic stack on [logit(mechplus), logit(llm)], 5-fold CV grouped by table
    inside the held-out set, against mechplus alone
  * 1,000-draw bootstrap resampling TABLES: P(stack AP > mechplus AP)

Usage
  python score_heldout_eval.py --scores heldout_scores.jsonl [--out eval264.json]
  python score_heldout_eval.py --selftest     (synthesises a noisy-label score file and runs)
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import argparse
import json
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import render_hand as R                                                      # noqa: E402

CV = R.CV
OUT = CV + r'\experiments\step264_llm_hand_judge_20260918'
KEYS = OUT + r'\keys_heldout.parquet'
OOF254 = CV + r'\experiments\step254_mechanical_tensor_20260917\hand_oof_step254.parquet'
K = ['pair_id', 'hand_id']
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
EPS = 1e-6
BOOT = 1000


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def tpr_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0:
        return float('nan')
    thr = neg[min(int(fpr * len(neg)), len(neg) - 1)]
    return float((s[y == 1] > thr).mean())


def pair_metrics(y, s, tag=''):
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {'n_pos': int(y.sum()), 'n_neg': int((1 - y).sum()), 'note': 'degenerate ' + tag}
    return {'n_pos': int(y.sum()), 'n_neg': int((1 - y).sum()),
            'ap': round(float(average_precision_score(y, s)), 5),
            'auc': round(float(roc_auc_score(y, s)), 5),
            'tpr@1e-3': round(tpr_at_fpr(y, s, 1e-3), 4),
            'tpr@1e-2': round(tpr_at_fpr(y, s, 1e-2), 4)}


def load(scores_path: str) -> pl.DataFrame:
    keys = pl.read_parquet(KEYS)
    with open(scores_path, encoding='utf-8') as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    sc = pl.DataFrame({'uid': [r['uid'] for r in rows],
                       'mean_conf': [float(r['mean_conf']) for r in rows],
                       'k': [int(r.get('k', 0)) for r in rows]})
    top = {}
    for r in rows:
        vc = r.get('verdict_counts')
        top[r['uid']] = (max(vc, key=lambda kk: vc[kk]) if isinstance(vc, dict) and vc else None)
    sc = sc.with_columns(pl.Series('top_verdict', [top[u] for u in sc['uid'].to_list()],
                                   dtype=pl.String))
    assert sc['uid'].n_unique() == sc.height, 'duplicate uid in the score file'
    unknown = set(sc['uid'].to_list()) - set(keys['uid'].to_list())
    assert not unknown, f'{len(unknown)} scored uids are not held-out uids'
    d = keys.join(sc, on='uid', how='left')
    miss = d['mean_conf'].null_count()
    if miss:
        print(f'WARNING: {miss} of {d.height} held-out prompts have no score; they are dropped')
        d = d.filter(pl.col('mean_conf').is_not_null())
    oof = pl.read_parquet(OOF254, columns=K + ['mechplus_p_fmax'])
    d = d.join(oof, on=K, how='left')
    assert d['mechplus_p_fmax'].null_count() == 0, 'mechplus join left nulls'
    return d.with_columns((pl.col('mean_conf') / 100.0).alias('llm'))


def evaluate(d: pl.DataFrame) -> dict:
    res: dict = {'n_scored': d.height,
                 'by_class': {c: int(n) for c, n in
                              d.group_by('source_class').len().sort('source_class').iter_rows()},
                 'k_values': sorted({int(x) for x in d['k'].to_list()})}

    ev = d.filter(pl.col('source_class').is_in(['POS', 'NEG_RANDOM', 'NEG_HARD']))
    y = ev['label'].to_numpy().astype(int)
    llm, mech = ev['llm'].to_numpy(), ev['mechplus_p_fmax'].to_numpy()
    res['event_detection'] = {'llm': pair_metrics(y, llm, 'llm'),
                              'mechplus': pair_metrics(y, mech, 'mech')}
    neg = ev.filter(pl.col('label') == 0)
    res['event_detection_per_family'] = {}
    for f in FAMS:
        sub = pl.concat([ev.filter((pl.col('label') == 1) & (pl.col('family') == f)), neg])
        yy = sub['label'].to_numpy().astype(int)
        res['event_detection_per_family'][f] = {
            'llm': pair_metrics(yy, sub['llm'].to_numpy(), f),
            'mechplus': pair_metrics(yy, sub['mechplus_p_fmax'].to_numpy(), f)}
    # hard negatives only, the honest comparison
    hard = ev.filter((pl.col('label') == 1) | (pl.col('source_class') == 'NEG_HARD'))
    yh = hard['label'].to_numpy().astype(int)
    res['event_detection_vs_hard_negatives'] = {
        'llm': pair_metrics(yh, hard['llm'].to_numpy(), 'llm-hard'),
        'mechplus': pair_metrics(yh, hard['mechplus_p_fmax'].to_numpy(), 'mech-hard')}

    # ---- within-pair: a colluding pair's listed hands against its certified ordinary hands
    wp = d.filter(pl.col('source_class').is_in(['POS', 'NEG_WITHIN']))
    out_wp = {}
    for name, col in (('llm', 'llm'), ('mechplus', 'mechplus_p_fmax')):
        per = []
        for pid, g in wp.group_by('pair_id'):
            yy = g['label'].to_numpy().astype(int)
            if yy.sum() and (1 - yy).sum():
                per.append(float(roc_auc_score(yy, g[col].to_numpy())))
        yy = wp['label'].to_numpy().astype(int)
        out_wp[name] = {'per_pair_mean_auc': round(float(np.mean(per)), 5) if per else None,
                        'n_pairs': len(per),
                        'pooled': pair_metrics(yy, wp[col].to_numpy(), 'within')}
    res['within_pair'] = out_wp

    # ---- 2-feature logistic stack, cross-fitted by table inside the held-out set
    X = np.column_stack([logit(mech), logit(llm)])
    g = ev['table_id'].to_numpy()
    oof_stack = np.full(len(y), np.nan)
    coefs = []
    gkf = GroupKFold(n_splits=min(5, len(np.unique(g))))
    for tr, te in gkf.split(X, y, groups=g):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        oof_stack[te] = m.predict_proba(X[te])[:, 1]
        coefs.append([round(float(c), 4) for c in m.coef_[0]] + [round(float(m.intercept_[0]), 4)])
    assert not np.isnan(oof_stack).any()
    res['stack'] = {'coefs_per_fold [logit_mech, logit_llm, intercept]': coefs,
                    'stack': pair_metrics(y, oof_stack, 'stack'),
                    'mechplus_alone': pair_metrics(y, mech, 'mech'),
                    'llm_alone': pair_metrics(y, llm, 'llm')}
    res['stack']['delta_ap'] = round(res['stack']['stack']['ap']
                                     - res['stack']['mechplus_alone']['ap'], 5)

    # ---- bootstrap by table
    tables = np.unique(g)
    idx_of = {t: np.flatnonzero(g == t) for t in tables}
    rng = np.random.default_rng(264)
    wins = 0
    deltas = []
    for _ in range(BOOT):
        pick = rng.choice(len(tables), len(tables), replace=True)
        rows = np.concatenate([idx_of[tables[j]] for j in pick])
        yy = y[rows]
        if yy.sum() == 0 or (1 - yy).sum() == 0:
            continue
        a = average_precision_score(yy, oof_stack[rows])
        b = average_precision_score(yy, mech[rows])
        deltas.append(a - b)
        wins += a > b
    res['bootstrap_tables'] = {
        'draws': len(deltas), 'p_stack_beats_mechplus': round(wins / max(len(deltas), 1), 4),
        'delta_ap_mean': round(float(np.mean(deltas)), 5) if deltas else None,
        'delta_ap_q05': round(float(np.quantile(deltas, .05)), 5) if deltas else None,
        'delta_ap_q95': round(float(np.quantile(deltas, .95)), 5) if deltas else None}

    # ---- optional: family choice on the positives
    if d['top_verdict'].null_count() < d.height:
        posr = d.filter((pl.col('label') == 1) & pl.col('top_verdict').is_not_null())
        if posr.height:
            want = {'directed_transfer': 'DIRECTED_TRANSFER', 'soft_play': 'SOFT_PLAY',
                    'coordinated_isolation': 'COORDINATED_ISOLATION'}
            hit = [want.get(f) == v for f, v in zip(posr['family'].to_list(),
                                                    posr['top_verdict'].to_list())]
            res['family_choice_on_positives'] = {'n': len(hit),
                                                 'accuracy': round(float(np.mean(hit)), 4)}
    # ---- audit rows are scored but carry no positive label; report their score distribution
    aud = d.filter(pl.col('source_class') == 'AUDIT')
    if aud.height:
        res['audit_rows'] = {'n': aud.height,
                             'llm_mean': round(float(aud['llm'].mean()), 4),
                             'llm_q90': round(float(np.quantile(aud['llm'].to_numpy(), .9)), 4),
                             'note': 'stage-1 top-20 candidates that are not listed hands; '
                                     'includes ambiguous unlisted hands of 5-listed pairs, '
                                     'so a high score here is not necessarily an error'}
    return res


def _synthetic(path: str, noise: float = .55, seed: int = 9) -> None:
    keys = pl.read_parquet(KEYS)
    rng = np.random.default_rng(seed)
    y = keys['label'].to_numpy().astype(float)
    conf = np.clip(100 * (y * (1 - noise) + noise * rng.random(len(y))), 0, 100)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for u, c in zip(keys['uid'].to_list(), conf):
            vc = {'NONE': 3, 'SOFT_PLAY': 1} if c < 50 else {'SOFT_PLAY': 3, 'NONE': 1}
            fh.write(json.dumps({'uid': u, 'mean_conf': round(float(c), 2), 'k': 4,
                                 'confidences': [round(float(c), 2)] * 4,
                                 'verdict_counts': vc}) + '\n')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', default=None)
    ap.add_argument('--out', default=OUT + r'\eval264.json')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)
    if a.selftest:
        import tempfile
        p = os.path.join(tempfile.mkdtemp(prefix='step264_eval_'), 'synthetic_scores.jsonl')
        _synthetic(p)
        a.scores = p
        a.out = os.path.splitext(p)[0] + '_eval.json'
        print('SELFTEST on a synthetic score file (LLM score = noisy copy of the label):', p)
    assert a.scores, 'pass --scores heldout_scores.jsonl'
    res = evaluate(load(a.scores))
    json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(res, indent=1))
    print('wrote', a.out)
    if a.selftest:
        assert res['event_detection']['llm']['auc'] > .9, 'a noisy label copy should rank well'
        assert res['stack']['stack']['ap'] >= res['stack']['mechplus_alone']['ap'] - .02
        assert res['bootstrap_tables']['draws'] > 900
        assert res['within_pair']['llm']['n_pairs'] > 0
        print('score_heldout_eval.py selftest: 4 assertions passed')


if __name__ == '__main__':
    main()
