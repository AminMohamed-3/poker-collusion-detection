"""Step 84 (C): the ORACLE CEILING of the pair lane on the PU-AP surface, and what PU-AP is.

author: claude (agent)

Follow-up to step83.  Three questions, in the order the coordinator set them.

(1) THE CEILING.  Frozen ranking, then oracle-rescue the k lowest-risk labelled positives to the
    top of the list, k = 0, 5, 10, 16, 33, 82, 164.  Report average precision at each rung.

    A WARNING THAT HAS TO COME BEFORE THE LADDER, because it changes what the ladder means.
    OBS 12 / 12.1 already established, with four label-free tests, that the DISCLOSED LABELS ARE A
    SAMPLE of the planted colluders, not a census:
        "Separability positives vs unlabelled-high: AUC .46 (all 897) / .46 (outside top-100) vs
         .997 for positives vs labelled negatives -- one population"
        "The .771 'AP' is a lower bound that treats real colluders as errors."
    So on the raw PU surface the unlabelled rows sitting ABOVE our labelled positives are, in the
    main, real colluders that were never disclosed.  A raw PU-AP deficit therefore has two parts
    that must not be confused:
        (a) labelled positives ranked too low        -- real, recoverable model error
        (b) unlabelled true colluders ranked high    -- an artefact of the label sample
    The ladder is computed BOTH ways: raw, and PU-safe (unlabelled rows at risk >= .9 censored out
    of the negative pool rather than counted as errors).  The PU-safe rung is the honest ceiling.

(2) is deferred to a later step and is only run if (1) is large.

(3) WHAT IS PU-AP, RELATIVE TO PairAP.  Same estimator (checked against sklearn here), different
    population.  Measured, not asserted: how AP moves with the size of the negative pool at a fixed
    positive set, and whether it is invariant when positives and negatives scale together.  Plus an
    analytic check of the coordinator's arithmetic: what AP does "94 % ranked perfectly, 6 % buried"
    actually produce?

Surface: labelled positives vs every other row of the deduplicated fold-1 + fold-2 all-pairs frames
at shared_hands >= 38.  67,463 rows, 164 positives, 650 labelled negatives, prevalence .243 %.
Training-only.  Held folds 0/1/2 truth is never read; no evaluation label is used anywhere.
Output: experiments/pair_side_C_20260912/audit-C-oracle-ladder.json
"""
import json
import time

import numpy as np
import polars as pl

from evidence_aug.paths import CV, check_protected, write_json

PU = CV / 'experiments' / 'pu_pairs_20260911'
OUT = CV / 'experiments' / 'pair_side_C_20260912'
MIN_SH = 38
RNG = np.random.default_rng(84)
EVAL_PAIRS = 112540  # the competition's evaluation pair count, used only to state the extrapolation


def log(*a):
    print(*a, flush=True)


def ap(y, s):
    """Average precision, stable order.  Cross-checked against sklearn below."""
    o = np.argsort(-s, kind='stable')
    yy = y[o]
    c = np.cumsum(yy)
    return float((c / np.arange(1, len(yy) + 1) * yy).sum() / max(1.0, yy.sum()))


def per_positive_precision(y, s):
    """Precision at the rank of each positive, in rank order.  AP is the mean of this vector."""
    o = np.argsort(-s, kind='stable')
    yy = y[o]
    c = np.cumsum(yy)
    p = c / np.arange(1, len(yy) + 1)
    return p[yy == 1]


def load():
    v = (pl.concat([pl.read_parquet(PU / f'fold{f}' / 'scores.parquet') for f in (1, 2)])
         .unique(subset=['pair_id'], keep='first', maintain_order=True)
         .sort('pair_id'))
    return v.filter(pl.col('shared_hands') >= MIN_SH)


def main():
    t0 = time.time()
    res = {'author': 'claude (agent)', 'step': 'step84_C_oracle_ladder',
           'question': 'oracle ceiling of the pair lane, and the PU-AP / PairAP relationship',
           'training_only': True,
           'label_sample_warning': (
               'OBS 12 / 12.1: the disclosed labels are a SAMPLE of the planted colluders, not a '
               'census.  Separability of labelled positives vs unlabelled risk>=.9 is AUC .46-.62 '
               'on all 897 features against .997-.998 for positives vs labelled negatives -- one '
               'population.  Raw PU-AP therefore counts real colluders as errors and is a LOWER '
               'BOUND.  Both a raw and a PU-safe ladder are reported.'),
           'protected_before': check_protected()}

    e = load()
    y = (e['label'] == 1).to_numpy().astype(float)
    r = e['risk'].to_numpy()
    lab = e['label'].to_numpy()
    tab = e['table_id'].to_numpy()
    n = len(y)
    log('surface', n, 'positives', int(y.sum()), 'labelled negatives', int((lab == 0).sum()))

    # ------------------------------------------------------------------ (3a) is my AP an AP?
    ctl = {'my_ap': round(ap(y, r), 6)}
    try:
        from sklearn.metrics import average_precision_score
        ctl['sklearn_average_precision_score'] = round(float(average_precision_score(y, r)), 6)
        ctl['abs_difference'] = round(abs(ctl['my_ap'] - ctl['sklearn_average_precision_score']), 8)
        ctl['same_estimator'] = ctl['abs_difference'] < 1e-6
    except Exception as exc:                                    # pragma: no cover
        ctl['sklearn_error'] = str(exc)
    # ties would be the only way the two could disagree; count them
    ctl['n_distinct_scores'] = int(len(np.unique(r)))
    ctl['n_rows'] = n
    ctl['note'] = ('PU-AP and PairAP are the SAME estimator -- average precision over a full '
                   'ranking with the labelled positives as the positive class.  They differ only '
                   'in the population that supplies the rows and in who owns the truth column.')
    res['ap_definition_control'] = ctl

    # ------------------------------------------------------------------ contamination ledger
    unl = lab < 0
    led = {'n_unlabelled': int(unl.sum()), 'n_tables': int(len(np.unique(tab))),
           'n_labelled_positives': int(y.sum()), 'n_labelled_negatives': int((lab == 0).sum())}
    for th in (.5, .9, .99, .999):
        led[f'unlabelled_ge_{th}'] = int((unl & (r >= th)).sum())
        led[f'unlabelled_ge_{th}_per_table'] = round(float((unl & (r >= th)).sum() / led['n_tables']), 3)
        led[f'labelled_positives_ge_{th}'] = int((y == 1).sum() - ((y == 1) & (r < th)).sum())
        led[f'labelled_negatives_ge_{th}'] = int(((lab == 0) & (r >= th)).sum())
    led['obs_12_reference'] = ('fold 1: 49 unlabelled >= .9 per 100 tables, fold 2: 51; '
                               'the two folds share 29 tables so the dedup rate should land near .5/table')
    res['contamination_ledger'] = led

    # ------------------------------------------------------------------ (1) the oracle ladder
    order_pos = np.argsort(r[y == 1], kind='stable')            # positives, worst risk first
    pos_idx = np.where(y == 1)[0][order_pos]
    ladder_ks = [0, 5, 10, 16, 33, 50, 82, 123, int(y.sum())]

    def rescue_scores(k, base=None):
        s = (r.copy() if base is None else base.copy())
        if k:
            s[pos_idx[:k]] = np.inf                             # oracle: straight to the top
        return s

    def build_ladder(mask_keep, tag):
        """mask_keep selects the rows that stay in the scored population."""
        yk, rk = y[mask_keep], r[mask_keep]
        keep_pos = np.where(y == 1)[0]
        remap = -np.ones(n, dtype=np.int64)
        remap[np.where(mask_keep)[0]] = np.arange(int(mask_keep.sum()))
        rows = []
        for k in ladder_ks:
            s = rk.copy()
            sel = [remap[i] for i in pos_idx[:k] if remap[i] >= 0]
            if sel:
                s[np.array(sel)] = np.inf
            rows.append({'k_rescued': k, 'ap': round(ap(yk, s), 5),
                         'delta_vs_frozen': None})
        base = rows[0]['ap']
        for row in rows:
            row['delta_vs_frozen'] = round(row['ap'] - base, 5)
        return {'population_rows': int(mask_keep.sum()),
                'population_positives': int(yk.sum()),
                'population_prevalence': round(float(yk.mean()), 5),
                'ladder': rows,
                'n_labelled_positives_total': int(len(keep_pos))}

    res['oracle_ladder_raw'] = build_ladder(np.ones(n, dtype=bool), 'raw')
    # PU-safe: censor unlabelled rows that OBS 12 shows are the positive population
    for th in (.9, .5):
        keep = ~(unl & (r >= th))
        res[f'oracle_ladder_pu_safe_censor_unlabelled_ge_{th}'] = build_ladder(keep, f'safe{th}')

    # ------------------------------------------------------------------ deficit decomposition
    pp = per_positive_precision(y, r)                            # in rank order, worst last
    pp_sorted = np.sort(pp)
    dec = {'frozen_ap': round(float(pp.mean()), 5),
           'ap_is_mean_of_per_positive_precision': True,
           'per_positive_precision_quantiles': {
               q: round(float(np.quantile(pp, float(q))), 4) for q in ('0.05', '0.1', '0.25', '0.5', '0.75', '0.9')},
           'n_positives_with_precision_above_.9': int((pp > .9).sum()),
           'n_positives_with_precision_below_.5': int((pp < .5).sum()),
           'n_positives_with_precision_below_.1': int((pp < .1).sum())}
    # how much of the 1 - AP deficit does each slice of positives own?
    tot = 1.0 - float(pp.mean())
    for k in (5, 10, 16, 33, 82):
        own = float((1.0 - pp_sorted[:k]).sum() / len(pp))
        dec[f'deficit_owned_by_worst_{k}_positives'] = round(own, 5)
        dec[f'share_of_total_deficit_worst_{k}'] = round(own / tot, 3) if tot > 0 else None
    dec['total_deficit_1_minus_ap'] = round(tot, 5)
    res['deficit_decomposition'] = dec

    # contamination oracle: delete the top-M unlabelled rows and re-score
    cont = {}
    ord_all = np.argsort(-r, kind='stable')
    unl_ranked = [i for i in ord_all if unl[i]]
    for m in (0, 25, 50, 100, 200, 400, 800):
        drop = np.zeros(n, dtype=bool)
        if m:
            drop[np.array(unl_ranked[:m])] = True
        keep = ~drop
        cont[str(m)] = {'ap': round(ap(y[keep], r[keep]), 5),
                        'min_risk_of_dropped': round(float(r[np.array(unl_ranked[:m])].min()), 4) if m else None}
    res['contamination_oracle_drop_top_unlabelled'] = cont

    # ------------------------------------------------------------------ (3b) population scaling
    scal = {'note': 'AP with the positive set FIXED and the unlabelled pool subsampled -- shows how '
                    'much of PU-AP is a property of the pool size rather than of the ranking.',
            'fixed_positives_vary_pool': {}, 'scale_both_together': {}}
    unl_i = np.where(unl)[0]
    keep_lab = np.where(~unl)[0]
    for f in (.25, .5, .75, 1.0):
        aps = []
        for _ in range(20):
            sub = RNG.choice(unl_i, size=int(len(unl_i) * f), replace=False)
            idx = np.concatenate([keep_lab, sub])
            aps.append(ap(y[idx], r[idx]))
        scal['fixed_positives_vary_pool'][str(f)] = {
            'pool_rows': int(len(keep_lab) + len(unl_i) * f),
            'ap_mean': round(float(np.mean(aps)), 5), 'ap_sd': round(float(np.std(aps)), 5)}
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    for f in (.25, .5, .75, 1.0):
        aps = []
        for _ in range(20):
            sp = RNG.choice(pos_i, size=max(5, int(len(pos_i) * f)), replace=False)
            sn = RNG.choice(neg_i, size=int(len(neg_i) * f), replace=False)
            idx = np.concatenate([sp, sn])
            aps.append(ap(y[idx], r[idx]))
        scal['scale_both_together'][str(f)] = {
            'rows': int(len(pos_i) * f + len(neg_i) * f),
            'ap_mean': round(float(np.mean(aps)), 5), 'ap_sd': round(float(np.std(aps)), 5)}
    scal['evaluation_pairs'] = EVAL_PAIRS
    scal['our_pool_rows'] = n
    scal['pool_ratio_evaluation_over_ours'] = round(EVAL_PAIRS / n, 3)
    res['population_scaling'] = scal

    # ------------------------------------------------------------------ (3c) the arithmetic check
    # "94 % of positives ranked perfectly, 6 % buried" -- what AP is that, exactly?
    arith = {'note': 'synthetic ranking: P positives, N negatives; a fraction q of positives sit in '
                     'the very top block, the rest sit at rank fraction b down the list.',
             'grid': []}
    P, N = 372, EVAL_PAIRS
    for q in (.90, .94, .96, .98):
        for b in (.001, .01, .05, .5):
            top = int(P * q)
            buried = P - top
            ys = np.zeros(N, dtype=float)
            ys[:top] = 1.0
            spot = int(N * b)
            ys[spot:spot + buried] = 1.0
            arith['grid'].append({'frac_perfect': q, 'buried_at_rank_fraction': b,
                                  'n_buried': buried, 'ap': round(ap(ys, -np.arange(N, dtype=float)), 4)})
    arith['public_pair_ap_reference'] = '.9568-.9603 public; OBS 14.9 quotes .938 at an earlier submission'
    res['arithmetic_check_buried_positive_model'] = arith

    res['protected_after'] = check_protected()
    res['seconds'] = round(time.time() - t0, 1)
    write_json(OUT / 'audit-C-oracle-ladder.json', res)
    log(json.dumps({k: res[k] for k in ('ap_definition_control', 'contamination_ledger',
                                        'oracle_ladder_raw', 'deficit_decomposition')},
                   indent=1, default=str))
    log('written', OUT / 'audit-C-oracle-ladder.json', res['seconds'])


if __name__ == '__main__':
    main()
