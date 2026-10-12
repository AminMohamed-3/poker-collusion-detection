"""Step 86 (C): how sensitive the local ceiling is to the contamination assumption, and whether
truncation buries the SAME positives or new random ones.  Plus the full two-fold 4-point curve.

author: claude (agent)

NEW FILE, not an overwrite.  experiments/pair_side_C_20260912/ is append-only from here on.

(A) PHI SENSITIVITY.  Step 84's PU-safe surface censors all 89 unlabelled rows at risk >= .9 on the
    strength of OBS 12 / 12.1 ("one population", separability AUC .46-.62 against .997-.998 for
    labelled negatives).  Censoring assumes every one of the 89 is a real colluder; the presumed
    surface assumes the same thing with the opposite sign.  That is ONE assumption and it has never
    been stress-tested.  Here phi = the fraction of the 89 that are really colluders.  A random
    phi-subset is treated as real (censored, or relabelled positive) and the remaining (1-phi) stay
    in the pool as genuine negatives.  50 draws per phi.  phi = 1 must reproduce .97660 and phi = 0
    must reproduce .71475; those two are the arithmetic checks on the interpolation.

(B) BURIAL OVERLAP.  If truncation buries a RANDOM extra set of positives, the extra ones are not
    samples from the population we care about and the measurement-power idea is fake.  If it buries
    the same ones deeper plus near-misses of the same kind, the idea is sound.  Tested by set
    nesting across the four prefixes and by a rank null on the newly-buried positives.

(C) The full two-fold four-point curve, now that all eight fold x prefix builds exist.

Training-only throughout.  Held folds 0/1/2 truth is never read; no evaluation label is used.
Output: experiments/pair_side_C_20260912/audit-C-phi-overlap-and-full-curve.json
"""
import json
import time

import numpy as np
import polars as pl

from evidence_aug.paths import CV, check_protected, write_json

PU = CV / 'experiments' / 'pu_pairs_20260911'
OUT = CV / 'experiments' / 'pair_side_C_20260912'
MIN_SH = 38
RNG = np.random.default_rng(86)
N_DRAW = 50
N_BOOT = 400
N_NULL = 20000


def log(*a):
    print(*a, flush=True)


def ap(y, s):
    o = np.argsort(-s, kind='stable')
    yy = y[o]
    c = np.cumsum(yy)
    return float((c / np.arange(1, len(yy) + 1) * yy).sum() / max(1.0, yy.sum()))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


BASE = ['pair_id', 'table_id', 'label', 'behavior_family', 'shared_hands', 'risk']


def cols(fold, tag):
    f = pl.read_parquet(PU / (f'fold{fold}' + tag) / 'scores.parquet')
    want = BASE + [c for c in ('p1', 'p2', 'p3') if c in f.columns]
    return f.select(want)


def load_view(tag):
    """Deduplicated fold1+fold2 view, ELIGIBLE ROWS ONLY (shared_hands >= MIN_SH)."""
    d = [PU / (f'fold{f}' + tag) / 'scores.parquet' for f in (1, 2)]
    if not all(p.exists() for p in d):
        return None
    v = (pl.concat([cols(f, tag) for f in (1, 2)])
         .unique(subset=['pair_id'], keep='first', maintain_order=True)
         .sort('pair_id')
         .filter(pl.col('shared_hands') >= MIN_SH))          # eligibility floor, OBS 17
    if 'p1' in v.columns:
        v = v.with_columns((pl.max_horizontal('p1', 'p2', 'p3') /
                            (pl.col('p1') + pl.col('p2') + pl.col('p3')).clip(1e-12)).alias('purity'))
    return v


def main():
    t0 = time.time()
    res = {'author': 'claude (agent)', 'step': 'step86_C_phi_and_overlap',
           'question': 'contamination-assumption sensitivity, burial overlap under truncation, '
                       'and the completed two-fold four-point curve',
           'training_only': True,
           'append_only_policy': 'new file; audit-C-pair-truncation.json and audit-C-oracle-ladder.json '
                                 'are never rewritten',
           'protected_before': check_protected()}

    full = load_view('')
    lab = full['label'].to_numpy()
    r = full['risk'].to_numpy()
    d = full['purity'].to_numpy()
    tab = full['table_id'].to_numpy()
    n = len(r)
    y0 = (lab == 1).astype(float)
    hi = np.where((lab < 0) & (r >= .9))[0]
    log('rows', n, 'labelled pos', int(y0.sum()), 'unlabelled>=.9', len(hi))
    assert n == 67463 and int(y0.sum()) == 164 and len(hi) == 89, (n, int(y0.sum()), len(hi))

    def w_blend(rr, dd, w=.25):
        z = (dd - dd.mean()) / (dd.std() + 1e-12)
        return logit(rr) + w * z

    # ------------------------------------------------------------------ (A) phi sensitivity
    phi_out = {'n_unlabelled_high': int(len(hi)), 'n_draws_per_phi': N_DRAW,
               'arithmetic_anchors': {'phi_1_must_equal_pu_safe_.97660': None,
                                      'phi_0_must_equal_raw_.71475': None},
               'censor_variant': {}, 'relabel_positive_variant': {}}
    for phi in (0.0, .25, .5, .75, 1.0):
        k = int(round(phi * len(hi)))
        cen, rel = [], []
        for _ in range(max(1, N_DRAW if 0 < phi < 1 else 1)):
            real = RNG.choice(hi, size=k, replace=False) if k else np.array([], dtype=np.int64)
            # censor variant: the "real" ones leave the pool; the rest stay as negatives
            keep = np.ones(n, dtype=bool)
            keep[real] = False
            yk, rk, dk = y0[keep], r[keep], d[keep]
            base = ap(yk, rk)
            s_or = rk.copy()
            s_or[(yk == 1) & (rk < .9)] = np.inf
            cen.append((base, ap(yk, s_or) - base, ap(yk, w_blend(rk, dk)) - base))
            # relabel variant: the "real" ones become positives, the rest stay negatives
            y2 = y0.copy()
            y2[real] = 1.0
            base2 = ap(y2, r)
            s2 = r.copy()
            s2[(y2 == 1) & (r < .9)] = np.inf
            rel.append((base2, ap(y2, s2) - base2, ap(y2, w_blend(r, d)) - base2))
        for tag, arr in (('censor_variant', cen), ('relabel_positive_variant', rel)):
            a = np.array(arr)
            phi_out[tag][str(phi)] = {
                'n_treated_as_real': k,
                'frozen_ap_mean': round(float(a[:, 0].mean()), 5),
                'frozen_ap_sd': round(float(a[:, 0].std()), 5),
                'oracle_ceiling_mean': round(float(a[:, 1].mean()), 5),
                'oracle_ceiling_sd': round(float(a[:, 1].std()), 5),
                'w0.25_delta_mean': round(float(a[:, 2].mean()), 5),
                'w0.25_delta_sd': round(float(a[:, 2].std()), 5),
                'w0.25_as_share_of_ceiling': (round(float(a[:, 2].mean() / a[:, 1].mean()), 4)
                                              if a[:, 1].mean() > 0 else None)}
    phi_out['arithmetic_anchors']['phi_1_must_equal_pu_safe_.97660'] = \
        phi_out['censor_variant']['1.0']['frozen_ap_mean']
    phi_out['arithmetic_anchors']['phi_0_must_equal_raw_.71475'] = \
        phi_out['censor_variant']['0.0']['frozen_ap_mean']
    # cheap independent read on phi with the columns we have (OBS 12 did the 897-feature version)
    posf = full.filter(pl.col('label') == 1)
    hif = full[hi.tolist()]
    negf = full.filter(pl.col('label') == 0)
    phi_out['who_are_the_89'] = {
        'note': 'OBS 12/12.1 settled this on 897 features (AUC .46-.62 vs positives, .997-.998 vs '
                'labelled negatives). This is only a cheap consistency read on the score columns.',
        'median_shared_hands': {'labelled_positives': float(posf['shared_hands'].median()),
                                'unlabelled_high': float(hif['shared_hands'].median()),
                                'labelled_negatives': float(negf['shared_hands'].median())},
        'median_purity': {'labelled_positives': round(float(posf['purity'].median()), 4),
                          'unlabelled_high': round(float(hif['purity'].median()), 4),
                          'labelled_negatives': round(float(negf['purity'].median()), 4)},
        'predicted_family_counts_unlabelled_high': {
            str(k): int(v) for k, v in zip(*np.unique(
                full['p1'].to_numpy()[hi] * 0 + np.argmax(
                    np.c_[full['p1'].to_numpy()[hi], full['p2'].to_numpy()[hi],
                          full['p3'].to_numpy()[hi]], axis=1) + 1, return_counts=True))}}
    res['phi_sensitivity'] = phi_out
    log('phi done', round(time.time() - t0, 1))

    # ------------------------------------------------------------------ (C) full two-fold curve
    views = {}
    for pct, tag in ((100, ''), (80, '-prefix80'), (60, '-prefix60'), (40, '-prefix40')):
        v = load_view(tag)
        if v is not None:
            views[pct] = v
    res['views_available'] = sorted(views, reverse=True)
    uni40 = set(views[min(views)]['pair_id'])  # load_view already applied the floor
    curve, colr = {}, {}
    for k, v in sorted(views.items(), reverse=True):
        e = v.filter(pl.col('pair_id').is_in(uni40)).sort('pair_id')
        yy = (e['label'] == 1).to_numpy().astype(float)
        ss = e['risk'].to_numpy()
        colr[k] = ss
        pp = e.filter(pl.col('label') == 1)
        curve[str(k)] = {'n_rows': int(e.height), 'n_pos': int(yy.sum()),
                         'pu_ap': round(ap(yy, ss), 5),
                         'median_shared_hands': float(e['shared_hands'].median()),
                         'pos_below_.5': int((pp['risk'] < .5).sum()),
                         'pos_below_.1': int((pp['risk'] < .1).sum()),
                         'recall_at_300': round(float(yy[np.argsort(-ss, kind='stable')][:300].sum()
                                                      / max(1.0, yy.sum())), 4)}
    b = views[100].filter(pl.col('pair_id').is_in(uni40)).sort('pair_id')
    tb = b['table_id'].to_numpy()
    uq = np.unique(tb)
    ix = {t: np.where(tb == t)[0] for t in uq}
    yb = (b['label'] == 1).to_numpy().astype(float)
    bo = {str(k): [] for k in views if k != 100}
    for _ in range(N_BOOT):
        pick = RNG.choice(uq, size=len(uq), replace=True)
        rows = np.concatenate([ix[t] for t in pick])
        yy = yb[rows]
        if yy.sum() < 5:
            continue
        a0 = ap(yy, colr[100][rows])
        for k in bo:
            bo[k].append(ap(yy, colr[int(k)][rows]) - a0)
    res['full_two_fold_curve'] = {
        'universe': 'pairs eligible at the two-fold 40 % prefix, every arm scored by the frozen model',
        'curve': curve,
        'bootstrap_delta_vs_100': {
            k: {'mean': round(float(np.mean(v)), 5),
                'ci95': [round(float(np.quantile(v, .025)), 5), round(float(np.quantile(v, .975)), 5)]}
            for k, v in bo.items() if v}}
    log('curve done', round(time.time() - t0, 1))

    # ------------------------------------------------------------------ (B) burial overlap
    # universe as the coordinator framed it: pairs eligible at the two-fold 60 % prefix
    uni60 = set(views[60]['pair_id'])  # load_view already applied the floor
    pos_ids = views[100].filter((pl.col('pair_id').is_in(uni60)) & (pl.col('label') == 1))['pair_id'].to_list()
    risk_by = {}
    for k, v in views.items():
        f = v.filter(pl.col('pair_id').is_in(set(pos_ids)))
        risk_by[k] = dict(zip(f['pair_id'].to_list(), f['risk'].to_list()))
    meta = views[100].filter(pl.col('pair_id').is_in(set(pos_ids)))
    fam = dict(zip(meta['pair_id'].to_list(), meta['behavior_family'].to_list()))
    shh = dict(zip(meta['pair_id'].to_list(), meta['shared_hands'].to_list()))

    def buried(k, cut=.5):
        return {p for p in pos_ids if risk_by[k].get(p, 1.0) < cut}

    ov = {'universe_rows': int(len(uni60)), 'n_positives': len(pos_ids)}
    sets = {k: buried(k) for k in sorted(views, reverse=True)}
    ov['buried_below_.5'] = {str(k): sorted(v) for k, v in sets.items()}
    ov['counts_below_.5'] = {str(k): len(v) for k, v in sets.items()}
    ov['nesting_100_subset_of_80_subset_of_60_subset_of_40'] = {
        '100_in_80': sets[100] <= sets[80] if 80 in sets else None,
        '80_in_60': sets[80] <= sets[60] if 80 in sets and 60 in sets else None,
        '60_in_40': sets[60] <= sets[40] if 40 in sets else None,
        '100_in_60': sets[100] <= sets[60]}
    ov['overlap_100_and_60'] = {
        'both': len(sets[100] & sets[60]), 'only_100': len(sets[100] - sets[60]),
        'only_60': len(sets[60] - sets[100])}
    # relate to the 10 below risk .9 on the full dedup surface
    ten = set(full.filter((pl.col('label') == 1) & (pl.col('risk') < .9))['pair_id'].to_list())
    ov['full_surface_below_.9'] = {
        'n': len(ten), 'n_inside_this_universe': len(ten & set(pos_ids)),
        'n_of_those_buried_below_.5_at_100': len(ten & sets[100]),
        'n_of_those_buried_below_.5_at_60': len(ten & sets[60]),
        'newly_buried_at_60_that_were_already_below_.9_at_full': len((sets[60] - sets[100]) & ten)}
    # are the newly buried NEAR-MISSES, or a random draw?
    order = sorted(pos_ids, key=lambda p: risk_by[100][p])                # lowest full risk first
    rank = {p: i + 1 for i, p in enumerate(order)}
    new6 = sorted(sets[60] - sets[100], key=lambda p: rank[p])
    pool = [p for p in pos_ids if p not in sets[100]]
    obs_mean_rank = float(np.mean([rank[p] for p in new6])) if new6 else None
    null = []
    pr = np.array([rank[p] for p in pool], dtype=float)
    for _ in range(N_NULL):
        null.append(RNG.choice(pr, size=len(new6), replace=False).mean())
    ov['newly_buried_at_60'] = {
        'n': len(new6),
        'full_history_risk_rank_of_each_out_of_%d' % len(pos_ids): [rank[p] for p in new6],
        'full_history_risk_of_each': [round(risk_by[100][p], 4) for p in new6],
        'family_of_each': [fam[p] for p in new6],
        'shared_hands_of_each': [int(shh[p]) for p in new6],
        'observed_mean_rank': round(obs_mean_rank, 2) if new6 else None,
        'null_mean_rank': round(float(np.mean(null)), 2),
        'null_p05_rank': round(float(np.quantile(null, .05)), 2),
        'p_value_random_draw_this_low_or_lower': round(float(np.mean(np.array(null) <= obs_mean_rank)), 5)
        if new6 else None,
        'reading': 'a low observed mean rank means truncation bured the positives that were ALREADY '
                   'nearest the surface at full history, i.e. near-misses of the same kind'}
    # rank correlation of the positive ordering across prefixes
    def spear(a, b_):
        ra = np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(float)
        rb = np.argsort(np.argsort(b_, kind='stable'), kind='stable').astype(float)
        return round(float(np.corrcoef(ra, rb)[0, 1]), 4)
    v100 = np.array([risk_by[100][p] for p in pos_ids])
    ov['spearman_positive_risk_100_vs'] = {
        str(k): spear(v100, np.array([risk_by[k].get(p, np.nan) for p in pos_ids]))
        for k in sorted(views, reverse=True) if k != 100}
    res['burial_overlap'] = ov

    res['protected_after'] = check_protected()
    res['seconds'] = round(time.time() - t0, 1)
    write_json(OUT / 'audit-C-phi-overlap-and-full-curve.json', res)
    log(json.dumps({'phi': res['phi_sensitivity']['censor_variant'],
                    'curve': res['full_two_fold_curve'],
                    'overlap': {k: v for k, v in ov.items() if k != 'buried_below_.5'}},
                   indent=1, default=str))
    log('written', OUT / 'audit-C-phi-overlap-and-full-curve.json', res['seconds'])


if __name__ == '__main__':
    main()
