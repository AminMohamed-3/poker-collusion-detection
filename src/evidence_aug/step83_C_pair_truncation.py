"""Step 83 (author: claude (agent)): the PAIR/RISK side under truncation, and who the missed positives are.

Two questions, both on the pair side, both at realistic prevalence.

Q1 (secondary).  OBS 18.17 (plan item E4) refuted the length shift "by a second and independent
construction".  That study is `src/evidence_aug/step74_length_shift.py`, which contains zero
occurrences of the string `risk`: it truncates the DEVELOPMENT window, recomputes the three chrono
columns, and scores EvidenceMAP@5 of the stage-2 evidence lists on the 162 training-only audit pairs.
It never touches the pair/risk model and never computes an AP.  So the E4 null is an evidence-side
null.  This step measures the pair side of the same construction: the frozen per-fold production pair
model scored on genuine temporal prefixes of each table's development window (step 40 frames), plus
the refit-on-truncated arms that step 41 already fitted.

Q2 (primary).  Public PairAP ~.958 against local labelled AP .9995, and OBS 14.9 attributes the gap
to missed positives, not false positives.  Every pair-side study since 18.2 attacked the
false-positive direction.  This step asks the opposite: WHICH true positives does the frozen risk
model rank low, and does anything characterise them?

Measurement surface, declared before any number below was computed.  Labelled AP on development is
useless here (.9995, saturated).  The surface is PU-AP: the 164 labelled positives against every
other row of the deduplicated fold-1 + fold-2 all-pairs frames, restricted by the evaluation's own
eligibility rule (shared_hands >= 38).  67,463 rows, prevalence .24 %.  Its dynamic range on this
population is [~.002 (random), 1.0]; the frozen model sits at .715 and the recorded prefix-60 value
is .6757 (OBS 18.8), so roughly .28 of headroom is visible, against .0005 for labelled AP.  The
per-positive statistic is the percentile rank of a positive inside the eligible population.

Controls (OBS 19.5's lesson: a naive contrast conflates the property with whatever drives risk).
Three, reported side by side for every property:
  naive     missed positives vs the other positives, unstratified;
  matched   the same, stratified on (behaviour family x within-family shared-hands tercile);
  decisive  missed positives vs risk-matched UNLABELLED pairs (25 nearest neighbours in logit risk).
The decisive contrast is the one that answers "is there a recoverable signal": if a property cannot
tell a missed positive from an ordinary pair the model scores the same, it cannot lift it.
A permutation null over all 897 production features (500 draws, max |AUC - .5| across features)
gives the threshold any single-feature finding has to beat.

Reproduction controls (these must match OBSERVATIONS or the harness is not the same harness):
  67,463 eligible deduplicated pairs                              OBS 15.3
  50,761 eligible prefix-60 rows with 138 labelled positives      OBS 18.8
  frozen prefix-60 PU-AP .6757                                    OBS 18.8
  labelled-negative out-of-fold risk quantiles .0006 / .001       OBS 18.15
  labelled AP ~.9995                                              OBS 18.15 / plan
No held-fold evidence pool is opened and no held read is consumed; the only labels used are the
public development pair labels in data/development_labels.csv.

Output: experiments/pair_side_C_20260912/audit-C-pair-truncation.json
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import CV, write_json, check_protected

PU = CV / 'experiments' / 'pu_pairs_20260911'
OUT = CV / 'experiments' / 'pair_side_C_20260912'
MIN_SH = 38
RNG = np.random.default_rng(83)
N_PERM = 500
N_MATCH = 25
FAMILIES = ('directed_transfer', 'soft_play', 'coordinated_isolation')

PANEL = ['shared_hands', 'net_bb_gap_mean', 'net_bb_gap_max', 'contribution_bb_gap_mean',
         'joint_win_share_mean', 'joint_win_share_max', 'went_to_showdown_sum_mean',
         'players_at_showdown_mean', 'pot_bb_mean', 'ca_surprise_mean_mean',
         'ca_residual_0_mean_mean', 'ca_fold_benefit_mean_mean', 'v2_weak_call_sum_mean',
         'v2_soft_favorable_fold_sum_mean', 'ca_actions_mean']


def log(*a):
    print(*a, flush=True)


def ap(y, s):
    """Average precision, stable order (ties broken by row order, frames are pair_id-sorted)."""
    o = np.argsort(-s, kind='stable')
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1.0, y.sum()))


def auc(pos, neg):
    """Mann-Whitney AUC with ties at .5.  Returns nan if either side is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind='stable')
    ranks = np.empty(len(allv), dtype=float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = .5 * (i + j) + 1.0
        i = j + 1
    r = ranks[:len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def strat_auc(x, m, strata):
    """Stratified AUC: concordant comparisons pooled over strata (Mantel-Haenszel style)."""
    num = 0.0
    den = 0.0
    for s in np.unique(strata):
        k = strata == s
        p = x[k & m]
        n = x[k & ~m]
        if len(p) == 0 or len(n) == 0:
            continue
        a = auc(p, n)
        num += a * len(p) * len(n)
        den += len(p) * len(n)
    return float(num / den) if den else float('nan')


def strat_auc_all(X, m, blocks):
    """Vectorised stratified AUC over every column of X at once (ties count .5)."""
    num = np.zeros(X.shape[1])
    den = 0.0
    for k in blocks:
        P = X[k & m]
        N = X[k & ~m]
        if len(P) == 0 or len(N) == 0:
            continue
        gt = (P[:, None, :] > N[None, :, :]).sum(axis=(0, 1))
        eq = (P[:, None, :] == N[None, :, :]).sum(axis=(0, 1))
        num += gt + .5 * eq
        den += len(P) * len(N)
    return num / den if den else np.full(X.shape[1], np.nan)


def matched_auc_all(T, groups):
    """Per-target matched AUC: each target row vs its OWN control rows, averaged over targets.

    T is a list of (target_vector, control_matrix) so the comparison is conditional on the match,
    not pooled across targets with different match values.
    """
    tot = np.zeros(groups[0][1].shape[1])
    n = 0
    for t, C in groups:
        if C.shape[0] == 0:
            continue
        gt = (t[None, :] > C).sum(axis=0)
        eq = (t[None, :] == C).sum(axis=0)
        tot += (gt + .5 * eq) / C.shape[0]
        n += 1
    return tot / max(1, n)


def load_view(tag):
    parts = []
    for fold in (1, 2):
        d = PU / (f'fold{fold}' + tag)
        if not (d / 'scores.parquet').exists():
            return None
        parts.append(pl.read_parquet(d / 'scores.parquet')
                     .select('pair_id', 'table_id', 'label', 'behavior_family', 'shared_hands', 'risk')
                     .with_columns(pl.lit(fold).alias('fold')))
    return pl.concat(parts).unique(subset=['pair_id'], keep='first', maintain_order=True).sort('pair_id')


def load_view_fold(tag, fold):
    """Single-fold view -- used for the dose-response curve when only fold 1 has every prefix."""
    d = PU / (f'fold{fold}' + tag)
    if not (d / 'scores.parquet').exists():
        return None
    return (pl.read_parquet(d / 'scores.parquet')
            .select('pair_id', 'table_id', 'label', 'behavior_family', 'shared_hands', 'risk')
            .sort('pair_id'))


def surface(frame):
    e = frame.filter(pl.col('shared_hands') >= MIN_SH)
    y = (e['label'] == 1).to_numpy().astype(float)
    s = e['risk'].to_numpy()
    lab = e.filter(pl.col('label') >= 0)
    return {'n_rows': int(e.height), 'n_pos': int(y.sum()),
            'n_labelled_neg': int((e['label'] == 0).sum()),
            'prevalence': round(float(y.mean()), 5),
            'pu_ap': round(ap(y, s), 4),
            'labelled_ap': round(ap((lab['label'] == 1).to_numpy().astype(float), lab['risk'].to_numpy()), 6),
            'median_shared_hands': float(e['shared_hands'].median()),
            'median_shared_hands_pos': float(e.filter(pl.col('label') == 1)['shared_hands'].median()),
            'pos_risk_q05_q10_q25_q50': [round(float(e.filter(pl.col('label') == 1)['risk'].quantile(q)), 4)
                                         for q in (.05, .10, .25, .50)],
            'recall_at_300': round(float(y[np.argsort(-s, kind='stable')][:300].sum() / max(1.0, y.sum())), 4)}


def percentiles(frame):
    """Percentile rank of every row inside its eligible population (1 = top)."""
    e = frame.filter(pl.col('shared_hands') >= MIN_SH)
    s = e['risk'].to_numpy()
    order = np.argsort(s, kind='stable')
    pct = np.empty(len(s))
    pct[order] = (np.arange(len(s)) + .5) / len(s)
    return e.with_columns(pl.Series('pct', pct))


def features_frame(pair_ids):
    keep = set(pair_ids)
    parts = []
    for fold in (1, 2):
        for p in sorted((PU / f'fold{fold}').glob('batch_*-pairs.parquet')):
            d = pl.read_parquet(p)
            d = d.filter(pl.col('pair_id').is_in(keep))
            if d.height:
                parts.append(d)
    return pl.concat(parts).unique(subset=['pair_id'], keep='first').sort('pair_id')


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {'author': 'claude (agent)', 'step': 'step83_C_pair_truncation',
           'question': 'Does 18.17 cover the pair model, and which true positives does the risk model rank low?',
           'training_only': False,
           'surface_note': 'development pair labels (data/development_labels.csv) + frozen per-fold out-of-fold risk; '
                           'no held-fold evidence pool opened, no held read consumed',
           'protected_before': check_protected()}

    # ---------------------------------------------------------------- Q0 what 18.17 actually tested
    src = CV / 'src' / 'evidence_aug'
    res['obs_18_17_scope'] = {
        f: {'mentions_risk': sum(1 for line in (src / f).read_text(encoding='utf-8').splitlines() if 'risk' in line),
            'mentions_map5': sum(1 for line in (src / f).read_text(encoding='utf-8').splitlines() if 'map5' in line.lower()),
            'mentions_average_precision': sum(1 for line in (src / f).read_text(encoding='utf-8').splitlines()
                                              if 'average_precision' in line or 'pair_ap' in line)}
        for f in ('step74_length_shift.py', 'step74b_length_shift_contrasts.py')}

    # ---------------------------------------------------------------- Q1 truncation curve, pair side
    views = {}
    for pct, tag in ((100, ''), (80, '-prefix80'), (60, '-prefix60'), (40, '-prefix40')):
        v = load_view(tag)
        if v is not None:
            views[pct] = v
            log(f'view {pct}% loaded', v.height, round(time.time() - t0, 1))
    res['views_available'] = sorted(views, reverse=True)
    res['surface_own_eligibility'] = {str(k): surface(v) for k, v in sorted(views.items(), reverse=True)}

    # paired: same pair universe (eligible at the SHORTEST prefix built), each view scored on it
    shortest = min(views)
    universe = set(views[shortest].filter(pl.col('shared_hands') >= MIN_SH)['pair_id'])
    paired = {}
    full_pct = percentiles(views[100]).select('pair_id', pl.col('pct').alias('pct_full'), pl.col('risk').alias('risk_full'))
    for k, v in sorted(views.items(), reverse=True):
        e = v.filter(pl.col('pair_id').is_in(universe))
        y = (e['label'] == 1).to_numpy().astype(float)
        s = e['risk'].to_numpy()
        pos = e.filter(pl.col('label') == 1)
        order = np.argsort(s, kind='stable')
        pct = np.empty(len(s))
        pct[order] = (np.arange(len(s)) + .5) / len(s)
        pe = e.with_columns(pl.Series('pct', pct)).filter(pl.col('label') == 1)
        paired[str(k)] = {'n_rows': int(e.height), 'n_pos': int(y.sum()),
                          'pu_ap': round(ap(y, s), 4),
                          'median_shared_hands': float(e['shared_hands'].median()),
                          'pos_mean_percentile': round(float(pe['pct'].mean()), 5),
                          'pos_median_percentile': round(float(pe['pct'].median()), 5),
                          'pos_risk_median': round(float(pos['risk'].median()), 4),
                          'pos_below_.5': int((pos['risk'] < .5).sum()),
                          'pos_below_.1': int((pos['risk'] < .1).sum()),
                          'recall_at_300': round(float(y[np.argsort(-s, kind='stable')][:300].sum() / max(1.0, y.sum())), 4)}
    res['truncation_paired_same_pairs'] = paired

    # table-level bootstrap of the paired PU-AP deltas against the 100 % view
    base = views[100].filter(pl.col('pair_id').is_in(universe)).sort('pair_id')
    tables = base['table_id'].to_numpy()
    uniq = np.unique(tables)
    cols = {}
    for k, v in sorted(views.items(), reverse=True):
        cols[k] = v.filter(pl.col('pair_id').is_in(universe)).sort('pair_id')['risk'].to_numpy()
    yb = (base['label'] == 1).to_numpy().astype(float)
    boot = {str(k): [] for k in views if k != 100}
    for _ in range(400):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([np.where(tables == t)[0] for t in pick])
        y = yb[rows]
        if y.sum() < 5:
            continue
        a100 = ap(y, cols[100][rows])
        for k in boot:
            boot[k].append(ap(y, cols[int(k)][rows]) - a100)
    res['truncation_bootstrap_delta_vs_100'] = {
        k: {'mean': round(float(np.mean(v)), 4),
            'ci95': [round(float(np.quantile(v, .025)), 4), round(float(np.quantile(v, .975)), 4)]}
        for k, v in boot.items() if v}

    # fold-1-only DOSE RESPONSE.  fold 2 prefix80/prefix40 did not finish in the time available,
    # so the 4-point curve is measured inside fold 1 on a universe paired at the shortest prefix.
    f1 = {}
    for pct, tag in ((100, ''), (80, '-prefix80'), (60, '-prefix60'), (40, '-prefix40')):
        v1 = load_view_fold(tag, 1)
        if v1 is not None:
            f1[pct] = v1
    if len(f1) >= 3:
        uni1 = set(f1[min(f1)].filter(pl.col('shared_hands') >= MIN_SH)['pair_id'])
        dose, colf = {}, {}
        for k, v in sorted(f1.items(), reverse=True):
            e = v.filter(pl.col('pair_id').is_in(uni1)).sort('pair_id')
            y = (e['label'] == 1).to_numpy().astype(float)
            s1 = e['risk'].to_numpy()
            colf[k] = s1
            pos = e.filter(pl.col('label') == 1)
            dose[str(k)] = {'n_rows': int(e.height), 'n_pos': int(y.sum()),
                            'pu_ap': round(ap(y, s1), 4),
                            'median_shared_hands': float(e['shared_hands'].median()),
                            'pos_risk_median': round(float(pos['risk'].median()), 4),
                            'pos_below_.5': int((pos['risk'] < .5).sum()),
                            'pos_below_.1': int((pos['risk'] < .1).sum()),
                            'recall_at_300': round(float(y[np.argsort(-s1, kind='stable')][:300].sum()
                                                        / max(1.0, y.sum())), 4)}
        b1 = f1[100].filter(pl.col('pair_id').is_in(uni1)).sort('pair_id')
        tb1 = b1['table_id'].to_numpy()
        u1 = np.unique(tb1)
        idx1 = {t: np.where(tb1 == t)[0] for t in u1}
        y1 = (b1['label'] == 1).to_numpy().astype(float)
        bo1 = {str(k): [] for k in f1 if k != 100}
        for _ in range(400):
            pick = RNG.choice(u1, size=len(u1), replace=True)
            rows = np.concatenate([idx1[t] for t in pick])
            yy = y1[rows]
            if yy.sum() < 5:
                continue
            a100 = ap(yy, colf[100][rows])
            for k in bo1:
                bo1[k].append(ap(yy, colf[int(k)][rows]) - a100)
        res['truncation_fold1_dose_response'] = {
            'note': 'fold 1 only (fold 2 prefix80/prefix40 builds unfinished); universe paired at the '
                    'shortest fold-1 prefix built, every arm SCORED by the frozen fold-1 pair model.',
            'curve': dose,
            'bootstrap_delta_vs_100': {
                k: {'mean': round(float(np.mean(v)), 4),
                    'ci95': [round(float(np.quantile(v, .025)), 4), round(float(np.quantile(v, .975)), 4)]}
                for k, v in bo1.items() if v}}
    else:
        res['truncation_fold1_dose_response'] = {'note': 'fewer than 3 fold-1 prefixes on disk',
                                                 'available': sorted(f1, reverse=True)}

    # CONTROL for the truncation reading: is the loss a train/test SHIFT or just less information?
    # If it is information loss, full-history pairs that are naturally short should score about the
    # same as truncated pairs of the same length.  Bin the 100 % view by its own shared_hands.
    fullv = views[100].filter(pl.col('shared_hands') >= MIN_SH)
    bins = [(38, 60), (60, 80), (80, 100), (100, 130), (130, 1000)]
    res['length_response_control'] = {'full_view_by_own_length': {}, 'prefix_views_by_own_length': {}}
    for a, b in bins:
        g = fullv.filter((pl.col('shared_hands') >= a) & (pl.col('shared_hands') < b))
        y = (g['label'] == 1).to_numpy().astype(float)
        res['length_response_control']['full_view_by_own_length'][f'{a}_{b}'] = {
            'n': int(g.height), 'n_pos': int(y.sum()),
            'pu_ap': round(ap(y, g['risk'].to_numpy()), 4) if y.sum() else None,
            'pos_below_.5': int((g.filter(pl.col('label') == 1)['risk'] < .5).sum())}
    for k, v in sorted(views.items(), reverse=True):
        if k == 100:
            continue
        e = v.filter(pl.col('shared_hands') >= MIN_SH)
        row = {}
        for a, b in bins:
            g = e.filter((pl.col('shared_hands') >= a) & (pl.col('shared_hands') < b))
            y = (g['label'] == 1).to_numpy().astype(float)
            row[f'{a}_{b}'] = {'n': int(g.height), 'n_pos': int(y.sum()),
                               'pu_ap': round(ap(y, g['risk'].to_numpy()), 4) if y.sum() else None}
        res['length_response_control']['prefix_views_by_own_length'][str(k)] = row

    # refit-on-truncated: read step 41's existing arms (fitted ON the 60 % prefix frames)
    refit = {}
    for tag in ('step41_train1_test2_prefix60', 'step41_train2_test1_prefix60'):
        p = PU / (tag + '.json')
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            refit[tag] = {a: {'ap_all_unlabelled_neg': round(v['ap_all_unlabelled_neg'], 4),
                              'ap_labelled': round(v['ap_labelled'], 4)}
                          for a, v in d['arms'].items()}
    res['refit_on_truncated_step41'] = refit

    # ---------------------------------------------------------------- Q2 who is missed (FULL view)
    full = percentiles(views[100])
    pos = full.filter(pl.col('label') == 1).sort('risk')
    n_pos = pos.height
    defs = {'risk_lt_.5': (pos['risk'] < .5).to_numpy(),
            'risk_lt_.9': (pos['risk'] < .9).to_numpy(),
            'bottom_decile': np.arange(n_pos) < int(round(.10 * n_pos)),
            'bottom_quintile': np.arange(n_pos) < int(round(.20 * n_pos))}
    res['missed_definitions'] = {k: int(v.sum()) for k, v in defs.items()}

    fam = pos['behavior_family'].to_numpy()
    sh = pos['shared_hands'].to_numpy().astype(float)
    res['missed_descriptive'] = {}
    for name, m in defs.items():
        res['missed_descriptive'][name] = {
            'n': int(m.sum()),
            'family_counts': {f: int(((fam == f) & m).sum()) for f in FAMILIES},
            'family_share_missed': {f: round(float(((fam == f) & m).sum() / max(1, m.sum())), 3) for f in FAMILIES},
            'family_base_share': {f: round(float((fam == f).mean()), 3) for f in FAMILIES},
            'family_miss_rate': {f: round(float(((fam == f) & m).sum() / max(1, (fam == f).sum())), 3) for f in FAMILIES},
            'shared_hands_median_missed': float(np.median(sh[m])),
            'shared_hands_median_rest': float(np.median(sh[~m])),
            'shared_hands_auc_naive': round(auc(sh[m], sh[~m]), 3),
            'risk_max_in_set': round(float(pos['risk'].to_numpy()[m].max()), 4),
            'pct_median_missed': round(float(np.median(pos['pct'].to_numpy()[m])), 4)}

    # matched strata: family x within-family shared-hands tercile
    strata = np.empty(n_pos, dtype=object)
    for f in FAMILIES:
        k = fam == f
        q = np.quantile(sh[k], [1 / 3, 2 / 3])
        strata[k] = [f'{f}_{int(np.searchsorted(q, v))}' for v in sh[k]]
    strata = np.array([str(s) for s in strata])

    # decisive control: risk-matched unlabelled neighbours
    elig = full.filter(pl.col('label') < 0)
    unl_ids = list(elig['pair_id'])
    unl_risk = elig['risk'].to_numpy()
    lg = lambda v: np.log(np.clip(v, 1e-9, 1 - 1e-9) / (1 - np.clip(v, 1e-9, 1 - 1e-9)))
    unl_lg = lg(unl_risk)
    unl_order = np.argsort(unl_lg)
    unl_lg_sorted = unl_lg[unl_order]

    def neighbours(target_risks):
        """K nearest unlabelled pairs in logit risk, per target, kept separate (conditional match)."""
        out = []
        for t in lg(np.asarray(target_risks)):
            j = np.searchsorted(unl_lg_sorted, t)
            lo, hi = max(0, j - N_MATCH), min(len(unl_lg_sorted), j + N_MATCH)
            out.append(unl_order[lo:hi].tolist())
        return out

    # placebo: an equal-sized sample of NON-missed positives gets the same treatment.  If a property
    # separates them from their own risk-matched controls just as well, it is a "positives differ
    # from unlabelled pairs" effect, not a property of being missed.
    ctrl_of, placebo_of = {}, {}
    risks = pos['risk'].to_numpy()
    for name, m in defs.items():
        if m.sum() < 4:
            continue
        ctrl_of[name] = neighbours(risks[m])
        others = np.where(~m)[0]
        pick = RNG.choice(others, size=min(int(m.sum()), len(others)), replace=False)
        placebo_of[name] = (pick, neighbours(risks[pick]))

    # one pass over the batch frames for everything we need
    need = set(pos['pair_id'])
    for groups in list(ctrl_of.values()) + [g for _, g in placebo_of.values()]:
        for g in groups:
            need.update(unl_ids[q] for q in g)
    feat = features_frame(sorted(need))
    model_features = [c for c in feat.columns if c not in ('pair_id', 'table_id')]
    M = np.nan_to_num(feat.select(model_features).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    fmap = {r[0]: i for i, r in enumerate(feat.select('pair_id').iter_rows())}
    X = M[np.array([fmap[p] for p in pos['pair_id']])]
    blocks = [strata == s for s in np.unique(strata)]
    log('feature matrices', M.shape, X.shape, round(time.time() - t0, 1))

    scan = {}
    for name, m in defs.items():
        if name not in ctrl_of:
            continue
        groups = [(X[i], M[np.array([fmap[unl_ids[q]] for q in g])])
                  for i, g in zip(np.where(m)[0], ctrl_of[name])]
        pick, pg = placebo_of[name]
        pgroups = [(X[i], M[np.array([fmap[unl_ids[q]] for q in g])]) for i, g in zip(pick, pg)]
        dgap = [float(lg(risks[i]) - np.median(lg(unl_risk[np.array(g)]))) for i, g in zip(np.where(m)[0], ctrl_of[name])]
        a_naive = strat_auc_all(X, m, [np.ones(n_pos, dtype=bool)])
        a_matched = strat_auc_all(X, m, blocks)
        a_dec = matched_auc_all(groups, groups)
        a_plac = matched_auc_all(pgroups, pgroups)
        entry = {'n_missed': int(m.sum()),
                 'controls_per_target': int(np.median([g[1].shape[0] for g in groups])),
                 'match_quality_max_abs_logit_gap': round(float(np.max(np.abs(dgap))), 4),
                 'panel': {}}
        for col in PANEL:
            if col not in model_features:
                continue
            j = model_features.index(col)
            entry['panel'][col] = {'naive': round(float(a_naive[j]), 3),
                                   'matched_family_length': round(float(a_matched[j]), 3),
                                   'decisive_vs_risk_matched': round(float(a_dec[j]), 3),
                                   'placebo_nonmissed_positives': round(float(a_plac[j]), 3)}
        # permutation null over ALL 897 features for the matched contrast
        null_max = np.empty(N_PERM)
        for b in range(N_PERM):
            mp = np.zeros(n_pos, dtype=bool)
            for k in blocks:
                ix = np.where(k)[0]
                want = int(m[ix].sum())
                if want:
                    mp[RNG.choice(ix, size=want, replace=False)] = True
            null_max[b] = np.nanmax(np.abs(strat_auc_all(X, mp, blocks) - .5))
        thr = float(np.nanquantile(null_max, .95))
        dev = np.abs(np.nan_to_num(a_matched, nan=.5) - .5)
        devd = np.abs(np.nan_to_num(a_dec, nan=.5) - .5)
        devp = np.abs(np.nan_to_num(a_plac, nan=.5) - .5)
        top = np.argsort(-dev)[:8]
        topd = np.argsort(-devd)[:8]
        entry['scan_897'] = {
            'permutation_null_95pct_max_abs_auc_minus_half_all_897': round(thr, 3),
            'n_features_beating_null_matched': int((dev > thr).sum()),
            'top8_matched': [{'feature': model_features[j], 'matched': round(float(a_matched[j]), 3),
                              'decisive': round(float(a_dec[j]), 3), 'placebo': round(float(a_plac[j]), 3)}
                             for j in top],
            'top8_decisive': [{'feature': model_features[j], 'decisive': round(float(a_dec[j]), 3),
                               'placebo': round(float(a_plac[j]), 3),
                               'decisive_minus_placebo': round(float(a_dec[j] - a_plac[j]), 3)} for j in topd],
            'max_abs_decisive_minus_half': round(float(np.nanmax(devd)), 3),
            'median_abs_decisive_minus_half': round(float(np.nanmedian(devd)), 3),
            'max_abs_placebo_minus_half': round(float(np.nanmax(devp)), 3),
            'n_features_decisive_beats_placebo_by_.15': int((devd - devp > .15).sum())}
        # permutation null for the DECISIVE contrast: inside each matched set, promote a random
        # member to "target".  Null hypothesis: a missed positive is exchangeable with any pair the
        # model scores the same.  This is the null the 897-feature scan has to beat.
        pool = [np.vstack([t[None, :], C]) for t, C in groups]
        dnull = np.empty(N_PERM)
        for b in range(N_PERM):
            g2 = []
            for P in pool:
                i = RNG.integers(P.shape[0])
                g2.append((P[i], np.delete(P, i, axis=0)))
            dnull[b] = np.nanmax(np.abs(matched_auc_all(g2, g2) - .5))
        entry['scan_897']['permutation_null_95pct_max_abs_decisive_minus_half'] = round(float(np.quantile(dnull, .95)), 3)
        entry['scan_897']['n_features_beating_decisive_null'] = int((devd > np.quantile(dnull, .95)).sum())
        j_sh = model_features.index('shared_hands')
        entry['shared_hands_detail'] = {
            'missed_median': float(np.median(X[m, j_sh])),
            'risk_matched_control_median': float(np.median(np.concatenate([C[:, j_sh] for _, C in groups]))),
            'eligible_population_median': float(fullv['shared_hands'].median()),
            'decisive_auc': round(float(a_dec[j_sh]), 3),
            'decisive_null_95': round(float(np.quantile(dnull, .95)), 3)}
        scan[name] = entry
        log(f'scan {name} done', round(time.time() - t0, 1))
    res['missed_controls'] = scan

    # THE CONTROL THAT WOULD KILL IT.  If "at matched risk the missed positives have longer
    # histories" were a usable lever, adding log shared_hands to the frozen order would raise PU-AP
    # on the eligible population.  OBS 18.2 predicts it will not (every learned blend put +.18 to
    # +.29 on log shared_hands and was <= 0 on eligible rows).  Sweep it directly.
    z = np.log(fullv['shared_hands'].to_numpy().astype(float))
    z = (z - z.mean()) / z.std()
    r = fullv['risk'].to_numpy()
    lr = np.log(np.clip(r, 1e-9, 1 - 1e-9) / (1 - np.clip(r, 1e-9, 1 - 1e-9)))
    yv = (fullv['label'] == 1).to_numpy().astype(float)
    sweep = {'w_0': round(ap(yv, lr), 4)}
    for w in (.05, .1, .25, .5, 1., 2.):
        sweep[f'w_{w}'] = round(ap(yv, lr + w * z), 4)
    sweep['shared_hands_alone'] = round(ap(yv, z), 4)
    # and restricted to the low-risk band where the missed positives live
    low = fullv.filter(pl.col('risk') < .9)
    zl = np.log(low['shared_hands'].to_numpy().astype(float))
    zl = (zl - zl.mean()) / zl.std()
    yl = (low['label'] == 1).to_numpy().astype(float)
    sweep['low_band_n'] = int(low.height)
    sweep['low_band_n_pos'] = int(yl.sum())
    sweep['low_band_shared_hands_alone_auc'] = round(auc(zl[yl == 1], zl[yl == 0]), 3)
    sweep['low_band_pu_ap_risk'] = round(ap(yl, low['risk'].to_numpy()), 5)
    sweep['low_band_pu_ap_shared_hands'] = round(ap(yl, low['shared_hands'].to_numpy().astype(float)), 5)
    sweep['base_rate_of_the_property'] = {
        f'risk<.9 and shared_hands>={t}': {
            'unlabelled': int(low.filter((pl.col('label') < 0) & (pl.col('shared_hands') >= t)).height),
            'labelled_positives': int(low.filter((pl.col('label') == 1) & (pl.col('shared_hands') >= t)).height)}
        for t in (100, 120, 148)}
    res['shared_hands_rerank_control'] = sweep

    # table-level structure of the missed set
    tbl = full.group_by('table_id').agg(pl.len().alias('n_pairs'),
                                        (pl.col('label') == 1).sum().alias('n_pos_table'),
                                        (pl.col('risk') >= .5).sum().alias('n_block_table'),
                                        pl.col('risk').max().alias('max_risk_table'))
    pj = pos.with_row_index('_i').join(tbl, on='table_id', how='left').sort('_i')  # join must not reorder
    assert pj['pair_id'].to_list() == pos['pair_id'].to_list()
    res['missed_table_structure'] = {}
    for name, m in defs.items():
        res['missed_table_structure'][name] = {
            k: {'missed': round(float(pj[k].to_numpy()[m].mean()), 3),
                'rest': round(float(pj[k].to_numpy()[~m].mean()), 3),
                'auc': round(auc(pj[k].to_numpy()[m].astype(float), pj[k].to_numpy()[~m].astype(float)), 3)}
            for k in ('n_pos_table', 'n_block_table', 'max_risk_table')}

    # plant density: does a positive get harder to see when its table carries more colluders?
    dens = {}
    for k in (1, 2, 3, 4):
        g = pj.filter(pl.col('n_pos_table') == k) if k < 4 else pj.filter(pl.col('n_pos_table') >= 4)
        dens[f'{k}{"+" if k == 4 else ""}'] = {
            'n_positives': int(g.height),
            'median_risk': round(float(g['risk'].median()), 4) if g.height else None,
            'frac_below_.9': round(float((g['risk'] < .9).mean()), 3) if g.height else None,
            'median_shared_hands': float(g['shared_hands'].median()) if g.height else None}
    res['plant_density'] = dens

    # ---------------------------------------------------------------- family-head PURITY
    # The pair model is a 4-class head: p0 = clean, p1/p2/p3 = the three families, risk = 1 - p0.
    # PURITY = max(p1,p2,p3) / (p1+p2+p3) asks "given the model thinks this pair is dirty, how sure
    # is it WHICH family?".  It is a ratio taken INSIDE the non-clean mass, so it is not a monotone
    # re-description of risk -- that is the failure mode 18.15 caught (block impurity, rho .9435).
    raw = pl.concat([pl.read_parquet(PU / f'fold{f}' / 'scores.parquet') for f in (1, 2)])             .unique(subset=['pair_id'], keep='first', maintain_order=True).sort('pair_id')
    ez = raw.filter(pl.col('shared_hands') >= MIN_SH).with_columns(
        (pl.max_horizontal('p1', 'p2', 'p3') /
         (pl.col('p1') + pl.col('p2') + pl.col('p3')).clip(1e-12)).alias('purity'))
    yz = (ez['label'] == 1).to_numpy().astype(float)
    rz = ez['risk'].to_numpy()
    pz = ez['purity'].to_numpy()
    tz = ez['table_id'].to_numpy()
    fam_head = ez.filter(pl.col('label') == 1).group_by('behavior_family', 'predicted_class')                  .agg(pl.len().alias('n')).sort('behavior_family')
    ph = {'family_head_confusion_on_positives': [
              {'true': r[0], 'predicted_class': int(r[1]), 'n': int(r[2])} for r in fam_head.iter_rows()],
          'pu_ap_risk': round(ap(yz, rz), 5),
          'pu_ap_purity_alone': round(ap(yz, pz), 5),
          'corr_purity_risk_spearman': round(float(np.corrcoef(
              np.argsort(np.argsort(pz)), np.argsort(np.argsort(rz)))[0, 1]), 4),
          'inside_risk_band': {}, 'base_rates': {}, 'tail_rescue_sweep': {}}
    for cut in (.9, .99):
        b = rz < cut
        ph['inside_risk_band'][f'risk<{cut}'] = {
            'n': int(b.sum()), 'n_pos': int(yz[b].sum()),
            'pu_ap_risk': round(ap(yz[b], rz[b]), 5),
            'pu_ap_purity': round(ap(yz[b], pz[b]), 5),
            'auc_purity': round(auc(pz[b][yz[b] == 1], pz[b][yz[b] == 0]), 3),
            'purity_median_pos': round(float(np.median(pz[b][yz[b] == 1])), 4),
            'purity_median_unlabelled': round(float(np.median(pz[b][yz[b] == 0])), 4)}
        for th in (.8, .9, .95, .99):
            m = b & (pz > th)
            ph['base_rates'][f'risk<{cut} & purity>{th}'] = {
                'rows': int(m.sum()), 'positives': int(yz[m].sum()),
                'precision': round(float(yz[m].mean()), 5) if m.sum() else None}

    def rescue(cut, w):
        """Keep the risk order above cut; re-rank only inside the tail band."""
        lr = np.log(np.clip(rz, 1e-9, 1 - 1e-9) / (1 - np.clip(rz, 1e-9, 1 - 1e-9)))
        b = rz < cut
        out = rz.copy()
        if b.sum():
            zc = (pz[b] - pz[b].mean()) / (pz[b].std() + 1e-12)
            zr = (lr[b] - lr[b].mean()) / (lr[b].std() + 1e-12)
            o = np.argsort(w * zc + zr, kind='stable')
            q = np.empty(int(b.sum()))
            q[o] = (np.arange(int(b.sum())) + .5) / b.sum()
            out[b] = cut * q
        return out

    base_ap = ap(yz, rz)
    for cut in (.9, .99, .999):
        for w in (.5, 1, 2, 4):
            ph['tail_rescue_sweep'][f'cut{cut}_w{w}'] = round(ap(yz, rescue(cut, w)) - base_ap, 5)
    uq = np.unique(tz)
    ix = {t: np.where(tz == t)[0] for t in uq}
    arms = {'cut.99_w0.5': rescue(.99, .5), 'cut.99_w1': rescue(.99, 1), 'cut.9_w0.5': rescue(.9, .5)}
    db = {k: [] for k in arms}
    for _ in range(400):
        pick = RNG.choice(uq, size=len(uq), replace=True)
        rows = np.concatenate([ix[t] for t in pick])
        yy = yz[rows]
        if yy.sum() < 5:
            continue
        a0 = ap(yy, rz[rows])
        for k, sc in arms.items():
            db[k].append(ap(yy, sc[rows]) - a0)
    ph['tail_rescue_bootstrap'] = {
        k: {'mean': round(float(np.mean(v)), 5),
            'ci95': [round(float(np.quantile(v, .025)), 5), round(float(np.quantile(v, .975)), 5)]}
        for k, v in db.items() if v}
    ph['fraction_of_positives_below_risk_.9'] = round(float((rz[yz == 1] < .9).mean()), 4)
    ph['obs_14_9_reference'] = '~6 % of public positives sit at low risk (OBS 14.9)'
    res['family_head_purity'] = ph

    # does truncation create NEW missed positives, or lose the same ones?
    if 60 in views:
        p60 = percentiles(views[60]).filter(pl.col('label') == 1).select(
            'pair_id', pl.col('risk').alias('risk_60'), pl.col('pct').alias('pct_60'))
        j = pos.select('pair_id', 'behavior_family', 'shared_hands', 'risk', 'pct').join(p60, on='pair_id', how='inner')
        lost = j.filter((pl.col('risk') >= .5) & (pl.col('risk_60') < .5))
        res['truncation_new_misses'] = {
            'n_positives_in_both': int(j.height),
            'missed_at_100_and_60': int(((j['risk'] < .5) & (j['risk_60'] < .5)).sum()),
            'missed_only_at_60': int(lost.height),
            'missed_only_at_100': int(((j['risk'] < .5) & (j['risk_60'] >= .5)).sum()),
            'new_miss_family_counts': {f: int((lost['behavior_family'] == f).sum()) for f in FAMILIES},
            'new_miss_shared_hands_median_full': float(lost['shared_hands'].median()) if lost.height else None,
            'all_pos_shared_hands_median_full': float(j['shared_hands'].median()),
            'delta_pct_median': round(float((j['pct_60'] - j['pct']).median()), 5),
            'delta_pct_q10_q90': [round(float((j['pct_60'] - j['pct']).quantile(q)), 5) for q in (.1, .9)]}

    res['protected_after'] = check_protected()
    res['seconds'] = round(time.time() - t0, 1)
    write_json(OUT / 'audit-C-pair-truncation.json', res)
    log(json.dumps({k: res[k] for k in ('surface_own_eligibility', 'truncation_paired_same_pairs',
                                        'missed_descriptive')}, indent=1, default=str))
    log('written', OUT / 'audit-C-pair-truncation.json', res['seconds'])


if __name__ == '__main__':
    main()
