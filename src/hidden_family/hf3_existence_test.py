"""HF3: does a fourth coordination population exist in the development period at all?

WHY THIS IS THE RIGHT TEST.  The ca_ Mahalanobis anomaly (step 37) is UNSUPERVISED: it never
sees a label.  It nevertheless ranks all three disclosed families at AUC .990 / .996 / .998
with the median positive at the 99.77th percentile of the population.  Because it is
label-free, that performance is an unbiased estimate of its performance on a family with NO
labels -- which is exactly what `other_coordination` is.  This is strictly stronger than the
LOFO argument available to a supervised model (hf1): there is nothing to hold out.

So: if a fourth mechanism exists in development with an effect size comparable to the
disclosed three, at least three quarters of its pairs must sit above the 99.4th percentile of
this anomaly.  We can therefore go and look.

WHAT THE PRIOR ATTEMPT GOT WRONG (OBS 14.14/14.15).  It thresholded at the 99.9th percentile
of the WHOLE population, including pairs with 1-4 shared hands.  The normal-scores binning is
degenerate at tiny n, so the extreme tail filled up with 16 ultra-short pairs (shared-hands
quartiles 1 / 2.5 / 3.5 against a population median of 95) and the test was spent on noise.
Here the population is restricted to shared_hands >= 38 -- the host's own eligibility rule for
evaluation pairs -- and the normal scores and the covariance are re-estimated inside it.

PRE-REGISTERED GATES (frozen before any candidate set is inspected):
  G1 COHERENCE.  Mean pairwise cosine of the candidate set's z-vectors must exceed the 95th
     percentile of the same statistic over size-matched random draws from the confirmed
     non-targets.  Labelled positives score ~.57 within family; the population scores ~.00.
     A real mechanism is coherent; a tail of ordinary play is not.
  G2 NOVELTY.  The candidate centroid's cosine to all three disclosed-family centroids must be
     below the lowest family-vs-family cosine, otherwise the set is unlisted members of a
     disclosed family, not a fourth one.
  G3 SEPARABILITY.  Out-of-fold AUC (GroupKFold by table) of candidates vs confirmed negatives
     must exceed .70.  A coherent population generalises across tables; a tail artefact does not.
  G4 CLUSTERING.  Candidate pairs must concentrate in tables / players above a permutation
     null, as a coordinated mechanism planted per pool would.
Passing G1-G4 = a fourth population is present and characterisable.  Failing them = within the
measured power of this detector, no fourth mechanism of comparable effect size exists in the
development period, and that is itself a decision-relevant result.

Read-only on the original tree and on Codex's experiment outputs.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '4'
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from evidence_aug.paths import ORIGINAL, CV, write_json
from pu_pairs.step37_conditional_anomaly import normal_scores_binned

OUT = CV / 'experiments' / 'hidden_family_20260912'
PU = CV / 'experiments' / 'pu_pairs_20260911'
PRIOR = CV / 'experiments' / 'hidden_family_20260911'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
MIN_HANDS = 38          # host eligibility rule for evaluation pairs
RNG = np.random.default_rng(20260912)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mean_pairwise_cos(Z):
    """Mean off-diagonal cosine of the rows of Z."""
    if len(Z) < 2:
        return float('nan')
    U = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    G = U @ U.T
    n = len(Z)
    return float((G.sum() - np.trace(G)) / (n * (n - 1)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select(
        'pair_id', 'label', 'behavior_family')

    files = sorted(glob.glob(str(PU / 'fold1' / 'batch_*-pairs.parquet'))) + \
        sorted(glob.glob(str(PU / 'fold2' / 'batch_*-pairs.parquet')))
    sch = pl.scan_parquet(files[0]).collect_schema().names()
    ca_mean = [c for c in sch if c.startswith('ca_') and c.endswith('_mean')]
    dev = pl.concat([pl.read_parquet(f, columns=['pair_id', 'table_id', 'shared_hands', *ca_mean])
                     for f in files], how='vertical_relaxed')
    # folds 1 and 2 overlap: 86,922 rows carry only 74,326 distinct pairs, and every frame
    # below is duplicated the same way.  Deduplicate BEFORE joining or the joins multiply out
    # (and identical copies of a row have cosine 1.0, which would fake coherence).
    dev = dev.unique(subset='pair_id', keep='first')
    prior = pl.read_parquet(PRIOR / 'dev-anomaly.parquet').select(
        'pair_id', 'risk_known').unique(subset='pair_id', keep='first')
    dev = (dev.join(lab, on='pair_id', how='left')
              .join(prior, on='pair_id', how='left')
              .with_columns(pl.col('label').fill_null(-1)))
    players = pl.concat([pl.read_parquet(PU / f / 'scores.parquet',
                                         columns=['pair_id', 'player_1', 'player_2'])
                         for f in ['fold1', 'fold2']], how='vertical_relaxed')
    players = players.unique(subset='pair_id', keep='first')
    dev = dev.join(players, on='pair_id', how='left')
    assert dev['pair_id'].n_unique() == dev.height, 'duplicate pairs survived the joins'

    print(f'all development pairs in folds 1-2 : {dev.height}')
    dev = dev.filter(pl.col('shared_hands') >= MIN_HANDS)
    n_pos = int((dev['label'] == 1).sum())
    n_neg = int((dev['label'] == 0).sum())
    n_unl = int((dev['label'] == -1).sum())
    print(f'eligible (shared_hands >= {MIN_HANDS}) : {dev.height}  '
          f'[pos {n_pos}, neg {n_neg}, unlabelled {n_unl}]')

    # ---- re-estimate the anomaly INSIDE the eligible population ----
    n_h = dev['shared_hands'].to_numpy().astype(float)
    Z = normal_scores_binned(dev.select(ca_mean).to_numpy(), n_h)
    lw = LedoitWolf().fit(Z)
    d2 = lw.mahalanobis(Z)
    pct = np.argsort(np.argsort(d2, kind='mergesort'), kind='mergesort') / (len(d2) - 1.0)
    y = dev['label'].to_numpy()
    fam = dev['behavior_family'].to_numpy()
    risk_known = np.nan_to_num(dev['risk_known'].to_numpy(), nan=0.0)
    pos, neg, unl = y == 1, y == 0, y == -1

    print()
    print('=' * 92)
    print('POWER OF THE LABEL-FREE DETECTOR INSIDE THE ELIGIBLE POPULATION')
    print('=' * 92)
    power = {}
    for f in FAMS:
        m = (fam == f)
        sel = m | ~pos
        auc = float(roc_auc_score(m[sel], d2[sel]))
        qs = np.quantile(pct[m], [.10, .25, .50, .75])
        power[f] = {'n': int(m.sum()), 'auc_vs_rest': auc, 'pct_q10': float(qs[0]),
                    'pct_q25': float(qs[1]), 'pct_median': float(qs[2])}
        print(f'  {f:<24} n={int(m.sum()):>4}  AUC={auc:.4f}  '
              f'anomaly pct q10={qs[0]:.4f} q25={qs[1]:.4f} median={qs[2]:.4f}')
    floor = float(np.median([power[f]['pct_q25'] for f in FAMS]))
    print()
    print(f'  => a family of comparable effect size puts >=75% of its pairs '
          f'above percentile {floor:.4f}')

    # ---- family centroids in z-space ----
    cent = {f: Z[fam == f].mean(axis=0) for f in FAMS}
    fam_cos = {}
    for i, a in enumerate(FAMS):
        for b in FAMS[i + 1:]:
            fam_cos[a + '|' + b] = cos(cent[a], cent[b])
    print('  family-vs-family centroid cosine: ' +
          '  '.join(k[:2] + '-' + k.split('|')[1][:2] + '=' + format(v, '.3f')
                    for k, v in fam_cos.items()))
    g2_threshold = min(fam_cos.values())

    # ---- candidate sets: high anomaly, unlabelled, low supervised risk ----
    print()
    print('=' * 92)
    print('CANDIDATE SETS  (high anomaly, not labelled, supervised risk < .5)')
    print('=' * 92)
    hdr = (f'{"top-q":>8}{"thr pct":>9}{"n_band":>8}{"lab_pos":>9}{"lab_neg":>9}'
           f'{"unl_hi":>8}{"CAND":>7}{"coh":>8}{"null95":>9}{"cosDT":>7}{"cosSP":>7}{"cosCI":>7}')
    print(hdr)
    results = []
    neg_idx = np.where(neg)[0]
    for topq in [0.0010, 0.0025, 0.0050, 0.0100, 0.0200]:
        thr = 1.0 - topq
        band = pct >= thr
        cand = band & unl & (risk_known < 0.5)
        n_lab_pos, n_lab_neg = int((band & pos).sum()), int((band & neg).sum())
        n_unl_hi = int((band & unl & (risk_known >= 0.5)).sum())
        idx = np.where(cand)[0]
        coh = mean_pairwise_cos(Z[idx]) if len(idx) >= 2 else float('nan')

        null = []
        if len(idx) >= 2 and len(neg_idx) > len(idx):
            for _ in range(400):
                pick = RNG.choice(neg_idx, size=len(idx), replace=False)
                null.append(mean_pairwise_cos(Z[pick]))
        null95 = float(np.quantile(null, .95)) if null else float('nan')
        c_cent = Z[idx].mean(axis=0) if len(idx) else np.zeros(Z.shape[1])
        cf = {f: cos(c_cent, cent[f]) for f in FAMS}
        results.append({'top_q': topq, 'threshold_pct': thr, 'n_band': int(band.sum()),
                        'labelled_pos': n_lab_pos, 'labelled_neg': n_lab_neg,
                        'unlabelled_high_risk': n_unl_hi, 'n_candidates': len(idx),
                        'coherence': coh, 'null_p95': null95,
                        'null_mean': float(np.mean(null)) if null else float('nan'),
                        'cos_to_family': cf})
        print(f'{topq:>8.4f}{thr:>9.4f}{int(band.sum()):>8}{n_lab_pos:>9}{n_lab_neg:>9}'
              f'{n_unl_hi:>8}{len(idx):>7}{coh:>8.3f}{null95:>9.3f}'
              f'{cf["directed_transfer"]:>7.3f}{cf["soft_play"]:>7.3f}'
              f'{cf["coordinated_isolation"]:>7.3f}')

    coh_pos = {f: mean_pairwise_cos(Z[fam == f]) for f in FAMS}
    coh_all_pos = mean_pairwise_cos(Z[pos])
    coh_neg = mean_pairwise_cos(Z[neg])
    print()
    print('  reference coherence:  positives by family ' +
          '  '.join(f[:2] + '=' + format(v, '.3f') for f, v in coh_pos.items()) +
          f'  |  all positives={coh_all_pos:.3f}  confirmed negatives={coh_neg:.3f}')

    write_json(OUT / 'hf3-existence-test.json',
               {'min_shared_hands': MIN_HANDS, 'eligible_pairs': dev.height,
                'composition': {'pos': n_pos, 'neg': n_neg, 'unlabelled': n_unl},
                'power': power, 'detector_floor_pct': floor,
                'family_centroid_cos': fam_cos, 'g2_threshold': g2_threshold,
                'reference_coherence': {'by_family': coh_pos, 'all_positives': coh_all_pos,
                                        'confirmed_negatives': coh_neg},
                'bands': results})
    print()
    print(f'written {OUT / "hf3-existence-test.json"}')

    pl.DataFrame({'pair_id': dev['pair_id'], 'table_id': dev['table_id'],
                  'player_1': dev['player_1'], 'player_2': dev['player_2'],
                  'shared_hands': dev['shared_hands'], 'label': y,
                  'behavior_family': dev['behavior_family'],
                  'risk_known': risk_known, 'anom': d2, 'anom_pct': pct}).write_parquet(
        OUT / 'hf3-dev-eligible.parquet')
    np.save(OUT / 'hf3-dev-z.npy', Z.astype(np.float32))
    print(f'written {OUT / "hf3-dev-eligible.parquet"} and hf3-dev-z.npy')


if __name__ == '__main__':
    main()
