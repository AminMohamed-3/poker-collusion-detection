"""Step 74 (plan E4): length-shift robustness of stage 2.

Evaluation histories are shorter than development ones -- the 16,060 selected pairs have a median of
about 78 shared hands against about 121 in development -- and read 2 of P0 landed in the branch where
the structural stage-2 increment did not transfer beyond r382.  18.4 refuted length shift by
RE-WEIGHTING the audit towards short pairs, which is a different thing from actually shortening a
history: re-weighting keeps every pair's chronology intact and only changes which pairs are counted.
This step does the real construction.

Two of the three chrono columns stage 2 carries are length-dependent by definition -- `chrono_rank` is
an absolute position and `n_hands` an absolute count -- while `chrono_frac` is a ratio and is not.  A
stage 2 trained where n_hands is 121 and applied where it is 78 is being asked to extrapolate on two
of its features.  That is the mechanism this step tests.

Construction.  For each prefix p in (1.0, .75, .6, .5): rank every hand of a table's DEVELOPMENT
window by `started_at` (the table's chronology, so all pairs at a table lose the same window, as a
shorter history really would), keep the hands below p of that window's length, then recompute `chrono_rank`, `n_hands`,
`chrono_frac` within each pair on what survives, re-take the top-20 by the unchanged stage-1 score,
drop listed hands outside the prefix from the truth and drop pairs with no listed hand left.  At
p = 1.0 the recomputation must reproduce the stored columns exactly; that assertion is the control.

Variants: {ref, r382} x {chrono3, chrono1} x {trained on full histories, trained on truncated
histories}, every one evaluated on the truncated pairs.  Training rows come from the training tables
of the same 5-way pool-hash split, so "trained on full" and "trained on truncated" differ only in
which version of the same tables the classifier saw.  The hybrid that production actually ships
(r382 for transfer and soft play, ref for isolation) is formed from the per-pair results afterwards.

Why the development window and not the whole table: every one of the 400 tables is exactly 60 %
development hands followed by 40 % evaluation hands, with no interleaving at all (0 violations).  So
the evaluation window is 40/60 = 2/3 as wide as the development one, which is the whole of the length
shift this step exists to test -- the observed median 78 against 121 shared hands is 0.645.  A prefix
of .667 reproduces the evaluation width exactly; the pre-registered decisive prefix .6 is just inside
it.  Ranking inside the whole table instead would place every development hand below .6 and truncate
nothing, which is what a first run of this script did.

Stage 1 is held fixed at its full-history per-hand scores, as the plan specifies.  That isolates
stage 2's length sensitivity; it also means this step does NOT measure the length sensitivity of the
ranker itself, which carries chrono_frac too.  Recorded as a limit.

Decision (pre-registered, written before any number below was computed): a variant is proposed for
production only if it beats the chrono3-full reference by more than +.010 MAP@5 on the truncated
audits with the paired bootstrap lower bound above 0, at the .6 prefix (the one closest to the
evaluation shift) and without a family below -.005.  The production branch is open because read 2 of
P0 reported no transfer.  Otherwise: record only.

Training-only (the 162 audit pairs, 172 tables outside every held fold).
Output: experiments/label_process_20260911/audit-length-shift.json
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl

from evidence_aug.fold import load_fold
from evidence_aug.frame import FAMILIES
from evidence_aug.paths import ORIGINAL, check_protected, write_json
from evidence_aug.step14_audit_truncation import CLF, OUT, TOP, SPLITS, split_of
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step65b_family_stage2 import boot

PREFIXES = (1.0, .75, .6, .5)
SEEDS = (11, 12, 13)
CHRONO3 = ['chrono_rank', 'chrono_frac', 'n_hands']
CHRONO1 = ['chrono_frac']
S1 = ['oof', 'oof_rank']
CI = 'coordinated_isolation'
REF = 'ref_full_chrono3'
GATE = {'beats_reference_by': .010, 'ci95_lower_gt': 0.0, 'family_floor': -.005,
        'decisive_prefix': .6, 'reference': REF,
        'production_branch': 'open: read 2 of P0 reported no transfer of the struct stage-2 increment',
        'note': 'written before any number in this file was computed'}


def rechrono(d):
    """recompute the three chrono columns within each pair, and the stage-1 rank, on whatever survives"""
    d = d.with_columns(
        pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'),
        pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'),
        pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.UInt32).alias('oof_rank'))
    return d.with_columns(pl.when(pl.col('n_hands') > 1)
                          .then(pl.col('chrono_rank') / (pl.col('n_hands') - 1))
                          .otherwise(0.0).alias('chrono_frac'))


def fit_predict(dtr, dte, cols, topk, seeds=SEEDS, params=CLF):
    """per-family, 5-way pool-hash OOF: fit on the training tables' rows of `dtr`, predict `dte`."""
    pred = np.full(dte.height, -1e6)
    ftr, fte = dtr['behavior_family'].to_numpy(), dte['behavior_family'].to_numpy()
    str_, ste = dtr['split'].to_numpy(), dte['split'].to_numpy()
    ytr = dtr['relevance'].to_numpy().astype(float)
    ktr = (dtr['oof_rank'] <= topk).to_numpy()
    kte = (dte['oof_rank'] <= topk).to_numpy()
    Xtr_all = dtr.select(cols).to_numpy().astype(np.float64)
    Xte_all = dte.select(cols).to_numpy().astype(np.float64)
    for fam in FAMILIES:
        for s in range(SPLITS):
            tr = np.where((ftr == fam) & ktr & (str_ != s))[0]
            te = np.where((fte == fam) & kte & (ste == s))[0]
            if len(te) == 0 or len(tr) == 0 or ytr[tr].sum() == 0:
                continue
            acc = np.zeros(len(te))
            for seed in seeds:
                clf = lgb.LGBMClassifier(**params, random_state=seed)
                clf.fit(Xtr_all[tr], ytr[tr])
                acc += clf.predict_proba(Xte_all[te])[:, 1] / len(seeds)
            pred[te] = acc
    return pred


def evaluate(dte, pred, topk, truths, labels):
    cands = (dte['oof_rank'] <= topk).to_numpy()
    z = np.where(cands, pred + 10.0, dte['oof'].to_numpy() - 1e6)
    key = dte.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', z))
    return summarize(key, truths, labels)


def build_d0():
    """the audit frame plus structural, ranker, chronology and table-position columns.

    Extracted from main() unchanged so step 74b can rebuild exactly the same frame; step 74b asserts
    reproduction of this file's stored prefix-1.0 numbers, which is the check that it stayed the same.
    """
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    feat = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    d0 = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')

    data, features, _, _ = load_fold(0)
    rcols = [f for f in features if not f.startswith('family_') and f in data.columns
             and f not in struct + CHRONO3 + S1]
    d0 = d0.join(data.select(['pair_id', 'hand_id'] + rcols), on=['pair_id', 'hand_id'],
                 how='left', validate='1:1').with_columns(pl.col(rcols).fill_null(-999.0))

    started = (pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').select('hand_id', 'started_at')
               .join(d0.select('hand_id').unique().lazy(), on='hand_id').collect())
    d0 = d0.join(started, on='hand_id', validate='m:1')
    # the TABLE's chronology: position of each hand among all hands of its table
    # Every table is exactly 60 % development hands followed by 40 % evaluation hands, with no
    # interleaving (verified: 0 of 400 tables violate it).  The window to truncate is therefore the
    # table's DEVELOPMENT portion; ranking inside the whole table would put every development hand
    # below .6 and truncate nothing.
    tbl = (pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet')
           .select('hand_id', 'table_id', 'started_at', 'phase')
           .filter(pl.col('table_id').is_in(d0['table_id'].unique().to_list())
                   & (pl.col('phase') == 'development')).collect())
    tbl = tbl.with_columns(
        pl.col('started_at').rank('ordinal').over('table_id').cast(pl.Float64).sub(1).alias('t_rank'),
        pl.len().over('table_id').cast(pl.Float64).alias('t_n'))
    d0 = d0.join(tbl.select('hand_id', 't_rank', 't_n'), on='hand_id', validate='m:1')
    d0 = d0.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))

    return d0, struct, rcols


def frame_at(d0, p):
    """the p-prefix of every table's development window, chrono recomputed within each surviving pair"""
    d = d0.filter(pl.col('t_rank') < p * pl.col('t_n')) if p < 1.0 else d0
    d = rechrono(d.sort('pair_id', 'hand_id'))
    keep = set(d.filter(pl.col('relevance') == 1)['pair_id'].to_list())
    return d.filter(pl.col('pair_id').is_in(list(keep))).sort('pair_id', 'hand_id')


def designs_of(struct, rcols):
    return {'ref_chrono3': struct + CHRONO3 + S1, 'ref_chrono1': struct + CHRONO1 + S1,
            'r382_chrono3': struct + CHRONO3 + S1 + rcols, 'r382_chrono1': struct + CHRONO1 + S1 + rcols}


def main():
    t0 = time.time()
    check_protected()
    d0, struct, rcols = build_d0()
    designs = designs_of(struct, rcols)
    res = {'training_only': True, 'gate': GATE, 'prefixes': list(PREFIXES),
           'n_structural': len(struct), 'n_ranker': len(rcols), 'seeds': list(SEEDS), 'top': TOP,
           'population': {}, 'variants': {}, 'paired_vs_reference': {}, 'hybrid': {}}
    print('=' * 100)
    print('STEP 74 (E4).  DOES STAGE 2 SURVIVE A SHORTER HISTORY?')
    print('=' * 100)

    per_pair_all, full_frame = {}, None
    for p in PREFIXES:
        d = frame_at(d0, p)
        if p == 1.0:                    # control: the recomputation must reproduce the stored columns
            for c in CHRONO3 + ['oof_rank']:
                a, b = d[c].to_numpy(), d0.sort('pair_id', 'hand_id')[c].to_numpy()
                assert np.allclose(a.astype(float), b.astype(float)), f'prefix 1.0 changed {c}'
            full_frame = d
        truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
        labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
        ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
        pools = sorted(set(ptab.values()))
        nh = d.select('pair_id', 'n_hands').unique()['n_hands'].to_numpy()
        res['population'][str(p)] = {
            'pairs': len(truths), 'hands': d.height,
            'median_n_hands': float(np.median(nh)), 'mean_n_hands': round(float(nh.mean()), 1),
            'listed_hands': int((d['relevance'] == 1).sum()),
            'pairs_dropped': 162 - len(truths),
            'listed_in_top20': round(float(((d['relevance'] == 1) & (d['oof_rank'] <= TOP)).sum()
                                           / max(1, (d['relevance'] == 1).sum())), 4)}
        print(f"\n  prefix {p}: {len(truths)} pairs, {d.height} hands, median n_hands "
              f"{res['population'][str(p)]['median_n_hands']:.0f}, "
              f"{res['population'][str(p)]['listed_hands']} listed, "
              f"{100 * res['population'][str(p)]['listed_in_top20']:.1f} % of listed inside the top-20")

        for dname, cols in designs.items():
            for train_on in ('full', 'trunc'):
                if p == 1.0 and train_on == 'trunc':
                    continue
                tag = f"{dname.split('_')[0]}_{train_on}_{dname.split('_')[1]}"
                dtr = full_frame if train_on == 'full' else d
                t1 = time.time()
                pred = fit_predict(dtr, d, cols, TOP)
                summ, per_pair = evaluate(d, pred, TOP, truths, labels)
                per_pair_all[(p, tag)] = per_pair
                res['variants'].setdefault(str(p), {})[tag] = {
                    'map5': round(summ['map5'], 4),
                    'by_family': {f: round(v, 4) for f, v in summ['by_family'].items()}}
                log(f'p={p} {tag}', json.dumps({'map5': round(summ['map5'], 4),
                                                'fam': {f[:2]: round(v, 4) for f, v in summ['by_family'].items()},
                                                'sec': round(time.time() - t1)}))

        # the shipped hybrid: r382 for transfer and soft play, ref for isolation
        for train_on in ('full', 'trunc'):
            if p == 1.0 and train_on == 'trunc':
                continue
            for ch in ('chrono3', 'chrono1'):
                a, b = per_pair_all[(p, f'r382_{train_on}_{ch}')], per_pair_all[(p, f'ref_{train_on}_{ch}')]
                hyb = {q: (b[q] if labels[q] == CI else a[q]) for q in a}
                per_pair_all[(p, f'hybrid_{train_on}_{ch}')] = hyb
                res['hybrid'].setdefault(str(p), {})[f'hybrid_{train_on}_{ch}'] = round(
                    float(np.mean(list(hyb.values()))), 4)

        ref_pp = per_pair_all[(p, REF)]
        for tag, pp in [(k[1], v) for k, v in per_pair_all.items() if k[0] == p]:
            if tag == REF:
                continue
            bb = boot(ref_pp, pp, ptab, pools)
            byf = {}
            for f in FAMILIES:
                fp = [q for q in pp if labels[q] == f]
                byf[f] = round(float(np.mean([pp[q] - ref_pp[q] for q in fp])), 4) if fp else None
            res['paired_vs_reference'].setdefault(str(p), {})[tag] = {
                'mean_delta': round(bb['mean_delta'], 4), 'ci95': [round(v, 4) for v in bb['ci95']],
                'by_family': byf}

    # ---------- verdict ----------
    dp = str(GATE['decisive_prefix'])
    verdict = {}
    for tag, row in res['paired_vs_reference'][dp].items():
        fam_ok = all(v is None or v >= GATE['family_floor'] for v in row['by_family'].values())
        verdict[tag] = {'mean_delta': row['mean_delta'], 'ci95': row['ci95'],
                        'beats_by': row['mean_delta'] >= GATE['beats_reference_by'],
                        'ci_lower_gt_0': row['ci95'][0] > GATE['ci95_lower_gt'],
                        'family_floor_ok': fam_ok}
        verdict[tag]['gate_met'] = bool(verdict[tag]['beats_by'] and verdict[tag]['ci_lower_gt_0'] and fam_ok)
    res['verdict_at_decisive_prefix'] = verdict
    res['any_gate_met'] = any(v['gate_met'] for v in verdict.values())

    print()
    print(f'  paired against {REF} at prefix {dp} (the decisive one):')
    for tag, v in sorted(verdict.items(), key=lambda kv: -kv[1]['mean_delta']):
        print(f"    {tag:<26} {v['mean_delta']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
              f"{'GATE MET' if v['gate_met'] else ''}")
    print()
    print(f"  ANY GATE MET: {res['any_gate_met']}")

    res['limits'] = [
        'Stage 1 is held at its full-history per-hand scores, so this measures stage 2 only; the '
        'ranker also carries chrono_frac and is not tested here.',
        'The truncation is of the table history, so pairs lose different fractions of their own hands.',
        '162 audit pairs; the truncated populations are smaller still (see population).',
        'Structural and ranker features are NOT recomputed on the prefix -- they are per-hand and do '
        'not depend on the history length, but any pair-level normalisation inside them would.']
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-length-shift.json', res)
    check_protected()
    print(f"\n  written {OUT / 'audit-length-shift.json'}  ({res['seconds']} s)")
    print('DONE')


if __name__ == '__main__':
    main()
