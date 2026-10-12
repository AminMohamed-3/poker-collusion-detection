"""step 85b [C].  The control that beat the treatment: is stage-1 CAPACITY a real lever?  And if a
better stage 1 does not pay, WHERE does the gain go?

author: claude (agent)

Step 85 asked whether an 8-seed ensemble beats the single stage-1 draw that ships.  Answer: no
(min-across-partition downstream delta -0.0011, verdict DEAD).  One of the CONTROLS in that file
was a capacity confound check -- "an 8-seed ensemble is 8x the compute; does one bigger model buy
the same lift?"  It did, and then some.  At salt audit-trunc-v1, stage-1 MAP@5:

    single seed 7329, n_estimators=238 (production)   .5574   recall@20 .9710
    8-seed score ensemble  (8x compute)               .5582   recall@20 .9660   (+.0008)
    n_estimators=476       (2x compute)               .5771   recall@20 .9723   (+.0197)
    n_estimators=952       (4x compute)               .5850   recall@20 .9736   (+.0276)
    n_estimators=1904      (8x compute)               .5930   recall@20 .9748   (+.0356)

That is 25-45x the ensemble's effect for the same or less compute, with bootstrap CIs excluding
zero, and -- unlike the ensemble -- it moves recall@20 UP.  Recall@20 is the one channel through
which a stage-1 change can reach stage 2 at all: stage 2 only ever sees the top-20 candidates
(OBS 18.14: a genuinely better stage 1, +.0167 held, bought -.0040 end-to-end).  The ensemble
lowers recall@20 and therefore cannot pay.  The tree-count arms raise it, so they could.

This file applies to the capacity lead the discipline that killed the ensemble: all three pool
partitions carrying the MINIMUM, end-to-end with the incumbent stage 2 unchanged and re-partitioned
to match the salt, pool-bootstrap CIs on every delta, composite units with the audit->held
transfer discount.

It also adds section C, the question the first attempt raised and could not answer.  Across the 8
of 9 arms that completed before the interruption noted below, stage-1 MAP@5 rose in 8 of 8 and
recall@20 rose in 8 of 8, yet the end-to-end hybrid FELL in 7 of 8 (the one exception, v1/trees_4x
at +.0001, is noise).  Those counts come from the interrupted log and are stated here only as the
motivation; every count reported in the artefact is recounted from this run's own 9 cells.  So
"recall@20 is the channel" is not enough: recall@20 went up everywhere and the hybrid still went
down.  The
hypothesis section C tests directly is that a stronger stage 1 does not merely ADD listed hands to
the top-20, it CHURNS the top-20, and the candidates it evicts include ones the incumbent stage 2
was already converting into correct final picks.  Section C measures that churn as a number and
splits it into what was lost and what was gained:

    top20_churn_fraction                      how much of the retrieved set is different at all
    listed_dropped / listed_added             the recall ledger behind the recall@20 delta
    correct_final_picks_lost_to_eviction      baseline stage-2 top-5 hits the arm evicted from the
                                              top-20, which stage 2 can then never recover
    correct_final_picks_gained_from_new       newly retrieved listed hands stage 2 promoted into
                                              its final five
    net_correct_final_picks                   gained minus lost; this is what MAP@5 tracks

PROVENANCE.  A first execution of this file was interrupted by the coordinator at 11 of 12 arms
(everything through audit-resplit-c/trees_4x; audit-resplit-c/trees_8x missing) and wrote no
artefact.  Nothing from that partial log is reused: this run recomputes all three salts end to end
so every number in the JSON comes from one coherent execution.  The interruption is recorded in
the artefact under `provenance`.

Reused, not refitted: the per-seed production stage-1 OOF vectors for all three salts are already
in stage1-oof-matrix.parquet from step 85, so the single-seed-7329 baseline at each salt is read
from disk.  A refit identity check at v1 guards that reuse and asserts on any mismatch.

Surface: the 162-pair training-only audit set.  Held folds 0/1/2 are never read.  No leaderboard
or evaluation label is used anywhere.  Nothing here is proposed for submission.
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from evidence_aug.frame import FAMILIES
from evidence_aug.fold import fold_pools, load_fold
from evidence_aug.paths import check_protected, write_json
from evidence_aug.step14_audit_truncation import OUT as LP, TOP
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step74_length_shift import (CHRONO3, CI, S1, build_d0, evaluate, fit_predict,
                                              frame_at)
from evidence_aug.step85_C_stage1_ensemble import (AUTHOR, LEVER, OUTDIR, RESOLUTION, SALTS,
                                                   TRANSFER, pair_slices, recalls, resplit,
                                                   stage1_oof, wrank)

ARMS = {'trees_2x': dict(n_estimators=476),
        'trees_4x': dict(n_estimators=952),
        'trees_8x': dict(n_estimators=1904)}
GATE = {'primary': 'end-to-end audit MAP@5 delta vs the production single-seed stage 1, incumbent '
                   'stage 2 unchanged, MINIMUM across the three pool partitions',
        'real_requires': ['min-across-salt downstream delta > 0',
                          'min-across-salt stage-1 delta > 0',
                          'min-across-salt recall@20 delta > 0'],
        'material_requires': f'{LEVER} * min_delta * {TRANSFER} >= {RESOLUTION} '
                             f'(min_delta >= {round(RESOLUTION / (LEVER * TRANSFER), 4)})',
        'verdict': 'ALIVE only if real AND material',
        'note': 'written before any number in this file was computed',
        'prior_under_test': 'OBS 18.14 says a better stage 1 need not pay; the weaker form is that '
                            'recall@20 is the channel that transmits. Section C tests whether a '
                            'recall@20 rise is SUFFICIENT, by measuring top-20 churn and the '
                            'correct final picks lost to eviction.'}
PROVENANCE = {
    'first_attempt': 'interrupted by the coordinator at 8 of 9 arms (through '
                     'audit-resplit-c/trees_4x; the missing arm was audit-resplit-c/trees_8x); '
                     'no artefact was written',
    'first_attempt_pattern': 'stage-1 MAP@5 up in 8/8, recall@20 up in 8/8, end-to-end up in 1/8 '
                             '(v1/trees_4x, +.0001); minimum -0.0166 at audit-resplit-b/trees_8x',
    'reuse_of_partial_log': 'none -- all three salts recomputed end to end in this run; every '
                            'count in this artefact is recounted from this run',
    'why_recorded': 'so the interruption is on the record rather than invisible'}


def main():
    t0 = time.time()
    check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTDIR / 'prediction-capacity-followup.json',
               {'author': AUTHOR, 'step': '85b', 'written_before_any_fit': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                'gate': GATE, 'arms': ARMS, 'salts': list(SALTS), 'provenance': PROVENANCE,
                'lead_from_step85_v1_only': {'trees_2x': .0197, 'trees_4x': .0276,
                                             'trees_8x': .0356},
                'churn_metrics_declared': ['top20_churn_fraction', 'listed_dropped', 'listed_added',
                                           'correct_final_picks_lost_to_eviction',
                                           'correct_final_picks_gained_from_new',
                                           'net_correct_final_picks']})

    print('=' * 100)
    print('STEP 85b [C].  DOES THE STAGE-1 CAPACITY LEAD SURVIVE A RE-SALT AND REACH STAGE 2?')
    print('=' * 100)

    # ---- frames: identical construction to step 85 --------------------------------------------
    audit = pl.read_parquet(LP / 'audit-oof.parquet')
    data, features, _, _ = load_fold(0)
    d = (audit.select('pair_id', 'hand_id')
         .join(data, on=['pair_id', 'hand_id'], how='left', validate='1:1')
         .join(audit.select('pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance',
                            'chrono_rank', 'chrono_frac', 'n_hands',
                            pl.col('oof').alias('oof_saved')),
               on=['pair_id', 'hand_id'], validate='1:1')
         .sort('pair_id', 'hand_id'))
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    assert not (set(d['table_id']) & held), 'audit surface overlaps a held fold -- stop'

    d0, struct, rcols = build_d0()
    d2 = frame_at(d0, 1.0)
    assert d.select('pair_id', 'hand_id').equals(d2.select('pair_id', 'hand_id')), 'frame keys differ'

    pid = d['pair_id'].to_numpy()
    rel = d['relevance'].to_numpy()
    famv = d['behavior_family'].to_numpy()
    sl = pair_slices(pid)
    key = d.select('pair_id', 'hand_id', 'table_id')
    truths = {k[0]: set(g['hand_id'])
              for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    stage2_r382 = struct + CHRONO3 + S1 + rcols
    stage2_ref = struct + CHRONO3 + S1
    prod_design = list(features) + ['chrono_frac']
    is_ci = famv == CI
    listed = rel == 1

    def s1_metrics(v):
        summ, per_pair = summarize(key.with_columns(pl.Series('score', v)), truths, labels)
        row = {'map5': round(summ['map5'], 4),
               'by_family': {f: round(x, 4) for f, x in summ['by_family'].items()}}
        row.update(recalls(rel, famv, v, sl))
        return row, per_pair

    def hybrid(oof_vec, split):
        """the incumbent stage 2, unchanged, on a supplied stage-1 vector, partitioned by `split`.

        Returns the MAP, the per-pair APs, and the FINAL combined score `z` -- the same quantity
        step74.evaluate ranks -- so section C can recover which hands stage 2 actually picked."""
        dd = d2.with_columns(pl.Series('oof', oof_vec), pl.Series('split', split))
        dd = dd.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id')
                             .cast(pl.UInt32).alias('oof_rank'))
        pr = fit_predict(dd, dd, stage2_r382, TOP)
        pf = fit_predict(dd, dd, stage2_ref, TOP)
        _, ppr = evaluate(dd, pr, TOP, truths, labels)
        _, ppf = evaluate(dd, pf, TOP, truths, labels)
        pp = {q: (ppf[q] if labels[q] == CI else ppr[q]) for q in ppr}
        cands = (dd['oof_rank'] <= TOP).to_numpy()
        ov = dd['oof'].to_numpy()
        z = np.where(is_ci, np.where(cands, pf + 10.0, ov - 1e6),
                     np.where(cands, pr + 10.0, ov - 1e6))
        return float(np.mean(list(pp.values()))), pp, z

    def churn(vb, zb, va, za):
        """where the stage-1 gain goes: what the arm evicted from the top-20, and what that cost.

        `vb`/`va` are baseline/arm stage-1 scores; `zb`/`za` the final stage-2 combined scores."""
        b20, a20 = wrank(vb, sl) <= TOP, wrank(va, sl) <= TOP
        b5, a5 = wrank(zb, sl) <= 5, wrank(za, sl) <= 5
        dropped, added = b20 & ~a20, a20 & ~b20
        slots = int(b20.sum())
        lost = int((b5 & dropped & listed).sum())
        gained = int((added & listed & a5).sum())
        return {
            'top20_slots': slots,
            'top20_churn_hands': int(dropped.sum()),
            'top20_churn_fraction': round(float(dropped.sum()) / slots, 4),
            'top20_churn_mean_hands_per_pair': round(float(dropped.sum()) / len(sl), 3),
            'listed_dropped': int((dropped & listed).sum()),
            'listed_added': int((added & listed).sum()),
            'net_listed_retrieved': int((added & listed).sum()) - int((dropped & listed).sum()),
            'baseline_top5_hands_evicted': int((b5 & dropped).sum()),
            'correct_final_picks_lost_to_eviction': lost,
            'correct_final_picks_gained_from_new': gained,
            'net_correct_final_picks': gained - lost,
            'baseline_correct_final_picks': int((b5 & listed).sum()),
            'arm_correct_final_picks': int((a5 & listed).sum()),
            'newly_listed_not_converted': int((added & listed).sum()) - gained,
            'reading': 'lost = hands the incumbent stage 2 had already placed in its final five AND '
                       'that were truly listed, which the arm removed from the top-20 so stage 2 '
                       'can never see them again'}

    # ---- reuse the stored single-seed baselines, with an identity guard -----------------------
    mat = pl.read_parquet(OUTDIR / 'stage1-oof-matrix.parquet')
    assert mat.select('pair_id', 'hand_id').equals(d.select('pair_id', 'hand_id')), 'matrix keys'
    dv1 = resplit(d, SALTS[0])
    fresh = stage1_oof(dv1, prod_design, 7329)
    stored = mat[f'{SALTS[0]}|prod|7329'].to_numpy()
    guard = float(np.max(np.abs(fresh - stored)))
    assert guard == 0.0, f'stored stage-1 matrix does not reproduce ({guard}) -- do not reuse it'
    log('guard stored-vs-refit max_abs', guard)

    res = {'author': AUTHOR, 'step': '85b', 'training_only': True, 'held_folds_touched': False,
           'gate': GATE, 'provenance': PROVENANCE, 'arms': ARMS, 'salts': list(SALTS),
           'reuse_guard_max_abs': guard, 'by_salt': {}}

    for salt in SALTS:
        spl = resplit(d, salt)['split'].to_numpy()
        base = mat[f'{salt}|prod|7329'].to_numpy()
        mb, ppb = s1_metrics(base)
        hb, pphb, zb = hybrid(base, spl)
        row = {'single_seed_7329': {'stage1_map5': mb['map5'],
                                    'recall_at_20': mb['recall_at_20']['pooled'],
                                    'hybrid_map5': round(hb, 4)},
               'arms': {}}
        log(f'{salt} baseline', json.dumps({'s1': mb['map5'], 'r20': mb['recall_at_20']['pooled'],
                                            'hybrid': round(hb, 4)}))
        for name, extra in ARMS.items():
            t1 = time.time()
            v = stage1_oof(resplit(d, salt), prod_design, 7329, extra=extra)
            m, pp = s1_metrics(v)
            b1 = boot(ppb, pp, ptab, pools)
            h, pph, za = hybrid(v, spl)
            b2 = boot(pphb, pph, ptab, pools)
            ch = churn(base, zb, v, za)
            row['arms'][name] = {
                'stage1_map5': m['map5'], 'stage1_delta': round(b1['mean_delta'], 4),
                'stage1_ci95': [round(x, 4) for x in b1['ci95']],
                'recall_at_20': m['recall_at_20']['pooled'],
                'recall_at_20_delta': round(m['recall_at_20']['pooled']
                                            - mb['recall_at_20']['pooled'], 4),
                'hybrid_map5': round(h, 4), 'downstream_delta': round(b2['mean_delta'], 4),
                'downstream_ci95': [round(x, 4) for x in b2['ci95']],
                'by_family': {f: round(float(np.mean([pph[p] for p in pph if labels[p] == f])), 4)
                              for f in FAMILIES},
                'churn': ch,
                'seconds': round(time.time() - t1)}
            log(f'{salt} {name}', json.dumps(
                {'s1': m['map5'], 's1_d': row['arms'][name]['stage1_delta'],
                 'r20_d': row['arms'][name]['recall_at_20_delta'],
                 'hybrid': round(h, 4), 'down_d': row['arms'][name]['downstream_delta'],
                 'churn': ch['top20_churn_fraction'],
                 'lost': ch['correct_final_picks_lost_to_eviction'],
                 'gained': ch['correct_final_picks_gained_from_new'],
                 'sec': row['arms'][name]['seconds']}))
        res['by_salt'][salt] = row

    # ---- minimums across partitions -----------------------------------------------------------
    res['resalt'] = {}
    for name in ARMS:
        s1d = [res['by_salt'][s]['arms'][name]['stage1_delta'] for s in SALTS]
        dwn = [res['by_salt'][s]['arms'][name]['downstream_delta'] for s in SALTS]
        r20 = [res['by_salt'][s]['arms'][name]['recall_at_20_delta'] for s in SALTS]
        mind = min(dwn)
        res['resalt'][name] = {
            'stage1_deltas': s1d, 'stage1_min': min(s1d),
            'recall20_deltas': r20, 'recall20_min': min(r20),
            'downstream_deltas': dwn, 'downstream_min': mind,
            'real': bool(mind > 0 and min(s1d) > 0 and min(r20) > 0),
            'material': bool(LEVER * mind * TRANSFER >= RESOLUTION),
            'composite_undiscounted': round(LEVER * mind, 5),
            'composite_discounted': round(LEVER * mind * TRANSFER, 5),
            'clears_resolution': bool(LEVER * mind * TRANSFER >= RESOLUTION)}

    # ---- section C rollup: does a recall@20 rise transmit? -------------------------------------
    cells = [(s, n) for s in SALTS for n in ARMS]
    s1_up = sum(res['by_salt'][s]['arms'][n]['stage1_delta'] > 0 for s, n in cells)
    r20_up = sum(res['by_salt'][s]['arms'][n]['recall_at_20_delta'] > 0 for s, n in cells)
    down_up = sum(res['by_salt'][s]['arms'][n]['downstream_delta'] > 0 for s, n in cells)
    res['recall_sufficiency'] = {
        'question': 'OBS 18.14 weaker form says recall@20 is the channel that transmits a stage-1 '
                    'gain to stage 2. Is a recall@20 rise SUFFICIENT?',
        'cells': len(cells),
        'stage1_map5_up': s1_up, 'recall_at_20_up': r20_up, 'downstream_up': down_up,
        'mean_top20_churn_fraction': round(float(np.mean(
            [res['by_salt'][s]['arms'][n]['churn']['top20_churn_fraction'] for s, n in cells])), 4),
        'total_correct_final_picks_lost': int(sum(
            res['by_salt'][s]['arms'][n]['churn']['correct_final_picks_lost_to_eviction']
            for s, n in cells)),
        'total_correct_final_picks_gained': int(sum(
            res['by_salt'][s]['arms'][n]['churn']['correct_final_picks_gained_from_new']
            for s, n in cells)),
        'total_listed_added': int(sum(
            res['by_salt'][s]['arms'][n]['churn']['listed_added'] for s, n in cells)),
        'total_listed_dropped': int(sum(
            res['by_salt'][s]['arms'][n]['churn']['listed_dropped'] for s, n in cells)),
        'verdict': None}
    rs = res['recall_sufficiency']
    rs['verdict'] = ('recall@20 rise is NOT sufficient' if r20_up == len(cells) and down_up < len(cells)
                     else 'inconclusive on this surface')

    best = max(ARMS, key=lambda n: res['resalt'][n]['downstream_min'])
    alive = bool(res['resalt'][best]['real'] and res['resalt'][best]['material'])
    res['verdict'] = {
        'best_arm_by_downstream_min': best,
        'deciding_number': res['resalt'][best]['downstream_min'],
        'deciding_number_is': 'minimum across the three pool partitions of the end-to-end audit '
                              'MAP@5 delta, capacity arm vs production single-seed stage 1, '
                              'incumbent stage 2 unchanged',
        'real': res['resalt'][best]['real'], 'material': res['resalt'][best]['material'],
        'ALIVE': alive}
    res['limits'] = [
        'n_estimators was varied on the audit surface; the arms were not chosen against any held '
        'fold or leaderboard score, but they were chosen after seeing v1, so v1 is not independent '
        'evidence -- resplit-b and -c are.',
        'Stage 2 is held at the incumbent recipe; a stage 2 re-tuned or re-fit for a deeper stage 1 '
        'is not tested, and the churn result is a statement about the INCUMBENT stage 2 only.',
        'The churn ledger counts hands, not AP; MAP@5 weights a hit by its rank and by '
        'min(n_relevant,5), so lost/gained counts explain the direction, not the exact magnitude.',
        'Audit->held transfer has run 46-52 % historically; the discounted composite applies .49.',
        '162 audit pairs. Nothing here is proposed for submission.']
    res['seconds'] = round(time.time() - t0)

    print()
    print('  STAGE-1 AND END-TO-END BY PARTITION (production single seed 7329 = baseline)')
    print(f"    {'partition':18s} {'arm':10s} {'s1':>8s} {'s1 d':>8s} {'r20 d':>8s} "
          f"{'hybrid':>8s} {'down d':>8s} {'churn':>7s} {'lost':>5s} {'gain':>5s}")
    for salt in SALTS:
        r = res['by_salt'][salt]
        print(f"    {salt:18s} {'single':10s} {r['single_seed_7329']['stage1_map5']:8.4f} "
              f"{0:8.4f} {0:8.4f} {r['single_seed_7329']['hybrid_map5']:8.4f} {0:8.4f}")
        for name in ARMS:
            a = r['arms'][name]
            c = a['churn']
            print(f"    {'':18s} {name:10s} {a['stage1_map5']:8.4f} {a['stage1_delta']:+8.4f} "
                  f"{a['recall_at_20_delta']:+8.4f} {a['hybrid_map5']:8.4f} "
                  f"{a['downstream_delta']:+8.4f} {c['top20_churn_fraction']:7.3f} "
                  f"{c['correct_final_picks_lost_to_eviction']:5d} "
                  f"{c['correct_final_picks_gained_from_new']:5d}")
    print()
    print('  MINIMUM ACROSS THE THREE PARTITIONS')
    for name in ARMS:
        z = res['resalt'][name]
        print(f"    {name:10s} stage1 {z['stage1_min']:+.4f}  recall@20 {z['recall20_min']:+.4f}  "
              f"downstream {z['downstream_min']:+.4f}  composite(disc) {z['composite_discounted']:+.5f}"
              f"  {'real' if z['real'] else 'not real'}")
    print()
    print('  IS A RECALL@20 RISE SUFFICIENT?')
    print(f"    of {rs['cells']} cells: stage-1 MAP@5 up in {rs['stage1_map5_up']}, "
          f"recall@20 up in {rs['recall_at_20_up']}, end-to-end up in {rs['downstream_up']}")
    print(f"    mean top-20 churn {rs['mean_top20_churn_fraction']:.3f} of retrieved slots; "
          f"listed added {rs['total_listed_added']} vs dropped {rs['total_listed_dropped']}")
    print(f"    correct final picks: gained {rs['total_correct_final_picks_gained']}, "
          f"lost to eviction {rs['total_correct_final_picks_lost']}")
    print(f"    -> {rs['verdict']}")
    v = res['verdict']
    print(f"\n  VERDICT: {'ALIVE' if alive else 'DEAD'}  -- best arm {best}, deciding number "
          f"{v['deciding_number']:+.4f} (min-across-partition end-to-end delta)")

    write_json(OUTDIR / 'audit-C-capacity-followup.json', res)
    check_protected()
    print(f"\n  wrote {OUTDIR / 'audit-C-capacity-followup.json'}  ({res['seconds']} s)")
    print('DONE')


if __name__ == '__main__':
    main()
