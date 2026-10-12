"""Step 68b: why did the struct stage 2 score zero on the public split?

Step 68 killed the first suspect.  Re-weighting the held pairs to the evaluation's candidate-set
size distribution leaves the struct gain at +.0400 (pooled +.0411), and on evaluation-like pairs
(38-100 hands) it is +.0564 -- LARGER, not smaller.  Length is not the explanation, so E4 is
demoted.

The second suspect is ROUTING.  Every stage-2 lever is per family: the held read routed each pair
by the fold behaviour model's predicted family, production routes by the `predicted_behavior`
column of the submission, which comes from the v12 conditional behaviour model.  Two different
routers.  And the struct gain is not spread evenly -- it is +.115 on coordinated isolation
against +.021 / +.017 on the other two, so it survives only if isolation pairs are actually
routed to the isolation model.  The r382 hybrid, by contrast, only ever touches predicted
transfer / soft play, which is exactly the part that DID transfer (88 %).

Three measurements, all label-free on the evaluation side:
  A. the predicted-family mix of every population that matters, weighted by risk where relevant;
  B. the held confusion matrix (true family x predicted family) and the MAP@5 of each cell under
     chrono1 / struct / the r382 hybrid -- what a stage-2 model does when it is handed the wrong
     family (this is also plan item B1);
  C. how much the production stage 2 actually MOVES the evaluation lists, per predicted family,
     against how much it moves the held lists.  Equal churn with no gain is a different disease
     from no churn.

Read-only apart from `experiments/public_ledger_20260912/`.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob
import numpy as np
import polars as pl

from evidence_aug.paths import ORIGINAL, CV, check_protected, write_json

OUT = CV / 'experiments' / 'public_ledger_20260912'
FS2 = CV / 'experiments' / 'family_stage2_20260911'
R382 = CV / 'experiments' / 'family_stage2_r382_20260911'
BASE = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
SEL = ORIGINAL / 'reports' / 'replay-v13' / 'selected-pairs.parquet'
S_FILE = CV / 'outputs_blend_20260911' / 'submission_blend_w0.5_plus_R1.csv'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}
EV = [f'evidence_hand_{i}' for i in range(1, 6)]


def mix(name, fam, w=None):
    w = np.ones(len(fam)) if w is None else np.asarray(w, float)
    tot = w.sum()
    return {'population': name, 'n': int(len(fam)),
            **{SHORT[f]: float(w[fam == f].sum() / tot) for f in FAMS}}


def part_a():
    print('=' * 100)
    print('A.  PREDICTED-FAMILY MIX -- what each stage-2 model is actually handed')
    print('=' * 100)
    held = pl.read_csv(FS2 / 'pairs.csv').filter(pl.col('candidate') == 'chrono1')
    sub = pl.read_csv(S_FILE, columns=['pair_id', 'risk_score', 'predicted_behavior'],
                      schema_overrides={'risk_score': pl.Float64})
    sel = pl.read_parquet(SEL).select('pair_id').unique()
    s = sub.join(sel, on='pair_id', how='inner')
    r = s['risk_score'].to_numpy()
    pf = s['predicted_behavior'].to_numpy()
    rows = [mix('development held pairs, TRUE family', held['family'].to_numpy()),
            mix('development held pairs, PREDICTED (fold router)', held['predicted'].to_numpy()),
            mix('evaluation, all 112,540 pairs', sub['predicted_behavior'].to_numpy()),
            mix('evaluation, 16,060 selected pairs', pf),
            mix('evaluation, selected, risk-weighted', pf, r),
            mix('evaluation, block risk >= .5', pf[r >= .5]),
            mix('evaluation, risk >= .9', pf[r >= .9])]
    print(f'{"population":<48}{"n":>8}' + ''.join(f'{f:>9}' for f in ('DT', 'SP', 'CI')))
    for x in rows:
        print(f'{x["population"]:<48}{x["n"]:>8}'
              + ''.join(f'{100 * x[f]:>8.1f}%' for f in ('DT', 'SP', 'CI')))
    print()
    print('  The evidence component is an average over the TRUE positives.  The risk-weighted and')
    print('  block rows are the best label-free proxies for that population.')
    return rows


def part_b():
    print()
    print('=' * 100)
    print('B.  HELD CONFUSION MATRIX AND WHAT A MIS-ROUTED STAGE 2 DOES (plan item B1)')
    print('=' * 100)
    a = pl.read_csv(FS2 / 'pairs.csv')
    b = pl.read_csv(R382 / 'pairs.csv').filter(
        ~pl.col('candidate').is_in(['chrono1', 'allfam_top20_struct']))
    w = pl.concat([a, b]).pivot(values='map5',
                                index=['fold', 'pair_id', 'family', 'predicted'], on='candidate')
    tru = w['family'].to_numpy()
    prd = w['predicted'].to_numpy()
    c1 = w['chrono1'].to_numpy()
    st = w['allfam_top20_struct'].to_numpy()
    r3 = w['allfam_top20_r382'].to_numpy()
    hyb = np.where(np.isin(prd, ['directed_transfer', 'soft_play']), r3, st)
    print(f'{"true \\ predicted":<26}' + ''.join(f'{SHORT[f]:>8}' for f in FAMS)
          + f'{"other":>8}{"acc":>8}')
    conf = {}
    for f in FAMS:
        m = tru == f
        cells = [int((m & (prd == g)).sum()) for g in FAMS]
        other = int(m.sum()) - sum(cells)
        conf[f] = dict(zip([SHORT[g] for g in FAMS], cells), other=other, n=int(m.sum()))
        print(f'{f:<26}' + ''.join(f'{c:>8}' for c in cells) + f'{other:>8}'
              + f'{cells[FAMS.index(f)] / max(int(m.sum()), 1):>8.3f}')
    ok = prd == tru
    print(f'\n  overall routing accuracy on held pairs: {ok.mean():.3f} '
          f'({int(ok.sum())}/{len(ok)})')
    print()
    print(f'{"cell":<34}{"n":>5}{"chrono1":>10}{"struct":>10}{"delta":>9}'
          f'{"hybrid":>10}{"delta":>9}')
    cells = []
    for f in FAMS:
        for g in FAMS:
            m = (tru == f) & (prd == g)
            if m.sum() < 3:
                continue
            row = {'true': f, 'predicted': g, 'n': int(m.sum()), 'chrono1': float(c1[m].mean()),
                   'struct': float(st[m].mean()), 'd_struct': float((st - c1)[m].mean()),
                   'hybrid': float(hyb[m].mean()), 'd_hybrid': float((hyb - st)[m].mean())}
            cells.append(row)
            tag = f'{SHORT[f]} routed to {SHORT[g]}' + ('  (correct)' if f == g else '')
            print(f'{tag:<34}{row["n"]:>5}{row["chrono1"]:>10.4f}{row["struct"]:>10.4f}'
                  f'{row["d_struct"]:>+9.4f}{row["hybrid"]:>10.4f}{row["d_hybrid"]:>+9.4f}')
    mis = ~ok
    print(f'\n  mis-routed pairs (n={int(mis.sum())}): struct {(st - c1)[mis].mean():+.4f}, '
          f'hybrid over struct {(hyb - st)[mis].mean():+.4f}')
    print(f'  correctly routed  (n={int(ok.sum())}): struct {(st - c1)[ok].mean():+.4f}, '
          f'hybrid over struct {(hyb - st)[ok].mean():+.4f}')

    # what the gain would be if the evaluation's predicted mix replaced the held one
    print()
    print('  Re-weighting the held per-pair struct gain to an evaluation predicted-family mix:')
    sub = pl.read_csv(S_FILE, columns=['pair_id', 'risk_score', 'predicted_behavior'],
                      schema_overrides={'risk_score': pl.Float64})
    sel = pl.read_parquet(SEL).select('pair_id').unique()
    s = sub.join(sel, on='pair_id', how='inner').filter(pl.col('risk_score') >= .5)
    target = {f: float((s['predicted_behavior'] == f).mean()) for f in FAMS}
    have = {f: float((prd == f).mean()) for f in FAMS}
    wgt = np.array([target.get(p, 0.0) / max(have.get(p, 1e-9), 1e-9) for p in prd])
    wgt = wgt / wgt.mean()
    rw_struct = float(((st - c1) * wgt).mean())
    rw_hyb = float(((hyb - st) * wgt).mean())
    print(f'    held predicted mix   ' + '  '.join(f'{SHORT[f]} {100 * have[f]:.0f}%' for f in FAMS))
    print(f'    block predicted mix  ' + '  '.join(f'{SHORT[f]} {100 * target[f]:.0f}%' for f in FAMS))
    print(f'    struct - chrono1 reweighted to the block mix: {rw_struct:+.4f} '
          f'(unweighted {(st - c1).mean():+.4f})')
    print(f'    hybrid - struct  reweighted to the block mix: {rw_hyb:+.4f} '
          f'(unweighted {(hyb - st).mean():+.4f})')
    return {'confusion': conf, 'accuracy': float(ok.mean()), 'cells': cells,
            'reweighted_to_block_mix': {'struct': rw_struct, 'hybrid': rw_hyb},
            'held_mix': have, 'block_mix': target}


def lists_from_scores(path):
    d = pl.read_parquet(path)
    return (d.sort(['pair_id', 'score'], descending=[False, True])
            .group_by('pair_id', maintain_order=True).head(5)
            .group_by('pair_id').agg(pl.col('hand_id')))


def churn(a, b, key='hand_id'):
    """set overlap and rank displacement between two top-5 list frames"""
    j = a.join(b, on='pair_id', suffix='_b')
    ov, moved = [], []
    for x, y in zip(j[key].to_list(), j[key + '_b'].to_list()):
        ov.append(len(set(x) & set(y)) / max(len(x), 1))
        moved.append(0.0 if x == y else 1.0)
    return float(np.mean(ov)), float(np.mean(moved)), len(ov)


def part_c():
    print()
    print('=' * 100)
    print('C.  HOW MUCH DOES THE STAGE 2 ACTUALLY MOVE THE LISTS?  (held vs production)')
    print('=' * 100)
    print(f'{"population":<44}{"top-5 overlap":>15}{"lists changed":>15}{"n":>8}')
    rows = []
    for k in (0, 1, 2):
        o, m, n = churn(lists_from_scores(FS2 / 'scores' / f'fold{k}-chrono1.parquet'),
                        lists_from_scores(FS2 / 'scores' / f'fold{k}-allfam_top20_struct.parquet'))
        rows.append({'population': f'held fold {k}: chrono1 -> struct', 'overlap': o,
                     'changed': m, 'n': n})
        print(f'{f"held fold {k}: chrono1 -> struct":<44}{o:>15.3f}{m:>15.3f}{n:>8}')
    # production: incumbent (stage 1) -> chrono3 and -> r382, per predicted family
    inc = pl.concat([pl.read_parquet(f) for f in
                     sorted(glob.glob(str(CV / 'outputs_stage2_20260911' / 'batches'
                                          / 'batch_*-incumbent.parquet')))])
    c3 = pl.concat([pl.read_parquet(f) for f in
                    sorted(glob.glob(str(CV / 'outputs_stage2_20260911' / 'batches'
                                         / 'batch_*-chrono3.parquet')))])
    r3 = pl.concat([pl.read_parquet(f) for f in
                    sorted(glob.glob(str(CV / 'outputs_stage2_r382_20260911' / 'batches'
                                         / 'batch_*-r382.parquet')))])
    sub = pl.read_csv(S_FILE, columns=['pair_id', 'risk_score', 'predicted_behavior'],
                      schema_overrides={'risk_score': pl.Float64})
    for name, frame in [('stage1 -> struct(chrono3)', c3), ('stage1 -> r382 hybrid', r3)]:
        j = inc.join(frame, on='pair_id', suffix='_b').join(sub, on='pair_id')
        for label, f in [('all selected', None)] + [(SHORT[f], f) for f in FAMS] + [('block .5+', 'BLOCK')]:
            if f == 'BLOCK':
                sub_j = j.filter(pl.col('risk_score') >= .5)
            elif f is None:
                sub_j = j
            else:
                sub_j = j.filter(pl.col('predicted_behavior') == f)
            if sub_j.height == 0:
                continue
            A = np.column_stack([sub_j[c].to_numpy() for c in EV])
            B = np.column_stack([sub_j[c + '_b'].to_numpy() for c in EV])
            ov = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(A, B)])
            ch = np.mean([not np.array_equal(a, b) for a, b in zip(A, B)])
            rows.append({'population': f'eval {name}, {label}', 'overlap': float(ov),
                         'changed': float(ch), 'n': sub_j.height})
            print(f'{f"eval {name}, {label}":<44}{ov:>15.3f}{ch:>15.3f}{sub_j.height:>8}')
    print()
    print('  If the evaluation churn matches the held churn but the score does not move, the')
    print('  models are firing normally and the ORDER they produce is simply not better there.')
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_protected()
    a = part_a()
    b = part_b()
    c = part_c()
    write_json(OUT / 'step68b-routing.json', {'mix': a, 'held': b, 'churn': c})
    check_protected()
    print()
    print(f'written {OUT / "step68b-routing.json"}')


if __name__ == '__main__':
    main()
