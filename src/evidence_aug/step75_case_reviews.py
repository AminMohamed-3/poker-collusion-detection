"""Step 75 (plan O2, second half): five descriptive case reviews of evaluation block pairs.

Winner verification asks for a writeup, reproducible code and a handful of worked cases.  This writes
the cases: five pairs from the evaluation block (production risk >= .5) of the shipped chain, two
predicted directed transfer, two predicted soft play, one predicted coordinated isolation, each with
the five hands the submission lists as evidence, their position in the pair's own chronology, the
structural columns the stage-2 model actually sees, and the gameplay those columns summarise.

There are NO evaluation labels.  Nothing here asserts that a pair is coordinating; the reviews are
descriptive readings of what the model selected and why, which is what a verifier needs in order to
follow the chain from raw hands to a submitted evidence list.

Selection rule, fixed before looking at any pair: inside each predicted family, order the block by
risk descending and take the highest-risk pair; for the two two-case families also take the pair at
the median of that family's block.  This gives one confident case and one marginal case per mechanism
family rather than five easy ones.

Reads only; writes research/case_reviews_20260912/*.md.
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from evidence_aug.paths import CV, ORIGINAL, check_protected, sha

CAND = CV / 'outputs_stage2_r382_20260911' / 'submission_S_stage2_r382DTSP.csv'
OUTDIR = CV / 'research' / 'case_reviews_20260912'
EV = ['evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4', 'evidence_hand_5']
PICK = {'directed_transfer': 2, 'soft_play': 2, 'coordinated_isolation': 1}
CORE = ['pot_bb', 'max_street', 'dealt', 'n_to_flop', 'members_to_flop', 'others_to_flop',
        'A_pos', 'B_pos', 'A_chen', 'B_chen', 'A_net_bb', 'B_net_bb', 'A_put_bb', 'B_put_bb',
        'A_folded', 'B_folded', 'A_sd', 'B_sd']
FAMCOLS = {
    'directed_transfer': ['pair_net_bb', 'pair_net_gap', 'pair_folded', 'pair_sd',
                          'post_victim_fold_to_pair_bet', 'post_partner_fold_to_pair_bet',
                          'post_pair_calls', 'post_pair_aggr', 'pair_won_no_sd'],
    'soft_play': ['hu_pair_only', 'pf_pair_vs_pair_raise', 'post_pair_aggr', 'post_pair_calls',
                  'post_cbet', 'pair_sd', 'pair_won', 'pair_won_no_sd'],
    'coordinated_isolation': ['pf_open_pair', 'pf_limpers_before_open', 'pf_iso_raise', 'pf_n_raises',
                              'pf_pair_raises', 'pf_other_raises', 'pf_others_fold', 'pf_partner_call',
                              'opener_won', 'opener_net_bb', 'pair_won_preflop']}
STREETS = ['preflop', 'flop', 'turn', 'river']


def fmt(v, nd=2):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.{nd}f}'.rstrip('0').rstrip('.') if abs(v) < 1e6 else f'{v:.2e}'
    return str(v)


def narrate(acts, a, b, bb):
    """one line per street: what each member did, and what the table did around them"""
    out = []
    for st in STREETS:
        rows = [r for r in acts if (r['street'] or '').lower() == st]
        if not rows:
            continue
        def seq(pid):
            return ' '.join(f"{r['action']}{'' if not r['amount'] else '(' + fmt((r['amount'] or 0) / bb, 1) + 'bb)'}"
                            for r in rows if r['player_id'] == pid) or '-'
        others = [r for r in rows if r['player_id'] not in (a, b)]
        oc = {}
        for r in others:
            oc[r['action']] = oc.get(r['action'], 0) + 1
        out.append(f"    {st:<8} A: {seq(a):<34} B: {seq(b):<34} others: "
                   f"{', '.join(f'{k} x{v}' for k, v in sorted(oc.items())) or '-'}")
    return out


def main():
    check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sub = pl.read_csv(CAND)
    pairs = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv')
    blk = (sub.filter(pl.col('risk_score') >= .5)
           .join(pairs, on='pair_id', validate='1:1').sort('risk_score', descending=True))
    print(f'block: {blk.height} pairs of {sub.height}')

    chosen = []
    for fam, k in PICK.items():
        f = blk.filter(pl.col('predicted_behavior') == fam)
        idx = [0] if k == 1 else [0, f.height // 2]
        for i in idx:
            chosen.append((fam, i, f.height, f.row(i, named=True)))
    print('chosen:', [(c[0], c[1], c[3]['pair_id'], round(c[3]['risk_score'], 4)) for c in chosen])

    hand_ids = sorted({r[3][e] for r in chosen for e in EV if r[3][e]})
    members = {r[3]['pair_id']: (r[3]['player_1'], r[3]['player_2']) for r in chosen}
    allp = sorted({p for v in members.values() for p in v})

    # started_at is tz-aware and this venv has no tzdata, so polars cannot materialise it into
    # Python objects; it is only ever needed for sorting (done in-engine below), never per row.
    hands = (pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet')
             .filter(pl.col('hand_id').is_in(hand_ids))
             .select('hand_id', 'table_id', 'players_dealt', 'big_blind', 'final_pot',
                     'board_cards').collect())
    seats_all = (pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet')
                 .filter(pl.col('player_id').is_in(allp)).collect())
    acts = (pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet')
            .filter(pl.col('hand_id').is_in(hand_ids)).sort('hand_id', 'action_no').collect())
    struct = (pl.scan_parquet(sorted((CV / 'experiments' / 'family_stage2_20260911' /
                                      'eval_structural').glob('batch_*.parquet')))
              .filter(pl.col('hand_id').is_in(hand_ids)).collect())
    # every named column must really exist -- a typo here would otherwise be swallowed by the
    # `if c in sr` guards below and silently drop a whole feature block from the review
    missing = [c for c in CORE + sum(FAMCOLS.values(), []) if c not in struct.columns]
    assert not missing, f'columns named but absent from eval_structural: {missing}'

    index = ['# Case reviews of five evaluation block pairs (Claude Version, 2026-09-12)', '',
             f'Chain: `{CAND.name}` (S pair order + r382 hybrid stage-2 evidence). '
             f'SHA-256 `{sha(CAND)}`.', '',
             'No evaluation labels exist. These are descriptive readings of what the submitted chain '
             'selected, not claims about the pairs. Selection rule: inside each predicted family, the '
             'highest-risk block pair and (for the two-case families) the pair at the median of that '
             "family's block.", '',
             '| file | pair | predicted | risk | block rank in family | shared hands |',
             '|---|---|---|---:|---|---:|']

    for fam, i, n, row in chosen:
        pid = row['pair_id']
        a, b = row['player_1'], row['player_2']
        # The pair's own chronology, restricted to the ELIGIBLE pool.  Every table is 60 % development
        # hands followed by 40 % evaluation hands, and an evaluation pair is scored only on the
        # evaluation ones -- `shared_hands` in evaluation_pairs.csv counts exactly those.  Ranking over
        # the pair's whole seated history would divide by the wrong denominator and report a
        # chrono_frac the model never saw.
        both = (seats_all.filter(pl.col('player_id').is_in([a, b]))
                .group_by('hand_id').agg(pl.len().alias('k')).filter(pl.col('k') == 2))
        allsh = (pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet')
                 .filter(pl.col('hand_id').is_in(both['hand_id'].to_list()))
                 .select('hand_id', 'started_at', 'phase').collect().sort('started_at'))
        sh = allsh.filter(pl.col('phase') == 'evaluation')
        n_dev = allsh.height - sh.height
        order = {h: j for j, h in enumerate(sh['hand_id'].to_list())}
        nh = sh.height
        assert nh == row['shared_hands'], f'{pid}: {nh} eligible vs shared_hands {row["shared_hands"]}'

        L = [f'# {pid} -- predicted {fam}', '',
             f'- members A = `{a}`, B = `{b}`; {nh} eligible (evaluation-phase) shared hands, '
             f'which is the pool the evidence is chosen from; the two also share {n_dev} '
             f'development hands, which are not scorable and not listable',
             f'- submitted risk {row["risk_score"]:.6f}; rank {i + 1} of {n} in the '
             f'{fam} block (block = risk >= .5)',
             f'- evidence hands, in submitted order: ' + ', '.join(f'`{row[e]}`' for e in EV), '',
             '## The five listed hands', '']
        for e in EV:
            hid = row[e]
            if not hid:
                continue
            h = hands.filter(pl.col('hand_id') == hid).row(0, named=True)
            s = struct.filter((pl.col('hand_id') == hid) & (pl.col('pair_id') == pid))
            sr = s.row(0, named=True) if s.height else {}
            pos = order.get(hid)
            frac = (pos / (nh - 1)) if (pos is not None and nh > 1) else None
            L += [f'### {e.replace("evidence_hand_", "hand ")}: `{hid}`',
                  f'- chronology: position {pos if pos is not None else "?"} of {nh} eligible hands '
                  f'(chrono_frac {fmt(frac, 3)}); table `{h["table_id"]}`; '
                  f'{h["players_dealt"]} dealt, big blind {h["big_blind"]}, '
                  f'final pot {h["final_pot"]}, board `{h["board_cards"] or "-"}`',
                  '- core: ' + ', '.join(f'{c} {fmt(sr.get(c))}' for c in CORE if c in sr),
                  f'- {fam} block: ' + ', '.join(f'{c} {fmt(sr.get(c))}' for c in FAMCOLS[fam] if c in sr),
                  '- action:', '```']
            L += narrate(acts.filter(pl.col('hand_id') == hid).to_dicts(), a, b, h['big_blind'] or 1)
            L += ['```', '']
        L += ['## Reading', '', '_(filled in below by hand; two sentences, descriptive only)_', '']
        fn = f'{fam}_{"top" if i == 0 else "median"}_{pid}.md'
        (OUTDIR / fn).write_text('\n'.join(L) + '\n', encoding='utf-8')
        index.append(f'| [{fn}]({fn}) | `{pid}` | {fam} | {row["risk_score"]:.4f} | '
                     f'{i + 1} of {n} | {row["shared_hands"]} |')
        print(f'  wrote {fn}')

    (OUTDIR / 'README.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    check_protected()
    print(f'written {OUTDIR}')
    print('DONE')


if __name__ == '__main__':
    main()
