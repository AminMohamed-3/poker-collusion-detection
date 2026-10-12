"""Step 16: training-only qualitative dump. For a few audit pairs (outside all held folds), print the listed
hands and the highest-scoring UNLISTED hands side by side with raw attributes (position, pot, both players'
contributions/net/fold/showdown, hole cards, board, action sequence), so a human can look for a rule the
feature space misses. Also saves the audit out-of-fold frame for reuse.
Output: Claude Version/experiments/label_process_20260911/{audit-oof.parquet, qualitative.txt}
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL
from evidence_aug.frame import training_relevance, attach_relevance, score, family_columns, FAMILIES
from evidence_aug.fold import load_fold, fold_pools
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step14_audit_truncation import split_of, OUT, SPLITS, add_chronology
import numpy as np
import polars as pl

PER_FAMILY = 3
TOP_UNLISTED = 6


def main():
    data, features, _, _ = load_fold(0)
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    audit = data.filter(~pl.col('table_id').is_in(sorted(held)))
    pools = sorted(set(audit['table_id']))
    evidence = training_relevance(set(pools))
    audit = add_chronology(attach_relevance(audit, evidence))
    audit = audit.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
    oof = np.full(audit.height, np.nan)
    for s in range(SPLITS):
        idx = np.where(audit['split'].to_numpy() == s)[0]
        model = fit_ranker(audit.filter(pl.col('split') != s), features)
        oof[idx] = score(model, family_columns(audit[idx], 'behavior_family'), features)
    audit = audit.with_columns(pl.Series('oof', oof)).with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').alias('oof_rank'))
    keep = ['pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance', 'chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank']
    audit.select(keep).write_parquet(OUT / 'audit-oof.parquet')
    ev = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'player_1', 'player_2')
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet', columns=['hand_id', 'button_seat', 'big_blind', 'board_cards', 'final_pot', 'players_dealt', 'players_at_showdown'])
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet')
    actions = pl.read_parquet(ORIGINAL / 'data' / 'actions.parquet')
    lines = []
    rng = np.random.default_rng(5)
    for fam in FAMILIES:
        pairs = sorted(set(audit.filter(pl.col('behavior_family') == fam)['pair_id']))
        for pid in rng.choice(pairs, PER_FAMILY, replace=False):
            p1, p2 = labels.filter(pl.col('pair_id') == pid).select('player_1', 'player_2').row(0)
            frame = audit.filter(pl.col('pair_id') == pid).select(keep).join(ev, on=['pair_id', 'hand_id'], how='left')
            shown = pl.concat([frame.filter(pl.col('relevance') == 1), frame.filter((pl.col('relevance') == 0) & (pl.col('oof_rank') <= TOP_UNLISTED + 5)).sort('oof_rank').head(TOP_UNLISTED)]).sort('chrono_rank')
            lines.append(f"\n===== {fam} pair {pid} ({p1} vs {p2}), {int(frame['n_hands'][0])} shared hands, listed ranks in top-k: {sorted(frame.filter(pl.col('relevance') == 1)['oof_rank'].to_list())}")
            for row in shown.iter_rows(named=True):
                hid = row['hand_id']
                h = hands.filter(pl.col('hand_id') == hid).row(0, named=True)
                st = seats.filter((pl.col('hand_id') == hid) & pl.col('player_id').is_in([p1, p2]))
                bb = h['big_blind']
                who = {p1: 'A', p2: 'B'}
                seat_txt = ' | '.join(f"{who[r['player_id']]} seat{r['seat_no']} {r['hole_card_1']}{r['hole_card_2']} stack{r['starting_stack']/bb:.0f}bb put{r['total_contribution']/bb:.1f} net{r['net_chips']/bb:+.1f} fold{int(r['folded'])} sd{int(r['went_to_showdown'])} win{r['won_share']:.2f}" for r in st.sort('seat_no').iter_rows(named=True))
                others = seats.filter((pl.col('hand_id') == hid) & ~pl.col('player_id').is_in([p1, p2]))
                other_txt = ' '.join(f"{('x' if not r['went_to_showdown'] else r['hole_card_1'] + r['hole_card_2'])}:{r['net_chips']/bb:+.0f}" for r in others.sort('seat_no').iter_rows(named=True))
                acts = actions.filter(pl.col('hand_id') == hid).sort('action_no')
                act_txt = ' '.join(f"{('A' if r['player_id'] == p1 else 'B' if r['player_id'] == p2 else 'o')}{r['street'][0]}:{r['action'][:2]}{('' if r['amount'] in (None, 0) else format(r['amount'] / bb, '.0f'))}" for r in acts.iter_rows(named=True))
                tag = f"LISTED#{row['evidence_rank']}" if row['relevance'] == 1 else 'unlisted'
                lines.append(f"  [{tag:9s}] t={int(row['chrono_rank']):3d}/{int(row['n_hands'])} rank={int(row['oof_rank']):3d} oof={row['oof']:+.2f} pot={h['final_pot']/bb:.1f}bb dealt={h['players_dealt']} sd={h['players_at_showdown']} board={h['board_cards'] or '-'}")
                lines.append(f"      {seat_txt}")
                lines.append(f"      others: {other_txt}")
                lines.append(f"      {act_txt}")
    text = '\n'.join(lines)
    (OUT / 'qualitative.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
