"""STEP195: packets for an LLM to act as an ANNOTATOR, not to induce a rule.

THE DEFECT THIS ATTACKS. 340 of the 372 labelled pairs have exactly 5 evidence hands; 32 have 3
or 4. So 5 is a CAP, not a quota -- annotators stopped early when the evidence ran out, which
means the 340 five-hand lists are CENSORED. The 6th-best hand of a censored pair is real evidence
sitting in our training data labelled as a negative, indistinguishable from the 180th-best hand.
STEP133 measured +.034 decoded MAP@5 per doubling of labelled lists, and this is the one place
where more real labels can actually be created: not by generating hands (unlabellable to
tournament standard, STEP136) and not by self-labelling (NULL, STEP137), but by recovering the
positives the cap threw away.

THE CANDIDATE SET is the production top-20, which holds 97.2 % of all listed hands and is exactly
the stage-2 training universe -- a label outside it could not be consumed anyway.

TWO PHASES, AND THE FIRST ONE GATES THE SECOND.

  calib   Show the annotator's own top-3 evidence hands for a pair and hide ranks 4 and 5. Ask
          for 2 more. The truth is known, so this MEASURES whether the model's annotation
          judgement matches the humans'. The competitor is production's own s1_rank: if the LLM
          only matches it, its labels carry nothing production does not already know, which is
          exactly how STEP137's pseudo-labels failed. It has to BEAT production to be worth
          training on.

  extend  Only built if calib clears the bar. Show all 5 and ask for ranks 6-10, doubling the
          labelled list length per pair.

s1_rank is deliberately NOT shown and candidates are presented chronologically, so the judgement
is independent of production's opinion rather than anchored to it.

HARD CONSTRAINTS honoured: players.parquet is never opened; player_ids become per-hand aliases
P1/P2/O1..On; development phase only; no evaluation pair is labelled or shown.

Usage: python step195_C_annotator_extension_packets.py PHASE OUTDIR [FAMILY]
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
from evidence_aug import step193_C_rule_induction_packets as S193
from evidence_aug import step167_C_lp_production as S167

DATA = POKER / 'data'
BURNED = S193.BURNED
N_HIDE = 2


def build(phase, outdir, family=None):
    outdir.mkdir(parents=True, exist_ok=True)
    S108, _ = S167.modules()
    CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
    frame, _, _, _, _ = CSP.development_frame()
    top = frame.filter(pl.col('s1_rank') <= TOP).select('pair_id', 'hand_id', 's1_rank', 'relevance')

    lab = pl.read_csv(DATA / 'development_labels.csv').filter(pl.col('label') == 1)
    ev = pl.read_csv(DATA / 'development_evidence.csv')
    if family:
        lab = lab.filter(pl.col('behavior_family') == family)
    keep = set(lab['pair_id'])
    top = top.filter(pl.col('pair_id').is_in(list(keep)))

    rank_of = {}
    for r in ev.iter_rows(named=True):
        rank_of.setdefault(r['pair_id'], {})[r['hand_id']] = r['evidence_rank']

    # calibration needs every one of the five listed hands inside the candidate set, or "recover
    # the hidden two" is unanswerable from what the packet shows
    got = top.group_by('pair_id').agg(pl.col('relevance').sum().alias('n'))
    eligible = set(got.filter(pl.col('n') == 5)['pair_id']) if phase == 'calib' else keep
    pairs = sorted(p for p in keep if p in eligible)
    print(f'{phase}: {len(pairs)} eligible pairs'
          + (f' (family {family})' if family else ''), flush=True)

    fam_of = dict(zip(lab['pair_id'], lab['behavior_family']))
    pl1 = dict(zip(lab['pair_id'], lab['player_1']))
    pl2 = dict(zip(lab['pair_id'], lab['player_2']))
    cand = {p: g.sort('s1_rank') for (p,), g in top.group_by('pair_id')}
    allh = sorted({h for p in pairs for h in cand[p]['hand_id']})
    print(f'{len(allh)} candidate hands to narrate', flush=True)

    H = S193.detz(pl.scan_parquet(DATA / 'hands.parquet')
                  .filter(pl.col('hand_id').is_in(allh)).collect())
    S = S193.detz(pl.scan_parquet(DATA / 'seats.parquet')
                  .filter(pl.col('hand_id').is_in(allh)).collect())
    A = S193.detz(pl.scan_parquet(DATA / 'actions.parquet')
                  .filter(pl.col('hand_id').is_in(allh)).sort('hand_id', 'action_no').collect())
    hmap = {r['hand_id']: r for r in H.iter_rows(named=True)}
    smap = {k[0]: g for k, g in S.sort('seat_no').group_by('hand_id')}
    amap = {k[0]: g for k, g in A.group_by('hand_id')}

    rng = np.random.default_rng(1950)
    key, manifest_pairs = {}, []
    for pid in pairs:
        c = cand[pid]
        hs = sorted(c['hand_id'].to_list(), key=lambda h: (hmap[h]['started_at'], h))
        ranks = rank_of.get(pid, {})
        listed = [h for h in hs if h in ranks]
        if phase == 'calib':
            hide = sorted(listed, key=lambda h: ranks[h])[-N_HIDE:]
            shown = {h: ranks[h] for h in listed if h not in hide}
            # hand_order lets the scorer map the packet's positional hand_index back to hand_id;
            # s1_rank is kept here (never in the packet) so production can be scored as the
            # competitor on exactly the same choice the model was given.
            key[pid] = {'hidden': hide, 'hidden_ranks': [ranks[h] for h in hide],
                        'candidates_offered': [h for h in hs if h not in shown],
                        'hand_order': hs,
                        's1_rank': {h: int(v) for h, v in zip(c['hand_id'], c['s1_rank'])}}
        else:
            hide, shown = [], {h: ranks[h] for h in listed}

        lines = [f'PAIR {pid}   family {fam_of[pid]}',
                 f'{len(hs)} candidate hands, chronological order.',
                 (f'The annotators listed {len(shown)} of these as their TOP {len(shown)} evidence '
                  f'hands (ranks shown). Your job: name {N_HIDE} MORE, ranked.'
                  if phase == 'calib' else
                  f'The annotators listed these {len(shown)} as evidence, in rank order. '
                  f'Your job: name 5 MORE, ranked 6 to 10.'), '']
        recs = []
        for i, hid in enumerate(hs):
            seats_h = smap[hid]
            alias, o = {}, 0
            for s in seats_h.iter_rows(named=True):
                p = s['player_id']
                if p == pl1[pid]:
                    alias[p] = 'P1'
                elif p == pl2[pid]:
                    alias[p] = 'P2'
                else:
                    o += 1
                    alias[p] = f'O{o}'
            acts_h = amap.get(hid, A.head(0))
            lines.append(S193.fmt_hand(hmap[hid], seats_h, acts_h, alias, shown.get(hid), i, len(hs)))
            rec = S193.hand_record(hmap[hid], seats_h, acts_h, alias, shown.get(hid), i)
            rec['chrono_frac'] = round(i / max(1, len(hs) - 1), 6)
            recs.append(rec)
        (outdir / f'pair_{pid}.txt').write_text('\n'.join(lines), encoding='utf-8')
        (outdir / f'pair_{pid}.json').write_text(json.dumps(
            {'pair_id': pid, 'family': fam_of[pid], 'phase': phase, 'n_candidates': len(hs),
             'n_shown_listed': len(shown), 'n_to_name': N_HIDE if phase == 'calib' else 5,
             'hands': recs}), encoding='utf-8')
        manifest_pairs.append(pid)

    manifest = {'phase': phase, 'family': family or 'all', 'n_pairs': len(manifest_pairs),
                'pairs': manifest_pairs, 'n_to_name': N_HIDE if phase == 'calib' else 5,
                'candidate_set': 'production top-20 (97.2% of all listed hands); s1_rank is NOT '
                                 'shown and hands are in chronological order, so the judgement is '
                                 'independent of production rather than anchored to it',
                'burned_pairs_present': sorted(set(manifest_pairs) & BURNED),
                'note': 'players.parquet never opened; aliases are per-hand and carry no identity'}
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    if phase == 'calib':
        (outdir.parent / f'ANSWER_KEY_calib{"_" + family if family else ""}.json').write_text(
            json.dumps({'phase': phase, 'family': family or 'all', 'hidden_by_pair': key}, indent=1),
            encoding='utf-8')
        assert not (outdir / 'ANSWER_KEY_calib.json').exists()
    print(f'wrote {len(manifest_pairs)} packets to {outdir}', flush=True)


def main():
    phase, outdir = sys.argv[1], Path(sys.argv[2])
    family = sys.argv[3] if len(sys.argv) > 3 else None
    assert phase in ('calib', 'extend')
    build(phase, outdir, family)


if __name__ == '__main__':
    main()
