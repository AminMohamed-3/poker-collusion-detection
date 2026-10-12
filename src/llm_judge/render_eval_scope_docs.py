"""STEP267: render the EVALUATION-phase documents of the STEP259b production fusion scope.

THIS IS THE PRODUCTION PATH AND IT IS NOT RUN UNTIL GATE 2 PASSES (registration267.json,
"production_feasibility": "the eval-scope renderer is written but NOT run until gate 2 passes").
Only --dry-run has been executed: it prints the scope row count and renders 20 documents.

THE SCOPE.  outputs_step259b_20260918/scope_keys.parquet, written by
src/evidence_aug/step259b_scope_keys.py: the 321,200 (pair_id, hand_id) rows of the STEP259
builder's fusion scope (DT 184,600 / SP 90,580 / other 46,020; 16,060 pairs, 239,845 distinct
hands), with each row's predicted_behavior and the column it is fused on.  The production third
factor would have to cover the DT + SP part (275,180 rows); this script renders the whole scope so
that the 'all families' variant is also possible.

THE EVALUATION-PHASE CONTEXT.  render_hand.HandContext asserts
``(hands['phase'] == 'development').all()`` and reads three DEVELOPMENT feature tables
(step237_features.parquet, step254_features.parquet, step254_chrono.parquet).  None of them exists
for the evaluation phase, so EvalHandContextV2 below builds the same numbers for evaluation hands
with the project's own phase-parameterised builders, with NO cross-phase join anywhere:

  hands / seats / actions   data/*.parquet, restricted to the scope's hand_ids, phase asserted
                            == 'evaluation' on every hand
  pair membership           data/evaluation_pairs.csv ONLY (never development_labels.csv)
  the 20 STEP237 columns    evidence_aug.step246_invisible_features._task, i.e. STEP237's own
                            hand_tables / pairhand_features, called with phase='evaluation'
  the 5 STEP254 columns     evidence_aug.step257_mech_eval.build_chunk(..., 'evaluation', ...),
                            i.e. step254_features.process_hand unchanged
                            (t2_eq_pre_A / t2_eq_flop_A / t2_eq_turn_A / t2_eq_river_A /
                            t5_transfer_bb - all five are in-hand columns, so the chronology
                            family is NOT needed by the renderer)
  p1_is_A                   step257_prep_eval's rule, recomputed here over the pair's COMPLETE
                            evaluation hand list (outputs_v2/hands_evaluation/batch_*.parquet):
                            p1_is_A = (sum of player_1's net_chips <= sum of player_2's)

  step254_pf169.npz is a property of the deck, not of a phase, and is reused unchanged - exactly
  what step257_mech_eval does.  experiments/step254_.../step254_features.parquet is NEVER read
  here: only F.COLS (the column NAMES) come from the frozen layout module.

OUTPUT (full run only)
  experiments/step267_text_rerankers_20260918/eval_scope_docs.jsonl   {"uid","doc"}
  experiments/step267_text_rerankers_20260918/keys_eval_scope.parquet  LOCAL ONLY
  uid = sha1('{pair_id}|{hand_id}|step264')[:16], the same function the development package uses,
  so a uid never collides across the two phases (the ids differ).

  a_is_player_1 is TRUE for every evaluation row: production scoring has no reason to randomise
  the A/B print order (that randomisation existed to stop a generative judge latching onto the
  named player), and a fixed order makes the render reproducible from the key alone.

Usage:  python render_eval_scope_docs.py --dry-run [--n 20]
        python render_eval_scope_docs.py --workers 8 --chunk 2000     (NOT run yet - see above)
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import argparse
import glob
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import render_hand as R                                                        # noqa: E402
import render_hand_v2 as V                                                     # noqa: E402
from build_reranker_data import uid_of                                         # noqa: E402

CV = R.CV
DATA = R.DATA
ROOT = R.ROOT
OUT = CV + r'\experiments\step267_text_rerankers_20260918'
SCOPE = CV + r'\outputs_step259b_20260918\scope_keys.parquet'
SCOPE_RECEIPT = CV + r'\outputs_step259b_20260918\scope_keys_receipt.json'
EVAL_PAIRS = DATA + r'\evaluation_pairs.csv'
EVAL_BATCHES = ROOT + r'\outputs_v2\hands_evaluation'
PF169 = CV + r'\experiments\step254_mechanical_tensor_20260917\step254_pf169.npz'
PHASE = 'evaluation'
N_SCOPE = 321200
F254_COLS = ['t2_eq_pre_A', 't2_eq_flop_A', 't2_eq_turn_A', 't2_eq_river_A', 't5_transfer_bb']
K = ['pair_id', 'hand_id']


def _src():
    """The phase-parameterised builders, imported lazily (they pull in lightgbm-free modules)."""
    sys.path.insert(0, CV + r'\src')
    sys.path.insert(0, CV + r'\src\evidence_aug')
    from evidence_aug import step246_invisible_features as A246
    import step257_mech_eval as M257
    return A246, M257


# ---------------------------------------------------------------------------- scope and pairs


def load_scope() -> pl.DataFrame:
    s = pl.read_parquet(SCOPE)
    assert s.height == N_SCOPE, ('scope_keys.parquet has %d rows, expected %d'
                                 % (s.height, N_SCOPE))
    assert s.select(K).n_unique() == s.height, 'duplicate (pair, hand) in the scope'
    if os.path.exists(SCOPE_RECEIPT):
        rec = json.load(open(SCOPE_RECEIPT, encoding='utf-8'))
        assert rec['rows'] == s.height, 'scope_keys.parquet disagrees with its receipt'
    return s.sort(K)


def eval_members(pairs: list[str] | None = None) -> dict[str, tuple[str, str]]:
    lab = pl.read_csv(EVAL_PAIRS).select('pair_id', 'player_1', 'player_2')
    if pairs is not None:
        lab = lab.join(pl.DataFrame({'pair_id': sorted(set(pairs))}), on='pair_id')
    return {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}


def pair_all_hands(pairs: list[str]) -> pl.DataFrame:
    """Every evaluation pair-hand of these pairs (the p1_is_A rule needs the COMPLETE list)."""
    want = pl.DataFrame({'pair_id': sorted(set(pairs))})
    parts = []
    for f in sorted(glob.glob(EVAL_BATCHES + r'\batch_*.parquet')):
        parts.append(pl.read_parquet(f, columns=K).join(want, on='pair_id'))
    out = pl.concat(parts)
    assert out.height, 'no evaluation pair-hands found for these pairs'
    return out.unique(subset=K)


def p1_is_A_map(pairs: list[str], members: dict) -> dict[str, int]:
    """step257_prep_eval's role rule: p1_is_A = (sum net_chips of player_1 <= that of player_2),
    summed over the pair's COMPLETE evaluation hand list."""
    allh = pair_all_hands(pairs)
    ids = allh.select('hand_id').unique()
    ph = (pl.scan_parquet(DATA + r'\hands.parquet').join(ids.lazy(), on='hand_id')
          .select('hand_id', 'phase').collect())
    assert (ph['phase'] == PHASE).all(), 'a pair hand is not %s-phase' % PHASE
    seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(ids.lazy(), on='hand_id')
             .select('hand_id', 'player_id', 'net_chips').collect())
    net = {(r[0], r[1]): float(r[2]) for r in seats.iter_rows()}
    tot: dict[str, list[float]] = {p: [0.0, 0.0] for p in set(allh['pair_id'].to_list())}
    for pid, hid in allh.iter_rows():
        p1, p2 = members[pid]
        tot[pid][0] += net.get((hid, p1), 0.0)
        tot[pid][1] += net.get((hid, p2), 0.0)
    return {p: int(v[0] <= v[1]) for p, v in tot.items()}


# ---------------------------------------------------------------------------- the two features


def build_f237(keys: pl.DataFrame, members: dict) -> dict:
    """The 20 STEP237 columns the v2 renderer uses, via STEP237's own functions."""
    A246, _ = _src()
    S = A246._mod()
    part = keys.select('hand_id').unique()
    meta = (pl.scan_parquet(DATA + r'\hands.parquet').join(part.lazy(), on='hand_id')
            .select('hand_id', 'phase', 'big_blind', 'board_cards', 'final_pot',
                    'players_dealt').collect())
    assert meta.height == part.height, 'hands missing'
    assert (meta['phase'] == PHASE).all(), 'a hand of this chunk is not %s-phase' % PHASE
    seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(part.lazy(), on='hand_id')
             .select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2', 'total_contribution',
                     'net_chips', 'folded', 'went_to_showdown', 'won_share').collect())
    acts = (pl.scan_parquet(DATA + r'\actions.parquet').join(part.lazy(), on='hand_id')
            .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                    'amount_to', 'pot_before', 'to_call').collect().sort('hand_id', 'action_no'))
    sm, am, jm = {}, {}, {}
    for r in seats.iter_rows():
        sm.setdefault(r[0], []).append(r)
    for r in acts.iter_rows(named=True):
        am.setdefault(r['hand_id'], []).append(r)
    for pid, hid in keys.select(K).iter_rows():
        jm.setdefault(hid, []).append((pid, tuple(members[pid])))
    nm = list(S.FEATURES)
    out = {}
    for mr in meta.iter_rows(named=True):
        h = mr['hand_id']
        for pid, hid, vals, _noact in A246._task((mr, sm[h], am.get(h, []), jm[h])):
            out[(pid, hid)] = dict(zip(nm, vals))
    assert len(out) == keys.height, 'a STEP237 row went missing'
    return out


def build_f254(keys: pl.DataFrame, members: dict, p1A: dict) -> dict:
    """The 5 STEP254 in-hand columns the v2 renderer uses, via step254_features.process_hand."""
    _, M257 = _src()
    PCT169, EQ169 = M257.load_pf169()
    kk = keys.select(K).with_columns(
        pl.Series('player_1', [members[p][0] for p in keys['pair_id'].to_list()]),
        pl.Series('player_2', [members[p][1] for p in keys['pair_id'].to_list()]),
        pl.Series('p1_is_A', [p1A[p] for p in keys['pair_id'].to_list()], dtype=pl.Int8))
    df = M257.build_chunk(kk, PHASE, PCT169, EQ169).select(*K, *F254_COLS)
    assert df.height == keys.height, 'a STEP254 row went missing'
    return {(r['pair_id'], r['hand_id']): r for r in df.iter_rows(named=True)}


# ---------------------------------------------------------------------------- the context


class EvalHandContextV2(V.HandContextV2):
    """HandContextV2 for EVALUATION-phase hands.  render_hand.HandContext.__init__ is bypassed
    because it asserts the development phase and reads the three development feature tables; every
    attribute it would set is set here from evaluation-phase sources only."""

    def __init__(self, keys: pl.DataFrame, members: dict, p1A: dict):
        k = keys.select(K).unique()
        ids = k.select('hand_id').unique()
        self.members = {p: members[p] for p in k['pair_id'].unique().to_list()}
        hands = (pl.scan_parquet(DATA + r'\hands.parquet').join(ids.lazy(), on='hand_id')
                 .select('hand_id', 'phase', 'button_seat', 'small_blind', 'big_blind',
                         'board_cards', 'final_pot', 'players_dealt',
                         'players_at_showdown').collect())
        assert hands.height == ids.height, 'missing hands'
        assert (hands['phase'] == PHASE).all(), 'a non-%s hand reached the render set' % PHASE
        assert (hands['players_dealt'] == 6).all(), 'not a 6-max hand'
        self.h = {r['hand_id']: r for r in hands.drop('phase').iter_rows(named=True)}
        seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(ids.lazy(), on='hand_id')
                 .select('hand_id', 'player_id', 'seat_no', 'starting_stack', 'hole_card_1',
                         'hole_card_2', 'total_contribution', 'net_chips', 'folded',
                         'went_to_showdown', 'won_share').collect().sort('hand_id', 'seat_no'))
        self.s = {}
        for r in seats.iter_rows(named=True):
            self.s.setdefault(r['hand_id'], []).append(r)
        acts = (pl.scan_parquet(DATA + r'\actions.parquet').join(ids.lazy(), on='hand_id')
                .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                        'amount_to', 'pot_before', 'to_call', 'players_active')
                .collect().sort('hand_id', 'action_no'))
        self.a = {}
        for r in acts.iter_rows(named=True):
            self.a.setdefault(r['hand_id'], []).append(r)
        self.f237 = build_f237(k, members)
        self.f254 = build_f254(k, members, p1A)
        self.p1_is_A = {key: int(p1A[key[0]]) for key in self.f254}
        self._strength, self._state, self._eqfield = {}, {}, {}


# ---------------------------------------------------------------------------- rendering


def _render_chunk(task):
    idx, rows, members, p1A = task
    os.environ['POLARS_MAX_THREADS'] = '1'
    kf = pl.DataFrame({'pair_id': [r[0] for r in rows], 'hand_id': [r[1] for r in rows]})
    ctx = EvalHandContextV2(kf, members, p1A)
    return idx, [(uid_of(p, h), V.body(ctx, p, h, True)) for p, h in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--n', type=int, default=20, help='documents to render in --dry-run')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--out', default=OUT + r'\eval_scope_docs.jsonl')
    a = ap.parse_args()

    t0 = time.time()
    scope = load_scope()
    fam = {f: int(n) for f, n in
           scope.group_by('predicted_behavior').len().sort('predicted_behavior').iter_rows()}
    fused = int(scope['fused_column'].is_not_null().sum())
    print('SCOPE %s: %d rows, %d pairs, %d distinct hands' %
          (SCOPE, scope.height, scope['pair_id'].n_unique(), scope['hand_id'].n_unique()))
    print('  predicted_behavior: %s' % json.dumps(fam))
    print('  rows on the DT/SP fusion scope (fused_column not null): %d' % fused)

    if a.dry_run:
        sub = scope.head(a.n)
        pairs = sub['pair_id'].unique().to_list()
        members = eval_members(pairs)
        assert len(members) == len(pairs), 'a scope pair is not in evaluation_pairs.csv'
        p1A = p1_is_A_map(pairs, members)
        print('  dry run: %d rows, %d pairs, p1_is_A computed over their complete evaluation '
              'hand lists' % (sub.height, len(pairs)))
        rows = list(zip(sub['pair_id'].to_list(), sub['hand_id'].to_list()))
        _i, docs = _render_chunk((0, rows, members, p1A))
        ln = np.array([len(d) for _u, d in docs], dtype=float)
        print('  rendered %d documents: chars min %d median %d max %d, est_tokens median %.0f'
              % (len(docs), ln.min(), np.median(ln), ln.max(), np.median(ln) / 3.6))
        print('  uid of the first row: %s' % docs[0][0])
        print('--- FIRST DOCUMENT ---')
        print(docs[0][1])
        print('--- END ---')
        print('DRY RUN COMPLETE in %.0fs; nothing written.' % (time.time() - t0))
        print('The full render is NOT run: registration267.json holds it until gate 2 passes.')
        return

    # ------------------------------------------------------------------ the full render
    pairs = scope['pair_id'].unique().to_list()
    members = eval_members(pairs)
    assert len(members) == len(pairs), 'a scope pair is not in evaluation_pairs.csv'
    print('computing p1_is_A over %d pairs ...' % len(pairs), flush=True)
    p1A = p1_is_A_map(pairs, members)
    rows = list(zip(scope['pair_id'].to_list(), scope['hand_id'].to_list()))
    tasks = [(i, rows[i:i + a.chunk], members, p1A) for i in range(0, len(rows), a.chunk)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = 0
    with open(a.out, 'w', encoding='utf-8', newline='\n') as fh, Pool(a.workers) as pool:
        for _idx, part in pool.imap_unordered(_render_chunk, tasks):
            for uid, doc in part:
                fh.write(json.dumps({'uid': uid, 'doc': doc}, ensure_ascii=True) + '\n')
            done += len(part)
            print('  %d/%d docs  %.0fs' % (done, len(rows), time.time() - t0), flush=True)
    scope.with_columns(
        pl.Series('uid', [uid_of(p, h) for p, h in rows]),
        pl.Series('a_is_player_1', [True] * len(rows)),
        pl.Series('p1_is_A', [p1A[p] for p, _h in rows])
    ).write_parquet(OUT + r'\keys_eval_scope.parquet')
    print('wrote %s and keys_eval_scope.parquet in %.0fs' % (a.out, time.time() - t0))


if __name__ == '__main__':
    main()
