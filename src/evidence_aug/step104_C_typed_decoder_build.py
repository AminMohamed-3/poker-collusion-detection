"""Step 104 (Claude): no-fit (type, time) listing decoder over the frozen step-98 held scores. Label-free build and freeze only.
Registration: experiments/typed_listing_decoder_C_20260913/prediction-typed-listing-decoder.json. Held truth is read only by step104b."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
import hashlib
import itertools
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from phevaluator import evaluate_cards

OUT = CV / 'experiments/typed_listing_decoder_C_20260913'
PREREG = OUT / 'prediction-typed-listing-decoder.json'
PREREG_SHA = 'db0561732bda44af2f4e2f654eea19f188ca77fbeb59dacba63bc3340bc93c87'
S98 = CV / 'experiments/prefix_strength_confirmation_C_20260913'
AUDIT_SCORES = CV / 'experiments/codex_ideas_sweep_C_20260913/scores-singles.parquet'
CODEX_READ = [HERE / 'src/audit_common.py']
KEYS = ['pair_id', 'hand_id']
STREET = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
BOARD_LEN = {0: 0, 1: 3, 2: 4, 3: 5}
DECK = [r + s for r in '23456789TJQKA' for s in 'cdhs']
AUDIT_POINT, AUDIT_TOL = .020765, .004


def aggressive(x):
    return x['action'] in ('bet', 'raise') or (x['action'] == 'all_in' and x['amount'] > x['to_call'])


def showdown(first, second, board):
    a, b = evaluate_cards(*first, *board), evaluate_cards(*second, *board)
    return 1. if a < b else (.5 if a == b else 0.)


def equity(first, second, board, seed):
    assert all(c in DECK for c in first + second + board)
    rest = [c for c in DECK if c not in set(first) | set(second) | set(board)]
    need = 5 - len(board)
    if need == 0:
        return showdown(first, second, board)
    if need <= 2:
        v = [showdown(first, second, board + list(x)) for x in itertools.combinations(rest, need)]
        return sum(v) / len(v)
    rng = random.Random(seed)
    return sum(showdown(first, second, board + rng.sample(rest, need)) for _ in range(600)) / 600


def fold_context(actions, folder, other):
    last, street, other_in = None, None, True
    for x in actions:
        if x['street'] != street:
            street, last = x['street'], None
        if x['player_id'] == other and x['action'] == 'fold':
            other_in = False
        if x['player_id'] == folder and x['action'] == 'fold':
            return STREET[street], last, other_in
        if aggressive(x):
            last = x['player_id']
    return -1, None, False


def type_attributes(keys, players):
    """Registered T1 for every (pair_id, hand_id) in keys. Reads hands, seats and actions only."""
    hids = keys['hand_id'].unique().to_list()
    boards = dict(pl.scan_parquet(ROOT / 'data/hands.parquet').filter(pl.col('hand_id').is_in(hids)).select('hand_id', 'board_cards').collect().iter_rows())
    seats = defaultdict(dict)
    for r in pl.scan_parquet(ROOT / 'data/seats.parquet').filter(pl.col('hand_id').is_in(hids)).select(
            'hand_id', 'player_id', 'net_chips', 'folded', 'hole_card_1', 'hole_card_2').collect().iter_rows(named=True):
        seats[r['hand_id']][r['player_id']] = r
    acts = defaultdict(list)
    for r in pl.scan_parquet(ROOT / 'data/actions.parquet').filter(pl.col('hand_id').is_in(hids)).select(
            'hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call').collect().sort('hand_id', 'action_no').iter_rows(named=True):
        acts[r['hand_id']].append(r)
    rows = []
    for pid, hid in keys.select(KEYS).iter_rows():
        a, b = players[pid]
        se = seats[hid]
        w, l = (se[a], se[b]) if se[a]['net_chips'] >= se[b]['net_chips'] else (se[b], se[a])
        street, last, other_in = fold_context(acts[hid], l['player_id'], w['player_id']) if l['folded'] else (-1, None, False)
        facing = 'nofold' if not l['folded'] else ('none' if last is None else 'partner' if last == w['player_id'] else 'outsider')
        eq = None
        if l['folded'] and other_in:
            board = (boards[hid] or '').split()[:BOARD_LEN[street]]
            seed = int(hashlib.sha256(f"{hid}|{l['player_id']}|{w['player_id']}".encode()).hexdigest()[:16], 16)
            eq = equity([l['hole_card_1'], l['hole_card_2']], [w['hole_card_1'], w['hole_card_2']], board, seed)
        rows.append({'pair_id': pid, 'hand_id': hid, 'facing': facing, 'fold_street': street, 'folder_equity': eq,
                     'T1': bool(facing == 'partner' and eq is not None and eq > .5)})
    schema = {'pair_id': pl.String, 'hand_id': pl.String, 'facing': pl.String, 'fold_street': pl.Int64, 'folder_equity': pl.Float64, 'T1': pl.Boolean}
    return pl.DataFrame(rows, schema=schema).sort(KEYS)


def less_than_five(ps):
    dist, out = np.zeros(6), np.empty(len(ps))
    dist[0] = 1.
    for i, p in enumerate(ps):
        out[i] = dist[:5].sum()
        new = dist * (1 - p)
        new[1:] += dist[:-1] * p
        new[5] += dist[5] * p
        dist = new
    return out


def chronology(frame):
    hands = pl.scan_parquet(ROOT / 'data/hands.parquet').select('hand_id', 'started_at').join(frame.select('hand_id').unique().lazy(), on='hand_id').collect()
    out = frame.join(hands, on='hand_id', validate='m:1').with_columns(pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'))
    ties = out.filter(pl.col('score') > 5).group_by('pair_id').agg((pl.col('started_at').n_unique() == pl.len()).alias('ok'))
    assert ties['ok'].all(), 'started_at tie inside a top-20 set'
    return out.drop('started_at').sort(KEYS)


def decode(frame, t1, rng=None):
    score = frame['score'].to_numpy()
    out = score.copy()
    pid, cr = frame['pair_id'].to_numpy(), frame['chrono_rank'].to_numpy()
    idx = np.flatnonzero(score > 5)
    _, first = np.unique(pid[idx], return_index=True)
    for g in np.split(idx, first[1:]):
        p = np.clip(score[g] - 10, 0, 1)
        t = rng.permutation(t1[g]) if rng is not None else t1[g]
        order = np.lexsort((cr[g], ~t))
        q = np.empty(len(g))
        q[order] = less_than_five(p[order])
        out[g] = 10 + p * q + 1e-9 * p
    return out


def align(frame, attrs):
    return frame.select(KEYS).join(attrs.select(KEYS + ['T1']), on=KEYS, how='left', validate='1:1')['T1'].fill_null(False).to_numpy()


def main():
    start = time.perf_counter()
    assert not (OUT / 'predictions-frozen.json').exists(), 'Preserve a completed freeze'
    assert sha(PREREG) == PREREG_SHA
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    for k, v in reg['inputs'].items():
        assert sha(ROOT / v['path']) == v['sha256'], k
    protected, snap0 = check_protected(), {str(p): sha(p) for p in CODEX_READ}
    players = {r[0]: (r[1], r[2]) for r in pl.read_csv(ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    d = pl.read_parquet(AUDIT / 'audit-oof.parquet', columns=KEYS + ['table_id', 'behavior_family', 'relevance', 'chrono_rank']).sort(KEYS)
    fr = chronology(pl.read_parquet(AUDIT_SCORES, columns=KEYS + ['table_id', 'B0']).rename({'B0': 'score'}).sort(KEYS))
    assert fr.select(KEYS).equals(d.select(KEYS))
    both = fr.select('pair_id', 'score', 'chrono_rank').with_columns(pl.Series('old', d['chrono_rank'].to_numpy())).filter(pl.col('score') > 5)
    both = both.with_columns(pl.col('chrono_rank').rank('ordinal').over('pair_id').alias('a'), pl.col('old').rank('ordinal').over('pair_id').alias('b'))
    assert both['a'].equals(both['b']), 'audit top-20 chronology order differs from audit-oof'
    attrs = type_attributes(fr.filter(pl.col('score') > 5).select(KEYS), players)
    attrs.write_parquet(OUT / 'audit-type-attributes.parquet')
    t1 = align(fr, attrs)
    c = evaluate(d, {'incumbent': fr['score'].to_numpy(), 'typed_B0': decode(fr, t1)})['comparisons']['typed_B0']
    replay = json.loads(json.dumps({k: c[k] for k in ('mean_delta', 'ci95', 'by_family', 'halves', 'gate_pass')}, default=lambda o: o.item()))
    replay.update(T1_rows=int(t1.sum()), within_tolerance=bool(abs(float(c['mean_delta']) - AUDIT_POINT) <= AUDIT_TOL))
    print('audit replay', json.dumps(replay), flush=True)
    if not replay['within_tolerance']:
        write(OUT / 'stopped-audit-replay.json', replay)
        raise SystemExit('audit replay outside tolerance; no held build')
    (OUT / 'scores').mkdir(exist_ok=True)
    frozen98 = json.loads((S98 / 'predictions-frozen.json').read_text(encoding='utf-8'))
    receipts = []
    for fold in (0, 1, 2):
        arms = {}
        for a in ('B0', 'incumbent', 'stage1_only'):
            f = S98 / 'scores' / f'fold{fold}-{a}.parquet'
            assert sha(f) == frozen98['scores'][f.name], f.name
            arms[a] = pl.read_parquet(f).sort(KEYS)
        assert all(arms[a].select(KEYS).equals(arms['B0'].select(KEYS)) for a in arms)
        assert np.array_equal(arms['B0']['score'].to_numpy() > 5, arms['incumbent']['score'].to_numpy() > 5)
        fb = chronology(arms['B0'])
        fi = fb.with_columns(pl.Series('score', arms['incumbent']['score'].to_numpy()))
        attrs = type_attributes(fb.filter(pl.col('score') > 5).select(KEYS), players)
        attrs.write_parquet(OUT / f'fold{fold}-type-attributes.parquet')
        t1 = align(fb, attrs)
        new = {'typed_B0': decode(fb, t1), 'time_only_B0': decode(fb, np.zeros(len(t1), bool)),
               'shuffled_type_B0': decode(fb, t1, np.random.default_rng(104)), 'typed_incumbent': decode(fi, t1)}
        for name, p in new.items():
            assert np.isfinite(p).all() and np.array_equal(p > 5, fb['score'].to_numpy() > 5)
            fb.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', p)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
        receipts.append({'fold': fold, 'rows': fb.height, 'top20_rows': int((fb['score'] > 5).sum()), 'pairs': fb['pair_id'].n_unique(), 'T1_rows': int(t1.sum())})
        print('fold', fold, receipts[-1], round(time.perf_counter() - start, 1), flush=True)
    frozen = {'created_at': datetime.now(timezone.utc).isoformat(), 'registration_sha256': PREREG_SHA, 'audit_replay': replay,
              'base_scores_step98': {f'fold{f}-{a}.parquet': frozen98['scores'][f'fold{f}-{a}.parquet'] for f in (0, 1, 2) for a in ('B0', 'incumbent', 'stage1_only')},
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').iterdir())},
              'attributes': {p.name: sha(p) for p in sorted(OUT.glob('*-type-attributes.parquet'))}, 'fold_receipts': receipts,
              'script_sha256': sha(Path(__file__)), 'protected_unchanged': check_protected() == protected,
              'codex_files_unchanged': snap0 == {str(p): sha(p) for p in CODEX_READ}, 'seconds': time.perf_counter() - start, 'held_evidence_evaluation': False}
    assert frozen['protected_unchanged'] and frozen['codex_files_unchanged']
    write(OUT / 'predictions-frozen.json', frozen)
    print('PREDICTIONS FROZEN', round(frozen['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
