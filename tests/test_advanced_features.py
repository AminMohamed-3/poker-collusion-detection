"""Independent gameplay scenarios; all writes are to pytest temporary folders."""
from __future__ import annotations

import math

import polars as pl
import pytest
from phevaluator import evaluate_cards

from src import advanced_features as af


ACTION_SCHEMA = {
    'hand_id': pl.String, 'player_id': pl.String, 'action_no': pl.Int64,
    'street': pl.String, 'action': pl.String, 'amount': pl.Int64,
    'amount_to': pl.Int64, 'pot_before': pl.Int64, 'stack_before': pl.Int64,
    'to_call': pl.Int64, 'players_active': pl.Int64,
}


def game(board, holes, moves, stacks=None):
    """A/B post 1/2 blinds. Moves give a raise-to total, or an automatic call.

    Heads-up A is button, so B acts first postflop. With C present, C is
    button, so A acts first postflop. Full boards settle all-ins/side pots.
    """
    stacks = stacks or {p: 100 for p in holes}
    paid = {p: min(stacks[p], {'A': 1, 'B': 2}.get(p, 0)) for p in holes}
    street_paid = paid.copy()
    active, street, rows = set(holes), 'preflop', []
    for no, (new_street, actor, action, target) in enumerate(moves):
        assert actor in active
        if new_street != street:
            street_paid = {p: 0 for p in holes}
            street = new_street
        stack = stacks[actor] - paid[actor]
        price = min(stack, max(street_paid.values()) - street_paid[actor])
        if action in ('check', 'fold'):
            amount = 0
        elif action == 'call':
            amount = price
        elif action == 'all_in':
            amount = stack
        else:
            amount = target - street_paid[actor]
        assert 0 <= amount <= stack
        assert action != 'check' or price == 0
        rows.append({
            'hand_id': 'h', 'player_id': actor, 'action_no': no, 'street': street,
            'action': action, 'amount': amount,
            'amount_to': street_paid[actor] + amount,
            'pot_before': sum(paid.values()), 'stack_before': stack,
            'to_call': price, 'players_active': len(active),
        })
        paid[actor] += amount
        street_paid[actor] += amount
        if action == 'fold':
            active.remove(actor)
    # Return unmatched contributions and award each contested side pot.
    gross = {p: 0.0 for p in holes}
    previous = 0
    for level in sorted(set(paid.values())):
        contributors = [p for p in holes if paid[p] >= level]
        eligible = [p for p in contributors if p in active]
        pot = (level - previous) * len(contributors)
        if len(contributors) == 1:
            winners = contributors
        elif len(eligible) == 1:
            winners = eligible
        else:
            ranks = {p: evaluate_cards(*holes[p], *board.split()) for p in eligible}
            winners = [p for p in eligible if ranks[p] == min(ranks.values())]
        for p in winners:
            gross[p] += pot / len(winners)
        previous = level
    seats = pl.DataFrame([
        {'hand_id': 'h', 'player_id': p, 'hole_card_1': cards[0],
         'hole_card_2': cards[1], 'net_chips': gross[p] - paid[p]}
        for p, cards in holes.items()
    ])
    assert sum(seats['net_chips']) == pytest.approx(0)
    hands = pl.DataFrame({'hand_id': ['h'], 'board_cards': [board], 'big_blind': [2]})
    actions = pl.DataFrame(rows, schema=ACTION_SCHEMA)
    return hands, seats, actions


def build_game(tmp_path, monkeypatch, records, pairs=None):
    data, base, out = (tmp_path / name for name in ('data', 'base', 'advanced'))
    data.mkdir()
    (base / 'hands_development').mkdir(parents=True)
    pairs = pairs or [('AB', 'A', 'B')]
    pl.DataFrame(pairs, schema=['pair_id', 'player_1', 'player_2'], orient='row').write_csv(
        data / 'development_labels.csv')
    for name, frame in zip(('hands', 'seats', 'actions'), records):
        frame.write_parquet(data / f'{name}.parquet')
    pl.DataFrame({'pair_id': [p[0] for p in pairs], 'hand_id': ['h'] * len(pairs),
                  'table_id': ['pool'] * len(pairs), 'original_feature': [7.0] * len(pairs)}
                 ).write_parquet(base / 'hands_development/batch_000.parquet')
    monkeypatch.setattr(af, 'DATA', data)
    monkeypatch.setattr(af, 'BASE', base)
    monkeypatch.setattr(af, 'OUT', out)
    af.build('development')
    return pl.read_parquet(out / 'hands_development/batch_000.parquet')


def test_card_encoding_and_preflop_order_are_correct():
    assert len(set(af.CARDS.values())) == 52
    for rank, label in enumerate('23456789TJQKA', 2):
        ids = [af.CARDS[label + suit] for suit in 'cdhs']
        assert {card // 4 + 2 for card in ids} == {rank}
        assert len({card % 4 for card in ids}) == 4
    def value(first, second):
        return af.preflop_strength(af.CARDS[first], af.CARDS[second])
    assert value('Ac', 'Ad') > value('Kc', 'Kd') > value('Ac', 'Kc')
    assert value('Ac', 'Kc') > value('Ac', 'Kd') > value('7c', '2d')
    assert value('Ac', 'Kd') == value('Kd', 'Ac')


def test_strength_uses_only_board_prefix_and_missing_streets_stay_missing():
    seats = pl.DataFrame({'hand_id': ['h'], 'player_id': ['A'],
                          'hole_card_1': ['As'], 'hole_card_2': ['Ks']})
    def ranks(board):
        return af.strength_table(seats, pl.DataFrame({'hand_id': ['h'], 'board_cards': [board]})).row(0, named=True)
    short = ranks('Qs Js Ts')
    full = ranks('Qs Js Ts 2c 3d')
    altered = ranks('Qs Js Ts 9c 8d')
    assert short['rank_flop'] == full['rank_flop'] == altered['rank_flop'] == 1
    assert short['rank_turn'] is None and short['rank_river'] is None
    assert short['rank_preflop'] == ranks('2c 3d 4h 5s 6c')['rank_preflop']


def test_strength_does_not_use_later_improvement():
    s = pl.DataFrame({'hand_id': ['h'], 'player_id': ['A'],
                      'hole_card_1': ['Ac'], 'hole_card_2': ['Ad']})
    a = af.strength_table(s, pl.DataFrame({'hand_id': ['h'], 'board_cards': ['2c 3d 7h As Ah']})).row(0, named=True)
    b = af.strength_table(s, pl.DataFrame({'hand_id': ['h'], 'board_cards': ['2c 3d 7h 8s 9h']})).row(0, named=True)
    assert a['rank_flop'] == b['rank_flop'] == evaluate_cards('Ac', 'Ad', '2c', '3d', '7h')
    assert a['rank_river'] < b['rank_river']


def test_opponent_attribution_pair_symmetry_and_sum_partition(tmp_path, monkeypatch):
    records = game('2h 5s 8d 9c Ts', {'A': ('Ac', 'Ad'), 'B': ('Kc', 'Kd'), 'C': ('Qc', 'Qd')}, [
        ('preflop', 'C', 'call', None), ('preflop', 'A', 'call', None), ('preflop', 'B', 'check', None),
        ('flop', 'A', 'bet', 4), ('flop', 'B', 'raise', 12), ('flop', 'C', 'raise', 24),
        ('flop', 'A', 'call', None), ('flop', 'B', 'call', None),
        ('turn', 'A', 'check', None), ('turn', 'B', 'check', None), ('turn', 'C', 'check', None),
        ('river', 'A', 'check', None), ('river', 'B', 'check', None), ('river', 'C', 'check', None),
    ])
    result = build_game(tmp_path, monkeypatch, records, [('AB', 'A', 'B'), ('BA', 'B', 'A'), ('AC', 'A', 'C')])
    rows = {r['pair_id']: r for r in result.to_dicts()}
    cols = [c for c in result.columns if c.startswith('v2_')]
    assert len(cols) == 56
    assert [rows['AB'][c] for c in cols] == [rows['BA'][c] for c in cols]
    assert rows['AB']['v2_partner_call_count_sum'] == 0
    assert rows['AB']['v2_isolation_reraise_sum'] == 1
    assert rows['AC']['v2_partner_call_count_sum'] == 1
    for row in rows.values():
        assert row['original_feature'] == 7
        assert row['v2_weak_call_sum'] == pytest.approx(row['v2_weak_partner_call_sum'] + row['v2_weak_other_call_sum'])
        assert all(math.isfinite(row[c]) and row[c] >= 0 for c in cols)
        assert all(row[c] <= row[c[:-4] + '_sum'] for c in cols if c.endswith('_max'))


def test_partner_is_inactive_after_folding(tmp_path, monkeypatch):
    records = game('2h 5s 8d 9c Ts', {'A': ('Ac', 'Ad'), 'B': ('Kc', 'Kd'), 'C': ('Qc', 'Qd')}, [
        ('preflop', 'C', 'call', None), ('preflop', 'A', 'call', None), ('preflop', 'B', 'check', None),
        ('flop', 'A', 'check', None), ('flop', 'B', 'bet', 4), ('flop', 'C', 'call', None), ('flop', 'A', 'fold', None),
        ('turn', 'B', 'check', None), ('turn', 'C', 'bet', 4), ('turn', 'B', 'call', None),
        ('river', 'B', 'check', None), ('river', 'C', 'check', None),
    ])
    result = build_game(tmp_path, monkeypatch, records, [('AB', 'A', 'B'), ('BC', 'B', 'C')])
    rows = {r['pair_id']: r for r in result.to_dicts()}
    assert rows['AB']['v2_soft_late_check_sum'] == 0
    assert rows['AB']['v2_soft_heads_up_check_sum'] == 0
    assert rows['AB']['v2_partner_fold_count_sum'] == 1
    assert rows['BC']['v2_soft_late_check_sum'] > 0
    # A's strong fold faces B's bet, even though C called in between.
    assert rows['AB']['v2_soft_ahead_fold_sum'] == pytest.approx(1 - 4 / (14 + 4))


def test_previous_aggressor_resets_at_new_street(tmp_path, monkeypatch):
    records = game('2h 5s 8d 9c Ts', {'A': ('Ac', 'Ad'), 'B': ('Kc', 'Kd'), 'C': ('Qc', 'Qd')}, [
        ('preflop', 'C', 'call', None), ('preflop', 'A', 'call', None), ('preflop', 'B', 'check', None),
        ('flop', 'A', 'bet', 4), ('flop', 'B', 'call', None), ('flop', 'C', 'call', None),
        ('turn', 'A', 'check', None), ('turn', 'B', 'bet', 4), ('turn', 'C', 'call', None), ('turn', 'A', 'call', None),
        ('river', 'A', 'check', None), ('river', 'B', 'check', None), ('river', 'C', 'check', None),
    ])
    row = build_game(tmp_path, monkeypatch, records).row(0, named=True)
    # B opens the turn, rather than reraising A's previous-street bet.
    assert row['v2_isolation_reraise_sum'] == 0
    assert row['v2_partner_call_count_sum'] == 2


def test_action_without_corresponding_board_prefix_is_rejected(tmp_path, monkeypatch):
    records = game('2h 5s 8d 9c Ts', {'A': ('Ac', 'Ad'), 'B': ('Kc', 'Kd')}, [
        ('preflop', 'A', 'call', None), ('preflop', 'B', 'check', None),
        ('flop', 'B', 'check', None), ('flop', 'A', 'check', None),
        ('turn', 'B', 'check', None), ('turn', 'A', 'check', None),
        ('river', 'B', 'check', None), ('river', 'A', 'check', None),
    ])
    # Intentionally malformed copy: the action log reaches streets absent in the hand row.
    hands, seats, actions = records
    hands = hands.with_columns(pl.lit('2h 5s 8d').alias('board_cards'))
    with pytest.raises(ValueError, match='no corresponding board prefix'):
        build_game(tmp_path, monkeypatch, (hands, seats, actions))


@pytest.mark.parametrize('stack,raises', [(12, False), (16, True)])
def test_all_in_call_and_raise_are_distinguished(tmp_path, monkeypatch, stack, raises):
    moves = [('preflop', 'A', 'call', None), ('preflop', 'B', 'check', None),
             ('flop', 'B', 'bet', 4), ('flop', 'A', 'call', None),
             ('turn', 'B', 'check', None), ('turn', 'A', 'check', None),
             ('river', 'B', 'bet', 6), ('river', 'A', 'all_in', None)]
    if raises:
        moves.append(('river', 'B', 'call', None))
    records = game('2h 5s 8d 9c Ts', {'A': ('Ac', 'Ad'), 'B': ('Kc', 'Kd')}, moves, {'A': stack, 'B': 100})
    row = build_game(tmp_path, monkeypatch, records).row(0, named=True)
    sr = {r['player_id']: r for r in af.strength_table(records[1], records[0]).to_dicts()}
    strength = lambda p: (7463 - sr[p]['rank_river']) / 7462
    expected = strength('B') + (strength('A') if raises else 0)
    assert row['v2_late_aggressive_strength_sum'] == pytest.approx(expected)
    assert row['v2_partner_ahead_call_count_sum'] == (1 if raises else 2)


@pytest.mark.parametrize('action', ['check', 'fold', 'call'])
def test_ties_do_not_count_as_strictly_ahead(action):
    hands = pl.DataFrame({'hand_id': ['h'], 'board_cards': ['Jd Th 9c 8s 7d']})
    seats = pl.DataFrame({'hand_id': ['h', 'h'], 'player_id': ['A', 'B'],
                          'hole_card_1': ['Ac', 'Kd'], 'hole_card_2': ['2c', '3d'],
                          'net_chips': [0, 0], 'fold_no': [1000000, 1000000]})
    si = af.strength_table(seats, hands).join(seats.select('hand_id', 'player_id', 'net_chips', 'fold_no'), on=['hand_id', 'player_id'])
    assert si['rank_river'].n_unique() == 1
    # A legal river decision fragment, with both players holding the board straight.
    bet = action != 'check'
    actions = pl.DataFrame({'hand_id': ['h'], 'player_id': ['A'], 'street': ['river'],
                           'action_no': [10], 'action': [action], 'previous_aggressor': ['B' if bet else None],
                           'to_call': [2 if bet else 0], 'pot_before': [12 if bet else 10],
                           'amount': [2 if action == 'call' else 0], 'big_blind': [2], 'stack_before': [100],
                           'aggressive': [False], 'players_active': [2]}, schema_overrides={'previous_aggressor': pl.String})
    keys = pl.DataFrame({'pair_id': ['AB'], 'hand_id': ['h'], 'player_1': ['A'], 'player_2': ['B']})
    row = af.directional_events(keys, actions, si, 'player_1', 'player_2').row(0, named=True)
    assert row['v2_soft_ahead_check'] == row['v2_soft_ahead_fold'] == row['v2_partner_ahead_call_count'] == 0
    assert row['v2_behind_partner_call'] == 0


def test_no_voluntary_actions_preserve_pair_row_and_fill_zero(tmp_path, monkeypatch):
    records = game('2h 5s 8d 9c Ts', {'A': ('Ac', 'Ad'), 'B': ('Kc', 'Kd')}, [], {'A': 1, 'B': 2})
    result = build_game(tmp_path, monkeypatch, records)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row['original_feature'] == 7
    assert len([c for c in result.columns if c.startswith('v2_')]) == 56
    assert all(v == 0 for k, v in row.items() if k.startswith('v2_'))
