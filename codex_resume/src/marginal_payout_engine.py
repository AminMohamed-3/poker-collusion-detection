"""Deterministic observer-side showdown payout proxy.

This evaluates disclosed cards and the board prefix available at an action.  It is
an observer proxy for the immediate call-versus-fold payout, not a player's
strategic EV: it has no ranges, rake, future betting, or belief model.  The full
board supplied at construction is deliberately sliced to ``street``; its later
suffix never enters rank generation or runout blocking.
"""
from __future__ import annotations

from itertools import combinations
import math
from numbers import Integral, Real

import numpy as np
from phevaluator import evaluate_cards

PREFLOP_SAMPLES = 4096
PREFLOP_SEED = 479203
_STREET_PREFIX = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
# Six two-card hands leave 40 unknown cards preflop.  Every hand uses these
# same permutation positions against its sorted remaining-card vector.
_PREFLOP_INDICES = np.argsort(
    np.random.default_rng(PREFLOP_SEED).random((PREFLOP_SAMPLES, 40)), axis=1
)[:, :5]


class MechanicalPayoutError(ValueError):
    """The supplied action-state accounting cannot form a legal payout layer."""


class HandPayouts:
    """Cache showdown rank arrays for six disclosed hole-card pairs.

    ``holes`` maps each of the six seat identifiers to two integer card IDs and
    ``full_board`` contains the disclosed terminal board prefix (0, 3, 4, or 5
    card IDs).  Player IDs are only output labels: simulation order is
    canonicalized by sorted hole cards.
    """

    def __init__(self, holes: dict, full_board):
        if len(holes) != 6:
            raise MechanicalPayoutError('exactly six disclosed seat hole-card pairs are required')
        self._holes = {player: self._card_pair(cards) for player, cards in holes.items()}
        if len(self._holes) != 6:
            raise MechanicalPayoutError('seat identifiers must be unique')
        self._board = tuple(full_board)
        if len(self._board) not in (0, 3, 4, 5):
            raise MechanicalPayoutError('full_board must contain 0, 3, 4, or 5 disclosed card IDs')
        self._validate_cards((*self._board, *(card for pair in self._holes.values() for card in pair)))
        # Distinct hole pairs make this total order independent of labels.
        self._players = tuple(sorted(self._holes, key=lambda player: tuple(sorted(self._holes[player]))))
        self._player_index = {player: index for index, player in enumerate(self._players)}
        self._rank_cache: dict[int, np.ndarray] = {}
        self._share_cache: dict[tuple[int, tuple[int, ...]], tuple[float, ...]] = {}

    @staticmethod
    def _card_pair(cards):
        pair = tuple(cards)
        if len(pair) != 2:
            raise MechanicalPayoutError('each seat must have exactly two hole cards')
        return pair

    @staticmethod
    def _validate_cards(cards):
        cards = tuple(cards)
        if any(not isinstance(card, Integral) or not 0 <= int(card) < 52 for card in cards):
            raise MechanicalPayoutError('cards must be integer IDs in [0, 51]')
        if len(set(cards)) != len(cards):
            raise MechanicalPayoutError('disclosed cards must be distinct')

    @staticmethod
    def _street_prefix(street):
        try:
            return _STREET_PREFIX[str(street).lower()]
        except KeyError as exc:
            raise MechanicalPayoutError('street must be preflop, flop, turn, or river') from exc

    def _runouts(self, prefix):
        board = tuple(int(card) for card in self._board[:prefix])
        known = set(board)
        # Every dealt hole card remains a blocker, even if that seat folded.
        known.update(card for pair in self._holes.values() for card in pair)
        remaining = tuple(card for card in range(52) if card not in known)
        if prefix == 0:
            if len(remaining) != 40:
                raise MechanicalPayoutError('six valid hole-card pairs must leave 40 preflop runout cards')
            return (tuple(remaining[index] for index in row) for row in _PREFLOP_INDICES)
        return combinations(remaining, 5 - prefix)

    def _ranks(self, street):
        prefix = self._street_prefix(street)
        if len(self._board) < prefix:
            raise MechanicalPayoutError(f'{street} requires {prefix} disclosed board cards; only {len(self._board)} are available')
        cached = self._rank_cache.get(prefix)
        if cached is not None:
            return cached
        board = tuple(int(card) for card in self._board[:prefix])
        rows = []
        for future in self._runouts(prefix):
            complete = board + future
            rows.append([evaluate_cards(*self._holes[player], *complete) for player in self._players])
        if not rows:
            raise MechanicalPayoutError('no legal showdown runouts')
        ranks = np.asarray(rows, dtype=np.int32)
        self._rank_cache[prefix] = ranks
        return ranks

    def _shares(self, eligible, street):
        eligible = tuple(eligible)
        prefix = self._street_prefix(street)
        indices = tuple(self._player_index[player] for player in eligible)
        key = (prefix, indices)
        cached = self._share_cache.get(key)
        if cached is None:
            ranks = self._ranks(street)[:, indices]
            best = ranks.min(axis=1, keepdims=True)
            winners = ranks == best
            cached = tuple((winners / winners.sum(axis=1, keepdims=True)).mean(axis=0).tolist())
            self._share_cache[key] = cached
        return dict(zip(eligible, cached))

    def _validate_state(self, contributions, live, mode):
        if mode not in {'single', 'layered'}:
            raise MechanicalPayoutError("mode must be 'single' or 'layered'")
        if set(contributions) != set(self._players):
            raise MechanicalPayoutError('contributions must contain every seat and no unknown seat')
        checked = {}
        for player in self._players:
            value = contributions[player]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
                raise MechanicalPayoutError('contributions must be finite, nonnegative real values')
            checked[player] = float(value)
        live = set(live)
        if not live or not live.issubset(self._player_index):
            raise MechanicalPayoutError('live must be a nonempty subset of the six seats')
        return checked, live

    def payouts(self, contributions: dict, live: set, street: str, mode: str = 'single') -> dict:
        """Return total seat payouts and separately identified uncalled refunds.

        In ``single`` mode every live seat shares the entire contributed pot.
        In ``layered`` mode each positive contribution level is a side-pot layer;
        only live contributors to that layer can win it.  A layer with one
        contributor is an uncalled-money refund.  A multi-contributor layer with
        no live contender is an explicit mechanical error.
        """
        contributions, live = self._validate_state(contributions, live, mode)
        payouts = {player: 0.0 for player in self._players}
        refunds = {player: 0.0 for player in self._players}
        total = math.fsum(contributions.values())
        layers = []
        if mode == 'single':
            eligible = tuple(player for player in self._players if player in live)
            shares = self._shares(eligible, street)
            for player, share in shares.items():
                payouts[player] += total * share
            layers.append({'value': total, 'contributors': self._players, 'eligible': eligible, 'refund': False})
        else:
            previous = 0.0
            for level in sorted({value for value in contributions.values() if value > 0.0}):
                contributors = tuple(player for player in self._players if contributions[player] >= level)
                value = (level - previous) * len(contributors)
                previous = level
                if len(contributors) == 1:
                    player = contributors[0]
                    payouts[player] += value
                    refunds[player] += value
                    layers.append({'value': value, 'contributors': contributors, 'eligible': (), 'refund': True})
                    continue
                eligible = tuple(player for player in contributors if player in live)
                if not eligible:
                    raise MechanicalPayoutError('multi-contributor payout layer has no live eligible seat')
                shares = self._shares(eligible, street)
                for player, share in shares.items():
                    payouts[player] += value * share
                layers.append({'value': value, 'contributors': contributors, 'eligible': eligible, 'refund': False})
        paid = math.fsum(payouts.values())
        if any(not math.isfinite(value) or value < -1e-12 for value in payouts.values()):
            raise MechanicalPayoutError('payouts must remain finite and nonnegative')
        if not math.isclose(paid, total, rel_tol=1e-12, abs_tol=1e-9):
            raise MechanicalPayoutError('payout conservation failure')
        return {
            'payouts': payouts,
            'refunds': refunds,
            'layers': layers,
            'runouts': int(self._ranks(street).shape[0]),
            'street': str(street).lower(),
            'mode': mode,
        }

