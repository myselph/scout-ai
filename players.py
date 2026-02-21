# A Scout (card game) simulator.
import random
from typing import Self
from common import Player, InformationState, Scout, Show, ScoutAndShow, Move, Util, Card, Score
from collections.abc import Sequence


class RandomPlayer(Player):
    # A baseline player that randomly selects from the possible moves.
    def flip_hand(self, hand: Sequence[Card]) -> bool:
        return random.choice([True, False])

    def select_move(self, info_state: InformationState) -> Move:
        return random.choice(info_state.possible_moves())


class GreedyShowPlayer(Player):
    # A player that maximizes for gettng rid of its cards (ie it picks the
    # highest Show and ScoutAndShow moves).
    def flip_hand(self, hand: Sequence[Card]) -> bool:
        return random.choice([True, False])

    def select_move(self, info_state: InformationState) -> Move:
        # The linter really has a hard time with this function, and I don't
        # understand why, hence some ignore statements. I tried to improve
        # things by sprinkling asserts over the function but that didn't help
        # and just makes the code slower. And I didn't want to go as far as
        # initializing next_move with nonsensical values to make pylance happy.
        moves = info_state.possible_moves()
        scouts = [m for m in moves if isinstance(m, Scout)]
        shows = [m for m in moves if isinstance(m, Show)]
        scout_and_shows = [m for m in moves if isinstance(m, ScoutAndShow)]
        next_move = None
        if scouts:
            next_move = random.choice(scouts)
        if shows:
            next_move = max(shows, key=lambda m: m.length)
        if scout_and_shows:
            best_scout_and_show = max(
                scout_and_shows, key=lambda m: m.show.length)
            # Pick the Scout&Show over a Show only if we to dump at least 3
            # cards more - because we a) increase our hand by one b) can S&S
            # only once c) another player scores points.
            if not shows or next_move.length < best_scout_and_show.show.length + 2:  # type:ignore
                next_move = best_scout_and_show
        assert next_move
        return next_move


class GreedyShowPlayerWithFlip(GreedyShowPlayer):
    # Like GreedyShowPlayer, but with non-random flip - improves performance.
    def flip_hand(self, hand: Sequence[Card]) -> bool:
        up_value = self._hand_value([h[0] for h in hand])
        down_value = self._hand_value([h[1] for h in hand])
        return up_value < down_value

class EpsilonGreedyScorePlayer(Player):
    # A player that picks, with P(1-epsilon), the action that most improves its
    # score, and a random move with P(epsilon)
    _epsilon: float
    def __init__(self, epsilon=0.1):
        self._epsilon = epsilon

    def flip_hand(self, hand: Sequence[Card]) -> bool:
        up_value = self._hand_value([h[0] for h in hand])
        down_value = self._hand_value([h[1] for h in hand])
        return up_value < down_value

    def select_move(self, info_state: InformationState) -> Move:
        moves = info_state.possible_moves()
        if random.random() < self._epsilon:
            return random.choice(moves)
        
        scores = self._scores(info_state, moves)
        max_index = max(range(len(scores)), key = lambda i: scores[i])
        return moves[max_index]

    def _scores(self, info_state: InformationState, moves: tuple[Move, ...]) -> list[Score]:
        scores = []
        for move in moves:
            if isinstance(move, Scout):
                scores.append(-1)
            elif isinstance(move, Show):
                scores.append(len(info_state.table) + move.length)
            elif isinstance(move, ScoutAndShow):
                scores.append(len(info_state.table) + move.show.length - 1)
        return scores




class PlanningPlayer(GreedyShowPlayerWithFlip):
    # A player with a heuristic value function that simulates all moves and
    # picks the one with the highest value. Best performing heuristic player.
    # There are various knobs in the value function one could tune through RL or
    # grid search.
    c: float

    def __init__(self):
        self.c = 0.25  # found via grid search

    def select_move(self, info_state: InformationState) -> Move:
        moves = info_state.possible_moves()
        return max(moves, key=lambda m: self._value(info_state, m))

    def _value(self, info_state: InformationState, move: Move) -> float:
        # Calculates a heuristic value for the state of the game after the given
        # move. This involves simulating every move and calculating the new
        # value.
        hand_values = [h[0] for h in info_state.hand]
        if isinstance(move, Scout):
            hand_values_new = self._simulate_scout(
                hand_values, info_state.table, move)
            return self.c * \
                self._hand_value(hand_values_new) - len(hand_values_new) - 1
        elif isinstance(move, Show):
            hand_values_new = self._simulate_show(hand_values, move)
            return self.c * \
                self._hand_value(hand_values_new) - len(hand_values_new) + len(info_state.table)
        else:
            hand_values_new = self._simulate_scout(
                hand_values, info_state.table, move.scout)
            hand_values_new = self._simulate_show(hand_values_new, move.show)
            return self.c * \
                self._hand_value(hand_values_new) - len(hand_values_new) + len(info_state.table) - 1

    def _simulate_scout(
            self,
            hand_values: list[int],
            table: tuple[Card, ...],
            scout: Scout):
        card = table[0] if scout.first else table[-1]
        card_value = card[1] if scout.flip else card[0]
        return hand_values[:scout.insertPos] + \
            [card_value] + hand_values[scout.insertPos:]

    def _simulate_show(self, hand_values: list[int], show: Show):
        return hand_values[:show.startPos] + \
            hand_values[show.startPos + show.length:]


class CompositePlayer(Player):
    _cum_weights: list[float]
    _players: list[Player]

    def __init__(self, players: list[Player], weights: list[float]):
        assert len(weights) == len(players)
        assert sum(weights) <= 1.0
        self._players = players
        self._cum_weights = [weights[0]]
        for w in weights[1:-1]:
            self._cum_weights.append(w + self._cum_weights[-1])
      
    def select_move(self, info_state: InformationState) -> Move:
        p = random.random()
        for i, w in enumerate(self._cum_weights):
            if p < w:
                return self._players[i].select_move(info_state)
                
        return self._players[-1].select_move(info_state)
    

    def flip_hand(self, hand: Sequence[Card]) -> bool:
        p = random.random()
        for i, w in enumerate(self._cum_weights):
            if p < w:
                return self._players[i].flip_hand(hand)
                
        return self._players[-1].flip_hand(hand)
        