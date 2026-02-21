# A Scout (card game) simulator.
from __future__ import annotations  # for static factory method annotation
import random
from typing import Callable, cast
from common import Scout, Show, Move, InformationState, Util, RecordedMove, RecordedScoutAndShow, RecordedScout, RecordedShow, Card
import math


def _initial_cards_per_player(num_players: int) -> int:
    if num_players == 3:
        return 12
    elif num_players == 4:
        return 11
    elif num_players == 5:
        return 9
    raise NotImplementedError("Only 3-5 players supported")


def _generate_hands(num_players: int) -> list[list[Card]]:
    # Helper function to deal a whole deck.
    # The whole deck consists of all pairs of 1-10 (sampling w/o replacement), ie 45 cards:
    # [(1,2), (1,3), ..., (2,1), ..., (8,9), (8,10), (9,10)], but which cards are used
    # depends on the number of players. For now I only support 3-5.
    full_deck = [(i, j) for i in range(1, 10) for j in range(i, 11) if i != j]
    if num_players < 3 or num_players > 5:
        raise NotImplementedError("Only 3-5 players supported")
    if num_players == 3:
        # 36 cards, skip the 10s -> 12 cards/player
        N = _initial_cards_per_player(3)
        deck = [r for r in full_deck if r[1] != 10]
    elif num_players == 4:
        # skip (9,10) -> 11 cards / player
        N = _initial_cards_per_player(4)
        deck = full_deck[:-1]
    else:
        N = _initial_cards_per_player(5)
        deck = full_deck
    # Randomly flip, shuffle and serve.
    flips = random.choices([True, False], k=len(deck))
    deck = [c if not flip else (c[1], c[0]) for (c, flip) in zip(deck, flips)]
    random.shuffle(deck)
    return [deck[i * N:(i + 1) * N] for i in range(0, num_players)]


def _normalize_card(c: Card) -> Card:
    return c if c[0] < c[1] else (c[1], c[0])


# Simulate a set of recorded moves on a deck full of Nones, to propagate
# information from the recorded moves - we should end up with a set of
# hands with some known cards in the right positions, and a set of
# cards that have been removed from the game.
def _simulate(num_players: int,
              dealer: int,
              history: list[RecordedMove]) -> tuple[list[list[Card | None]],
                                                    list[Card]]:
    cards_per_player = _initial_cards_per_player(num_players)
    partial_hand: list[list[Card | None]] = [
        [None] * cards_per_player for _ in range(num_players)]
    player_index = dealer
    removed_cards = []
    for rm in history:
        if isinstance(rm, RecordedScout):
            card = rm.card
            card = card if not rm.move.flip else (card[1], card[0])
            partial_hand[player_index] = partial_hand[player_index][:rm.move.insertPos] + [
                card] + partial_hand[player_index][rm.move.insertPos:]
        elif isinstance(rm, RecordedShow):
            # NB we do not use the .shown attribute, and it could conceivably
            # be removed from RecordedShow, but it seems like it belongs there
            # and may be useful for other replays (eg visualization, neural
            # nets).
            removed_cards += rm.removed
            partial_hand[player_index] = partial_hand[player_index][:rm.move.startPos] + \
                partial_hand[player_index][rm.move.startPos + rm.move.length:]
        elif isinstance(rm, RecordedScoutAndShow):
            card = rm.scout.card
            card = card if not rm.scout.move.flip else (card[1], card[0])
            partial_hand[player_index] = partial_hand[player_index][:rm.scout.move.insertPos] + [
                card] + partial_hand[player_index][rm.scout.move.insertPos:]
            removed_cards += rm.show.removed
            partial_hand[player_index] = partial_hand[player_index][:rm.show.move.startPos] + \
                partial_hand[player_index][rm.show.move.startPos + rm.show.move.length:]
        else:
            raise NotImplementedError("Unknown move")
        player_index = (player_index + 1) % num_players
    return partial_hand, removed_cards


class GameState:
    num_players: int
    dealer: int  # index of first player
    current_player: int  # index of next player to call move() and info_state()
    # index of player who did the last Show or ScoutAndShow, for accounting.
    scout_benefactor: int
    hands: list[list[Card]]
    table: list[Card]
    # Per-player running scores: scout points + cards collected - cards in
    # hand.
    scores: list[int]
    # Whether a player has used their Scout & Show capability yet.
    can_scout_and_show: list[bool]
    history: list[RecordedMove]
    initial_flip_executed: bool  # Whether the initial flip has been executed.
    finished: bool  # Whether the game is over.

    def __init__(self, num_players: int, dealer: int, max_moves: int = 1000):
        self.num_players = num_players
        self.hands = _generate_hands(num_players)
        self.scores = [-len(h) for h in self.hands]
        self.can_scout_and_show = [True] * num_players
        self.dealer = dealer
        self.current_player = self.dealer
        self.scout_benefactor = -1
        self.table = []
        self.history = []
        self.initial_flip_executed = False
        self.finished = False
        self.moves_left = max_moves

    def move(self, m: Move):
        assert self.initial_flip_executed
        assert not self.finished
        assert Util.is_move_valid(self.hands[self.current_player], self.table,
                                  self.can_scout_and_show[self.current_player], m)
        if isinstance(m, Scout):
            scouted_card = self._scout(m)
            recorded_move = RecordedScout(m, scouted_card)
            if (self.current_player + 1) % self.num_players == self.scout_benefactor:
                self.finished = True
        elif isinstance(m, Show):
            (s, r) = self._show(m)[:]
            recorded_move = RecordedShow(m, tuple(s), tuple(r))
        else:
            scouted_card = self._scout(m.scout)
            (s, r) = self._show(m.show)[:]
            recorded_move = RecordedScoutAndShow(
                RecordedScout(
                    m.scout, scouted_card), RecordedShow(
                    m.show, tuple(s), tuple(r)))
            self.can_scout_and_show[self.current_player] = False
        if not self.hands[self.current_player]:
            self.finished = True
        self.history.append(recorded_move)
        self.current_player = (self.current_player + 1) % self.num_players
        # I added the max moves counter; there is no such thing in official
        # Scout, but it helps avoid enless loops in which deterministic players
        # can get stuck in. It would be preferable to have that check outside
        # since this is not an official rule, but I doubt it will ever cause
        # problems (famous last words).
        self.moves_left -= 1
        if self.moves_left == 0:
            self.finished = True

    def is_finished(self):
        return self.finished

    def maybe_flip_hand(self, flip_fns: list[Callable[[list[Card]], bool]]):
        assert not self.initial_flip_executed
        # Give each player the option to flip their hand
        assert not self.history
        assert len(flip_fns) == self.num_players
        for player in range(self.num_players):
            if flip_fns[player](self.hands[player]):
                self.hands[player] = list(
                    map(lambda c: (c[1], c[0]), self.hands[player]))
        self.initial_flip_executed = True

    def info_state(self):
        # Returns the information state for the current player.
        return InformationState(
            self.num_players, self.dealer, self.current_player,
            self.scout_benefactor, tuple(self.hands[self.current_player]),
            tuple(self.table),
            tuple([len(self.hands[i]) for i in range(self.num_players)]),
            tuple(self.scores),
            tuple(self.can_scout_and_show),
            tuple(self.history))

    @staticmethod
    def sample_from_info_state(info_state: InformationState) -> GameState:
        # Static factory method to create a GameState consistent with the
        # provided InformationState. Also returns the cardinality (how many
        # unique GameStates there are).
        # This function can be used to run ISMCTS explorations, i.e. sample many
        # game states and run simulations on each to get an aggregate tree.
        game_state = GameState(info_state.num_players, info_state.dealer)
        game_state.current_player = info_state.current_player
        game_state.scout_benefactor = info_state.scout_benefactor
        game_state.table = list(info_state.table)
        game_state.scores = list(info_state.scores)
        game_state.can_scout_and_show = list(info_state.can_scout_and_show)
        game_state.history = list(info_state.history)
        game_state.initial_flip_executed = True
        game_state.finished = False
        game_state.hands = [[] for _ in range(game_state.num_players)]

        # 1. Replay the recorded moves starting on a deck of Nones; this should
        #    give us a final deck with some known cards in the opponents hands.
        (partial_hands, removed_cards) = _simulate(
            info_state.num_players, info_state.dealer, game_state.history)

        # 2. Fill in the information about our own hand we know.
        partial_hands[info_state.current_player] = list(info_state.hand)

        # 3. Generate a random hand, and flatten it (get rid of assignments).
        # NB cards in random_deck are not normalized (ie some may have been
        # flipped).
        random_deck = _generate_hands(
            game_state.num_players)
        random_deck = [card for hand in random_deck for card in hand]
        # 4. Remove all cards in the partial hand and on the table.
        normalized_partial_hands = [_normalize_card(
            c) for h in partial_hands for c in h if c is not None]
        normalized_table = [_normalize_card(c) for c in info_state.table]
        random_deck = [
            c for c in random_deck if not _normalize_card(c) in normalized_partial_hands
            and not _normalize_card(c) in normalized_table]
        # 5. Remove cards that are not in the game anymore. NB both return values
        #    use normalized cards.
        normalized_removed_cards = [_normalize_card(c) for c in removed_cards]
        random_deck = [c for c in random_deck if not _normalize_card(
            c) in normalized_removed_cards]

        # 6. Distribute the remaining cards in random_deck across the players.
        card_index = 0
        for p in range(info_state.num_players):
            for i in range(len(partial_hands[p])):
                if not partial_hands[p][i]:
                    partial_hands[p][i] = random_deck[card_index]
                    card_index += 1

        game_state.hands = cast(list[list[Card]], partial_hands)
        assert card_index == len(random_deck)        
        return game_state

    def _scout(self, m: Scout) -> Card:
        hand = self.hands[self.current_player]
        if m.first:
            card = self.table[0]
            scouted_card = card
            self.table = self.table[1:]
        else:
            card = self.table[-1]
            scouted_card = card
            self.table = self.table[:-1]
        if m.flip:
            card = (card[1], card[0])
        hand.insert(m.insertPos, card)
        self.scores[self.scout_benefactor] += 1
        self.scores[self.current_player] -= 1
        return scouted_card

    def _show(self, m: Show) -> tuple[list[Card], list[Card]]:
        hand = self.hands[self.current_player]
        self.scores[self.current_player] += len(self.table) + m.length
        shown_cards = hand[m.startPos:m.startPos + m.length]
        removed_cards = self.table[:]
        self.table = shown_cards
        self.hands[self.current_player] = hand[:m.startPos] + \
            hand[m.startPos + m.length:]
        self.scout_benefactor = self.current_player
        return shown_cards, removed_cards
