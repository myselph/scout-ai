from collections.abc import Sequence
import time
from common import Move, Card, InformationState, Player, Util, StateAndScoreRecord, Score
import neural_value_function
from players import RandomPlayer
from game_state import GameState
from typing import Self, Callable
from math import inf, sqrt, log
from dataclasses import dataclass
import random
import pickle


def save_records(records: list[StateAndScoreRecord], filename):
    with open(filename, "wb") as f:
        pickle.dump(records, f)


@dataclass
class Node:
    possible_moves: tuple[Move, ...]
    untried_moves: list[Move]  # subset of possible_moves
    N: dict[Move, int]  # action -> visit count
    W: dict[Move, Score]  # action -> sum of points
    # action -> information_state hash -> Node
    children: dict[Move, dict[int, Self]]
    # If set, call this to record an update to W, then unset.
    record_score: Callable[[float], None] | None


def sample_move_uct(node: Node) -> Move:
    # Only call this once node is fully expanded (all actions have been tried.)
    max_uct_val = -inf
    best_move = None
    num_visits = sum(node.N.values())
    return max(
        node.possible_moves,
        key=lambda m: node.W[m] /
        node.N[m] + sqrt(2 * log(num_visits) / node.N[m]))


def backprop(path: list[tuple[Node, Move]], score: Score):
    for (node, move) in path:
        node.N[move] += 1
        node.W[move] += score
        # This callback is set on expansion, and is cleared after the first
        # call so it's only called once during expansion.
        # TODO: There are probably easier ways to do this than storing the cb
        # in the node.
        if node.record_score:
            node.record_score(score)
            node.record_score = None


def backprop_from_gamestate(path: list[tuple[Node, Move]],
                            game_state: GameState, my_index: int):
    my_score = game_state.scores[my_index]
    avg_opp_score = (sum(game_state.scores) - my_score) / \
        (len(game_state.scores) - 1)
    score = my_score - avg_opp_score
    backprop(path, score)


def gen_ssr(
        my_index: int,
        info_state: InformationState) -> StateAndScoreRecord:
    hand_values = tuple(c[0] for c in info_state.hand)
    return StateAndScoreRecord(my_index, hand_values, info_state.table,
                               info_state.num_cards, info_state.scores,
                               info_state.can_scout_and_show, inf)


def gen_score_recorder(expansion_recorder: Callable[[
                       StateAndScoreRecord], None],
                       info_state: InformationState,
                       my_index: int) -> Callable[[Score], None]:
    # Called from backprop() to update a previously generated
    # StateAndStoreRecord with the score, and record the resulting record.
    record = gen_ssr(my_index, info_state)

    def update_score(s):
        record.rollout_score = s
        return record
    return lambda s: expansion_recorder(update_score(s))


def ismcts(node: Node,
           game_state: GameState,
           players: list[Player],
           my_index: int,
           expansion_recorder: Callable[[StateAndScoreRecord],
                                        None] | None = None,
           value_fn: Callable[[list[StateAndScoreRecord]],
                              list[Score]] | None = None):
    # 1. Selection phase: Descend while nodes are expanded, chosing actions
    # via UCT.
    path: list[tuple[Node, Move]] = []
    while not node.untried_moves:
        move = sample_move_uct(node)
        path.append((node, move))
        game_state.move(move)
        if game_state.is_finished():
            backprop_from_gamestate(path, game_state, my_index)
            return
        for p in players[my_index + 1:] + players[:my_index]:
            game_state.move(p.select_move(game_state.info_state()))
            if game_state.is_finished():
                backprop_from_gamestate(path, game_state, my_index)
                return
        info_state = game_state.info_state()
        hash = info_state.__hash__()
        # NB a move having been tried does not imply that a child node was
        # created (the game can have ended before we reached a new info_state);
        # therefore, we need to be careful accessing node.children[move]
        if move in node.children and hash in node.children[move]:
            node = node.children[move][hash]
        else:
            # Expand.
            possible_moves = info_state.possible_moves()
            new_node = Node(
                possible_moves,
                list(possible_moves),
                {},
                {},
                {},
                gen_score_recorder(
                    expansion_recorder,
                    info_state,
                    my_index) if expansion_recorder else None)
            if move not in node.children:
                node.children[move] = {hash: new_node}
            else:
                node.children[move][hash] = new_node
            node = new_node

            # If we use a value function, we stop after expansions and
            # backprop.
            if value_fn:
                backprop(path, value_fn([gen_ssr(my_index, info_state)])[0])
                return

    # This is where things get slightly confusing. There should be at most one
    # expansion per roll-out (ie per invocation of this function).
    # There are three code paths that lead to this point; node can refer to one
    # of three situations:
    # 1. node == the root node, either
    #    a) first call to this function (counts as expansion), or
    #    b) root node not fully expanded -> try action, expand.
    # 2. An expansion in the above for loop - ie the node was just created in
    #    the above for-loop because we tried a previously explored action but
    #    added a new node for it (because we found a new info_state, or the
    #    previous action explorations ended in the finish state before it was
    #    our player's turn again.) Same as for #1: We have already expanded.
    # 3. A previously visited node with untried moves -> we pick an untried
    #    move and try to expand (no guarantee this will happen because we might
    #    hit the end of the game first.
    # In cases 1a and 2, we do not want to expand again. We can tell by looking
    # at N[]. Just roll out.
    # In cases 1a and 3, we do want to expand.
    # In any case, pick a random untried move and make it.
    should_expand = len(node.N) > 0
    move = random.choice(node.untried_moves)
    path.append((node, move))
    node.N[move] = 0
    node.W[move] = 0
    node.untried_moves.remove(move)
    game_state.move(move)

    # Should we try to expand? Let the other players move, add a new node if
    # the game didn't finish, and make a random move.
    if should_expand:
        if game_state.is_finished():
            backprop_from_gamestate(path, game_state, my_index)
            return
        for p in players[my_index + 1:] + players[:my_index]:
            game_state.move(p.select_move(game_state.info_state()))
            if game_state.is_finished():
                backprop_from_gamestate(path, game_state, my_index)
                return
        info_state = game_state.info_state()
        hash = info_state.__hash__()
        possible_moves = info_state.possible_moves()

        node.children[move] = {}
        node.children[move][hash] = Node(
            tuple(possible_moves), list(possible_moves), {}, {}, {},
            gen_score_recorder(
                expansion_recorder,
                info_state,
                my_index) if expansion_recorder else None)
        node = node.children[move][hash]
        # If we use a value function, we stop after expansions and backprop.
        if value_fn:
            backprop(path, value_fn([gen_ssr(my_index, info_state)])[0])
            return

        move = random.choice(node.untried_moves)
        path.append((node, move))
        node.N[move] = 0
        node.W[move] = 0
        node.untried_moves.remove(move)
        game_state.move(move)

    # Play the game till the end and return the player's score.
    if value_fn:
        # The only way to end up in this branch is if this is the first call to
        # ismcts (ie an empty root node) with a value_fn - if this wasn't the
        # first call, we'd expanded above and handled the value_fn call there.
        if game_state.is_finished():
            backprop_from_gamestate(path, game_state, my_index)
            return
        for p in players[my_index + 1:] + players[:my_index]:
            game_state.move(p.select_move(game_state.info_state()))
            if game_state.is_finished():
                backprop_from_gamestate(path, game_state, my_index)
                return
        info_state = game_state.info_state()
        backprop(path, value_fn([gen_ssr(my_index, info_state)])[0])

    p = (my_index + 1) % len(players)
    while not game_state.is_finished():
        game_state.move(players[p].select_move(game_state.info_state()))
        p = (p + 1) % len(players)

    # 4. Backprop phase - Increment N and possible W for all nodes visited.
    backprop_from_gamestate(path, game_state, my_index)


@dataclass
class IsmctsStats:
    # We use floats to support averages
    cards_left: float
    num_moves: float
    num_children: float
    max_move_visits: float
    max_move_children: float


class IsmctsPlayer(Player):
    # The number of roll-outs we perform.
    # On reasonable values:
    #   I measured the win rate as a function of number of simulations When
    #   playing against GreedyShowPlayerWithFlip. We get ~0 at 40, about 50% at
    #   120 rollouts, and 75% at 250; unsurprisingly it follows a logarithm
    #   curve. Using randomly selected moves,
    # On timing:
    #   10 games, 5 rounds each, 10 roll-outs per select_move() takes 30s, when
    #   playing against 4 GreedyShowPlayerWithFlip.
    #   So a single round with one roll-out takes about 60ms.
    #   100 games with 100 roll-outs thus take ~3000s.
    _num_simulations: int
    # List of simulated players.
    _players: list[Player]
    # Cached trees for the action last taken.
    # We hardly ever hit a cached tree, and when we do, it has a visit count
    # of 1 so we save ourselves a single roll-out which is hardly worth it.
    # But it doesn't hurt performance and might help in future scenarios when
    # the game state cardinality is significantly reduced, so keeping it for
    # now
    _cached_trees: dict[int, Node]
    # If >0, select_move will finish after exceeding this many seconds.
    _move_time_limit_seconds: int
    # If not None or "", record expansions (for neural net training).
    _expansion_file_prefix: str
    _record_expansions: bool
    _expansions: list[StateAndScoreRecord]
    _num_expansion_files: int

    _value_fn: Callable[[list[StateAndScoreRecord]], list[Score]] | None

    def __init__(
            self,
            num_players: int,
            num_simulations: int = 2_000,
            generate_player_fn: Callable[[],
                                         Player] = lambda: RandomPlayer(),
            move_time_limit_seconds: int = 0,
            expansion_file_prefix: str = "",
            use_value_fn: bool = False):
        self._num_simulations = num_simulations
        self._players = [generate_player_fn() for _ in range(num_players)]
        self._cached_trees = {}
        self._move_time_limit_seconds = move_time_limit_seconds
        self._expansion_file_prefix = expansion_file_prefix
        self._expansions = []
        if expansion_file_prefix:
            self._expansion_recorder = lambda r: self._expansions.append(r)
        else:
            self._expansion_recorder = None

        self._num_expansion_files = 0
        if use_value_fn:
            self._value_fn = neural_value_function.create_inference_model(
                'ismcts_value_function.pth')
        else:
            self._value_fn = None

    def flip_hand(self, hand: Sequence[Card]) -> bool:
        up_value = self._hand_value([h[0] for h in hand])
        down_value = self._hand_value([h[1] for h in hand])
        return up_value < down_value

    def select_move(self, info_state: InformationState) -> Move:
        start_time = time.time()
        possible_moves = info_state.possible_moves()
        my_index = info_state.current_player
        hash = info_state.__hash__()

        if hash in self._cached_trees:
            root = self._cached_trees[hash]
        else:
            root = Node(possible_moves, list(possible_moves), {}, {}, {},
                        gen_score_recorder(
                self._expansion_recorder,
                info_state,
                my_index) if self._expansion_recorder else None)

        for _ in range(self._num_simulations):
            game_state = GameState.sample_from_info_state(info_state)
            ismcts(root, game_state, self._players, my_index,
                   self._expansion_recorder, self._value_fn)
            if self._move_time_limit_seconds > 0 and time.time(
            ) - start_time > self._move_time_limit_seconds:
                break

        # Find the most visited move.
        move = max(root.N.keys(), key=lambda k: root.N[k])
        if move in root.children:
            self._cached_trees = root.children[move]
        else:
            self._cached_trees = {}

        if self._expansion_file_prefix and len(self._expansions) >= 10_000:
            save_records(
                self._expansions, f"{self._expansion_file_prefix}_{
                    self._num_expansion_files}.pkl")
            self._num_expansion_files += 1
            self._expansions.clear()
        return move
