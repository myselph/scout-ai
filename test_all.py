# Tests for scout
from dataclasses import dataclass
from abc import abstractmethod
from game_state import GameState
from common import Util, Scout, Show, ScoutAndShow, InformationState
from ismcts_player import IsmctsPlayer


def test_utils():
    # Tests for the group and run finding functions.
    assert not Util._find_groups([])
    assert not Util._find_groups([1])
    assert Util._find_groups([1], min_length=1) == {(0, 0)}
    assert Util._find_groups([1, 1]) == {(0, 1)}
    assert Util._find_groups([1, 1, 1]) == {(0, 2), (0, 1), (1, 2)}
    assert Util._find_groups([1, 1, 1], min_length=3) == {(0, 2)}
    assert Util._find_groups([1, 1, 2]) == {(0, 1)}
    assert Util._find_groups([2, 1, 1]) == {(1, 2)}
    assert Util._find_groups([2, 1, 1, 1, 1, 3, 5, 5, 2, 2, 4]) == {
        (1, 4), (6, 7), (8, 9), (1, 2), (2, 3), (3, 4), (1, 3), (2, 4)}

    # Tests for runs
    assert not Util._find_runs([])
    assert not Util._find_runs([1])
    assert Util._find_runs([1], min_length=1) == {(0, 0)}
    assert Util._find_runs([1, 2]) == {(0, 1)}
    assert Util._find_runs([1, 2, 3]) == {(0, 2), (0, 1), (1, 2)}
    assert Util._find_runs([1, 2, 3], min_length=3) == {(0, 2)}
    assert Util._find_runs([1, 2, 4]) == {(0, 1)}
    assert Util._find_runs([4, 2, 1]) == {(1, 2)}
    assert Util._find_runs([8, 1, 2, 3, 4, 7, 3, 2, 4, 5, 9]) == {
        (1, 4), (6, 7), (8, 9), (1, 2), (2, 3), (3, 4), (1, 3), (2, 4)}

    # Tests for find_shows.
    assert Util.find_shows([1, 1, 2], []) == {Show(
        0, 1), Show(1, 1), Show(2, 1), Show(0, 2), Show(1, 2)}
    assert Util.find_shows([1, 1, 2], [1]) == {
        Show(2, 1), Show(0, 2), Show(1, 2)}
    assert Util.find_shows([1, 1, 2], [3]) == {Show(0, 2), Show(1, 2)}
    assert not Util.find_shows([1, 1, 2], [1, 1])
    assert Util.find_shows([1, 1, 2], [2, 3]) == {Show(0, 2)}
    assert Util.find_shows([5, 4, 3, 2], [6, 6, 6]) == {Show(0, 4)}
    assert Util.find_shows([5, 4, 3, 2], [6, 5, 4]) == {Show(0, 4)}
    assert Util.find_shows([5, 4, 3, 2], [1, 2, 3, 4]) == {Show(0, 4)}
    assert Util.find_shows([5, 4, 3, 2], [4, 3, 2, 1]) == {Show(0, 4)}

    game_state = GameState(5, 0)
    # Tests for is_group, is_run.
    assert Util.is_group([])
    assert Util.is_group([1])
    assert Util.is_group([1, 1])
    assert not Util.is_group([1, 2])
    assert not Util.is_group([1, 1, 2])
    assert Util.is_run([])
    assert Util.is_run([1])
    assert Util.is_run([1, 2])
    assert Util.is_run([1, 2, 3])
    assert Util.is_run([3, 2, 1])
    assert not Util.is_run([1, 1])
    assert not Util.is_run([1, 3])
    assert not Util.is_run([1, 2, 4])
    assert not Util.is_run([1, 3, 3])

    # Tests for is_move_valid.
    # Scouts
    assert not Util.is_move_valid([(1, 2)], [], False, Scout(False, False, 0))
    assert not Util.is_move_valid([(1, 2)], [], False, Scout(False, False, 1))
    table = [(2, 9), (3, 4), (4, 1)]
    assert Util.is_move_valid([(5, 6)], table, False, Scout(True, True, 0))
    assert Util.is_move_valid([(5, 6)], table, False, Scout(True, True, 1))
    assert not Util.is_move_valid([(5, 6)], table, False, Scout(True, True, 2))
    # Shows - groups vs. run
    assert Util.is_move_valid(
        [(1, 6), (1, 7), (1, 8)], table, False, Show(0, 3))
    assert Util.is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                              table, False, Show(0, 3))
    assert Util.is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                              table, False, Show(0, 4))
    assert Util.is_move_valid([(1, 6), (1, 7), (1, 8), (1, 9)],
                              table, False, Show(1, 3))
    assert not Util.is_move_valid(
        [(1, 6), (1, 7), (1, 8), (1, 9)], table, False, Show(1, 2))
    # Shows - runs vs. run
    assert not Util.is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(0, 3))
    assert Util.is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                               (1, 9)], table, False, Show(1, 3))
    assert Util.is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                               (1, 9)], table, False, Show(0, 4))
    assert not Util.is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(1, 4))
    assert Util.is_move_valid([(2, 8), (3, 6), (4, 7), (5, 8),
                               (1, 9)], table, False, Show(0, 4))
    assert not Util.is_move_valid(
        [(2, 8), (3, 6), (4, 7), (5, 8), (1, 9)], table, False, Show(2, 3))

    # Scout and Show.
    hand = [(3, 1), (5, 8), (6, 9)]
    # 3,4,5,6 wins over 2,3
    assert Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 4)))
    assert not Util.is_move_valid(hand, table, False, ScoutAndShow(
        Scout(False, False, 1), Show(0, 4)))
    # 3,4,5 wins over 2,3
    assert Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 3)))
    # 3,4 wins over 2,3
    assert Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, False, 1), Show(0, 2)))
    # 1, 3 is not valid
    assert not Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(False, True, 1), Show(0, 2)))
    # 2,3 loses to 3,4
    assert not Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, False, 0), Show(0, 2)))
    # 5,6 wins over 3,4
    assert Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, False, 0), Show(2, 2)))
    # illegal to play 3,5,6
    assert not Util.is_move_valid(hand, table, True, ScoutAndShow(
        Scout(True, True, 0), Show(1, 3)))


def test_InformationState():
    # InformationState tests - specifically, the valid move generator.
    # 2 cards in hand, none on table -> only Shows.
    hand = ((4, 7), (5, 8))
    info_state = InformationState(
        5, 0, 0, -1, hand, (), (2, 2, 2, 2, 2), (0, 0, 0, 0, 0),
        (True, True, True, True, True), ())
    assert set(
        info_state.possible_moves()) == {
        Show(
            0, 1), Show(
                0, 2), Show(
                    1, 1)}

    # 2 cards in hand, 1 on table -> 6 Scouts, 3 Shows (4, 5, (4,5)),
    # For Scout & Shows:
    # There's a lot of coalescing going on inside possible_moves, and it's
    # best to just enumerate the possible options:
    # - single card shows:
    #   - scout 3, end up with 3,4; 4,3; 3,5; 5,3; 4,5 -> 5
    #   - scout 1, end up with 1,4; 4,1; 1,5; 5,1; 4,5 -> 5
    # - two card shows:
    #   - show 3,4 or 4,3; end up with 5 -> 2
    #   - show 4,5, end up with 1 or 3 -> 2
    # - three card show - end up with nothing -> 1
    # So overall, 15 S&S moves.
    info_state = InformationState(
        5, 0, 0, -1, hand, ((3, 1),), (2, 2, 2, 2, 2), (0, 0, 0, 0, 0),
        (True, True, True, True, True), ())
    moves = info_state.possible_moves()
    assert 6 == len([m for m in moves if isinstance(m, Scout)])
    assert 3 == len([m for m in moves if isinstance(m, Show)])
    assert 15 == len([m for m in moves if isinstance(m, ScoutAndShow)])

    # Expected: 12 scout moves; one show move (pair); and for S&S:
    # Again, lots of coalescing, let's enumerate possible end states:
    # - single card shows:
    #   - scout 2 -> hand 2,4; 4,2; 2,5; 5,2 -> 4
    #   - scout 1 of (2,1) -> hand 1,4; 4,1; 1,5; 5,1 -> 4
    #   - scout 1 of (3,1) -> hand 1,4; 4,1; 1,5; 5,1 -> 4.
    #   - scout 3 -> hand 3,4; 4,3; 3,5; 5,3; 4,5 -> 5
    # - two card shows:
    #   - scout 2 -> show 4,5 -> 1
    #   - scout 3 -> show 4,5; 3,4; 4,3 -> 3
    #   - scout 1 -> show 4,5 -> 2 (we can scout two 1s)
    # - three card shows: 1
    # Thus, we expect 24 Scout and Show moves.
    info_state = InformationState(
        5, 0, 0, 0, hand, ((2, 1), (3, 1)), (2, 2, 2, 2, 2), (0, 0, 0, 0, 0),
        (True, True, True, True, True), ())
    moves = info_state.possible_moves()
    assert 12 == len([m for m in moves if isinstance(m, Scout)])
    assert 1 == len([m for m in moves if isinstance(m, Show)])
    assert 24 == len([m for m in moves if isinstance(m, ScoutAndShow)])
    assert 17 == len([m for m in moves if isinstance(
        m, ScoutAndShow) and m.show.length == 1])
    assert 6 == len([m for m in moves if isinstance(
        m, ScoutAndShow) and m.show.length == 2])
    assert 1 == len([m for m in moves if isinstance(
        m, ScoutAndShow) and m.show.length == 3])


def test_GameState():
    # GameState tests. Should probably add test c'tor to inject my own decks;
    # for now, just test scouting and showing and that scoring works.
    game_state = GameState(5, 1)
    assert not game_state.table
    assert not game_state.initial_flip_executed
    game_state.maybe_flip_hand([lambda _: False] * 5)
    assert game_state.scores[1] == -9
    game_state.move(Show(0, 1))
    assert game_state.scores[1] == -8
    game_state.move(Scout(True, False, 0))
    assert game_state.scores[1] == -7
    assert game_state.scores[2] == -10

    # Test the GameState generator. Make a couple of moves, create a
    # determinization, and ensure it is a valid representation.
    game_state = GameState(5, 0)
    card1 = game_state.hands[0][0]
    game_state.maybe_flip_hand([lambda _: False] * 5)
    game_state.move(Show(0, 1))
    game_state.move(Scout(True, False, 0))
    card2 = game_state.hands[2][0]
    game_state.move(Show(0, 1))
    game_state.move(Scout(True, True, 3))
    card3 = game_state.hands[4][4]
    game_state.move(Show(4, 1))
    game_state.move(ScoutAndShow(Scout(True, True, 6), Show(6, 1)))
    game_state.move(Scout(True, True, 2))

    info_state = game_state.info_state()
    determinization = GameState.sample_from_info_state(info_state)
    assert [len(h) for h in determinization.hands] == [len(h)
                                                       for h in game_state.hands]
    assert not determinization.table
    assert card1 == determinization.hands[1][0]
    assert (card2[1], card2[0]) == determinization.hands[3][3]
    assert card3 == determinization.hands[1][2]
    assert info_state.hand == tuple(determinization.hands[2])


def test_IsmctsPlayer():
    game_state = GameState(5, 1)
    game_state.maybe_flip_hand([lambda _: True] * 5)
    ismcts_player = IsmctsPlayer(5, 2)
    game_state.move(Show(0, 1))
    move = ismcts_player.select_move(game_state.info_state())
    game_state.move(move)
