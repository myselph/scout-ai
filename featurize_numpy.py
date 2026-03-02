"""
Pure-numpy featurization for the SimplePolicyNet / SimpleAgent architecture.
No PyTorch dependency - safe to use in environments where torch is unavailable.

The torch-based training code in neural_value_function.py imports _featurize_single
and the normalization constants from here to avoid duplication.
"""
import numpy as np
from common import StateAndScoreRecord, Util


def _featurize_single(record: StateAndScoreRecord) -> list[float]:
    # Converts a single StateAndScoreRecord into unnormalized features.
    # Do not use this directly - it doesn't take care of normalization.
    # Use featurize() instead - it will be more efficient, too.
    num_players = [len(record.num_cards)]
    num_cards_rot = record.num_cards[record.my_player:] + \
        record.num_cards[:record.my_player]
    num_cards = [0.0] * 5
    num_cards[:len(num_cards_rot)] = num_cards_rot
    scores_rot = record.scores[record.my_player:] + \
        record.scores[:record.my_player]
    scores = [0.0] * 5
    scores[:len(scores_rot)] = scores_rot
    can_scout_and_show_rot = record.can_scout_and_show[record.my_player:] + \
        record.can_scout_and_show[:record.my_player]
    can_scout_and_show = [0.0] * 5
    can_scout_and_show[:len(can_scout_and_show_rot)] = can_scout_and_show_rot
    # The value the net should predict is the difference of the player's
    # score and the mean opponent score when the game is over. I suspect adding
    # the current difference may help.
    # NB we should divide the sum of opponent scores by the number of opponent
    # players, not the number of players; but changing features requires retraining,
    # and I think it won't affect performance much or at all, so keeping it as is.
    cur_score_diff = [scores_rot[0] - sum(scores_rot[1:])/num_players[0]]

    # Table features
    table_values = [c[0] for c in record.table]
    table_features = [
        1.0 if Util.is_group(table_values) else 0.0,
        len(table_values),
        max(table_values) if table_values else 0.0
    ]

    # Hand features
    hand_values = list(record.hand)
    runs = Util._find_runs(hand_values)
    run_lengths = [len(r) for r in runs]
    groups = Util._find_groups(hand_values)
    group_lengths = [len(g) for g in groups]
    hand_features = [
        len(hand_values),
        len(runs),
        len(groups),
        max([r[1] for r in runs]) if runs else 0,
        max([g[1] for g in groups]) if groups else 0,
        sum([r[1] for r in runs]) / len(runs) if runs else 0,
        sum([g[1] for g in groups]) / len(groups) if groups else 0,
    ] + [hand_values.count(i) for i in range(1, 11)] \
      + [run_lengths.count(i) for i in range(2, 11)] \
      + [group_lengths.count(i) for i in range(2, 11)]

    shows = Util.find_shows(hand_values, table_values)
    show_groups = [s for s in shows if Util.is_group(
        hand_values[s.startPos:s.startPos + s.length])]
    show_runs = [s for s in shows if Util.is_run(
        hand_values[s.startPos:s.startPos + s.length])]
    show_features = [
        max([s.length for s in show_groups]) if show_groups else 0,
        max([s.length for s in show_runs]) if show_runs else 0
    ]

    return (
        num_players +
        num_cards +
        scores +
        cur_score_diff +
        can_scout_and_show +
        table_features +
        hand_features +
        show_features)


# Normalization constants - kept here so both torch and numpy code can share them.
NORM_MEANS = np.array([
    2.5,                              # num players
    5, 5, 5, 5, 5,                    # num cards
    -1, -1, -1, -1, -1,               # scores
    0.5, 0.5, 0.5, 0.5, 0.5,          # scout & show
    4,                                # cur score diff
    0.5, 2, 5,                        # table stuff
    8, 2, 1, 4, 3, 3, 3,              # hand stuff
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    # hand val hist
    0, 0, 0, 0, 0, 0, 0, 0, 0,       # run len hist
    0, 0, 0, 0, 0, 0, 0, 0, 0,       # group len hist
    0, 0,                             # shows
], dtype=np.float32)

NORM_STDS = np.array([
    2,
    2, 2, 2, 2, 2,
    5, 5, 5, 5, 5,
    0.5, 0.5, 0.5, 0.5, 0.5,
    3,
    0.5, 1, 3,
    2, 2, 1, 3, 3, 2, 3,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1,
], dtype=np.float32)


def featurize(records: list[StateAndScoreRecord]) -> np.ndarray:
    """
    Featurize a list of StateAndScoreRecords into a normalized [N, 57] float32 array.
    """
    raw = [_featurize_single(r) for r in records]
    x = np.array(raw, dtype=np.float32)
    return (x - NORM_MEANS) / NORM_STDS
