from typing import Callable
from common import Util, StateAndScoreRecord, Score
import torch
import torch.nn as nn


def _featurize_single(record: StateAndScoreRecord) -> list[float]:
    # Converts a single StateAndScoreRecord into unnormalized features.
    # Do not use this directly - it doesn't take care of normalization.
    # Use the batched version instead - it will be more efficient, too.
    num_players = [len(record.num_cards)]
    num_cards_rot = record.num_cards[record.my_player:] + \
        record.num_cards[:record.my_player]
    num_cards = [0.0] * 5
    num_cards[:len(num_cards_rot)] = num_cards_rot
    scores_rot = record.scores[record.my_player:] + \
        record.scores[:record.my_player]
    scores = [0.0 * 5]
    scores[:len(scores_rot)] = scores_rot
    can_scout_and_show_rot = record.can_scout_and_show[record.my_player:] + \
        record.can_scout_and_show[:record.my_player]
    can_scout_and_show = [0.0] * 5
    can_scout_and_show[:len(can_scout_and_show_rot)] = can_scout_and_show_rot
    # The value the net should predict is the difference of the player's
    # score and the mean opponent score when the game is over. I suspect adding
    # the current difference may help.
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

    # TODO: Add num new groups, num new runs after best scout
    # Etc., in short how good moves are. Leaving this for now
    # to get started.
    return (
        num_players +
        num_cards +
        scores +
        cur_score_diff +
        can_scout_and_show +
        table_features +
        hand_features +
        show_features)


def featurize(records: list[StateAndScoreRecord]
              ) -> tuple[torch.Tensor, torch.Tensor]:
    # Featurizes all the StateAndScoreRecords, does some hardcoded mean/std
    # normalization, and returns a tuple of input & output tensor.
    feature_output_lists = [_featurize_single(r) for r in records]
    inputs = torch.tensor(feature_output_lists)
    outputs = torch.tensor([r.rollout_score for r in records])

    # Normalize inputs. We subtract the approximate mean, then divide by the
    # approximate std. I mostly decided on the manually based on known ranges;
    # eg for booleans (1 or 0) I always pick 0.5 and 1.0
    norm_means = torch.tensor([2.5, # num players
                               5, 5, 5, 5, 5,  # num cards
                               -1, -1, -1, -1, -1,  # scores
                               0.5, 0.5, 0.5, 0.5, 0.5,  # scout & show
                               4, # cur score diff. TODO: Update with mean
                               0.5, 2, 5,  # table stuff
                               8, 2, 1, 4, 3, 3, 3,  # hand stuff
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # hand val hist
                               0, 0, 0, 0, 0, 0, 0, 0, 0,  # run len hist
                               0, 0, 0, 0, 0, 0, 0, 0, 0,  # group len hist
                               0, 0])  # shows
    norm_stds = torch.tensor([2,
                              2, 2, 2, 2, 2,
                              5, 5, 5, 5, 5,
                              0.5, 0.5, 0.5, 0.5, 0.5,
                              3,
                              0.5, 1, 3,
                              2, 2, 1, 3, 3, 2, 3,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1])
    inputs = (inputs - norm_means) / norm_stds
    return inputs, outputs


class ScoutValueNet(torch.nn.Module):
    def __init__(self):
        super(ScoutValueNet, self).__init__()

        # Define the stack of layers
        self.layers = nn.Sequential(
            nn.Linear(57, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


def create_inference_model(
        filepath: str) -> Callable[[list[StateAndScoreRecord]], list[Score]]:
    model = ScoutValueNet()
    state_dict = torch.load(filepath, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return lambda in_list: model(featurize(in_list)[0]).squeeze(dim=1).tolist()

