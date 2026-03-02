from typing import Callable
from common import StateAndScoreRecord, Score
import torch
import torch.nn as nn
from featurize_numpy import _featurize_single, NORM_MEANS, NORM_STDS


def featurize(records: list[StateAndScoreRecord]
              ) -> tuple[torch.Tensor, torch.Tensor]:
    # Featurizes all the StateAndScoreRecords, does some hardcoded mean/std
    # normalization, and returns a tuple of input & output tensor.
    feature_output_lists = [_featurize_single(r) for r in records]
    inputs = torch.tensor(feature_output_lists)
    outputs = torch.tensor([r.rollout_score for r in records])
    norm_means = torch.from_numpy(NORM_MEANS)
    norm_stds = torch.from_numpy(NORM_STDS)
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

