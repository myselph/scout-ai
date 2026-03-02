"""
Numpy-only inference player for the trained SimplePolicyNet.

No PyTorch dependency - safe to deploy in size-constrained environments (e.g. Vercel).
Weights must first be exported to .npz format via export_simple_weights.py.

Usage from another project:
    import sys; sys.path.insert(0, "/path/to/scout-ai")
    from numpy_neural_player import load_default_player
    player = load_default_player()
    move = player.select_move(info_state)
"""
import os
from math import inf

import numpy as np

from common import InformationState, Move, StateAndScoreRecord
from featurize_numpy import featurize
from players import PlanningPlayer


class NumpySimplePolicyNet:
    """Numpy inference-only implementation of SimplePolicyNet (Linear-ReLU x2, Linear)."""

    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.w0 = data["net.0.weight"]  # [128, 57]
        self.b0 = data["net.0.bias"]    # [128]
        self.w2 = data["net.2.weight"]  # [64, 128]
        self.b2 = data["net.2.bias"]    # [64]
        self.w4 = data["net.4.weight"]  # [1, 64]
        self.b4 = data["net.4.bias"]    # [1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: [N, 57] float32 - batch of N normalized feature vectors
        returns logits: [N] float32
        """
        x = np.maximum(0.0, x @ self.w0.T + self.b0)  # [N, 128]
        x = np.maximum(0.0, x @ self.w2.T + self.b2)  # [N, 64]
        x = x @ self.w4.T + self.b4                    # [N, 1]
        return x.squeeze(1)                             # [N]


class NumpySimplePlayer(PlanningPlayer):
    """
    A Player that uses a trained NumpySimplePolicyNet for move selection.
    No PyTorch dependency.

    Featurizes all post-move InformationStates in a single batch,
    runs the network, samples from the resulting distribution.
    flip_hand() falls back to the heuristic inherited from PlanningPlayer.
    """

    def __init__(self, net: NumpySimplePolicyNet):
        self.net = net

    def select_move(self, info_state: InformationState) -> Move:
        moves, post_states = info_state.post_move_states()
        ssrs = [
            StateAndScoreRecord(
                s.current_player,
                tuple(c[0] for c in s.hand),
                s.table,
                s.num_cards,
                s.scores,
                s.can_scout_and_show,
                inf)
            for s in post_states
        ]
        features = featurize(ssrs)            # [N, 57]
        logits = self.net.forward(features)   # [N]
        # Numerically stable softmax then sample
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action_idx = int(np.random.choice(len(moves), p=probs))
        return moves[action_idx]


def load_default_player() -> NumpySimplePlayer:
    """Load the best bundled agent (agent 0, skill 1.95) from the package weights dir."""
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(
        base_dir, "simple_agent_weights", "simple_agent_0_it_79_skill_1.95.npz")
    return NumpySimplePlayer(NumpySimplePolicyNet(weights_path))


def load_player(weights_path: str) -> NumpySimplePlayer:
    """Load a player from an explicit .npz weights path."""
    return NumpySimplePlayer(NumpySimplePolicyNet(weights_path))


if __name__ == "__main__":
    import random
    from game_state import GameState

    print("Loading player...")
    player = load_default_player()

    print("Setting up game state...")
    rng_seed = 42
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    game = GameState(num_players=3, dealer=0)
    game.maybe_flip_hand([player.flip_hand, player.flip_hand, player.flip_hand])
    info_state = game.info_state()

    print(f"  Hand ({len(info_state.hand)} cards): {[c[0] for c in info_state.hand]}")
    print(f"  Table ({len(info_state.table)} cards): {[c[0] for c in info_state.table]}")

    moves, post_states = info_state.post_move_states()
    print(f"  Legal moves: {len(moves)}")

    # Run a batched forward pass and show distribution
    from featurize_numpy import featurize as np_featurize
    ssrs = [
        StateAndScoreRecord(
            s.current_player, tuple(c[0] for c in s.hand),
            s.table, s.num_cards, s.scores, s.can_scout_and_show, inf)
        for s in post_states
    ]
    features = np_featurize(ssrs)
    logits = player.net.forward(features)
    logits_shifted = logits - logits.max()
    probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
    print(f"  Features shape: {features.shape}")
    print(f"  Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  Top-3 move probs: {sorted(probs, reverse=True)[:3]}")

    move = player.select_move(info_state)
    print(f"  Selected move: {move}")
    print("Inference test passed!")
