# This file contains different agents for use in neural self-play.
# An agent is a wrapper around a neural policy and a (possibly shared) value
# network. We train the networks inside; at inference time, the agent is wrapped
# in a Player and used in tournaments.

import os
from collections.abc import Callable, Sequence
from math import inf
import torch
from torch import nn
from abc import ABC, abstractmethod

from .common import InformationState, Move, StateAndScoreRecord, Util
from . import neural_value_function
from .players import PlanningPlayer
from .featurize_numpy import NORM_MEANS as _NUMPY_SCALAR_MEANS, NORM_STDS as _NUMPY_SCALAR_STDS


class Agent(ABC):
    # This is a base class for all agents. Its interface is designed to provide
    # a common API across different policy+value net architectures (notably,
    # fixed vs variable length inputs, different featurizations, etc.) and to
    # be usable for both inference and training operations.

    # policy must have a forward() method that takes in:
    # - Fixed length: a [B, D] float tensor (B moves, D features), returns [B] logits.
    # - Variable length: a [B, L, C] float tensor (B moves, L padded sequence length,
    #   C channels) and a [B, L] boolean padding mask, returns [B] logits.
    #   The C channels are: 0=token ID, 1=within-span card position (1-indexed, 0=none),
    #   2=segment (0=none, 1=table, 2=hand), 3=card ordinal value (1-10, 0=non-card),
    #   4=normalized score value (float, 0.0 for non-score tokens).
    policy: nn.Module
    # value_fn has the same interface as policy, but returns [B] value predictions.
    value_fn: nn.Module

    def __init__(
            self,
            policy: nn.Module,
            policy_optim: torch.optim.Optimizer,
            value_fn: nn.Module,
            value_fn_optim: torch.optim.Optimizer,
            is_fixed_length: bool,
            device: torch.device = torch.device("cpu")):
        self.device = device
        self.policy = policy.to(device)
        self.policy_optim = policy_optim
        self.value_fn = value_fn.to(device)
        self.value_optim = value_fn_optim
        self.is_fixed_length = is_fixed_length

    @abstractmethod
    def featurize(
            self, info_states: tuple[InformationState, ...]) -> tuple[torch.Tensor, ...]:
        pass

    def invoke_net(
            self,
            net: nn.Module,
            states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Invokes policy or value_fn with appropriate batching.
        # Fixed length: each state is [D]; stacked to [B, D].
        # Variable length: each state is [L_i, C]; padded to [B, L_max, C],
        # with a [B, L_max] boolean mask marking padding positions.
        assert states, "Must have at least one state"
        if self.is_fixed_length:
            batch = torch.stack(states).to(self.device)  # [B, D]
            return net(batch)                             # [B]
        else:
            batch = torch.nn.utils.rnn.pad_sequence(
                list(states), batch_first=True, padding_value=PADDING_IDX).to(self.device)
            padding_mask = batch[:, :, 0] == PADDING_IDX  # [B, L_max]
            return net(batch, padding_mask)                # [B]

    def compute_logprobs(
            self,
            post_move_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Takes a B-tuple of [D] tensors (fixed length) or [L, C] tensors
        # (variable length), and returns logprobs of shape [B].
        return torch.log_softmax(
            self.invoke_net(
                self.policy,
                post_move_states), dim=-1)  # [B]

    def compute_logprobs_and_entropy_batched(
            self,
            post_move_states_list: list[tuple[torch.Tensor, ...]],
            action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Batched version of compute_logprobs for use during training.
        # Takes a list of A tuples, each containing B_i tensors of shape [D]
        # (fixed length) or [L_ij, C] (variable length).
        # action_indices: [A] - the chosen action index for each of the A samples.
        # Returns:
        #   new_logprobs: [A] - log prob of the chosen action per sample
        #   entropies:    [A] - entropy of the policy distribution per sample
        #
        # Rather than A separate forward passes (one per sample), we concatenate
        # all B_i action tensors into a single batch of size sum(B_i), run one
        # forward pass, then split and softmax per segment. For the fixed-length
        # case this requires no padding at all. For the variable-length case,
        # sequences are padded to the global max length across all A*B_i actions,
        # which in practice is close to the per-sample max since game states
        # within a microbatch have similar lengths.
        B_i_list = [len(states) for states in post_move_states_list]
        all_states = [s for states in post_move_states_list for s in states]

        if self.is_fixed_length:
            batch = torch.stack(all_states).to(self.device)  # [sum(B_i), D]
            logits = self.policy(batch)                       # [sum(B_i)]
        else:
            batch = torch.nn.utils.rnn.pad_sequence(
                all_states, batch_first=True, padding_value=PADDING_IDX).to(self.device)
            padding_mask = batch[:, :, 0] == PADDING_IDX       # [sum(B_i), L_max]
            logits = self.policy(batch, padding_mask)           # [sum(B_i)]

        per_sample_logits = torch.split(logits, B_i_list)

        new_logprobs = []
        entropies = []
        for i, sample_logits in enumerate(per_sample_logits):
            logprobs = torch.log_softmax(sample_logits, dim=0)  # [B_i]
            new_logprobs.append(logprobs[action_indices[i]])
            entropies.append(-(logprobs * torch.exp(logprobs)).sum())

        return torch.stack(new_logprobs), torch.stack(entropies)

    def compute_values(
            self,
            pre_move_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Takes a B-tuple of [D] tensors (fixed length) or [L, C] tensors
        # (variable length), and returns values of shape [B].
        return self.invoke_net(
            self.value_fn,
            pre_move_states)

    def select_action(
        self,
        post_move_states: tuple[torch.Tensor, ...]
    ) -> tuple[int, float]:
        """
        Compute distribution over legal moves and sample one.
        Returns:
            action_idx, logprob of sampled move
        """
        with torch.no_grad():
            logprobs = self.compute_logprobs(post_move_states)
            probs = torch.exp(logprobs)
            dist = torch.distributions.Categorical(probs)
            a = int(dist.sample().item())

        return a, float(logprobs[a].item())

    def value(self, pre_move_state: torch.Tensor) -> float:
        # Compute the value for a pre-move state.
        with torch.no_grad():
            value = self.compute_values((pre_move_state,))
        return float(value.item())


class NeuralPlayer(PlanningPlayer):
    # A Player wrapping an agent that wraps a neural policy and value network.
    def __init__(self, agent: Agent):
        self.agent = agent

    def select_move(self, info_state: InformationState) -> Move:
        moves, raw_post_move_states = info_state.post_move_states()
        post_move_states = self.agent.featurize(raw_post_move_states)
        action_idx, _ = self.agent.select_action(post_move_states)
        return moves[action_idx]

    def flip_hand(self, hand: Sequence[tuple[int, int]]) -> bool:
        # TODO: Implement. Would be nice to use value_fn for that, but I'm not
        # sure how to derive an information state.
        # For now, use PlanningPlayer's heuristic method.
        return super().flip_hand(hand)


class AgentCollection:
    @staticmethod
    def create_agents(num_agents: int) -> list[Agent]:
        raise NotImplementedError

    @classmethod
    def load_agents(
            cls,
            policy_paths: list[str],
            value_fn_path: str | None = "",
            device: torch.device = torch.device("cpu")) -> list[Agent]:
        agents = cls.create_agents(len(policy_paths), device)
        for i, policy_path in enumerate(policy_paths):
            agents[i].policy.load_state_dict(torch.load(policy_path, map_location=device))
        if value_fn_path:
            agents[0].value_fn.load_state_dict(torch.load(value_fn_path, map_location=device))
        return agents

    @staticmethod
    def save_agents(
            agents: list[Agent],
            policy_paths: list[str],
            value_fn_path: str):
        for i, agent in enumerate(agents):
            torch.save(agent.policy.state_dict(), policy_paths[i])
        torch.save(agents[0].value_fn.state_dict(), value_fn_path)


#############################################################################
# A simple-ish baseline policy and value network - lots of feature engineering,
# simple feedforward MLP.
# Reuses featurization code I've originally added for ISMCTS value functions.
##############################################################################
def _build_simple_mlp(state_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


class SimplePolicyNet(nn.Module):
    # The Policy network is a ranking model - it takes the state visible to
    # the current player, and the state after each possible move, and "ranks"
    # the moves by producing a logit for each post move state.
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = _build_simple_mlp(state_dim)

    def forward(self, post_move_states: torch.Tensor) -> torch.Tensor:
        return self.net(post_move_states).squeeze(1)  # [B, 1] -> [B]


class SimpleValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = _build_simple_mlp(state_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(1)  # [B, 1] -> [B]


class SimpleAgent(Agent):
    def __init__(
            self,
            state_dim: int,
            policy_lr: float,
            value_fn: nn.Module,
            value_optim: torch.optim.Optimizer,
            device: torch.device = torch.device("cpu")):
        policy = SimplePolicyNet(state_dim)
        super().__init__(
            policy=policy,
            policy_optim=torch.optim.Adam(policy.parameters(), lr=policy_lr),
            value_fn=value_fn,
            value_fn_optim=value_optim,
            is_fixed_length=True,
            device=device,
        )

    def featurize(
            self, info_states: tuple[InformationState, ...]) -> tuple[torch.Tensor, ...]:
        # For now, just return the current player's score repeated to fill; that
        # way, the network has something to learn on.
        # Next, I hope to move to the full featurization used in neural_value_function.
        # Finally, I hope to use Transformers or RNNs over annotated hand/table
        # sequences, e.g. "4,4,9,2,3,4,7" ->
        # "<set2>,4,<single>,9,<run3>,2,3,4,<single>,7" etc.
        # Unclear whether "3,2,1,2" should be run2,run2 or run3,single.

        ssrs = []
        for info_state in info_states:
            ssrs.append(StateAndScoreRecord(
                info_state.current_player,
                tuple(c[0] for c in info_state.hand),
                info_state.table,
                info_state.num_cards,
                info_state.scores,
                info_state.can_scout_and_show, inf))

        features = neural_value_function.featurize(ssrs)[0]
        return torch.unbind(features, dim=0)


class SimpleAgentCollection(AgentCollection):
    @staticmethod
    def load_default_agent(device: torch.device = torch.device("cpu")) -> Agent:
        base_dir = os.path.dirname(__file__)
        policy_path = os.path.join(base_dir, "simple_agent_weights", "simple_agent_0_it_79_skill_1.95.pth")
        return SimpleAgentCollection.load_agents([policy_path], device=device)[0]

    @staticmethod
    def create_agents(num_agents: int, device: torch.device = torch.device("cpu"),
                      policy_lr: float | None = None, value_lr: float | None = None) -> list[Agent]:
        state_dim = 57
        if policy_lr is None:
            policy_lr = 3e-3
        if value_lr is None:
            value_lr = 1e-3
        value_fn = SimpleValueNet(state_dim)
        value_optim = torch.optim.Adam(value_fn.parameters(), lr=value_lr)
        return [SimpleAgent(state_dim, policy_lr, value_fn, value_optim, device)
                for _ in range(num_agents)]


#############################################################################
# A Transformer based policy and value network. My primary hope is that these
# are more able to spot patterns such as "being close to a strong run", and
# eventually taking history (which cards have been played) into account.
# A secondary hope is that having other architectures in the mix adds diversity.
##############################################################################
PADDING_IDX = 0  # shared by pad_sequence, embedding tables, and featurize
# Number of scalar (non-hand) features taken from the first NUM_SCALAR_FEATURES
# elements of featurize_numpy: num_players(1), num_cards(5), scores(5),
# cur_score_diff(1), can_scout_and_show(5), table_features(3).
NUM_SCALAR_FEATURES = 20
_SCALAR_MEANS: list[float] = _NUMPY_SCALAR_MEANS[:NUM_SCALAR_FEATURES].tolist()
_SCALAR_STDS: list[float] = _NUMPY_SCALAR_STDS[:NUM_SCALAR_FEATURES].tolist()

# Token vocabulary for the transformer's hand input.
# 0=padding, 1=CLS, 2-11=card1-card10 (12 total).
_HAND_VOCAB_SIZE = 12
_CLS_TOKEN_IDX = 1
_CARD1_TOKEN_IDX = 2  # card i → _CARD1_TOKEN_IDX + i - 1

def map_card_to_token_idx(i: int) -> int:
    return _CARD1_TOKEN_IDX + i - 1


class TransformerNet(nn.Module):
    # Shared architecture for both the policy and value transformer networks.
    # Layout: [other_1..other_20, CLS, hand_1..hand_N, padding...]
    # Channels: [value/token_id, card_pos]
    def __init__(self, embed_dim: int, num_heads: int, dim_ffd: int, num_layers: int,
                 max_card_pos: int = 45):
        super().__init__()
        self.token_embedding = nn.Embedding(_HAND_VOCAB_SIZE, embed_dim, padding_idx=PADDING_IDX)
        self.card_pos_embedding = nn.Embedding(max_card_pos + 1, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim, num_heads, dim_ffd, dropout=0.0, batch_first=True, norm_first=True),
            num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + NUM_SCALAR_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        scalar_features = states[:, :NUM_SCALAR_FEATURES, 0]             # [B, 20]
        hand = states[:, NUM_SCALAR_FEATURES:, :]                        # [B, 1+N+pad, 2]

        token_ids = hand[:, :, 0].long()                                 # [B, 1+N+pad]
        card_pos  = hand[:, :, 1].long()                                 # [B, 1+N+pad]
        embedded = (self.token_embedding(token_ids)
                  + self.card_pos_embedding(card_pos))                   # [B, 1+N+pad, E]

        hand_padding_mask = token_ids == PADDING_IDX                     # [B, 1+N+pad]
        transformed = self.transformer(
            embedded, src_key_padding_mask=hand_padding_mask)            # [B, 1+N+pad, E]

        cls_out = transformed[:, 0, :]                                   # [B, E]
        combined = torch.cat([cls_out, scalar_features], dim=1)         # [B, E+20]
        return self.mlp(combined).squeeze(1)                             # [B]


class TransformerAgent(Agent):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_layers: int,
            dim_ffd: int,
            policy_lr: float,
            value_fn: nn.Module,
            value_optim: torch.optim.Optimizer,
            device: torch.device = torch.device("cpu"),
            max_card_pos: int = 45):
        policy = TransformerNet(embed_dim, num_heads, dim_ffd, num_layers, max_card_pos)
        super().__init__(
            policy=policy,
            policy_optim=torch.optim.Adam(policy.parameters(), lr=policy_lr),
            value_fn=value_fn,
            value_fn_optim=value_optim,
            is_fixed_length=False,
            device=device,
        )

    def featurize(self, info_states: tuple[InformationState, ...]) -> tuple[torch.Tensor, ...]:
        # Returns one [L, 2] float32 tensor per info_state.
        # L = NUM_SCALAR_FEATURES (OTHER tokens) + 1 (CLS) + len(hand).
        # Channels: [value/token_id, card_pos]
        #   - OTHER tokens: channel 0=normalized scalar feature value, 1=0
        #   - CLS: channel 0=_CLS_TOKEN_IDX, 1=0
        #   - hand cards: channel 0=token_idx (int), 1=card_pos (1-indexed)
        # Layout: [other_1..other_20, CLS, hand_1..hand_N]  (padding added by pad_sequence)
        results: list[torch.Tensor] = []

        for info_state in info_states:
            p = info_state.current_player
            num_players = len(info_state.num_cards)
            num_cards_rot = info_state.num_cards[p:] + info_state.num_cards[:p]
            scores_rot = info_state.scores[p:] + info_state.scores[:p]
            can_sas_rot = info_state.can_scout_and_show[p:] + info_state.can_scout_and_show[:p]

            # Build the same first NUM_SCALAR_FEATURES features as featurize_numpy,
            # in the same order: num_players, num_cards(5), scores(5), cur_score_diff,
            # can_scout_and_show(5), table_features(3).
            num_cards_feat = [0.0] * 5
            num_cards_feat[:len(num_cards_rot)] = [float(x) for x in num_cards_rot]
            scores_feat = [0.0] * 5
            scores_feat[:len(scores_rot)] = [float(x) for x in scores_rot]
            cur_score_diff = float(scores_rot[0] - sum(scores_rot[1:]) / num_players)
            can_sas_feat = [0.0] * 5
            can_sas_feat[:len(can_sas_rot)] = [float(x) for x in can_sas_rot]
            table_values = [c[0] for c in info_state.table]
            table_feat = [
                1.0 if Util.is_group(table_values) else 0.0,
                float(len(table_values)),
                float(max(table_values)) if table_values else 0.0,
            ]
            scalar_raw = (
                [float(num_players)] + num_cards_feat + scores_feat +
                [cur_score_diff] + can_sas_feat + table_feat
            )
            scalar_norm = [
                (scalar_raw[i] - _SCALAR_MEANS[i]) / _SCALAR_STDS[i]
                for i in range(NUM_SCALAR_FEATURES)
            ]

            rows: list[list[float]] = []
            # Scalar feature tokens first (fixed-length prefix).
            for val in scalar_norm:
                rows.append([val, 0.0])
            # CLS token followed by hand cards (variable-length suffix, padded by pad_sequence)
            rows.append([float(_CLS_TOKEN_IDX), 0.0])
            for pos, c in enumerate(info_state.hand):
                rows.append([float(map_card_to_token_idx(c[0])), float(pos + 1)])

            results.append(torch.tensor(rows, dtype=torch.float32))

        return tuple(results)


class TransformerAgentCollection(AgentCollection):
    @staticmethod
    def create_agents(num_agents: int, device: torch.device = torch.device("cpu"),
                      policy_lr: float | None = None, value_lr: float | None = None) -> list[Agent]:
        embed_dim = 16
        num_heads = 4
        num_layers = 2
        dim_ffd = 32
        max_card_pos = 45
        if policy_lr is None:
            policy_lr = 1e-4
        if value_lr is None:
            value_lr = 3e-4
        value_fn = TransformerNet(embed_dim, num_heads, dim_ffd, num_layers, max_card_pos)
        value_optim = torch.optim.Adam(value_fn.parameters(), lr=value_lr)
        return [
            TransformerAgent(
                embed_dim,
                num_heads,
                dim_ffd,
                num_layers,
                policy_lr,
                value_fn,
                value_optim,
                device,
                max_card_pos) for _ in range(num_agents)]
