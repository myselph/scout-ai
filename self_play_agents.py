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

from common import InformationState, Move, StateAndScoreRecord
import neural_value_function
from players import PlanningPlayer


class Agent(ABC):
    # This is a base class for all agents. Its interface is designed to provide
    # a common API across different policy+value net architectures (notably,
    # fixed vs variable length inputs, different featurizations, etc.) and to
    # be usable for both inference and training operations.

    # policy must have a forward() method that takes in
    # 1. a [B, D] tensor where B is the number of moves, and D is the length of
    #    either the feature vector (fixed length features) or the maximum
    #    sequence length (variable length)
    # 2. if variable length features: a [B, D] boolean padding mask tensor,
    #    and a [B] long tensor of sequence lengths.
    # The policy eithr returns a [B] tensor of logits (fixed length features) or
    # a [B, D] tensor of logits (variable length features).
    policy: nn.Module
    # value_fn has the same interface as policy, but returns [B] independent
    # value predictions.
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
        # A helper function to invoke the policy and value networks with the
        # appropriate handling of variable or fixed length inputs (padding etc.)        
        assert states, "Must have at least one state"
        assert states[0].ndim == 1
        if self.is_fixed_length:
            batch = torch.stack(states)  # [B, D]
            return net(batch)  # [B]
        else:
            padding_value = transformer_dict["<padding>"]
            lengths = torch.tensor([s.size(0)
                                   for s in states], dtype=torch.long, device=self.device)
            batch = torch.nn.utils.rnn.pad_sequence(
                list(states), batch_first=True, padding_value=padding_value)
            padding_mask = batch == padding_value # [B, D]
            return net(batch, padding_mask, lengths)  # [B]            

    def compute_logprobs(
            self,
            post_move_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Takes either a B-list of [D] tensors (fixed length features) or a
        # B-list of [L] tensors (variable length features), and returns
        # logprobs of shape [B].
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
        # (fixed length) or [L_ij] (variable length).
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
            batch = torch.stack(all_states)        # [sum(B_i), D]
            logits = self.policy(batch)            # [sum(B_i)]
        else:
            padding_value = transformer_dict["<padding>"]
            lengths = torch.tensor(
                [s.size(0) for s in all_states], dtype=torch.long, device=self.device)
            batch = torch.nn.utils.rnn.pad_sequence(
                all_states, batch_first=True, padding_value=padding_value)
            padding_mask = batch == padding_value  # [sum(B_i), L_max]
            logits = self.policy(batch, padding_mask, lengths)  # [sum(B_i)]

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
        # Takes a tensor of size [B, D] (fixed length features) or [B, L, D]
        # (variable length features), and returns values of shape [B].
        # For variable length features, padding_mask is boolean [B, L], and
        # lengths is long [B].
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

    @staticmethod
    def load_agents(
            policy_paths: list[str],
            value_fn_path: str | None = None) -> list[Agent]:
        raise NotImplementedError

    @staticmethod
    def save_agents(
            agents: list[Agent],
            policy_paths: list[str],
            value_fn_path: str):
        raise NotImplementedError


#############################################################################
# A simple-ish baseline policy and value network - lots of feature engineering,
# simple feedforward MLP.
# Reuses featurization code I've originally added for ISMCTS value functions.
##############################################################################
class SimplePolicyNet(nn.Module):
    # ----------------------------------------------------------------------
    # The Policy network is a ranking model - it takes the state visible to
    # the current player, and the state after each possible move, and "ranks"
    # the moves by producing a logit for each post move state.
    # ----------------------------------------------------------------------
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
            self,
            post_move_states: torch.Tensor) -> torch.Tensor:
        """
        post_move_states: [num_moves, D]
        return logits: [1, num_moves]
        """
        # Output: [N, 1] -> squeeze to [1, N] or [N]
        return self.net(post_move_states).squeeze(1)

class SimpleValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
            self,
            state: torch.Tensor) -> torch.Tensor:
        # padding_mask is ignored since the input is always the same length,
        # but we have it here to use the same interface that Transformers need.
        return self.net(state)[:, None]


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

        features = neural_value_function.featurize(ssrs)[0].to(self.device)
        return torch.unbind(features, dim=0)


class SimpleAgentCollection(AgentCollection):
    @staticmethod
    def load_default_agent(device: torch.device = torch.device("cpu")) -> Agent:
        base_dir = os.path.dirname(__file__)
        policy_path = os.path.join(base_dir, "simple_agent_weights", "simple_agent_0_it_79_skill_1.95.pth")
        return SimpleAgentCollection.load_agents([policy_path], device=device)[0]

    @staticmethod
    def create_agents(num_agents: int, device: torch.device = torch.device("cpu")) -> list[Agent]:
        state_dim = 57
        policy_lr = 3e-3
        value_fn = SimpleValueNet(state_dim)
        value_optim = torch.optim.Adam(value_fn.parameters(), lr=1e-3)
        return [SimpleAgent(state_dim, policy_lr, value_fn, value_optim, device)
                for _ in range(num_agents)]

    @staticmethod
    def load_agents(
            policy_paths: list[str],
            value_fn_path: str | None = "",
            device: torch.device = torch.device("cpu")) -> list[Agent]:
        # This function allows for loading agents from disk; this can be useful
        # for checkpointing/resumption, or simply to load a previously trained agent
        # for evaluation.
        # When used for evaluation, the value function is not used, so the
        # value_fn_path can be left empty, and the value function will be randomly
        # initialized.
        agents = SimpleAgentCollection.create_agents(len(policy_paths), device)
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
# A Transformer based policy and value network. My primary hope is that these
# are more able to spot patterns such as "being close to a strong run", and
# eventually taking history (which cards have been played) into account.
# A secondary hope is that having other architectures in the mix adds diversity.
##############################################################################
dict_tokens = [
    "<padding>",
    "<num_players>",
    "</num_players>",
    "<num_cards>",
    "</num_cards>",
    "<scores>",
    "</scores>",
    "<scout_and_show>",
    "</scout_and_show>",
    "<table>",
    "</table>",
    "<hand>",
    "</hand>",
    "True",
    "False",
    "<eos>",
    "<too_small>",
    "<too_large>",
]

transformer_dict = {token: i for i, token in enumerate(dict_tokens)}
next_index = len(transformer_dict)
for i in range(1,11):
    transformer_dict[f"<card{i}>"] = next_index
    next_index += 1
dict_int_limits = [-20, 20]
for i in range(dict_int_limits[0], dict_int_limits[1] + 1):
    transformer_dict[f"int_{i}"] = next_index
    next_index += 1

def map_card_to_tf_dict(i: int) -> int:
    return transformer_dict[f"<card{i}>"]

def map_int_to_tf_dict(i: int) -> int:
    if i < dict_int_limits[0]:
        return transformer_dict['<too_small>']
    elif i > dict_int_limits[1]:
        return transformer_dict['<too_large>']
    else:
        return transformer_dict[f"int_{i}"]


# TODO: I have some dimenion related bug in there. I found that when I call
# forward below with a 1xN pre_move_stat and a 2xM post_move_state where both
# rows of the post_move_state are identical, I get different logits.
class TransformerPolicyNet(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_ffd: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(len(transformer_dict), embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim, num_heads, dim_ffd, batch_first=True, norm_first=True), num_layers)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(
            self,
            post_move_states: torch.Tensor,
            padding_mask: torch.Tensor,
            lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(post_move_states)  # [B, D, E]
        transformed = self.transformer(embedded, src_key_padding_mask=padding_mask)  # [B, D, E]
        last_embeddings = transformed[torch.arange(transformed.size(0)), lengths - 1, :]  # [B, E]
        return self.output_layer(last_embeddings).squeeze(1)  # [B]        


class TransformerValueNet(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_ffd: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(len(transformer_dict), embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim, num_heads, dim_ffd, batch_first=True, norm_first=True), num_layers)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, states: torch.Tensor, padding_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(states)  # [M, D_embed]        
        transformed = self.transformer(embedded, src_key_padding_mask=padding_mask)  # [M, D_embed]
        last_embeddings = transformed[torch.arange(transformed.size(0)), lengths - 1, :]  # [B, E]
        return self.output_layer(last_embeddings).squeeze(1)  # [B]


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
            device: torch.device = torch.device("cpu")):
        policy = TransformerPolicyNet(embed_dim, num_heads, dim_ffd, num_layers)
        super().__init__(
            policy=policy,
            policy_optim=torch.optim.Adam(policy.parameters(), lr=policy_lr),
            value_fn=value_fn,
            value_fn_optim=value_optim,
            is_fixed_length=False,
            device=device,
        )

    def featurize(self, info_states: tuple[InformationState, ...]) -> tuple[torch.Tensor, ...]:
        # Not yet added: history, or some subset of it such as discarded cards
        # and known card -> opponent assignments.
        # TODO: Positional embeddings so order matters.
        results: list[torch.Tensor] = []
        TD: dict[str,int] = transformer_dict  # to make things more readable
        
        for info_state in info_states:
            num_cards_rot = info_state.num_cards[info_state.current_player:] + \
                info_state.num_cards[:info_state.current_player]
            scores_rot = info_state.scores[info_state.current_player:] + \
                info_state.scores[:info_state.current_player]
            can_scout_and_show_rot = info_state.can_scout_and_show[info_state.current_player:] + \
                info_state.can_scout_and_show[:info_state.current_player]
            
            result: list[int] = []
            result.append(TD["<num_players>"])
            result.append(map_int_to_tf_dict(info_state.num_players))
            result.append(TD["</num_players>"])
            result.append(TD["<num_cards>"])
            for i in num_cards_rot:
                result.append(map_int_to_tf_dict(i))
            result.append(TD["</num_cards>"])
            result.append(TD["<scores>"])
            for i in scores_rot:
                result.append(map_int_to_tf_dict(i))
            result.append(TD["</scores>"])
            result.append(TD["<scout_and_show>"])
            for i in can_scout_and_show_rot:
                if i:
                    result.append(TD["True"])
                else:
                    result.append(TD["False"])

            result.append(TD["</scout_and_show>"])
            result.append(TD["<table>"])
            for c in info_state.table:
                result.append(map_card_to_tf_dict(c[0]))
            result.append(TD["</table>"])
            result.append(TD["<hand>"])
            for c in info_state.hand:
                result.append(map_card_to_tf_dict(c[0]))
            result.append(TD["</hand>"])
            result.append(TD["<eos>"])

            results.append(torch.tensor(result, dtype=torch.int, device=self.device))

        return tuple(results)
        

class TransformerAgentCollection(AgentCollection):
    @staticmethod
    def create_agents(num_agents: int, device: torch.device = torch.device("cpu")) -> list[Agent]:
        embed_dim = 8
        num_heads = 2
        num_layers = 2
        dim_ffd = 32
        policy_lr = 3e-3
        value_fn = TransformerValueNet(embed_dim, num_heads, dim_ffd, num_layers)
        value_optim = torch.optim.Adam(value_fn.parameters(), lr=1e-3)
        return [
            TransformerAgent(
                embed_dim,
                num_heads,
                dim_ffd,
                num_layers,
                policy_lr,
                value_fn,
                value_optim,
                device) for _ in range(num_agents)]

    @staticmethod
    def load_agents(
            policy_paths: list[str],
            value_fn_path: str | None = "",
            device: torch.device = torch.device("cpu")) -> list[Agent]:
        agents = TransformerAgentCollection.create_agents(len(policy_paths), device)
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
