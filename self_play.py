from __future__ import annotations
import copy
import random
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Dict, Any
from common import InformationState, Move
from evaluation import play_game, rank_against_planning_player
from game_state import GameState
from main import play_tournament
from players import GreedyShowPlayerWithFlip, PlanningPlayer, RandomPlayer
import argparse

from self_play_agents import Agent, SimpleAgentCollection, TransformerAgentCollection
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch_size",
    type=int,
    help="Minibatch size for PPO updates",
    default=512
)
parser.add_argument(
    "--iterations",
    type=int,
    help="Number of training iterations",
    default=80
)
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs per training iteration",
    default=2
)
parser.add_argument(
    "--episodes",
    type=int,
    help="Number of episodes per training iteration",
    default=40
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# Data structures for rollout storage
# ----------------------------------------------------------------------


@dataclass
class Transition:
    """
    One step of experience.
    """
    pre_move_state: torch.Tensor
    action_idx: int             # index into the action list
    logprob: float              # scalar
    reward: float               # scalar
    value: float                # scalar
    done: bool                  # episode termination flag
    post_move_states: tuple[torch.Tensor, ...]


@dataclass
class Trajectory:
    """
    Sequence of Transitions for a single agent over one episode.
    """
    transitions: List[Transition]


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


# ----------------------------------------------------------------------
# PPO Loss
# ----------------------------------------------------------------------

def ppo_loss(
    agent: Agent,
    pre_move_states: list[torch.Tensor],
    post_move_states_list: list[tuple[torch.Tensor, ...]],
    action_idx: torch.Tensor,     # shape: [batch]
    old_logprob: torch.Tensor,    # shape: [batch]
    returns: torch.Tensor,        # shape: [batch]
    advantages: torch.Tensor,     # shape: [batch]
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    minibatch_size: int = 512
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO clipped surrogate objective.
    We assume:
    - policy(obs, move_batch) -> logits for each move in move_batch
    - value_fn(obs) -> scalar value prediction

    NOTE: Because legal action sets differ per item, we must compute
    action logprobs one sample at a time. I'm sure this could be optimized
    somehow, but it's not a bottleneck yet.
    """

    batch_size = len(pre_move_states)
    new_logprobs = []
    entropies = []

    for i in range(batch_size):
        post_move_states = post_move_states_list[i]
        logprobs = agent.compute_logprobs(
            post_move_states)  # shape: [num_actions]

        # Select logprob of performed action
        chosen = action_idx[i]
        new_logprobs.append(logprobs[chosen])

        # Entropy for this decision
        entropy = -(logprobs * torch.exp(logprobs)).sum()
        entropies.append(entropy)

    new_logprobs = torch.stack(new_logprobs)  # shape: [batch]
    entropies = torch.stack(entropies)        # shape: [batch]

    # Probability ratio
    ratio = torch.exp(new_logprobs - old_logprob)

    # PPO objective
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -torch.sum(torch.min(unclipped, clipped)) / minibatch_size

    # Value loss
    values = agent.compute_values(tuple(pre_move_states))
    value_loss = vf_coef * torch.sum((returns - values) ** 2) / minibatch_size

    # Entropy bonus
    entropy_bonus = ent_coef * entropies.sum() / minibatch_size

    total_loss = policy_loss + value_loss - entropy_bonus

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropies.mean().item()
    }

    return total_loss, metrics


# ----------------------------------------------------------------------
# Advantage calculation (GAE)
# ----------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,       # [T]
    values: torch.Tensor,        # [T+1]
    dones: torch.Tensor,         # [T]
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.
    NB the parameters are concatenations of trajectories for multiple episodes,
    and dones indicate episode boundaries.
    """
    T = rewards.size(0)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        # compute TD: difference between what we saw (reward_t + V(s_{t+1})) and
        # what we expected to see (V(s_t))
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return returns, advantages


# ----------------------------------------------------------------------
# Rollout generation
# ----------------------------------------------------------------------

def collect_episodes(
    agents: List[Agent],
    env_constructor: Callable[[int], GameState],
    min_episodes: int,
    num_players: int,
    min_examples_per_player: int
) -> Dict[int, List[Trajectory]]:
    """
    All agents play together in a multi-player environment.
    Returns dict: agent_id -> list of Trajectory (one per episode this agent
    played in).
    We play at least num_episodes, but possibly more to ensure we collect at
    least min_examples_per_player for each agent.
    """
    data: Dict[int, List[Trajectory]] = {i: [] for i in range(len(agents))}

    episode = 0
    while episode < min_episodes or any(not traj for traj in data.values()) or any(sum(
            len(traj.transitions) for traj in data[i]) < min_examples_per_player for i in data):
        episode += 1
        episode_agent_indices = random.sample(range(len(agents)), num_players)
        episode_agents = [agents[i] for i in episode_agent_indices]
        env = env_constructor(0)
        # For now, we just flip like PlanningPlayer would.
        # Eventually, we should learn a policy for that, but it complicates
        # training a bit because the flip decision requires another network.
        pp = PlanningPlayer()
        env.maybe_flip_hand([lambda h: pp.flip_hand(h)
                            for _ in episode_agents])
        traj = {i: [] for i in episode_agent_indices}

        done = False
        while not done:
            player = env.current_player
            agent = episode_agents[player]

            raw_pre_move_state = env.info_state()
            moves, raw_post_move_states = raw_pre_move_state.post_move_states()
            pre_move_state = agent.featurize((raw_pre_move_state,))[0]
            post_move_states = agent.featurize(raw_post_move_states)
            action_idx, logp = agent.select_action(post_move_states)
            value = agent.value(pre_move_state)
            env.move(moves[action_idx])
            reward = 0
            done = env.is_finished()

            traj[episode_agent_indices[player]].append(Transition(
                pre_move_state=pre_move_state,
                action_idx=action_idx,
                logprob=logp,
                reward=reward,
                value=value,
                done=done,
                post_move_states=post_move_states,
            ))
        # Game over - assign final rewards: the difference to the average opponent
        # score. This strikes a balance between using raw scores (not indicative
        # if we won or lost) and just win/loss (too sparse).
        sum_scores = sum(env.scores)
        for i, j in enumerate(episode_agent_indices):
            avg_opp_score = (sum_scores - env.scores[i]) / (num_players - 1)
            traj[j][-1].reward = env.scores[i] - avg_opp_score
        for i in traj:
            data[i].append(Trajectory(traj[i]))

    return data


# ----------------------------------------------------------------------
# Prepare minibatches from trajectories
# ----------------------------------------------------------------------

def flatten_trajectories(
    trajectories: Dict[int, List[Trajectory]]
) -> Dict[int, Dict[str, Any]]:
    """
    Flatten per-agent trajectories into tensors suitable for PPO updates.
    """
    out = {}

    for agent_id, traj_list in trajectories.items():
        pre_move_state_list: list[torch.Tensor] = []
        act_list = []
        logp_list = []
        rew_list = []
        val_list = []
        done_list = []
        post_move_states_list: list[tuple[torch.Tensor, ...]] = []

        for traj in traj_list:
            for tr in traj.transitions:
                pre_move_state_list.append(tr.pre_move_state)
                act_list.append(tr.action_idx)
                logp_list.append(tr.logprob)
                rew_list.append(tr.reward)
                val_list.append(tr.value)
                done_list.append(float(tr.done))
                post_move_states_list.append(tr.post_move_states)

        val_list.append(0.0)  # bootstrap value for final state

        actions = torch.tensor(act_list, dtype=torch.int16)   # [T]
        old_logprobs = torch.tensor(logp_list, dtype=torch.float32)  # [T]
        rewards = torch.tensor(rew_list, dtype=torch.float32)  # [T]
        dones = torch.tensor(done_list)                       # [T]
        values = torch.tensor(val_list, dtype=torch.float32)  # [T+1]

        returns, advantages = compute_gae(
            rewards, values, dones,
            gamma=0.99, lam=0.95
        )  # each: [T]

        out[agent_id] = {
            "pre_move_states": pre_move_state_list,
            "actions": actions,
            "post_move_states_list": post_move_states_list,
            "old_logprobs": old_logprobs,
            "returns": returns,
            "advantages": advantages,
        }
        num_steps = len(actions)
        num_steps_per_game = num_steps / len(traj_list)
        print(
            f"Agent {agent_id} - collected {len(out[agent_id]['actions'])} steps - {num_steps_per_game:.1f} steps per game.")

    return out


# ----------------------------------------------------------------------
# PPO Update
# ----------------------------------------------------------------------
def ppo_update(
    agents: List[Agent],
    data_by_agent: Dict[int, Dict[str, Any]],
    minibatch_size: int,
    epochs: int,
    microbatch_size: int = 32
):
    """
    Run PPO updates for each agent separately.
    NOTE: all agents share value_fn, so value_optimizer updates it globally.
    """

    for agent_id, data in data_by_agent.items():
        print(f"training agent {agent_id}")
        pre_move_states = data["pre_move_states"]
        actions = data["actions"]
        post_move_states_list = data["post_move_states_list"]
        old_logprobs = data["old_logprobs"]
        returns = data["returns"]
        advantages = data["advantages"]

        N = len(pre_move_states)
        for epoch in range(epochs):
            perm = torch.randperm(N)
            # TODO: We sample episodes until we have at least one minibatch
            # per agent; so for all agents we typically have just a little bit
            # more than one minibatch; and that last partial batch could be
            # ignored.
            for start in range(0, N, minibatch_size):
                end = min(start + minibatch_size, N)
                minibatch_indices = perm[start:end]

                # Zero gradients at start of minibatch
                agents[agent_id].policy_optim.zero_grad()
                agents[agent_id].value_optim.zero_grad()

                # Process minibatch in microbatches to save memory
                minibatch_len = len(minibatch_indices)
                for mb_start in range(0, minibatch_len, microbatch_size):
                    mb_end = min(mb_start + microbatch_size, minibatch_len)
                    mb_indices = minibatch_indices[mb_start:mb_end]

                    prms = [pre_move_states[i] for i in mb_indices]
                    psms = [post_move_states_list[i] for i in mb_indices]

                    # Compute loss for microbatch
                    loss, metrics = ppo_loss(
                        agents[agent_id],
                        pre_move_states=prms,
                        post_move_states_list=psms,
                        action_idx=actions[mb_indices],
                        old_logprob=old_logprobs[mb_indices],
                        returns=returns[mb_indices],
                        advantages=advantages[mb_indices],
                        minibatch_size=minibatch_len  # Normalize by full minibatch size
                    )
                    loss.backward()

                # Step optimizer after full minibatch
                agents[agent_id].policy_optim.step()
                agents[agent_id].value_optim.step()


# ----------------------------------------------------------------------
# High-level training loop
# ----------------------------------------------------------------------

def train(
    num_iterations: int,
    episodes_per_iter: int,
    num_players: int,
    minibatch_size: int = 512,
    epochs: int = 2
):
    agents = []
    # Number of agents we train in an iteration.
    num_agents_train = num_players
    # We keep copies of the best ones.
    num_best_agents = int(0.2 * num_agents_train)
    num_best_agents = 0
    # So overall there are num_agents + num_best_agents agents.
    # agents = SimpleAgentCollection.create_agents(num_agents_train)
    agents = TransformerAgentCollection.create_agents(num_agents_train)

    best_agents: dict[float, Agent] = {}

    def env_constructor(dealer): return GameState(
        num_players=num_players,
        dealer=dealer)

    for iteration in range(num_iterations):
        # 1. Self-play
        trajectories = collect_episodes(
            agents,
            env_constructor,
            episodes_per_iter,
            num_players,
            min_examples_per_player=minibatch_size
        )

        # 2. Flatten storage and compute GAE
        data = flatten_trajectories(trajectories)
        del trajectories

        # 3. PPO update
        ppo_update(
            agents,
            data,
            minibatch_size,
            epochs,
            microbatch_size=32
        )
        del data

        # 4. Evaluation & shuffling.
        if iteration % 5 == 0 and iteration > 0:
            agents_list = agents + list(best_agents.values())
            order, skills = rank_against_planning_player(
                [NeuralPlayer(a) for a in agents_list], num_players, num_games_per_player=50)
            agents = [agents_list[i] for i in order[:num_agents_train]]
            TransformerAgentCollection.save_agents(
                agents, [
                    f"transformer_agent_{i}_it_{iteration}_skill_{
                        skills[i]:.2f}.pth" for i in range(
                        len(agents))], f"transformer_agent_it_{iteration}_value_fn.pth")
            best_agents = {}
            for i in range(num_best_agents):
                agent_index = order[i]
                best_agents[skills[i]] = copy.deepcopy(
                    agents_list[agent_index])
            print(f"Best agents' skills: {list(best_agents.keys())}")

        print(f"Iteration {iteration} completed.")

    return agents


def main():
    num_players = 5
    agents = train(
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        num_players=num_players,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
    )

    # Find the best agent.
    print("Finding best agent...")
    players = [lambda: NeuralPlayer(agents[i]) for i in range(num_players)]
    wins = [0] * len(players)
    for reps in range(0, 50):
        scores = play_game([p() for p in players])
        winner_index = max(range(len(scores)), key=lambda i: scores[i])
        wins[winner_index] += 1
    print(f"Agent wins: {wins}")
    best_agent_index = max(range(len(wins)), key=lambda i: wins[i])
    print(f"Best agent is Agent {best_agent_index}.")

    # Play a tournament against RandomPlayer. This is a good sanity check that
    # training works - an untrained net shouldn't do any better than random,
    # while a trained net should win even with a very primitive feature set.
    print("Tournament against GreedyShowPlayerWithFlip:")
    play_tournament(
        player_a_factory_fn=lambda: NeuralPlayer(agents[best_agent_index]),
        player_b_factory_fn=lambda: GreedyShowPlayerWithFlip(),
        num_games=200
    )
    print("Tournament against PlanningPlayer:")
    play_tournament(
        player_a_factory_fn=lambda: NeuralPlayer(agents[best_agent_index]),
        player_b_factory_fn=lambda: PlanningPlayer(),
        num_games=200
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main()
