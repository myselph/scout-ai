from __future__ import annotations
import copy
import glob as glob_module
import os
import random
import torch
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Dict, Any
from .evaluation import play_game, rank_against_planning_player
from .game_state import GameState
from .main import play_tournament
from .players import GreedyShowPlayer, PlanningPlayer
import argparse

from .self_play_agents import Agent, NeuralPlayer, SimpleAgentCollection, TransformerAgentCollection
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
parser.add_argument(
    '--use_transformer',
    action='store_true',
    help='Use transformer-based agents instead of simple feedforward agents'
)
parser.add_argument(
    '--policy_lr',
    type=float,
    help='Learning rate for the policy network',
    default=None
)
parser.add_argument(
    '--value_lr',
    type=float,
    help='Learning rate for the value network',
    default=None
)
parser.add_argument(
    '--resume_dir',
    type=str,
    help='Directory containing .pth files to resume training from; '
         'derives number of agents from policy files found there',
    default=None
)
parser.add_argument(
    '--num_planning_players',
    type=int,
    help='Number of non-trainable PlanningPlayer opponents to include per game '
         'episode during training (must be less than num_players)',
    default=0
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
    """

    new_logprobs, entropies = agent.compute_logprobs_and_entropy_batched(
        post_move_states_list, action_idx)  # each: [batch]

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
    min_examples_per_player: int,
    num_static_per_game: int = 0,
) -> Dict[int, List[Trajectory]]:
    """
    All agents play together in a multi-player environment.
    Returns dict: agent_id -> list of Trajectory (one per episode this agent
    played in).
    We play at least num_episodes, but possibly more to ensure we collect at
    least min_examples_per_player for each agent.

    If num_static_per_game > 0, that many non-trainable PlanningPlayer opponents
    are randomly mixed into each episode. Their moves are executed but no
    transitions are recorded for them.
    """
    data: Dict[int, List[Trajectory]] = {i: [] for i in range(len(agents))}
    num_trainable_per_game = num_players - num_static_per_game

    episode = 0
    while episode < min_episodes or any(not traj for traj in data.values()) or any(sum(
            len(traj.transitions) for traj in data[i]) < min_examples_per_player for i in data):
        episode += 1

        # Sample trainable agents and (optionally) static players for this episode.
        sampled_agent_ids = random.sample(range(len(agents)), num_trainable_per_game)
        sampled_statics = [PlanningPlayer() for _ in range(num_static_per_game)]

        # Randomly assign participants to the num_players positions.
        combined_ids = sampled_agent_ids + [None] * num_static_per_game
        combined_participants = [agents[i] for i in sampled_agent_ids] + sampled_statics
        perm = list(range(num_players))
        random.shuffle(perm)
        episode_agent_ids = [None] * num_players      # agent_id or None per position
        episode_participants = [None] * num_players   # Agent or static Player per position
        for slot, pos in enumerate(perm):
            episode_agent_ids[pos] = combined_ids[slot]
            episode_participants[pos] = combined_participants[slot]

        env = env_constructor(0)
        # For now, we just flip like PlanningPlayer would.
        # Eventually, we should learn a policy for that, but it complicates
        # training a bit because the flip decision requires another network.
        pp = PlanningPlayer()
        env.maybe_flip_hand([lambda h: pp.flip_hand(h) for _ in range(num_players)])
        traj = {agent_id: [] for agent_id in episode_agent_ids if agent_id is not None}

        done = False
        while not done:
            player = env.current_player
            agent_id = episode_agent_ids[player]
            participant = episode_participants[player]

            if agent_id is None:
                # Static (non-trainable) player: pick a move but don't record a transition.
                move = participant.select_move(env.info_state())
                env.move(move)
                done = env.is_finished()
            else:
                raw_pre_move_state = env.info_state()
                moves, raw_post_move_states = raw_pre_move_state.post_move_states()
                pre_move_state = participant.featurize((raw_pre_move_state,))[0]
                post_move_states = participant.featurize(raw_post_move_states)
                action_idx, logp = participant.select_action(post_move_states)
                value = participant.value(pre_move_state)
                env.move(moves[action_idx])
                reward = 0
                done = env.is_finished()

                traj[agent_id].append(Transition(
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
        for pos, agent_id in enumerate(episode_agent_ids):
            if agent_id is not None:
                avg_opp_score = (sum_scores - env.scores[pos]) / (num_players - 1)
                traj[agent_id][-1].reward = env.scores[pos] - avg_opp_score
        for agent_id in traj:
            data[agent_id].append(Trajectory(traj[agent_id]))

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

    for agent_id in list(data_by_agent.keys()):
        print(f"training agent {agent_id}")
        data = data_by_agent.pop(agent_id)
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
                        old_logprob=old_logprobs[mb_indices].to(device),
                        returns=returns[mb_indices].to(device),
                        advantages=advantages[mb_indices].to(device),
                        minibatch_size=minibatch_len  # Normalize by full minibatch size
                    )
                    loss.backward()
                    del loss, metrics, prms, psms

                # Step optimizer after full minibatch
                agents[agent_id].policy_optim.step()
                agents[agent_id].value_optim.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del pre_move_states, post_move_states_list, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# High-level training loop
# ----------------------------------------------------------------------

def train(
    num_iterations: int,
    episodes_per_iter: int,
    num_players: int,
    minibatch_size: int = 512,
    epochs: int = 2,
    policy_lr: float | None = None,
    value_lr: float | None = None,
    resume_dir: str | None = None,
    num_planning_players: int = 0,
):
    agents = []
    # Number of agents we train in an iteration.
    num_agents_train = num_players
    # We keep copies of the best ones.
    # num_best_agents = int(0.2 * num_agents_train)
    num_best_agents = 0
    if resume_dir is not None:
        all_pth = sorted(glob_module.glob(os.path.join(resume_dir, '*.pth')))
        policy_paths = [p for p in all_pth if not p.endswith('_value_fn.pth')]
        value_fn_paths = [p for p in all_pth if p.endswith('_value_fn.pth')]
        value_fn_path = value_fn_paths[-1] if value_fn_paths else None
        num_agents_train = len(policy_paths)
        if num_agents_train < num_players:
            raise ValueError(
                f"resume_dir has {num_agents_train} policy files but num_players={num_players}; "
                "need at least num_players agents to run self-play"
            )
        if args.use_transformer:
            agents = TransformerAgentCollection.load_agents(policy_paths, value_fn_path, device)
        else:
            agents = SimpleAgentCollection.load_agents(policy_paths, value_fn_path, device)
        print(f"Resumed {num_agents_train} agents from {resume_dir} "
              f"(value fn: {value_fn_path})")
    elif args.use_transformer:
        agents = TransformerAgentCollection.create_agents(num_agents_train, device, policy_lr=policy_lr, value_lr=value_lr)
    else:
        agents = SimpleAgentCollection.create_agents(num_agents_train, device, policy_lr=policy_lr, value_lr=value_lr)

    best_agents: dict[float, Agent] = {}

    def env_constructor(dealer): return GameState(
        num_players=num_players,
        dealer=dealer)

    for iteration in range(1, num_iterations+1):
        t_start = time.time()
        # 1. Self-play
        trajectories = collect_episodes(
            agents,
            env_constructor,
            episodes_per_iter,
            num_players,
            min_examples_per_player=minibatch_size,
            num_static_per_game=num_planning_players,
        )
        t_collect_episodes = time.time()
        print(f"Episode collection took {t_collect_episodes - t_start:.2f} seconds.")

        # 2. Flatten storage and compute GAE
        data = flatten_trajectories(trajectories)
        del trajectories
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. PPO update
        ppo_update(
            agents,
            data,
            minibatch_size,
            epochs,
            microbatch_size=32
        )
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_ppo_update = time.time()
        print(f"PPO update took {t_ppo_update - t_collect_episodes:.2f} seconds.")

        # 4. Evaluation & shuffling.
        if iteration % 5 == 0:
            agents_list = agents + list(best_agents.values())
            order, skills = rank_against_planning_player(
                [NeuralPlayer(a) for a in agents_list], num_players, num_games_per_player=500)
            agents = [agents_list[i] for i in order[:num_agents_train]]
            if args.use_transformer:
                TransformerAgentCollection.save_agents(
                    agents, [
                        f"transformer_agent_{i}_it_{iteration}_skill_{
                            skills[i]:.2f}.pth" for i in range(
                            len(agents))], f"transformer_agent_it_{iteration}_value_fn.pth")
            else:
                SimpleAgentCollection.save_agents(
                    agents, [
                        f"simple_agent_{i}_it_{iteration}_skill_{
                            skills[i]:.2f}.pth" for i in range(
                            len(agents))], f"simple_agent_it_{iteration}_value_fn.pth")
            best_agents = {}
            for i in range(num_best_agents):
                agent_index = order[i]
                best_agents[skills[i]] = copy.deepcopy(
                    agents_list[agent_index])
            print(f"Best agents' skills: {list(best_agents.keys())}")
            t_evaluation = time.time()
            print(f"Evaluation took {t_evaluation - t_ppo_update:.2f} seconds.")

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
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        resume_dir=args.resume_dir,
        num_planning_players=args.num_planning_players,
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
    print("Tournament against GreedyShowPlayer:")
    play_tournament(
        player_a_factory_fn=lambda: NeuralPlayer(agents[best_agent_index]),
        player_b_factory_fn=lambda: GreedyShowPlayer(),
        num_games=200
    )
    print("Tournament against PlanningPlayer:")
    play_tournament(
        player_a_factory_fn=lambda: NeuralPlayer(agents[best_agent_index]),
        player_b_factory_fn=lambda: PlanningPlayer(),
        num_games=200
    )
    print("Tournament against baseline NeuralPlayer:")
    play_tournament(
        player_a_factory_fn=lambda: NeuralPlayer(agents[best_agent_index]),
        player_b_factory_fn=lambda: NeuralPlayer(SimpleAgentCollection.load_default_agent()),
        num_games=200
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main()
