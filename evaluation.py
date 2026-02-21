# This file holds code for evaluating the performance of players by letting
# them compete. Differenty types of evaluations exists for different purposes:
# * Evaluating a specific player against a known baseline to compute "win rates"
# * Playing num_player rounds with different dealers vs. playing single rounds.
# * Evaluating a set of players to estimate a ranking and skill levels.
# Many evaluation strategies exist, and - as in the real world - the primary
# difficulty is the trade-off between the number of games played and the
# accuracy of the resulting estimates; this holds even more so in a multi-player
# setting with a high degree of environmental randomness.

import random
import time
from typing import Callable
from pyparsing import Sequence
import torch

from common import Player
from game_state import GameState
from players import PlanningPlayer


def play_round(players: Sequence[Player], dealer: int) -> list[int]:
    # Plays a single "round" - that is, a single deck of cards - and returns the
    # scores.
    game_state = GameState(len(players), dealer)
    game_state.maybe_flip_hand([p.flip_hand for p in players])
    current_player = dealer
    while not game_state.is_finished():
        info_state = game_state.info_state()
        move = players[current_player].select_move(info_state)
        game_state.move(move)
        current_player = (current_player + 1) % len(players)
    return game_state.scores


def play_game(players: Sequence[Player]) -> list[int]:
    # Play a game - that is num_players rounds, with each player dealing once.
    # Return the cumulative scores.
    scores = []
    for dealer in range(0, len(players)):
        scores.append(play_round(players, dealer))
        print("#", end="", flush=True)
    return [sum(y) for y in zip(*scores)]

###############################################################################
# 1-vs-N evaluation
# The code below is for evaluating a single player against a known baseline.
# That is, a Player implementation A is pitted against N Player implementations
# B, varying the dealer sequence.
###############################################################################


def play_tournament(
        player_a_factory_fn: Callable[[],
                                      Player],
        player_b_factory_fn: Callable[[],
                                      Player],
        num_games: int = 100) -> float:
    # For now: let one player A compete against 4 player B's. There's a lot of
    # other types of setups we could use, such as 2A vs 3B, with different
    # player sequences; other numbers of players (3, 4); or average between
    # the 1-vs-4 and 4-vs-1 setting. But until I get a better
    # understanding for how to produce rankings for multi-player games, this
    # will do.
    # The resulting scores get normalized - that is, we multiply the win rate
    # of A by 4 before computing the ratio.
    # At the very least, this should be symmetric, ie playing A against B should
    # give the complement (1-p) of playing B against A.
    players = [player_a_factory_fn()] + [player_b_factory_fn()
                                         for _ in range(4)]
    wins = [0] * len(players)

    start_time = time.time()

    for reps in range(0, num_games):
        print(f"game {reps}/{num_games}: ", end="", flush=True)
        scores = play_game(players)
        winner_index = max(range(len(scores)), key=lambda i: scores[i])
        wins[winner_index] += 1
        print("")

    end_time = time.time()

    a_win_rate = wins[0] / (wins[0] + sum(wins[1:]) / 4)
    wins = list(map(lambda i: i / sum(wins), wins))
    print(
        f"wins %: {wins}, a_win_rate normalized: {
            a_win_rate:.3f}, dt/game: {
            (
                end_time -
                start_time) /
            num_games}")
    return a_win_rate


###############################################################################
# Evaluating a set of players with ranking and skill estimation
# The code below is for concurrent evaluation of a whole set of players at once,
# by letting them compete against each other and estimating their "skill level"
# (similar to ELO ratings).
# This works by picking random subsets of players, having them play rounds
# (not games - I expect the impact of being dealer or not gets averaged out),
# and using the Plackett-Luce model to estimate skill levels and thus a ranking.
# This is more appropriate when more than one player needs to be evaluated such
# as in self-training of a population of neural players.
###############################################################################
def rank_players(
        game_results: list[tuple[set[int], int]],
        num_players,
        first_player_skill_val=1.0,
        iterations=100,
        lr=0.1) -> tuple[list[float], list[int]]:
    # Rank the players using a Plackett-Luce model.
    # game_results is a list, each element containing the players that
    # participated in that game, and who won, as integer indices.
    # Player 0 is assumed to have a known skill level (first_player_skill_val),
    # while the others are learned, so the first player serves as a reference
    # point and will always have skill 1.0.
    # Returns: the skills of players in the original order (ie first element is
    # first_player_skill_val), and the player indices ordered in descending order
    # according to their skill values.
    skills_improving = torch.zeros(num_players - 1, requires_grad=True)

    # Only pass the improving agents to the optimizer
    optimizer = torch.optim.Adam([skills_improving], lr=lr)

    # Pre-wrap fixed skill as a non-grad tensor
    fixed_skill = torch.log(torch.tensor([float(first_player_skill_val)]))

    for i in range(iterations):
        optimizer.zero_grad()

        # Combine fixed agent (index 0) with improving agents
        # This creates a full vector [fixed, agent1, agent2, ...]
        all_skills = torch.cat([fixed_skill, skills_improving])
        total_nll = torch.tensor(0.0)

        for participants, winner_idx in game_results:
            # We index into the combined vector
            participant_skills = all_skills[list(participants)]
            winner_skill = all_skills[winner_idx]

            # NLL = log(sum(exp(participants))) - winner_skill
            log_prob = winner_skill - \
                torch.logsumexp(participant_skills, dim=0)
            total_nll -= log_prob

        total_nll.backward()
        optimizer.step()

    with torch.no_grad():
        final_skills_log = torch.cat([fixed_skill, skills_improving])
        final_skills_exp = torch.exp(final_skills_log)
        order = torch.argsort(final_skills_exp, descending=True)

    return final_skills_exp.tolist(), order.tolist()


def rank(players: list[Player],
         num_players_per_round: int,
         num_games_per_player: int = 500) -> tuple[list[int],
                                                   list[float]]:
    # Play rounds (not games - due to random selection, they should all get
    # their fair share of being dealer). Unfortunately, to get a reasonably precise
    # skill level for a player (that is, +- 10%), we need ~500 games per player.
    # For num_agents, where the selection probability is num_players/num_agents,
    # to aim for 500 games we need to play ~500*num_players/num_agents games.
    # This is due to the very high level of non-determinism in the game
    # environment.
    num_games = int(
        num_games_per_player *
        len(players) /
        num_players_per_round)
    game_results = []
    print("Evaluating agents...")
    for _ in range(num_games):
        selected_indices = random.sample(
            range(len(players)), num_players_per_round)
        selected_players = [players[i] for i in selected_indices]
        scores = play_round(selected_players, 0)
        winner_index = max(range(len(scores)), key=lambda i: scores[i])
        winning_player_index = selected_indices[winner_index]
        game_results.append((set(selected_indices), winning_player_index))
    # 3. Rank players using Plackett-Luce model
    skills, order = rank_players(game_results, len(players))
    order = [index for index in order if index != 0]
    # 4. Omit PlanningPlayer from returned list, return agents in ranked order.
    skills = [skills[i] for i in order]
    order = [i - 1 for i in order]
    ftd_skill_list = ", ".join([f"{x:.2f}" for x in skills])
    print(f"Skills + order (1.0 == PlanningPlayer): {ftd_skill_list}, {order}")

    return order, skills


def rank_against_planning_player(
        players: list[Player], num_players_per_round: int,
        num_games_per_player: int = 500) -> tuple[list[int], list[float]]:
    # 1. Create players from agents, and mix in PlanningPlayer as a baseline.
    return rank([PlanningPlayer()] + players, num_players_per_round,
                num_games_per_player=num_games_per_player)
