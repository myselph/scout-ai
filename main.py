# A Scout (card game) simulator.
from evaluation import play_tournament, rank_against_planning_player
from players import CompositePlayer, EpsilonGreedyScorePlayer, PlanningPlayer, GreedyShowPlayer, RandomPlayer
from self_play_agents import NeuralPlayer, SimpleAgentCollection
from ismcts_player import IsmctsPlayer
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_games_per_player",
    type=int,
    help="Minimum number of games per player in tournament",
    default=5000
)
parser.add_argument(
    '--fix_seed',
    type=bool,
    default=False,
    help="Fix the RNG seed - useful for comparing performance measurements"
)

def main():
    args = parser.parse_args()
    if args.fix_seed:
        random.seed(10)
    # Here we create a pool of 5 players (4 below, and rank_against_planning_player adds a 5th)
    # let them compete a bunch, then rank them. This method isn't perfect, and I suspect no
    # method is, because it's possible to have network effects such as third-order correlations
    # (eg two players of the same type may benefit or suffer from each other), but it seems
    # better than what I did before (1-vs-N eval) against a fixed baseline.
    # NB it can take a lot of games to get a good estimate of player's skills - even at 2,000 per
    # player I still see +-0.1 variations in the estimates. I suspect this is because luck plays
    # a big role in Scout, especially when players don't play optimally.
    players = {
        'Random Player': RandomPlayer(),
        'Greedy Show Player': GreedyShowPlayer(),
        'Planning Player (scout_penalty=1.5)': PlanningPlayer(scout_penalty=1.5),
        'Neural Player': NeuralPlayer(SimpleAgentCollection.load_default_agent()),
    }
    # Currently only testing 5-player games because the NeuralPlayer wasn't trained
    # on 3 or 4 player games.    
    for num_players in range(5, 6):
        print(f"Ranking using {num_players}-player games")
        order, skills = rank_against_planning_player(list(players.values()), num_players, num_games_per_player=args.num_games_per_player)
        for player_index, skill in zip(order, skills):
            print(f"{list(players.keys())[player_index]}: {skill}")

if __name__ == '__main__':
    main()
