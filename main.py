# A Scout (card game) simulator.
from dataclasses import dataclass
import time
from abc import abstractmethod
from typing import Callable, Sequence
from evaluation import play_tournament
from game_state import GameState
from common import Player
from players import CompositePlayer, EpsilonGreedyScorePlayer, PlanningPlayer, GreedyShowPlayerWithFlip, RandomPlayer
from ismcts_player import IsmctsPlayer
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_games",
    type=int,
    help="Number of games per setting",
    default=100
)
parser.add_argument(
    '--num_rollouts',
    type=lambda s: [int(item) for item in s.split(',')],
    default=[100],
    help="Comma-separated list of the number of MCTS rollouts to run"
)
parser.add_argument(
    '--fix_seed',
    type=bool,
    default=False,
    help="Fix the RNG seed - useful for comparing performance measurements"
)
parser.add_argument(
    "--expansion_file_prefix",
    type=str,
    help="If provided, record expansions and store using this prefix.",
    default=""
)
parser.add_argument(
    '--use_value_fn',
    action='store_true',
    help="Whether to use a value_fn in ISMCTS"
)

def main():
    args = parser.parse_args()
    if args.fix_seed:
        random.seed(10)
    for i in args.num_rollouts:
        awr = play_tournament(
            lambda: IsmctsPlayer(
                5, i,
                lambda: CompositePlayer([EpsilonGreedyScorePlayer(epsilon=0.5),
                                         PlanningPlayer()], weights = [0.4, 0.6]),
                0, args.expansion_file_prefix, args.use_value_fn),
            lambda: PlanningPlayer(),
            num_games = args.num_games)
        print(f"{i}: {awr}")


if __name__ == '__main__':
    main()
