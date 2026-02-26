# Scout-AI

Scout card game engine, AI players, and infra for training, eval etc.
This package is used for training, eval, simulating games. For an app
that allows humans to play against AIs, check out
https://github.com/myselph/scout-app, which imports this package here
as a dependency.

## Getting Started
This directory is a bit of a mess, because I developed various kinds of players over time, and the methods to evaluate them have improved over time.

* The easiest way to get started is to let two simple AI player implementations compete against each other in a tournament. Check out `main.py` and modify the call to `play_tournament` to e.g. let PlanningPlayer and another player (GreedyShowPlayerWithFlip? RandomPlayer?) compete against each other. Files of importance here are
   * `players.py` which contains various heuristic player implementations
   * `game_state.py` and `common.py` which contain the game engine (most of the complexity comes from determining what the legal moves are)
   * evaluation.py which contains code to let players compete against each other, and some code to rank them with an Elo-like rating system.
* Next, try and implement your own baseline player - look at `PlanningPlayer` in `players.py` and add your own version that uses different heuristics to pick moves. The Player API is real simple.
* Then there's more advanced players:
    1. The first one I implemented was ISMCTSPlayer, which is essentially Monte-Carlo tree search.
    1. I then extended that implementation to also support neural net value functions to stop roll-outs early and pick better moves. This works roughly as follows - I first run normal ISMCTS, recording the traces to disk, then merge those with `merge_pickles.py`, use that to train a neural network value function (`neural_value_function.py` for shared neural net infra like featurization, the neural net model; and `train_neural_value_function.py` for the training loop). The trained model can then be used at inference time by handing ISMCTSPlayer an argument pointing to the pth weight file.
    1. `NeuralPlayer` is a class of players that, for each possible move, estimate how good this move is and return a probability distribution over moves. These players are trained with RL (PPO) and self-play. The main files of importance here are `self_play.py` - entry point, and most of the training heavylifting; and `self_play_agents.py` - contains different neural nets (FFN, Transformers) that power NeuralPlayer.

## Evaluation
A note on evaluation. My original method of evaluation in `play_tournament` to compare player A with player B was to let one player A (the player of interest) compete against 4 player B instances (B usually being some form of baseline - at first, `RandomPlayer`; then eventually, `PlanningPlayer`) by playing something like 100 games a 5 rounds each, and measure the win rates; if in such a 5-player game player A would win 20% of the games, that means its win rate against player B if 50%. That is simple and works fairly well, but as I moved more towards neural self-play and evaluating more players at once (e.g. I train ten neural nets at once and need to rank them relative to each other), I came up with a more general procedure (see `rank()` and `rank_against_planning_player()`): I play a whole bunch of 5-player games, random dealer, random types of players (but including a known baseline such as PlanningPlayer), the calculate a "skill" level for each player using an Elo-like rating system (see `evaluation.py`); this can give you absolute skill levels by fixing a known baseline (usually `PlanningPlayer`) to have a skill of 1.0; and I think it's overall a better way of evaluating players than the old `play_tournament()` method, but it's also more complex.

## History

This was originally developed in https://github.com/myselph/ml, but I had to move it into a separate repository for vercel deployment; see that repo for commit history.
