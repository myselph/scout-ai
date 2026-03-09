# Scout-AI

Scout card game engine, AI players, and infra for training, eval etc.
This package is used for training, eval, simulating games. For an app
that allows humans to play against AIs, check out
https://github.com/myselph/scout-app, which imports this package here
as a dependency.

## Getting Started
This directory is a bit of a mess, because I developed various kinds of players over time, and the methods to evaluate them have improved over time.

* The easiest way to get started is to let two simple AI player implementations compete against each other in a tournament. Check out `main.py` and modify the call to `play_tournament` to e.g. let PlanningPlayer and another player (GreedyShowPlayer? RandomPlayer?) compete against each other. Files of importance here are
   * `players.py` which contains various heuristic player implementations
   * `game_state.py` and `common.py` which contain the game engine (most of the complexity comes from determining what the legal moves are)
   * evaluation.py which contains code to let players compete against each other, and some code to rank them with an Elo-like rating system.
* Next, try and implement your own baseline player - look at `PlanningPlayer` in `players.py` and add your own version that uses different heuristics to pick moves. The Player API is real simple.
* Then there's more advanced players:
    1. The first one I implemented was ISMCTSPlayer, which is essentially Monte-Carlo tree search.
    1. I then extended that implementation to also support neural net value functions to stop roll-outs early and pick better moves. This works roughly as follows - I first run normal ISMCTS, recording the traces to disk, then merge those with `merge_pickles.py`, use that to train a neural network value function (`neural_value_function.py` for shared neural net infra like featurization, the neural net model; and `train_neural_value_function.py` for the training loop). The trained model can then be used at inference time by handing ISMCTSPlayer an argument pointing to the pth weight file.
    1. `NeuralPlayer` is a class of players that, for each possible move, estimate how good this move is and return a probability distribution over moves. These players are trained with RL (PPO) and self-play. The main files of importance here are `self_play.py` - entry point, and most of the training heavylifting; and `self_play_agents.py` - contains different neural nets (FFN, Transformers) that power NeuralPlayer.

## Evaluation
A note on evaluation. I have tried a couple of different methods, but have converged to an Elo-like rating system with a fixed baseline. That gives not just rankings of different
players against each (as opposed to just pairwise comparisons), but also allows for determining absolute skill levels by fixing a known baseline (PlanningPlayer) to have a skill of 1.0. See `main.py` for an example of how to use this system to rank players.

## Results summary
In the following, I'll report "skills" as derived from the Placket Luce model
relative to a PlanningPlayer which is at 1.0, for 5-player-games.
* RandomPlayer is close to 0
* GreedyShowPlayer is ~0.1
* PlanningPlayer with a scout penalty of 1.5 (instead of 1) gets ~1.15. I'm sure there's other knobs in PlanningPlayer that cold be tuned to give better results still.
* NeuralPlayer with a simple neural net: I got to roughly 1.5 here
* Transformers: always got stuck at around 0.35.

For 3 and 5 players, the NeuralPlayer had a skill level of 1.4; I think the 
other players were similar. Didn't measure Transformer.

## Training tips
Learning the simple neural player was somewhat stable for 5 players with the
parameters I checked in, meaning it always will exceed PlanningPlayer. I found
it to be less stable or hard to converge for 4 players (I used a learning rate
schedule to drop the lr once PlanningPlayer performance was reached), and little
success with 3 players. Since the 5-player net does very well on 3-player games,
I did not pursue it any further. I found that the number of trainable players,
and whether or not non-trainable players are injected into training data collection,
did not matter that much, but also doesn't hurt (other than performance), so it's
probably a good thing to have in there anyway.
Re monitoring training progress - ideally, one competes against the PlanningPlayer
after every round, but since that takes significantly longer than

## Future Work
I never had success training Transformers; there is some convergence to a skill
of ~0.35 which may be trivial to accomplish (ie without understanding the cards
at all by just looking at a feature such as post-move-score). I tried a lot of
different feature embeddings (learnable per card, per segment, and position
embeddings; sinusoidal embeddings), widely different learning rates; injecting
PlanningPlayer into roll-out generation; larger and smaller batch sizes;
small (8) and large (64) embedding sizes; different feedforward sizes.
One thin that may be worth trying is to use the Transformer for hand (and maybe
table) cards only; concat that output embedding with the other features like we
do in the MLP. That may make it easier for it to learn groups, sets etc., and
I'm not sure that cramming things as different as scores and cards into the same
token sequence makes sense anyway.

Other than that, the usual - larger sweep over training hyperparams, dataset
sizes; larger models (they may be able to just memorize useful sequences); more
featurize engineering.

## History
This was originally developed in https://github.com/myselph/ml, but I had to move it into a separate repository for vercel deployment; see that repo for commit history.
