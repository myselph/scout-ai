# Description
This is Github repo of a Python package containing a simulator and AI agents for the Scout card game.
* `game_state.py`, `common.py` contain most of the game engine
* `players.py` contains baseline AI agents with hardcoded heuristics; `ismcts_player.py` contain a Monte-Carlo AI player that may optionally use neural nets (`neural_value_function.py`, `train_neural_value_function.py`)
* `evaluation.py`, `main.py` are used to evaluate AI agents
* `self_play.py` contains code to train neural net agents in `self_play_agents.py` using PPO and self-play; `neural_value_function.py` contains code for featurizing input for the `SimpleAgent` 

# Run instructions
* Scripts use relative imports and must be run as modules from the `scout-ai/` directory:
  * `python -m scout_engine.self_play` — train via neural self-play (PPO)
  * `python -m scout_engine.main` — run a tournament / evaluate players
* Use the Python venv in ../.venv.
