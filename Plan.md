Rough plan for implementing an AI that plays Scout.

1. Done - get a simulator (scoring, computing valid moves) up and running
   with random policies.
2. Done - Try some heuristics, and infra for comparing different policies.    
3. Done: ISMCTS experiments
4. Done: ISMCTS + value function
4. WIP: Neural policies value functions

# Neural self-play
## Policy net architecture
* The easiest architecture would be one that takes state and outputs a move, but
I have concerns around the net outputting illegal moves - it would need something
like 2-3 heads (for Scout, for Show, for Scout&Show), and if an illegal move is
generated, it's unclear what else to do (keep trying?). Sure, a good enough net
should not even generate illegal moves, and we could use RL to enforce that, but
that seems unnecessarily hard.
* The next idea I had was to instead have a net that ranks a set of supplied
legal moves and outputs logprobs for each one. What I don't like about this
solution is that the net needs to learn a world model (ie how a move would
affect its hand and the table and scores), which feels entirely unnecessary
since that is a deterministic and cheap operation.
* The third idea was to take the legal moves and calculate the state after making
that move, and let the network rank these states. This would allow the network
to focus on the things we can't easily compute/predict. This is very similar to
what PlanningPlayer does - only that the "value function" is learnt and more
complex. It's also not dissimilar to the value functions I have learnt; but it's
computed *after* a move, and unlike action-value functions it does not predict
a value, just rank actions.

## Features
Lots of possibilities. I'll start with a simple feedforward net like in 
neural_value_function.py which should be able to beat PlanningPlayer - this will
allow me to get the infrastructure working and making sure stuff works, before
diving into more advanced nets like Transformers, and taking things like history
into account.

## Strategy / TODO
1. Done: Try to get as good as PlanningPlayer with a very simple setup - just train
   5 policies.
1. Done: Track performance. Every 5 iterations, I play matches between the players
   (~50 / player) and the known baseline PlanningPlayer, then rank them +
   compute their skills using the Plackett-Luce model. This is very insightful
   in showing actual convergence (and divergence), but unfortunately is very
   slow. I also found that one needs a very high number of games played between
   players to get a stable ranking / skill level - about 500 - 1000 per player.
1. WIP: Experiment with a larger population of agents to add diversity, and
   keeping the best old players around.
   WIP: Baseline gets pretty good; quickly surpasses PlanningPlayer with the
   best agent, other agents follow quickly. Exciting! I may need a new heuristic
   baseline. E.g. one that has a curriculum of strategies. Or that takes into
   account other player's running scores. Or just keep the best neural player
   I've trained as new baseline.
   When using 10 instead of 5 agents + larger networks (128:64:1): first PP beat @ 15, 8/10 @20. Then regression (back to 2), recovery, regression.
   When using 10 instead of 5 agents + smaller network (32:8:1): first PP beat @ it 35; 5/10 beat it fter 50. Definitely slower progress, but gets there just the same.
   For both, feels like similar progress to just 5 agents - not sure diversity helps (yet).
   Kinda puzzled by the regressions. I expected the players to keep getting better,
   but they top out around 1.9, then regress, then recover. It may be they hit
   a natural limit with their architecture / feature set - I'm very curious now
   to try a Transformer or RNN. Or better features such as "almost-long-run".
   I have not yet tried keeping the best players around.
1. Add a new baseline? I like PlanningPlayer because it is stable and easy to
   understand; a neural player will require keeping weights and the exact
   featurization around. My concern around PP as baseline is that two other
   players A, B might have relative skills, but get the same win rates against
   PP (under a reasonably large number of games) because there's an upper bound
   of how much you can win against PP due to the inherent non-determinism.
1. Try Transformers. May benefit from initializing with imitation learning,
   unclear.

## Hyperparameter optimization & Learnings
* Tried different batch sizes; 128 didn't work as well, 256 & 512 better / about
  the same.
* With batch size fixed, tried different combos of learning rates, iterations,
  epochs, and episodes. LRs < 1e-3 seem to be too small; LRs 1e-2 seems to be
  too big. I stuck with 3e-3 for now but think LR schedule or 1e-3 with more
  data/iterations is still in the cards.
* Then for B=512, lr=1e-3, I tried to keep comp budget about constant by
  requiring iterations * epochs * episodes = 3200 (40 * 4 * 20), and varied
  those params, measuring how well they worked in win_rate against
  PlanningPlayer. 10 iterations never seems to be enough, so I omitted that
  from the table below. Got the follow results - columns iterations, rows epochs:

lr 1e-3:

|   | 20 | 40 | 80 |
| - | -  | -  | -  |
| 2 | 13 | 13 | 42 |
| 4 |  2 | 10 | 36 |
| 8 | 19 | 49 | 30 |

lr 3e-3
|   | 20 | 40 | 80 |
| - | -  | -  | -  |
| 2 | 37 | 51 | 54 |
| 4 | 32 | 49 | 52 |
| 8 | 11 | 17 | 13 |

* With roughly double the budget - lr 1e-3, 80 iterations, 2 epochs, 40 episodes,
  I get to a win rate of ~64% (with 80 episodes, same thing), which is great -
  I win against PlanningPlayer.
* Training is somewhat unstable, eg I found win rates against PlanningPlayer cna
  vary a fair bit (40%, 60%) across two runs if little data (100 examples per
  minibatch) is used; gets more reliable with more data. Also it seems the
  performance of players oscillates after a while (when looking at game length)
  but I need to add that progress tracking to really know.



# On ISMCTS
After a long run, and thinking through ISMCTS as applied to Scout, I came away
with the conclusion that it is not the appropriate tool for the job.
Take a game of Scout with five players, 45 cards distributed across players and
the table; and imagine a playout tree where each node corresponds to an
opportunity for the player to take an action, and edges between nodes represent
five actions (one by each player).
Consider the branch factor of the search tree:

1. There are on the order of O(10) possible moves every time an action is selected,
and for five players, that means we'd have on the order if 100k child nodes -
ballpark estimate, there are frequently more moves than just 10.
1. At each node, the player has incomplete information, and the number of
possible game states is huge (even after accounting for removed cards, and
know assignments of cards to players) - way beyond 1M. We'd have O(100k)
child nodes for each one of these states.
1. If we were to explore less and bias more towards picking strong actions, that
might reduce the branching factor by maybe 10x only.

What that means in practice is that if we perform N rollouts where N = O(1000),
we mostly visit child nodes once even at the first level of the tree, never
mind subsequent levels. That refutes the whole point of building a tree, which
is to aggregate (and possibly cache) information, and simply doing N independent
roll-outs from the root node (instead of all the ISMCTS overhead - UCT, the
different phases, expansion, buildign a tree) would give the same result.
The above back-of-an-envelope calculations mostly match actual runs - there is
little reuse of previously created nodes, and a "Flat Monte Carlo" approach
would suffice.

I realize that the ISMCTS investment was probably a waste of time, and I may end
up with what I wanted to do from day 1 - train policy networks with
self play.


# Appendix
## ISMCTS experiments
1. My original ISMCTS version used absolute scores and was fairly slow.
   It barely won 1% against PlanningPlayer even with 20k roll-outs, so I used
   GreedyShowPlayerWithFlip in the experiments here and below.
   For the original scoring and all others, I found a saturating relationship
   between rollouts and win rate. This one was roughly
   0% at 40, 50% at 125, and 75%
1. Next I sped things up 3x through different move generation and confirmed the
   win rate / rollout relationship was the same. So this CL here (1/16/26, 8pm)
   is a baseline working ISMCTS version:
   For 40, 70, 100, 130, 160, 200, 250, 300, respectively:
   0, 22, 40, 48, 64, 68, 77, 81.
1. Changing the scores from absolute to score := diff from opponent avg. made a
   huge difference, and is now the new score definition.
   For 40, 70, 100, 130, 160, 200, 250, 300, respectively:
   4, 56,  70,  75,  81,  83,   ?,  91
1. Using EpsilonGreedyPlayer during ISMCTS rollouts instead of randomly picking
   moves also helped:
      Using an EpsilonGreedy(eps=0.8), this gives
      for 40, 70, 100, 130, 160, 200, 250, 300:
           2, 58,  71,  81,  83,  88,  89,  94
      Using EpsilonGreedy(eps=0.5) is a bit better:
      for 40, 70, 100, 130, 160, 200, 250, 300:
           9, 57,  83,  85,  91,  94,  97,  96
      Using EpsilonGreedy(eps=0.2):
      for 40, 70, 100, 130, 160, 200, 250, 300:
           2, 61,  83,  91,  93,  97,   ?,   ?
      Not sure yet what to make of the EpsilonGreedy stuff. It works significantly
      better than just picking random moves (eps=1), but I'm a bit concerned about
      overfitting (does this extend to other players?)/underexploring, and
      having another hyperparameter (epsilon).
1. Experiments against PlanningPlayer: With the new scoring method and
   with EpsilonGreedyPlayer(0.5), I get for 150, 500, 1000, 2000, 3000, 5000 sims:
   2, 16, 43, 54, 62, 60. Great - had single digit percent rates when trying 20k
   moves before my improvements. However, given the duration of these
   simulations (collecting that data took >10h), it also begs the question
   about MCTS efficiency.
1. Neural value function experiments.
  1. I implemented recording traces, following ChatGPT's advice, which I have
     medium high confidence in.
     Algorithm: When expanding (whether it's a new or existing action), record
     the info state and the resulting score; collect 1M data points, train a
     neural net predicting the score. Use that net to essentially replace
     rollouts.
    * The details of the algorithm and implementation are a bit messy, and
       it's possible I got something wrong. The ISMCTS algorithm still proceeds
       through the stages of selection, expansion, rollout and backprop, but
       the roll-out is adjusted to stop as soon as possible, ie the first time
       it is the player's turn again.
    * On the data I collected (200 rollouts, 1M traces), the neural net MSE loss
      is not very confidence inspiring - a randomly initialized net gets ~45, merely
      predicting the mean score gives ~37, and a converged net gives ~29 on the
      validation set. I suspect that the problem of predicting scores is a very
      hard one due to the high variance. The usual "tricks" - different archs,
      nonlinearities, input normalization weight decay etc. all didn't help, and
      the net mostly converges to its lowest loss after a single epoch.
    * The unoptimized version is maybe 30% faster than using full roll-outs.
      It should be possible to improve this by using a significantly smaller
      network and batching NN calls (a bit tricky because ISMCTS and UCT are
      both inherently sequential, and would require rewriting ismcts()). For
      now, it makes more sense to focus on fully measuring win rates and first.
    * I compared ISMCTS with a GreedyEpsilonPlayer(0.5) against a PlanningPlayer
      for different roll-outs numbers in two configurations - without, and with
      the neural value function. The win rates are a bit puzzling.
      * w/o value function: for 40, 70, 100, 150, 200, 300, 400, 500, 1000, 2000,
      3000, 5000, I get 0,0,0,2,?,?,?,16,43,54,62,60.
      * w/ value function: for 40, 70, 100, 150, 200, 300, 400, 500, 1000, 2000,
      3000, 5000, I get 0,6,8,9,23,?,?,53,45,48,57.
      The value function curve looks much less smooth (as for all numbers above,
      I used 200 games, each 5 rounds) and gets worse after 500 rollouts. A bit
      hard to tell noise from trend. It is encouraging to see that for low
      numbers of roll-outs, the net adds value by providing more smoothed
      numbers (eg I get a win rate of 23% for 200 rollouts when using a neural
      value function vs. ? without) - that at least suggests it's learning
      something. But the breakdown shows that quickly, the neural net value
      function stops adding value and even gets worse than epsilon-greedy
      exploration. I can think of a couple of causes:
      * The net was trained on data from 200 rollouts and won't perform well on
        larger rollouts anymore, due to biases in the sampling procedure. I
        don't find this explanation very convincing - I expect increasing the 
        number of rollouts primarily creates more of the same training data, and
        whether I use 200 rollouts in 1000 games or 2000 rollouts in 100 games
        should not matter (ChatGPT strongly disagrees).
      * Roll-outs take advantage of more info than the neural net has; the
        determinization procedure takes into account what cards other players
        are known to have from scouting. It's possible that this gives it some
        edge the more we sample. Put differently, sampling more when using a
        neural value function gives you more of the same distribution; sampling
        more without a neural value function lets you converge towards the
        actual distribution. I can't think of a good way how to let the neural
        net make use of that info.
      * The neural net needs better features or more information. E.g. it does
        not currently consider how good a hand or scores are after taking
        actions (but this goes way into feature engineering). Eg I could add
        some info about the score diffs for the N best actions. But at that
        point, I would have effectively built PlanningPlayer++
      I am rather surprised how hard it seems to be to beat PlanningPlayer,
      because that is essentially a 1-level (40 moves?) search tree with a
      hard-coded value function, without any form of modeling opponent behavior.
      Prompting ChatGPT, it says that's expected, not the norm (but it says that
      about most observations I give it regardless of its prior predictions).
      But the arguments (obvious - how ISMCTS is handicapped, how a strong
      heuristic bypasses learning obvious things and works really well when
      my actions matter and dominate long term combinatorial effect).
      That's quite interesting, and may also explain heuristic approaches were
      (and still are!) so popular in game AIs. They may be really hard to beat
      if designed by somebody with a good understanding of the game; and while
      MCTS will always beat them in the limit of countless simulations, the
      number of sims you need to get to that point may be excessive (or at
      least hardly worth the risk). So it makes sense that combining the best
      of both worlds - strong hand-engineerd value or action-value functions
      plus MCTS - may provide the biggest bang for the buck.
      Of course I wanted a different answer - just throw a ton of data at it
      and get good. I am still pursuing that avenue - I plan to make the
      neural net function more powerful, and I then want to move on to neural
      policy functions.
    * I traced ISMCTS with stronger simulated players - 40% EpsilonGreedy, 60%
      PlanningPlayer - playing against PlanningPlayer's. The neural net trained
      on that exhibits lower loss (MSE=20, down from 29) so things become more
      predictable / lower variance. Results when using that value_fn are mixed/
      high variance - 0% for 200, 53% for 400.
    * An idea for how to speed up ISMCTS w/ neural value fns is to try all
      untried actions on a "virgin" node at once. In classic ISMCTS, we pick
      actions on unexpanded nodes (such as the first one) until there are no
      untried actions left, but each one of those looks the same, and we don't
      use UCT until a node is fully expanded. It will lead to different outcomes
      though - today, we can end up with many partially expanded nodes.
   1. Using stronger opponents during ISMCTS roll-outs:
     1. W/o value_fn:
     1. for value_fn training: MSE loss strongly reduced (now 20), meaning the
        scores are much better predictable.
   1. Next steps:
     1. Use stronger opponents during the ISMCTS roll-outs, but retain some
       randomness, e.g. by extending PlanningPlayer with epsilon-greedy
       selection, and/or combos strong and weak players (both with randomness).
     1. Use Transformer encoders for the table and hand, but only after above,
        because clean labels / non-garbage data needs to come before more
        advanced neural nets.
     1. Move on to neural policies and self-play.
1. Blog post. Contents?
  1. Rough journey
    1. How game works; what makes it interesting: no obvious strategy, yet one
    person consistently won. The randomness is high - lots placketof surprises, big
    changes in the end phase - so I wondered what a winning strategy looks like.
    Could let computers get good at it, than try and understand what they do.
    (not *why* they do it - just see if patterns emerge such as curricula, or
    greedy behavior, etc.).
    1. instead of starting right away with neural policies,
    co-designed w/ AI -> suggested reasonable path. heuristics, ISMCTS, w/
    neural value functions, full blown neural policies w/ self play.
    1. Then build basic infra - the interfaces Players need
    to implement, a simple random player, and a simulator. More work than I
    expected - in particular, finding all valid moves in an efficient manner
    wasn't exactly trivial and took a bit of iterating.
    1. Starting to rank players - 3 heuristic players. Very easy to implement,
    but as it turns out, strongest not easy to beat.
    1. ISMCTS. Type safety becoming crucial to get anything done, hashing 
    InformationStates. Sampling possible game states. Performance becoming a
    problem -> hot spot analysis, optimizations. Heavily relying on talking w/
    LLM about edge cases; in hindsight I feel I should have just read the paper
    to get a deeper understanding and feel more confident about my 
    implementation because occasional LLM slips. Hard to tell when to go that
    deep and when to just rely on AI. Slow but better than many players;
    redefining score to be relative was crucial.
    1. Hidden state / partial observable state seems to be core difficulty: it
    makes it very challenging or impossible to compute values of states because
    the variance is so high. Another difficulty is the number of players -
    between your current state, and the next time you get to make a move, there
    are N-1 other player actions based on hidden state, leading to extremely
    high branching factors.
    rather different setting - for each node there's pretty much an infinite
    number of child nodes, because a) there are 5 players, each with O(10)
    moves, especially early on when card counts are high - that's 100k child
    nodes, and we have a rollout of 200! b) the variance (difference in scores
    when picking an action and playing out results) is very high - even towards
    the end of the game. So I'm not sure how useful UCT is as a strategy as
    compared to either just random sampling (then playouts with strong policies)
    or a heuristic and randomized policy chosing strong moves.
    1. Use of neural value functions - trace, train, skip rollouts. Worked
    better for low number of roll-outs by reducing variance, and offers 
    speedups by skipping rollouts and allowing for batching, but still hard to
    compete against best heuristic player. At this point, impressed how far the
    classic methods go; it took me minutes to write the heuristic policy, and
    I could easily make it better by incorporating more of my own player
    behaviors. Even just replicating this with tree search, and neural value
    functions, is a significant endeavour in terms of engineering and
    computational requirements. Depending on what the goal is - build the best
    possible player for a competitive environment? build a player good enough
    for entertainment purposes (hobby players, runs on phones) - a heuristic
    player may be far preferable. I was somehow hoping or expecting that thanks
    to self-play and neural net advances, I would very quickly surpass hand-
    written functions or players.
    1. Realization that ISMCTS is the wrong tool (or rather offers no advantage
    over flat Monte Carlo). Next up, neural policies. PPO. Reward hacking problems.
    Need to rank players (Plackett-Luce: like Elo, but for multi-player, and batch-wise, not online), also how to track progress (rewards / loss not
    helpful).