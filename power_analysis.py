################################################################
# This file is used to rank players and compute skills, and do
# power analysis to see how many games are needed to get a
# reliable ranking. Specifically, it loads a pickle file containing
# game results of the form [[a,b,c,d,e],x] where a,b,c,d,e are player
# indices and x is the winner (one of a-e), which should be a very
# large number of games - think ~500 - 1000 per player - computes
# skills and order; then samples subsets and repeats the process to
# see how much things vary. That should give us a rough idea of how
# many games are needed to get a reliable ranking and skill estimate.
# I suspect the answer depends a bit on how close players are in
# skill, but I found that to be within 10% of the actual skill,
# you will want 500-1000 games per player, which is a lot.
################################################################
import pickle
import random
import sys
import numpy as np
from self_play import rank_players
 

def get_skills_and_order(results: list[tuple[list[int],int]], num_players: int) -> tuple[list[float], str]:
    # Compute skills and order, return skills in original player order.
    skills, order = rank_players(results, num_players)
    skills_str = ", ".join(f"{s:.2f}" for s in skills)
    return skills, skills_str

def main():
    results: list[tuple[list[int],int]] = []
    with open(sys.argv[1], "rb") as f:
        results = pickle.load(f)
    pass
    print(len(results))

    # 1. Establish the baseline rank and skills
    num_players = max(max(idx) for idx, _ in results) + 1
    _, skills_str = get_skills_and_order(results, num_players)
    print(f"Baseline skills: {skills_str}")
    print("=============================")

    # Try different subset sizes to get an idea of how much things vary.
    for i in [2, 4, 8, 16, 32]:
        N = int(len(results) / i)
        print("Subset size:", N)
        skills_list = []
        reps = 10
        for j in range(reps):
            # Sample subset of games, make sure all players are covered.
            coverage = False
            while not coverage:
                subset_indices = random.sample(range(len(results)), N)
                coverage = all(
                    any(p in results[k][0] for k in subset_indices)
                    for p in range(num_players))
                
            skills, skills_str = get_skills_and_order(
                [results[k] for k in subset_indices], num_players)
            print(f"    Sample {j}: skills: {skills_str}")
            skills_list.append(skills)
            
        skills = np.array(skills_list)
        skills_avg = np.mean(skills, axis=0)
        skills_std = np.std(skills, axis=0)
        print(f"  Avg/Std: {skills_avg} +- {skills_std}")
        print("-----------------------------")
    
            




    

if __name__ == "__main__":
    main()
