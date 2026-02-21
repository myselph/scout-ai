import pickle
from common import StateAndScoreRecord
import os

def main():
    limit = 1_000_000
    pickle_files = [f for f in os.listdir() if f.startswith('expansions_')]
    records:list[StateAndScoreRecord] = []
    for f in pickle_files:
        with open(f, "rb") as pf:
            records += pickle.load(pf)
        if len(records) >= limit:
            break

    with open("expansions.pkl", "wb") as pf:
        pickle.dump(records, pf)

if __name__ == '__main__':
    main()