from typing import *
import json
import pytest
from tqdm import tqdm
from rich import print as pprint

from anuta.theory import Theory


@pytest.fixture
def cases() -> List[Dict[str, str]]:
    """Fixture to load the queries from the JSON file."""
    path = "tests/queries/cidds_queries.json"
    with open(path, 'r') as f:
        return json.load(f)

@pytest.fixture
def theory() -> Theory:
    # modelpath = 'results/cidds/nodc/learned_8192.pl'
    # modelpath = 'results/cidds/nodc/learned_1024_checked.pl'
    # modelpath = 'results/cidds/nodc/learned_256_checked.pl'
    # modelpath = 'results/cidds/dc/learned_256.pl'
    # modelpath = 'results/cidds/dc/learned_4096_checked.pl'
    # modelpath = 'learned_cidds_128.pl'
    modelpath = 'treerules_cidds_cat.pl'
    modelpath = 'treerules_cidds.pl'
    modelpath = 'arules_cidds.pl'
    modelpath = 'xgbrules_cidds.pl'
    # modelpath = 'rules/cidds/new/learned_cidds_8192_filtered.pl'
    # modelpath = 'rules/cidds/new/learned_cidds_8192_checked.pl'
    return Theory(modelpath)

def test_cidds(cases: List[Dict[str, str]], theory: Theory) -> None:
    """Test that prints the description of each query."""
    successes = 0
    total_queries = 0
    total_cases = len(cases)
    coverage = 0
    for i, q in enumerate(cases):
        cases = q['queries']
        nqueries = len(cases)
        total_queries += nqueries

        new_successes = 0
        for query in cases:
            entailed = theory.z3proves(query, verbose=False)
            # assert entailed, f"Failed Test #{i}"
            if entailed:
                new_successes += 1
            else:
                print(f"Failed Test #{i+1}:")
                pprint("\tQuery: ", query)
                print(f"\tMeaning: {q['description']}")
        if new_successes == nqueries:
            print(f"Passed Test #{i+1}")
            coverage += 1
        else:
            pass
        successes += new_successes

    print(f"Coverage={coverage/total_cases:.3f}, Recall={successes/total_queries:.3f}")