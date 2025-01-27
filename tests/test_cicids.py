from typing import *
import json
import pytest
from tqdm import tqdm
from rich import print as pprint

from anuta.theory import Theory


@pytest.fixture
def cases() -> List[Dict[str, str]]:
    """Fixture to load the queries from the JSON file."""
    path = "tests/queries/cicids_queries.json"
    with open(path, 'r') as f:
        return json.load(f)

@pytest.fixture
def theory() -> Theory:
    modelpath = 'rules/cicids/nodc/learned_cicids_2048.pl'
    # modelpath = 'rules/cicids/nodc/learned_cicids_128_checked.pl'
    # modelpath = 'rules/cicids/dc/learned_cicids_1024.pl'
    # modelpath = 'rules/cicids/dc/learned_cicids_4096.pl'
    return Theory(modelpath)

def test_cicids(cases: List[Dict[str, str]], theory: Theory) -> None:
    """Test that prints the description of each query."""
    successes = 0
    total_queries = 0
    total_cases = len(cases)
    coverage = 0
    for i, q in tqdm(enumerate(cases), total=len(cases)):
        cases = q['queries']
        nqueries = len(cases)
        total_queries += nqueries

        new_successes = 0
        for query in cases:
            entailed = theory.proves(query, verbose=False)
            negative_query = 'Wrong' in q['description']
            if (entailed and not negative_query) or (not entailed and negative_query):
                new_successes += 1
            else:
                print(f"Failed Test #{i+1}")
                pprint("\t", query)
                print(f"\t{q['description']}")
        if new_successes == nqueries:
            coverage += 1
        else:
            pass
        successes += new_successes

    print(f"Coverage={coverage/total_cases: .3f} Recall={successes/total_queries: .3f}")