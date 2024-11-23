from multiprocess import Pool
import numpy as np
import psutil
import sys
from rich import print as pprint
import pandas as pd
from tqdm import tqdm
import sympy as sp
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import *
from copy import deepcopy

from grammar import Anuta, log
import json


anuta : Anuta = None

def test_constraints(worker_idx: int, dfpartition: pd.DataFrame) -> List[int]:
    global anuta
    assert anuta, "anuta is not initialized."
    
    log.info(f"Worker {worker_idx+1} started.")
    #* 1: Violated, 0: Not violated
    violations = [0 for _ in range(len(anuta.kb))]
    
    for _, sample in tqdm(dfpartition.iterrows(), total=len(dfpartition)):
        # print(f"Testing with sample {j+1}/{len(dfpartition)}.", end='\r')
        assignments = {}
        for name, val in sample.items():
            if name not in anuta.variables:
                continue
            assignments[anuta.variables[name]] = val
        
        for k, constraint in enumerate(anuta.kb):
            if violations[k]: 
                #* This constraint has already been violated.
                continue
            #* Evaluate the constraint with the given assignments
            sat = constraint.subs(assignments | anuta.constants)
            if not sat:
                violations[k] = 1
    log.info(f"Worker {worker_idx+1} finished.")
    return violations

def save_constraints(constraints, fname: str='constraints'):
    # Convert expressions to strings
    expressions_str = [str(expr) for expr in constraints]

    # Save to a JSON file
    with open(f"{fname}.json", 'w') as f:
        json.dump(expressions_str, f)
    
def main(metadf: pd.DataFrame, label: str):
    global anuta
    assert anuta, "anuta is not initialized."
    
    #* Prepare arguments for parallel processing
    core_count = psutil.cpu_count()
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(metadf, core_count)]
    args = [(i, df) for i, df in enumerate(dfpartitions)]

    # Use multiprocessing Pool to test constraints in parallel
    print(f"Testing constraints in parallel ...")
    pool = Pool(core_count)
    violation_indices = pool.starmap(test_constraints, args)
    # pool.close()

    log.info(f"All workers finished.")
    aggregated_violations = np.logical_or.reduce(violation_indices)
    learned_kb = []
    log.info(f"Aggregating violations ...")
    #* Update learned_kb based on the violated constraints
    for index, is_violated in tqdm(enumerate(aggregated_violations), total=len(aggregated_violations)):
        if not is_violated:
            learned_kb.append(anuta.kb[index])

    removed_count = len(anuta.kb) - len(learned_kb)
    print(f"{len(learned_kb)=}, {len(anuta.kb)=} ({removed_count=})\n")
    save_constraints(learned_kb, f'learned_{label}')
    
    log.info(f"Filtering redundant constraints ...")
    assumptions = [v >= 0 for v in anuta.variables.values()]
    cnf = sp.And(*(learned_kb + assumptions))
    simplified_logic = cnf.simplify()
    reduced_kb = list(simplified_logic.args) \
        if isinstance(simplified_logic, sp.And) else [simplified_logic]
    filtered_count = len(learned_kb) - len(reduced_kb)
    print(f"{len(learned_kb)=}, {len(reduced_kb)=} ({filtered_count=})\n")
    
    save_constraints(reduced_kb, f'reduced_{label}')
    

if __name__ == '__main__':
    file = f"./data/meta_{sys.argv[1]}.csv"
    print(f"Loading data from {file}")
    metadf = pd.read_csv(file)
    
    variables = []
    for col in metadf.iloc[:, 2:]:
        if not 'egress' in col and not 'out' in col:
            # if len(col.split('_')) > 1 and col.split('_')[1].isdigit(): continue
            variables.append(col)

    constants = {
        'burst_threshold': round(2891883 / 7100), # round(0.5*metadf.ingressBytes_sampled.max().item()),
    }
    # print(f"{variables=} {constants=}")
    
    anuta = Anuta(variables, constants, operators=[0, 1, 2])
    pprint(anuta.variables)
    pprint(anuta.constants)
    
    anuta.populate_kb()
    
    main(metadf, sys.argv[1])
    sys.exit(0)