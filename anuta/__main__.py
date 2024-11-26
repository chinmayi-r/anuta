from multiprocess import Pool, Manager, managers
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
from time import perf_counter

from grammar import Anuta, log, IntBounds
import json


window = 10
anuta : Anuta = None

def test_constraints(worker_idx: int, dfpartition: pd.DataFrame) -> List[int]:
    global anuta
    assert anuta, "anuta is not initialized."
    
    # var_bounds = anuta.bounds
    
    log.info(f"Worker {worker_idx+1} started.")
    #* 1: Violated, 0: Not violated
    violations = [0 for _ in range(len(anuta.kb))]
    
    for i, sample in tqdm(dfpartition.iterrows(), total=len(dfpartition)):
        # print(f"Testing with sample {j+1}/{len(dfpartition)}.", end='\r')
        canary_max = sample.iloc[-window:].max()
        canary_premise = i % 2 #* 1: old index, 0: even index
        canary_conclusion = anuta.constants['burst_threshold'] + 1 if canary_premise else 0
        assignments = {}
        #* Assign the canary variables
        for cannary in anuta.canary_vars:
            if 'max' in cannary.name:
                assignments[cannary] = canary_max
            elif 'premise' in cannary.name:
                assignments[cannary] = canary_premise
            elif 'conclusion' in cannary.name:
                assignments[cannary] = canary_conclusion
        for name, val in sample.items():
            var = anuta.variables.get(name)
            if not var: continue
            assignments[var] = val
            
            # bounds = var_bounds.get(name)
            # if bounds and (val < bounds.lb or val > bounds.ub):
            #     #* Learn the bounds
            #     # mutex.acquire()
            #     # print(f"Updating bounds for {name}: [{bounds.lb}, {bounds.ub}]")
            #     var_bounds[name] = IntBounds(
            #         min(bounds.lb, val), max(bounds.ub, val))
            #     # print(f"Updated bounds for {name}: [{var_bounds[name].lb}, {var_bounds[name].ub}]")
            #     # mutex.release()
        
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
    global anuta
    # Convert expressions to strings
    expressions_str = [str(expr) for expr in constraints]

    with open(f"{fname}.json", 'w') as f:
        json.dump(expressions_str, f, indent=4, sort_keys=True)
        #* Save the variable bounds
        print(f"Saved to {fname}.json")
    # with open(f"{fname}_bounds.json", 'w') as f:
    #     json.dump({k: (v.lb, v.ub) for k, v in bounds.items()}, f)
    #     print(f"Saved to {fname}_bounds.json")
    
def main(metadf: pd.DataFrame, label: str):
    global anuta
    assert anuta, "anuta is not initialized."
    
    start = perf_counter()
    
    #* Prepare arguments for parallel processing
    core_count = psutil.cpu_count()
    # mutex = Manager().Lock()
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(metadf, core_count)]
    args = [(i, df) for i, df in enumerate(dfpartitions)]

    # Use multiprocessing Pool to test constraints in parallel
    print(f"Testing constraints in parallel ...")
    pool = Pool(core_count)
    # violation_indices, bounds_array = pool.starmap(test_constraints, args)
    violation_indices = pool.starmap(test_constraints, args)
    # violation_indices = [r[0] for r in results]
    # bounds_array = [r[1] for r in results]
    # pool.close()

    log.info(f"All workers finished.")
    aggregated_violations = np.logical_or.reduce(violation_indices)
    # aggregated_bounds = {k: IntBounds(sys.maxsize, 0) for k in anuta.bounds.keys()}
    learned_kb = []
    log.info(f"Aggregating violations ...")
    #* Update learned_kb based on the violated constraints
    for index, is_violated in tqdm(enumerate(aggregated_violations), total=len(aggregated_violations)):
        if not is_violated:
            learned_kb.append(anuta.kb[index])
    
    end = perf_counter()
    # log.info(f"Aggregating bounds ...")
    #* Update the bounds based on the learned bounds
    # for bounds in bounds_array:
    #     for k, v in bounds.items():
    #         if v.lb < aggregated_bounds[k].lb:
    #             aggregated_bounds[k].lb = v.lb
    #         if v.ub > aggregated_bounds[k].ub:
    #             aggregated_bounds[k].ub = v.ub

    removed_count = len(anuta.kb) - len(learned_kb)
    # pprint(aggregated_bounds)
    print(f"{len(learned_kb)=}, {len(anuta.kb)=} ({removed_count=})")
    save_constraints(learned_kb, f'learned_{label}')
    print(f"Time taken: {end-start:.2f}s\n\n")
    
    # log.info(f"Filtering redundant constraints ...")
    # assumptions = [v >= 0 for v in anuta.variables.values()]
    # cnf = sp.And(*(learned_kb + assumptions))
    # simplified_logic = cnf.simplify()
    # reduced_kb = list(simplified_logic.args) \
    #     if isinstance(simplified_logic, sp.And) else [simplified_logic]
    # filtered_count = len(learned_kb) - len(reduced_kb)
    # print(f"{len(learned_kb)=}, {len(reduced_kb)=} ({filtered_count=})\n")
    
    # save_constraints(reduced_kb, f'reduced_{label}')
    

if __name__ == '__main__':
    boundsfile = f"./data/meta_bounds.json"
    file = f"./data/meta_w10_s5_{sys.argv[1]}.csv"
    print(f"Loading data from {file}")
    metadf = pd.read_csv(file)
    
    variables = []
    for col in metadf.columns:
        if col not in ['server_hostname', 'window', 'stride']:
            # if len(col.split('_')) > 1 and col.split('_')[1].isdigit(): continue
            variables.append(col)
    constants = {
        'burst_threshold': round(2891883 / 7200), # round(0.5*metadf.ingressBytes_sampled.max().item()),
    }
    
    canaries = {
        'canary_max10': (0, metadf.ingressBytes_aggregate.max().item()),
        #^ Max(u1, u2, ..., u10) == canary_max10
        'canary_premise': (0, 1),
        'canary_conclusion': (constants['burst_threshold']+1, constants['burst_threshold']+1),
        #^ (canary_premise > 0) => (canary_max10 + 1 ≥ burst_threshold)
    }
    variables.extend(canaries.keys())

    #* Load the bounds directly from the file
    with open(boundsfile, 'r') as f:
        bounds = json.load(f)
        bounds = {k: IntBounds(v[0], v[1]) for k, v in bounds.items()}
    # bounds = {}
    # for col in metadf.columns:
    #     if col in ['server_hostname', 'window', 'stride']: 
    #         continue
    #     bounds[col] = IntBounds(metadf[col].min().item(), metadf[col].max().item())
    for n, c in constants.items():
        bounds[n] = IntBounds(c, c)
    for n, c in canaries.items():
        bounds[n] = IntBounds(c[0], c[1])
     
    anuta = Anuta(variables, bounds, constants, operators=[0, 1, 2])
    pprint(anuta.variables)
    pprint(anuta.constants)
    pprint(anuta.bounds)
    
    anuta.populate_kb()
    
    main(metadf, sys.argv[1])
    sys.exit(0)