from rich import print as pprint
from time import perf_counter
from multiprocess import Pool
from tqdm import tqdm
from typing import *
import pandas as pd
import numpy as np
import sympy as sp
import psutil

from grammar import AnutaMilli, Anuta
from constructor import Constructor
from utils import log, save_constraints


anuta : Anuta = None
window = 10

def test_millisampler_constraints(worker_idx: int, dfpartition: pd.DataFrame) -> List[int]:
    global anuta
    assert anuta, "anuta is not initialized."
    
    # var_bounds = anuta.bounds
    
    log.info(f"Worker {worker_idx+1} started.")
    #* 1: Violated, 0: Not violated
    violations = [0 for _ in range(len(anuta.initial_kb))]
    
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
        
        for k, constraint in enumerate(anuta.initial_kb):
            if violations[k]: 
                #* This constraint has already been violated.
                continue
            #* Evaluate the constraint with the given assignments
            sat = constraint.subs(assignments | anuta.constants)
            if not sat:
                violations[k] = 1
    log.info(f"Worker {worker_idx+1} finished.")
    return violations

def test_cidds_constraints(worker_idx: int, dfpartition: pd.DataFrame) -> List[int]:
    global anuta
    assert anuta, "anuta is not initialized."
    
    # var_bounds = anuta.bounds
    
    log.info(f"Worker {worker_idx+1} started.")
    #* 1: Violated, 0: Not violated
    violations = [0 for _ in range(len(anuta.initial_kb))]
    
    for _, sample in tqdm(dfpartition.iterrows(), total=len(dfpartition)):
        assignments = {}
        for name, val in sample.items():
            name = name.replace(' ', '_')
            var = anuta.variables.get(name)
            if not var: continue
            assignments[var] = val
        
        # pprint(assignments)
        for k, constraint in enumerate(anuta.initial_kb):
            if violations[k]: 
                #* This constraint has already been violated.
                continue
            #* Evaluate the constraint with the given assignments
            sat = constraint.subs(assignments)
            # print(f"{sat=}")
            # print(f"\t{bool(sat)=}")
            if not sat:
                # if constraint == sp.And(sp.Eq(anuta.variables['Flags'], 1), sp.Eq(anuta.variables['Proto'], 0)):
                    # print(f"Violated: {sample['Flags']=}, {sample['Proto']=}")
                    # pprint(constraint)
                violations[k] = 1
    log.info(f"Worker {worker_idx+1} finished.")
    return violations


def miner(constructor: Constructor, label: str):
    global anuta
    anuta = constructor.anuta
    start = perf_counter()
    
    #* Prepare arguments for parallel processing
    core_count = psutil.cpu_count()
    # mutex = Manager().Lock()
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(constructor.df, core_count)]
    args = [(i, df) for i, df in enumerate(dfpartitions)]

    # Use multiprocessing Pool to test constraints in parallel
    print(f"Testing constraints in parallel ...")
    pool = Pool(core_count)
    # violation_indices, bounds_array = pool.starmap(test_constraints, args)
    # violation_indices = pool.starmap(test_millisampler_constraints, args)
    violation_indices = pool.starmap(test_cidds_constraints, args)
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
            learned_kb.append(anuta.initial_kb[index])
    
    end = perf_counter()
    # log.info(f"Aggregating bounds ...")
    #* Update the bounds based on the learned bounds
    # for bounds in bounds_array:
    #     for k, v in bounds.items():
    #         if v.lb < aggregated_bounds[k].lb:
    #             aggregated_bounds[k].lb = v.lb
    #         if v.ub > aggregated_bounds[k].ub:
    #             aggregated_bounds[k].ub = v.ub

    removed_count = len(anuta.initial_kb) - len(learned_kb)
    # pprint(aggregated_bounds)
    print(f"{len(learned_kb)=}, {len(anuta.initial_kb)=} ({removed_count=})")
    save_constraints(learned_kb + anuta.prior_kb, f'learned_{label}')
    print(f"Learning time: {end-start:.2f}s\n\n")
    
    if len(learned_kb) > 200: 
        #* Skip pruning if the number of constraints is too large
        return
    
    start = perf_counter()
    log.info(f"Pruning redundant constraints ...")
    # assumptions = [v >= 0 for v in anuta.variables.values()]
    assumptions = []
    cnf = sp.And(*(learned_kb + assumptions))
    simplified_logic = cnf.simplify()
    reduced_kb = list(simplified_logic.args) \
        if isinstance(simplified_logic, sp.And) else [simplified_logic]
    filtered_count = len(learned_kb) - len(reduced_kb)
    end = perf_counter()
    print(f"{len(learned_kb)=}, {len(reduced_kb)=} ({filtered_count=})\n")
    print(f"Pruning time: {end-start:.2f}s\n\n")
    
    anuta.learned_kb = reduced_kb + anuta.prior_kb
    save_constraints(anuta.learned_kb, f'reduced_{label}')