from collections import defaultdict, Counter
from rich import print as pprint
from time import perf_counter
from multiprocess import Pool
from copy import deepcopy
from tqdm import tqdm
from typing import *
import pandas as pd
import numpy as np
import sympy as sp
import psutil
import warnings
warnings.filterwarnings("ignore")

from grammar import AnutaMilli, Anuta, Kind
from constructor import Constructor, DomainCounter
from model import Model, Constraint
from utils import log, desugar, save_constraints


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

def levelwise_search(
        worker_idx: int, dfpartition: pd.DataFrame, 
        indexset: dict[str, dict[str, np.ndarray]], 
        fcount: dict[str, dict[str, DomainCounter]], 
        limit: int, #* limit ≤ len(dfpartition)
    ) -> List[int]:
    global anuta
    assert anuta, "anuta is not initialized."
    
    log.info(f"Worker {worker_idx+1} started.")
    #* 1: Violated, 0: Not violated
    violations = [0 for _ in range(len(anuta.candidates))]
    exhausted_values = defaultdict(list)
    exhausted_values[f"Worker {worker_idx}"] = 'Exhausted Domain Values'
    
    for i in tqdm(range(limit), total=limit):
        '''Domain Counting'''
        #* Get the vars at every iteration to account for the changes in the indexset.
        indexed_vars = list(fcount.keys())
        #* Cycle through the vars, treating them equally (no bias).
        nxt_var = indexed_vars[i % len(indexed_vars)]
        #* Find the least frequent value of the next variable.
        least_freq_val = min(fcount[nxt_var], key=fcount[nxt_var].get)
        #& Get the 1st from the indices of least frequent value (inductive bias).
        #TODO: Choose randomly from the indices?
        indices = indexset[nxt_var][least_freq_val]
        #^ Somehow ndarray passes by value (unlike list) ...
        index, indexset[nxt_var][least_freq_val] = indices[0], indices[1: ]
        if indexset[nxt_var][least_freq_val].size == 0:
            #* Remove the corresponding counter if the value is exhausted 
            #* to prevent further sampling (from empty sets).
            del fcount[nxt_var][least_freq_val]
            exhausted_values[nxt_var].append(least_freq_val)
            if not fcount[nxt_var]:
                del fcount[nxt_var]
        
        sample = dfpartition.iloc[index]
        assignments = {}
        for name, val in sample.items():
            var = anuta.variables.get(name)
            if not var: continue
            assignments[var] = val
            
            #* Increment the frequency count of the value of the var.
            if name in fcount and val in fcount[name]:
                fcount[name][val].count += 1
        
        for k, constraint in enumerate(anuta.candidates):
            if violations[k]: 
                #* This constraint has already been violated.
                continue
            #* Evaluate the constraint with the given assignments
            sat = constraint.expr.subs(assignments)
            if not sat:
                # log.info(f"Violated: {constraint}")
                violations[k] = 1
        
    log.info(f"Worker {worker_idx+1} finished.")
    pprint(exhausted_values)
    return violations


def miner(constructor: Constructor, limit: int = 0):
    global anuta
    #* Use a global var to prevent passing the object to each worker.
    anuta = constructor.anuta
    label = str(limit)
    ARITY_LIMIT = 3
    start = perf_counter()
    
    #* Prepare arguments for parallel processing
    core_count = psutil.cpu_count()
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(constructor.df, core_count)]
    indexsets, fcounts = zip(*[constructor.get_indexset_and_counter(df, anuta.domains) for df in dfpartitions])
    
    pprint(fcounts[0])
    
    while anuta.search_arity < ARITY_LIMIT:
        anuta.propose_new_candidates()
        
        log.info(f"Started searching for arity-{anuta.search_arity} constraints.")
        if not anuta.candidates:
            log.error(f"No new candidates found.")
            break
        
        log.info(f"Testing {len(anuta.candidates)} arity-{anuta.search_arity} candidates ...")
        args = [(i, df, indexset, fcount, limit//core_count) 
                for i, (df, indexset, fcount) in enumerate(
                    # zip(dfpartitions, deepcopy(indexsets), deepcopy(fcounts)))]
                    zip(dfpartitions, deepcopy(indexsets), deepcopy(fcounts)))]
                    #? Create new DC counters or keep the counts from the level?
                    #^ I think we should NOT keep the counts from the previous level,
                    #^ because an unviolated example could eliminate a candidate on the next level.
                    
        log.info(f"Spawning {core_count} workers ...")
        pool = Pool(core_count)
        log.info(f"Testing arity-{anuta.search_arity} constraint in parallel ...")
        violation_indices = pool.starmap(levelwise_search, args)
        log.info(f"All workers finished.")
        pool.close()
        
        log.info(f"Aggregating violations ...")
        aggregated_violations = np.logical_or.reduce(violation_indices)
        assert len(aggregated_violations) == len(anuta.candidates), \
            f"{len(aggregated_violations)=} != {len(anuta.candidates)=}"
        pprint(Counter(aggregated_violations))
        
        log.info(f"Removing violated constraints ...")
        old_size = len(anuta.kb)
        new_candidates = deepcopy(anuta.candidates)
        for idx, is_violated in enumerate(aggregated_violations):
            candidate = anuta.candidates[idx]
            if is_violated:
                new_candidates.remove(candidate)
                anuta.num_candidates_rejected += 1
                #& Learn the ancestors of the violated candidate!
                anuta.kb |= candidate.ancestors #* Dedupe is handled by the set.
            else:
                #* A constraint is only learned if one of its successors is violated.
                pass
        log.info(f"Removed {len(anuta.candidates)-len(new_candidates)} candidates.")
        anuta.candidates = new_candidates
        new_size = len(anuta.kb)
        log.info(f"Learned {new_size-old_size} candidates.")
    #> End of while loop
    end = perf_counter()
    
    #& Learn all remaining candidates (not violated by any example at the last level).
    anuta.kb |= set(anuta.candidates)
    log.info(f"Finished mining all constraints up to arity {anuta.search_arity}.")
    
    # aggregated_violations = np.logical_or.reduce(violation_indices)
    # # aggregated_bounds = {k: IntBounds(sys.maxsize, 0) for k in anuta.bounds.keys()}
    # learned_kb = []
    # log.info(f"Aggregating violations ...")
    # #* Update learned_kb based on the violated constraints
    # for index, is_violated in tqdm(enumerate(aggregated_violations), total=len(aggregated_violations)):
    #     if not is_violated:
    #         learned_kb.append(anuta.initial_kb[index])

    # pprint(aggregated_bounds)
    print(f"Total proposed: {anuta.num_candidates_proposed}")
    print(f"Total rejected: {anuta.num_candidates_rejected} ({anuta.num_candidates_rejected/anuta.num_candidates_proposed:.2%})")
    print(f"Total prior: {len(anuta.prior)}")
    print(f"Total learned: {len(anuta.kb)}")
    #* Prior: [(X=2 | X=3 | ...), (Y=100 | Y=200 | ...)]
    Model.save_constraints(anuta.kb | anuta.prior, f'learned_{label}.rule')
    print(f"Runtime: {end-start:.2f}s\n\n")
    
    if len(anuta.kb) <= 200: 
        #^ Skip pruning if the number of constraints is too large
        log.info(f"Pruning redundant constraints ...")
        start = perf_counter()
        # assumptions = [v >= 0 for v in anuta.variables.values()]
        assumptions = anuta.prior
        cnf = sp.And(*(anuta.kb | set(assumptions)))
        simplified_logic = sp.to_cnf(desugar(cnf))
        reduced_kb = list(simplified_logic.args) \
            if isinstance(simplified_logic, sp.And) else [simplified_logic]
        pruned_count = len(anuta.kb) - len(reduced_kb)
        end = perf_counter()
        print(f"{len(anuta.kb)=}, {len(reduced_kb)=} ({pruned_count=})\n")
        print(f"Pruning time: {end-start:.2f}s\n\n")
        
        Model.save_constraints(anuta.kb, f'pruned_{label}.rule')