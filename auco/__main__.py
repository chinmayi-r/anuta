from multiprocess import Pool
import psutil
import sys
from rich import print as pprint
import pandas as pd
from tqdm import tqdm
import sympy as sp
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from grammar import Auco
import json

g_df = None

def test_constraint(i, constraint, auco):
    global g_df
    
    learned_kb = auco.kb.copy()
    print(f"Testing constraint {i+1}/{len(auco.kb)}: {constraint}")
    for _, sample in g_df.iterrows():
        assignments = {}
        for name, val in sample.items():
            if name not in auco.variables:
                continue
            assignments[auco.variables[name]] = val
        
        # Evaluate the constraint with the given assignments
        sat = constraint.subs(assignments | auco.constants)
        
        # If constraint is violated, mark for removal
        if not sat and constraint in learned_kb:
            learned_kb.remove(constraint)
            print(f"Violation occurred: {constraint} removed.")
            return i  # Return the index of the violated constraint
    
    return None  # No violation found for this constraint

def save_constraints(constraints, fname: str='constraints'):
    # Convert expressions to strings
    expressions_str = [str(expr) for expr in constraints]

    # Save to a JSON file
    with open(f"{fname}.json", 'w') as f:
        json.dump(expressions_str, f)
    
def main(auco: Auco):
    # learned_kb = auco.kb.copy()
    # removed_count = 0
    # for i, constraint in enumerate(auco.kb):
    #     if constraint not in learned_kb: 
    #         #* Already removed.
    #         continue
        
    #     print(f"Testing constraint {i+1}/{len(auco.kb)}:")
    #     for _, sample in tqdm(metadf.iterrows(), total=len(metadf)):
    #         assignments = {}
    #         for name, val in sample.items():
    #             if name not in auco.variables: continue
    #             #! The substitution has to be done with the var itself.
    #             #! Var name str is insufficient since the var has assumptions
    #             #! e.g., ≥0 and being int.
    #             assignments[auco.variables[name]] = val
    #         sat = constraint.subs(assignments | auco.constants)
            
    #         if not sat and constraint in learned_kb:
    #             learned_kb.remove(constraint)
    #             removed_count += 1
    #             # print('Highlighting violation ...')
    #             print(f"\n\t\tViolation occurred: {constraint} removed.")
    #             break
    # print(f"{len(learned_kb)=}, {len(auco.kb)=} ({removed_count=})\n")
    removed_count = 0

    # Prepare arguments for parallel processing
    args = [(i, constraint, auco) for i, constraint in enumerate(auco.kb)]

    # Use multiprocessing Pool to test constraints in parallel
    print(f"Testing constraints in parallel ...")
    core_count = psutil.cpu_count()
    pool = Pool(core_count)
    violation_indices = pool.starmap(test_constraint, args)

    learned_kb = auco.kb.copy()
    # Update learned_kb based on the violated constraints
    for index in violation_indices:
        if index:
            learned_kb.remove(auco.kb[index])
            removed_count += 1

    print(f"{len(learned_kb)=}, {len(auco.kb)=} ({removed_count=})\n")
    save_constraints(learned_kb, 'learned')
    
    print(f"Filtering redundant constraints ...")
    assumptions = [v >= 0 for v in auco.variables.values()]
    cnf = sp.And(*(learned_kb + assumptions))
    simplified_logic = cnf.simplify()
    reduced_kb = list(simplified_logic.args) \
        if isinstance(simplified_logic, sp.And) else [simplified_logic]
    filtered_count = len(learned_kb) - len(reduced_kb)
    print(f"{len(learned_kb)=}, {len(reduced_kb)=} ({filtered_count=})\n")
    
    save_constraints(reduced_kb, 'reduced')
    

if __name__ == '__main__':
    metadf = pd.read_csv('data/meta_test.csv')
    g_df = metadf
    variables = []
    for col in metadf.iloc[:, 2:]:
        if not 'egress' in col and not 'out' in col:
            # if len(col.split('_')) > 1 and col.split('_')[1].isdigit(): continue
            variables.append(col)

    constants = {
        'burst_threshold': round(0.5*metadf.ingressBytes_sampled.max().item()),
    }
    # print(f"{variables=} {constants=}")
    
    auco = Auco(variables, constants, operators=[0, 1, 2])
    pprint(auco.variables)
    pprint(auco.constants)
    
    auco.populate_kb()
    
    main(auco)
    sys.exit(0)