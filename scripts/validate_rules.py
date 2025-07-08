import z3
import pandas as pd
import numpy as np
from tqdm import tqdm
import sympy as sp
import multiprocessing as mp

from anuta.known import *
from anuta.utils import *
from anuta.theory import Theory


# evalmap = None 

def complete_evalmap(df, evalmap, sep=' '):
    _evalmap = {}
    for col in df.columns:
        varname = to_big_camelcase(col, sep=sep) if sep else col
        if df.dtypes[col] == np.float64:
            _evalmap[varname] = z3.Real(varname)
        else:
            _evalmap[varname] = z3.Int(varname)
    return _evalmap | evalmap


def build_evalmap_from_dtypes(dtypes, sep=' '):
    evalmap = {}
    for col, dtype in dtypes.items():
        varname = to_big_camelcase(col, sep=sep) if sep else col
        if dtype == np.float64:
            evalmap[varname] = z3.Real(varname)
        else:
            evalmap[varname] = z3.Int(varname)
    return evalmap | z3evalmap


def check_rule_validity_worker(args):
    idx, rule_str, df_dicts, dtypes = args
    evalmap = build_evalmap_from_dtypes(dtypes, None)
    try:
        z3rule = eval(rule_str, evalmap)
    except Exception as e:
        print(f"Failed to eval rule {idx}: {e}")
        return idx  # Consider rule invalid if it can't even be evaluated
    for example in df_dicts:
        assignment = [(evalmap[varname], z3.IntVal(value)) for varname, value in example.items()]
        substituted = z3.substitute(z3rule, *assignment)
        if not z3.is_true(z3.simplify(substituted)):
            return idx
    return None


if __name__ == '__main__':
    # Load data and rules
    metadf = pd.read_csv("data/metadc_test_10racks.csv").drop(columns=['rackid', 'hostid'])
    metadc_rules = []
    path = "rules/metadc/dt_metadc_train_all.pl"

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            expr: sp.Expr = sp.sympify(line.strip())
            metadc_rules.append(expr)
            print(f"Loaded # of rules:\t{i+1}", end='\r')

    evalmap = complete_evalmap(metadf, z3evalmap, None)

    # Preprocess dataframe into list of dictionaries for fast parallel access
    dtypes = metadf.dtypes
    df_dicts = metadf.to_dict(orient='records')
    tasks = [(i, str(rule), df_dicts, dtypes) for i, rule in enumerate(metadc_rules)]

    # Use multiprocessing to validate rules
    print(f"Validating {len(tasks)} rules using {mp.cpu_count()} cores...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(check_rule_validity_worker, tasks), total=len(tasks)))

    # Collect invalid rule indices
    invalid_rule_ids = sorted([r for r in results if r is not None])
    print(f"\nInvalid rules: {len(invalid_rule_ids)} out of {len(metadc_rules)}")

    # Filter and save valid rules
    valid_rules = [rule for i, rule in enumerate(metadc_rules) if i not in invalid_rule_ids]
    Theory.save_constraints(valid_rules, path="dt_metadc_train_validated.pl")