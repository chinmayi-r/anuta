import z3
import pandas as pd
from tqdm import tqdm
import numpy as np

from anuta.known import *
from anuta.utils import *
from anuta.theory import Theory


def complete_evalmap(df, evalmap, sep=' '):
    _evalmap = {}
    #* First, complete the evalmap for sp to z3 conversion.
    for col in df.columns:
        varname = to_big_camelcase(col, sep=sep) if sep else col
        if df.dtypes[col] == np.float64:
            _evalmap[varname] = z3.Real(varname)
        else:
            _evalmap[varname] = z3.Int(varname)
        # match dataset:
        #     case 'cidds':
        #         #* Update the evalmap with vars.
        #         if varname in cidds_ints:
        #             _evalmap[varname] = z3.Int(varname)
        #         elif varname in cidds_reals:
        #             _evalmap[varname] = z3.Real(varname)
        #     case 'cicids':
        #         if varname in constructor.anuta.domains:
        #             domain = constructor.anuta.domains[varname]
        #             if varname == 'Protocol':
        #                 _evalmap[varname] = z3.Int(varname)
        #             elif type(domain.bounds.lb)==float or type(domain.bounds.ub)==float:
        #                 _evalmap[varname] = z3.Real(varname)
        #             else:
        #                 _evalmap[varname] = z3.Int(varname)
    return _evalmap | evalmap

metadf = pd.read_csv("data/metadc_test_10racks.csv").drop(columns=['rackid', 'hostid'])
metadc_rules = []
path = "rules/metadc/dt_metadc_train_all.pl"
# path = "/home/hh1789/Projects/REaLTabFormer/rules/learned_cidds_8192_checked.pl"
with open(f"{path}", 'r') as f:
    for i, line in enumerate(f):
        expr: sp.Expr = sp.sympify(line.strip())
        metadc_rules.append(expr)
        # cic_rules.append(line.strip())
        print(f"Loaded # of rules:\t{i+1}", end='\r')

evalmap = complete_evalmap(metadf, z3evalmap, None)
metaz3rules = []
for rule in metadc_rules:
    z3rule = eval(str(rule), evalmap)
    metaz3rules.append(z3rule)

invalid_rule_ids = []
# total = 5 # len(metadf)
# metadf = metadf.sample(n=total, random_state=42).reset_index(drop=True)
for ridx, rule in enumerate(tqdm(metaz3rules)):
    for i, row in metadf.iterrows():
        example = row.to_dict()
        assignment = []
        for varname, value in example.items():
            assignment.append((evalmap[varname], z3.IntVal(value)))
        
        substituted = z3.substitute(rule, *assignment)
        # s = z3.Solver()
        # s.add(z3.Not(substituted))
        # if s.check() != z3.unsat:
        if not z3.is_true(z3.simplify(substituted)):
            invalid_rule_ids.append(ridx)
            break
print(f"Invalid rules: {len(invalid_rule_ids)} out of {len(metaz3rules)}")

#* Fileter out the invalid rules
valid_rules = [rule for i, rule in enumerate(metadc_rules) if i not in invalid_rule_ids]
Theory.save_constraints(valid_rules, path="dt_metadc_train_validated.pl")
