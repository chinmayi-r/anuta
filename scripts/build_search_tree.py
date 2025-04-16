import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import sympy as sp
from IPython.display import display
import z3
import pickle
from time import perf_counter

from anuta.theory import Theory
from anuta.utils import to_big_camelcase, z3evalmap
from anuta.known import cidds_flag_map, cidds_proto_map, cidds_ip_map, cidds_ports, cidds_ints, cidds_reals


def get_path_to_root(G: nx.DiGraph, node):
    path = [node]
    while True:
        preds = list(G.predecessors(node))
        if not preds:
            break
        assert len(preds)==1
        node = preds[0] 
        path.append(node)
    return list(reversed(path))

def map_to_z3var_value(var_name, value, dtype, dataset):
    var_val = None
    match dataset:
        case 'cidds':
            #TODO: Wrap the mapping from generated value to rule encoding in a function.
            if var_name=='Flags':
                var_val = z3.IntVal(cidds_flag_map(value))
            elif var_name=='Proto':
                var_val = z3.IntVal(cidds_proto_map(value))
            elif 'ip' in var_name.lower():
                var_val = z3.IntVal(cidds_ip_map(value))
            elif 'pt' in var_name.lower():
                value = int(value[:-2]) \
                    if type(value)==str and value[-2:] == 'pt' else int(value)
                var_val = z3.IntVal(value)
            else:
                try:
                    if '.' in value:
                        var_val = z3.RealVal(float(value))
                    elif '_' not in value:
                        var_val = z3.IntVal(int(value))
                except ValueError:
                    var_val = value
        case 'cicids':
            if var_name == 'Protocol':
                var_val = z3.IntVal(int(value))
            elif dtype == np.float64:
                var_val = z3.RealVal(float(value))
            else:
                var_val = z3.IntVal(int(value))
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    assert var_val is not None, f"Rule value not mapped for {var_name}={value}"
    return var_val

def get_domain_constraints(series, varname, evalmap):
    domain_constraints = []
    if varname not in evalmap:
        return domain_constraints
    z3_var = evalmap[varname]
    #& For numerical vars.
    if series.dtype == np.int64:
        domain_constraints.append(z3_var >= z3.IntVal(series.min()))
        domain_constraints.append(z3_var <= z3.IntVal(series.max()))
    elif series.dtype == np.float64:
        domain_constraints.append(z3_var >= z3.RealVal(series.min()))
        domain_constraints.append(z3_var <= z3.RealVal(series.max()))
    else:
        #! Adding domain constraints for categorical vars may lead to unsatisfiability for some reason...
        pass
        #& For categorical vars or vars with predefined values.
        # assert len(domain.values)>0
        # if any(type(val)!=np.int64 for val in domain.values):
        #     domain_constraints.append(z3.Or([z3_var==z3.RealVal(val) for val in domain.values]))
        # else:
        #     domain_constraints.append(z3.Or([z3_var==z3.IntVal(val) for val in domain.values]))

    return domain_constraints

def get_relevant_rules(df, dataset, rules_sp):
    evalmap = z3evalmap.copy()
    #* First, complete the evalmap for sp to z3 conversion.
    for col in df.columns:
        varname = to_big_camelcase(col) if dataset == 'cidds' else to_big_camelcase(col, '_')
        match dataset:
            case 'cidds':
                #* Update the evalmap with vars.
                if varname in cidds_ints:
                    evalmap[varname] = z3.Int(varname)
                elif varname in cidds_reals:
                    evalmap[varname] = z3.Real(varname)
            case 'cicids':
                # if varname in constructor.anuta.domains:
                #     domain = constructor.anuta.domains[varname]
                if varname == 'Protocol':
                    evalmap[varname] = z3.Int(varname)
                elif df.dtypes[col] == np.float64:
                    evalmap[varname] = z3.Real(varname)
                else:
                    evalmap[varname] = z3.Int(varname)

    relevant_rules = defaultdict(list)
    for i, col in enumerate(df.columns):
        relevant = []
        filtered_relevant = []
        varname = to_big_camelcase(col) if dataset == 'cidds' else to_big_camelcase(col, '_')
        prefix_vars = set(sp.symbols([to_big_camelcase(name) for name in df.columns[: i]]))\
            if dataset == 'cidds' else \
                set(sp.symbols([to_big_camelcase(name, '_') for name in df.columns[: i]]))

        cur_var = sp.symbols(varname)
        included_vars = prefix_vars | {cur_var}
        for rule in rules_sp:
            variables = rule.free_symbols
            num_vars = len(variables)

            # #& All connected rules.
            # if len(variables & included_vars) > 0:
            #     relevant.append(rule)
            #     #* Accumulate vars from included rules.
            #     included_vars |= variables

            # #& Related rules.
            # if len(variables & (prefix_vars | {cur_var})) > 0:
            #     #* As long as the rule contains ≥1 variable from the current and/or one of the prefix vars.
            #     relevant.append(rule)

            #& Prefix rules only.
            if cur_var in variables and len(variables & (prefix_vars | {cur_var})) == num_vars:
                #* rule has to contain current var AND the rest of vars are all prefixes
                relevant.append(rule)
        print(f"Found {len(relevant)} relevant rules for {varname}.")

        #* Filter redundant rules
        for rule in relevant:
            if isinstance(rule, sp.Implies):
                precedent, consequent = rule.args
                if isinstance(precedent, sp.And):
                    p1, p2 = precedent.args
                    # pprint(rule)
                    # pprint([p1.free_symbols, p2.free_symbols])
                    if p1.free_symbols == p2.free_symbols:
                        # pprint(rule)
                        continue
            filtered_relevant.append(rule)
        filtered_relevant = sorted(filtered_relevant, key=lambda r: str(r))
        # print(f"Filtered to {len(filtered_relevant)} relevant rules for {varname}.")

        # coalesced_rules = coalesce(filtered_relevant)
        # print(f"Coalesced to {len(coalesced_rules)} rules for {varname}.")
        # # if varname == 'SrcPt':
        # #     for rule in coalesced_rules:
        # #         # display(rule)
        # #         print(rule)

        # rules_z3 = [eval(str(rule), evalmap) for rule in coalesced_rules]
        rules_z3 = [eval(str(rule), evalmap) for rule in filtered_relevant]
        relevant_rules[varname] = rules_z3

    for i, (varname, rules) in enumerate(relevant_rules.items()):
        checked_rules = set()
        for rule in rules:
            isvalid = True
            solver = z3.Solver()
            rule = z3.simplify(rule)
            solver.add(~rule)
            if solver.check() == z3.unsat:
                isvalid = False
                # print(f"Tautology:")
                # display(rule)

            solver = z3.Solver()
            solver.add(rule)
            if solver.check() == z3.unsat:
                isvalid = False
                # print(f"Contradiction:")
                # display(rule)

            if isvalid:
                checked_rules.add(rule)
        print(f"{varname}: {len(rules)-len(checked_rules)}/{len(rules)} redundant rules.")

        checked_rules = list(checked_rules)
        series = df.iloc[:, i]
        domain_constraints = get_domain_constraints(series, varname, evalmap,)
        #* Constraint the feasible tokens within the var's domain.
        checked_rules.extend(domain_constraints)
        #* Combine all rules into a single theorem.
        combined_rule = z3.And(checked_rules)
        relevant_rules[varname] = combined_rule
    return relevant_rules

if __name__ == "__main__":
    ciddf = pd.read_csv('data/cidds_wk3_all.csv')
    cidds_rules = Theory.load_constraints('rules/cidds/new/learned_cidds_8192_checked.pl')
    cidds_relevant_rules = get_relevant_rules(ciddf, 'cidds', cidds_rules)

    start = perf_counter()
    G = nx.DiGraph()
    parents = []
    nodeid = 0
    for col in ciddf.columns[1:]:
        print(f"Processing {col} (#nodes={nodeid}):")
        domain = ciddf[col].unique()
        dtype = ciddf.dtypes[col]
        varname = to_big_camelcase(col)
        z3var = z3.Real(varname) if dtype==np.float64 else z3.Int(varname)
        
        relevant_rule = cidds_relevant_rules[varname]
        #* Compute the common path (only once).
        common_path = get_path_to_root(G, parents[0])[:-1] if parents else []
        common_path_values = []
        for n in common_path:
            node = G.nodes[n]
            if node['value'] is not None:
                var = z3.Int(node['varname'])
                #! Assuming values are all integers and that floats are with bounds.
                val = z3.IntVal(node['value'])
                common_path_values.append((var, val))
            else:
                #* Add bounding constraints.
                assert node['bounds'] is not None
                var = z3.Int(node['varname']) \
                    if dtype==np.int64 else z3.Real(node['varname'])
                lb, ub = node['bounds']
                #* Amend the relevant rule to include the bounds.
                relevant_rule = z3.And(relevant_rule, var >= lb)
                relevant_rule = z3.And(relevant_rule, var <= ub)
        
        old_id = nodeid
        if dtype in [np.float64, np.int64] and 'Pt' not in col:
            relevant_rule = cidds_relevant_rules[varname]
            bounds = min(domain), max(domain)
            for p in tqdm(parents, total=len(parents)):
                path_values = common_path_values.copy()
                node = G.nodes[p]
                if node['value'] is not None:
                    var = z3.Int(node['varname'])
                    #! Assuming values are all integers and that floats are with bounds.
                    val = z3.IntVal(node['value'])
                    path_values.append((var, val))
                else:
                    #* Add bounding constraints.
                    assert node['bounds'] is not None
                    var = z3.Int(node['varname']) \
                        if dtype==np.int64 else z3.Real(node['varname'])
                    lb, ub = node['bounds']
                    #! Instantiate a new rule for each parent, otherwise accumulating constraints.
                    path_rule = z3.And(relevant_rule, var >= lb)
                    path_rule = z3.And(path_rule, var <= ub)
                
                opt = z3.Optimize()
                substituted = z3.simplify(z3.substitute(path_rule, *path_values))
                opt.add(substituted)
                lb = opt.minimize(z3var)
                if opt.check() == z3.sat:
                    lb = lb.value()
                    if isinstance(lb, z3.RatNumRef):
                        lb = float(lb.as_fraction())
                    elif isinstance(lb, z3.IntNumRef):
                        lb = lb.as_long()
                    else:
                        assert isinstance(lb, z3.ArithRef), f"Unknown {type(lb)=} for {lb=}"
                        lb = z3.RealVal(bounds[0]) \
                            if dtype==np.float64 else z3.IntVal(bounds[0])
                        display(substituted)
                else:
                    print(f"[Optm] Can't obtain logit lower bound for {z3var}")
                    lb = bounds[0]
                    display(substituted)
                
                opt = z3.Optimize()
                opt.add(substituted)
                ub = opt.maximize(z3var)
                if opt.check() == z3.sat:
                    ub = ub.value()
                    if isinstance(ub, z3.RatNumRef):
                        ub = float(ub.as_fraction())
                    elif isinstance(ub, z3.IntNumRef):
                        ub = ub.as_long()
                    else:
                        assert isinstance(ub, z3.ArithRef), f"Unknown {type(ub)=} for {ub=}"
                        ub = bounds[1]
                        display(substituted)
                else:
                    print(f"[Optm] Can't obtain logit upper bound for {z3var}")
                    ub = bounds[1]
                    display(substituted)
                
                bounds = (lb, ub)
                G.add_node(nodeid, varname=varname, value=None, bounds=bounds)
                G.add_edge(p, nodeid)
                nodeid += 1
            if not parents:
                #* If it's root, bounds are the domain.
                G.add_node(nodeid, varname=varname, value=None, bounds=bounds)
                nodeid += 1
        else:
            varvals = set()        
            for val in domain:
                varval = map_to_z3var_value(varname, val, dtype, 'cidds')    
                varvals.add(varval)
            
            for i, z3val in tqdm(enumerate(varvals), total=len(varvals)):
                for p in parents:
                    path_values = common_path_values.copy()    
                    node = G.nodes[p]
                    if node['value'] is not None:
                        var = z3.Int(node['varname'])
                        #* For some reason, numerics in value pairs have to be explicitly converted, 
                        #*  while other places don't require.
                        path_values.append((var, z3.IntVal(node['value'])))
                    else:
                        #* Add bounds for numeric vars.
                        assert node['bounds'] is not None
                        var = z3.Int(node['varname']) \
                            if dtype==np.int64 else z3.Real(node['varname'])
                        lb, ub = node['bounds']
                        # print(f"Add bounds {lb, ub}")
                        #! Instantiate a new rule for each parent, otherwise accumulating constraints.
                        path_rule = z3.And(relevant_rule, var >= lb)
                        path_rule = z3.And(path_rule, var <= ub)
                        
                    #* Check if the path together with the new value is satisfiable.
                    s = z3.Solver()
                    substituted = z3.simplify(z3.substitute(path_rule, (z3var, z3val), *path_values))
                    s.add(substituted)
                    if s.check() == z3.sat:
                        #* Treat nodes with the same value different, do NOT have shared nodes for ease of traversal.
                        G.add_node(nodeid, varname=varname, value=z3val.as_long(), bounds=None)
                        G.add_edge(p, nodeid)
                        nodeid += 1
                    # else:
                    #     print(f"Invalid path: {path_values}->{(varname, domain[i])}", end='\r')
                if not parents:
                    G.add_node(nodeid, varname=varname, value=z3val.as_long(), bounds=None)
                    nodeid += 1
        parents = [nid for nid in range(old_id, nodeid)]
    
    end = perf_counter()
    print(f"Built search tree in {end-start:.2f}s.")
    #* Save the graph to a file.
    save_path = 'results/cidds_search_tree.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved search tree to {save_path}.")
        