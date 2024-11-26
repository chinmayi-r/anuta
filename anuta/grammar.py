from itertools import combinations
from typing import *
import sympy as sp
import sys
from dataclasses import dataclass

from enum import Enum, auto

import logging


log = logging.getLogger("anuta")
handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[%(name)s @ %(asctime)s] %(levelname)-8s | %(message)s", 
    datefmt="%H:%M:%S"  # Format to show only hour, minute, and second
)
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)


def consecutive_combinations(lst):
    ccombo = []
    n = len(lst)
    
    #* Start with combinations of size 2 up to size n
    for size in range(2, n + 1):  
        #* Ensure that the combination is consecutive
        for start in range(n - size + 1):  
            ccombo.append(lst[start: start+size])
    
    return ccombo

@dataclass
class IntBounds(object):
    lb: int
    ub: int

class Operator(Enum):
    NOP = 0
    PLUS = auto()
    MAX = auto()

class Anuta(object):
    def __init__(self, variables: List[str], bounds: Dict[str, IntBounds], constants: Dict[str, int]=None, operators: List[int]=None):
        variables = sp.symbols(' '.join(variables), integer=True, nonnegative=True)
        self.variables = {v.name: v for v in variables}
        self.constants = constants
        self.bounds = bounds
        self.operators = list(Operator) if not operators else \
            [Operator(o) for o in operators]
        
        
        self.varlabel = 'aggregate'
        self.static_vars = [v for v in self.variables.values() if self.varlabel in v.name and 
                            'host' not in v.name and 'stride' not in v.name and 'window' not in v.name and 'canary' not in v.name]
        self.dynamic_vars = [v for v in self.variables.values() if self.varlabel not in v.name and 'canary' not in v.name]
        self.canary_vars = [v for v in self.variables.values() if 'canary' in v.name]
        self.kb = []
    
    def generate_expressions(self) -> Generator[Tuple[Operator, sp.Expr], None, None]:
        #& Limit to homogeneous expressions only (i.e., no mixed operators).
        if Operator.NOP in self.operators:
            for v in self.dynamic_vars:
                yield (Operator.NOP, v)
                
        if Operator.PLUS in self.operators:
            #! Too expansive:
            # for k in range(1, len(self.variables)+1):
            #     for combo in combinations(self.variables.values(), k):
            #         yield sum(combo)
            #! Too expansive:
            # for k in range(2, len(measurement_vars)+1):
            #     for combo in combinations(measurement_vars.values(), k):
            #         yield sum(combo)
            
            #* Isolate the interactions to within static vars and dynamic vars.
            #! Omit interactions within static vars.
            # for k in range(2, len(self.static_vars)+1):
            #     #* All combinations of sampled vars
            #     for combo in combinations(self.static_vars, k):
            #         yield sum(combo)
            #& Limit to consecutive combinations only.
            for combo in consecutive_combinations(self.dynamic_vars):
                yield (Operator.PLUS, sum(combo))
                    
        if Operator.MAX in self.operators:
            #! Too expansive:
            # #* Max over ≥2 args.
            # for k in range(2, len(self.variables)+1):
            #     for combo in combinations(self.variables.values(), k):
            #         yield sp.Max(*combo)
            #& Limit to consecutive combinations only.
            yield (Operator.MAX, sp.Max(*self.dynamic_vars))
    
    def generate_constraints(self):
        for op, expr in self.generate_expressions():
            #& Restrict the bounds to constants and static vars only.
            bounds = self.static_vars + [sp.symbols(name) for name in self.constants]
            for b in bounds:
                match op:
                    case _:
                        yield expr >= b
                        yield expr <= b
            #* Add canary max constraint
            if op == Operator.MAX:
                canary_max = [v for v in self.canary_vars if 'max' in v.name][0]
                print(type(canary_max))
                yield expr >= canary_max
                yield expr <= canary_max
            
            #* Implications
            triggers = self.static_vars + self.dynamic_vars
            for trigger in triggers:
                for name in self.constants:
                    const = sp.symbols(name)
                    #* Only consider premises of the form: (var > 0)
                    #* Only consider conclusions of the form: (expr {} const)
                    yield (trigger > 0) >> (expr >= const)
                    yield (trigger > 0) >> (expr <= const)
        #* Add canary implications
        canary_premise = [v for v in self.canary_vars if 'premise' in v.name][0]
        canary_conclusion = [v for v in self.canary_vars if 'conclusion' in v.name][0]
        yield (canary_premise > 0) >> (canary_conclusion >= sp.symbols('burst_threshold'))
            

    def interval_filter(self, constraint: sp.Expr) -> bool:
        #* Interval arithmetic
        args = constraint.args
        match type(constraint):
            case sp.GreaterThan:
                operation = args[0]
                operands = operation.args
                comparator = args[1]
                assert not comparator.args, f"Comparator is not a singleton: {comparator}"
                
                rhs_bounds = self.bounds[comparator.name]
                if isinstance(operation, sp.Add):
                    lhs_ub = 0
                    for operand in operands:
                        lhs_ub += self.bounds[operand.name].ub
                    return lhs_ub < rhs_bounds.lb
                elif isinstance(operation, sp.Max):
                    lhs_ubs = [self.bounds[operand.name].ub for operand in operands]
                    return max(lhs_ubs) < rhs_bounds.lb
            case sp.LessThan:
                operation = args[0]
                operands = operation.args
                comparator = args[1]
                assert not comparator.args, f"Comparator is not a singleton: {comparator}"
                
                rhs_bounds = self.bounds[comparator.name]
                if isinstance(operation, sp.Add):
                    lhs_lb = 0
                    for operand in operands:
                        lhs_lb += self.bounds[operand.name].lb
                    return lhs_lb > rhs_bounds.ub
                elif isinstance(operation, sp.Max):
                    # #! Need to double-check this
                    # lhs_ubs = [self.bounds[operand.name].ub for operand in operands]
                    # return min(lhs_ubs) > rhs_bounds.ub
                    lhs_lbs = [self.bounds[operand.name].lb for operand in operands]
                    return max(lhs_lbs) > rhs_bounds.ub
            case sp.Implies:
                #* Assume the premise is always true, and check if the conclusion is valid.
                conclusion = args[1]
                return self.interval_filter(conclusion)
            case _:
                raise NotImplementedError(f"Unsupported constraint type: {type(constraint)}")
    
    def populate_kb(self):
        num_trivial = 0
        num_constraints = 0
        interval_filtered = 0
        for constraint in self.generate_constraints():
            # log.info(type(constraint))
            if isinstance(constraint, sp.logic.boolalg.BooleanTrue):
                num_trivial += 1
            else:
                if self.interval_filter(constraint):
                    interval_filtered += 1
                    continue
                self.kb.append(constraint)
                num_constraints += 1
                if num_constraints % 10_000 == 0:
                    log.info(f"Generated {num_constraints} constraints.")
        log.info(f"Skipped {num_trivial} trivial constraints.")
        log.info(f"Interval-filtered {interval_filtered} constraints.")
        log.info(f"Populated KB with {len(self.kb)} constraints.\n")
        
        #! Too expansive, and the reduction is little (26/149720)
        # log.info(f"Filtering redundant constraints ...")
        # # Nonnegative integers
        # assumptions = [v >= 0 for v in self.variables.values()]
        # cnf = sp.And(*(self.kb + assumptions))
        # simplified_logic = sp.simplify_logic(cnf)
        # #! Too expansive
        # # simplified = simplified_logic.simplify()
        # reduced_constraints = list(simplified_logic.args) \
        #     if isinstance(simplified_logic, sp.And) else [simplified_logic]
        # num_redundant = len(self.kb) - len(reduced_constraints)
        # #* Update KB
        # self.kb = reduced_constraints
        # log.info(f"Removed {num_redundant} redundant constraints.\nFinal KB size: {len(self.kb)}")