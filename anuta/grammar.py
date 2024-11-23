from itertools import combinations
from typing import *
import sympy as sp

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
            ccombo.append(lst[start:start + size])
    
    return ccombo


class Operator(Enum):
    NOP = 0
    PLUS = auto()
    MAX = auto()

class Anuta(object):
    def __init__(self, variables: List[str], constants: Dict[str, int]=None, operators: List[int]=None):
        variables = sp.symbols(' '.join(variables), integer=True, nonnegative=True)
        self.variables = {v.name: v for v in variables}
        self.constants = constants
        self.operators = list(Operator) if not operators else \
            [Operator(o) for o in operators]
        
        self.kb = []
    
    def generate_expressions(self) -> Generator[sp.Expr, None, None]:
        sampled_vars = [v for v in self.variables.values() if 'sampled' in v.name]
        measurement_vars = [v for v in self.variables.values() if 'sampled' not in v.name]
        if Operator.NOP in self.operators:
            for v in self.variables.values():
                yield v
                
        if Operator.PLUS in self.operators:
            #! Too expansive:
            # for k in range(1, len(self.variables)+1):
            #     for combo in combinations(self.variables.values(), k):
            #         yield sum(combo)
            #* Isolate the interactions to within sampled vars and measurement vars.
            for k in range(2, len(sampled_vars)+1):
                #* All combinations of sampled vars
                for combo in combinations(sampled_vars, k):
                    yield sum(combo)
            #! Too expansive:
            # for k in range(2, len(measurement_vars)+1):
            #     for combo in combinations(measurement_vars.values(), k):
            #         yield sum(combo)
            #* Limit to consecutive combinations only.
            for combo in consecutive_combinations(measurement_vars):
                yield sum(combo)
                    
        if Operator.MAX in self.operators:
            #! Too expansive:
            # #* Max over ≥2 args.
            # for k in range(2, len(self.variables)+1):
            #     for combo in combinations(self.variables.values(), k):
            #         yield sp.Max(*combo)
            yield sp.Max(*measurement_vars)
    
    def generate_constraints(self):
        sampled_vars = [v for v in self.variables.values() if 'sampled' in v.name]
        for expr in self.generate_expressions():
            #! Too expansive:
            # #* Relationships among variables.
            # for v in self.variables.values():
            #     if expr.equals(v): 
            #         #* Omit X {} X
            #         continue
            #     yield expr >= v
            #     yield expr <= v
                # yield sp.Eq(expr, v)
            #* Restrict the bounds to sampled vars only.
            for v in sampled_vars:
                #! Can get stuck in equality check for some reason.
                # if expr.equals(v): 
                #     #* Omit X {} X
                #     continue
                yield expr >= v
                yield expr <= v
                
            # #* Relationships between vars and conts.
            # for name in self.constants:
            #     const = sp.symbols(name)
            #     yield expr >= const
            #     yield expr <= const
            #     # yield sp.Eq(expr, const)
            
            #* Implications
            for v in self.variables.values():
                for name in self.constants:
                    const = sp.symbols(name)
                    #* Only consider premises of the form: (var > 0)
                    #* Only consider conclusions of the form: (expr {} const)
                    yield (v > 0) >> (expr >= const)
                    yield (v > 0) >> (expr <= const)

        
    def populate_kb(self):
        num_trivial = 0
        num_constraints = 0
        for constraint in self.generate_constraints():
            # log.info(type(constraint))
            if isinstance(constraint, sp.logic.boolalg.BooleanTrue):
                num_trivial += 1
            else:
                self.kb.append(constraint)
                num_constraints += 1
                if num_constraints % 10_000 == 0:
                    log.info(f"Generated {num_constraints} constraints.")
        log.info(f"Skipped {num_trivial} trivial constraints.")
        log.info(f"Populated KB with {len(self.kb)} constraints.")
        
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