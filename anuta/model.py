import json
import sympy as sp
from sympy.logic.inference import satisfiable
from typing import *

from known import cidds_conversions
from utils import log


class Model(object):
    def __init__(self, path_to_constraints: str):
        self._model = sp.And(*self.load_constraints(path_to_constraints))
        
    def interpret(self, constraint: sp.Expr, dataset: str='cidds', reverse=False) -> sp.Expr:
        #TODO: Yield the semantic model the formula given the mappings in domain knowledge.
        pass
    
    def entails(self, query: str) -> bool:
        query = sp.sympify(query)
        if type(query) == sp.Equivalent:
            #* Needs manual interpretation due to sympy's limitation.
            lhs, rhs = query.args 
            query = [sp.Or(~lhs, rhs), sp.Or(~rhs, lhs)]
        else:
            query = [sp.simplify_logic(query)]
        
        sats = [
            satisfiable(
                sp.Not(sp.Or(~self._model, q))
            )
            for q in query
        ]
        #* If the negation is not satisfiable (i.e., there is no interpretation
        #* KB is true but query is false), then the entailment holds.
        return not any(sats)
    
    @staticmethod
    def save_constraints(constraints: List[sp.Expr], path: str='constraints.rule'):
        with open(f"{path}", 'w') as f:
            for constraint in constraints:
                f.write(sp.srepr(constraint) + '\n')

        expressions_str = [str(expr) for expr in constraints]
        with open(f"{path}.json", 'w') as f:
            json.dump(expressions_str, f, indent=4, sort_keys=True)
        log.info(f"Constraints saved to {path}.rule/json")

    @staticmethod
    def load_constraints(path: str='constraints.rule'):
        constraints = []
        with open(f"{path}", 'r') as f:
            for line in f:
                constraints.append(sp.sympify(line.strip()))
        log.info(f"Constraints loaded from {path}")
        return constraints