import json
import sympy as sp
from sympy.logic.inference import satisfiable
from typing import *
from rich import print as pprint
from tqdm import tqdm

from known import cidds_conversions
from utils import log, desugar, is_purely_or


class Constraint(object):
    #^ Can't inherit from sympy.Expr as it causes AttributeError on newly defined attributes.
    def __init__(self, expr: sp.Expr):
        # super().__init__()
        #* Semantic expression: original (sugared) formula for readability.
        self.expr: sp.Expr = expr
        # self.ancestors: Set['Constraint'] = set()
        #* Use the syntactic model to identify the constraint.
        self.id = hash(sp.srepr(desugar(self.expr)))
        
    def __hash__(self) -> int:
        #* Define the identity of the constraint for easy set operations.
        return self.id
    
    def __eq__(self, another: 'Constraint') -> bool:
        assert isinstance(another, Constraint), "Can only compare with another 'Constraint'."
        return self.id == another.id
    
    def __repr__(self) -> str:
        return f"Constraint: {self.expr}"
    
    def __str__(self) -> str:
        return self.__repr__()

class Model(object):
    def __init__(self, path_to_constraints: str):
        self.constraints = Model.load_constraints(path_to_constraints)
        self._model = Model.create(self.constraints)
        
    def interpret(self, constraint: sp.Expr, dataset: str='cidds', reverse=False) -> sp.Expr:
        #TODO: Yield the semantic model the formula given the mappings in domain knowledge.
        pass
    
    def entails(self, query: str) -> bool:
        query = sp.sympify(query)
        pprint(query)
        if type(query) == sp.Equivalent:
            #* Needs manual interpretation due to sympy's limitation.
            lhs, rhs = query.args 
            query = [sp.Or(~lhs, rhs), sp.Or(~rhs, lhs)]
        else:
            query = [desugar(query)]
        
        sats = [
            satisfiable(
                desugar(sp.Not(sp.Or(~self._model, q)))
            )
            for q in query
        ]
        #* If the negation is not satisfiable (i.e., there is no interpretation
        #* KB is true but query is false), then the entailment holds.
        return not any(sats)
    
    @staticmethod
    def create(constraints: List[sp.Expr|Constraint] | Set[sp.Expr|Constraint]) -> 'Model':
        log.info("Creating model...")
        constraints = list(constraints)
        if type(constraints[0]) == Constraint:
            constraints = [constraint.expr for constraint in constraints]
        
        simplified = []
        for constraint in tqdm(constraints):
            # simplified.append(constraint)
            #! To create syntactic model from semantic constraints, we must desugar them.
            #* Semantic land -> Syntactic land
            simplified.append(desugar(constraint))
        return sp.And(*simplified)
    
    @staticmethod
    def save_constraints(constraints: List[sp.Expr]|Set[sp.Expr], path: str='constraints.rule'):
        assert len(constraints) > 0, "No constraints to save."
        constraints = list(constraints)
        if type(constraints[0]) == Constraint:
            constraints = [constraint.expr for constraint in constraints]
        
        with open(f"{path}", 'w') as f:
            for constraint in constraints:
                f.write(sp.srepr(constraint) + '\n')

        expressions_str = [str(expr) for expr in constraints]
        with open(f"{path}.json", 'w') as f:
            json.dump(expressions_str, f, indent=4, sort_keys=True)
        log.info(f"Constraints saved to {path}/json")

    @staticmethod
    def load_constraints(path: str='constraints.rule'):
        constraints = []
        with open(f"{path}", 'r') as f:
            for i, line in enumerate(f):
                expr = sp.sympify(line.strip())
                # if not is_purely_or(expr):
                #     constraints.append(expr)
                #     print(f"Loaded {i+1} constraints", end='\r')
                # else:
                #     print(f"Skipping DNF clause: {expr}")
                constraints.append(expr)
                print(f"Loaded {i+1} constraints", end='\r')
        log.info(f"Loaded {len(constraints)} constraints from {path}")
        return constraints