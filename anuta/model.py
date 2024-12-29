import json
import sympy as sp
from sympy.logic.inference import satisfiable
from typing import *
from rich import print as pprint
from tqdm import tqdm
from time import perf_counter

from utils import *


class Constraint(object):
    #^ Can't inherit from sympy.Expr as it causes AttributeError on newly defined attributes.
    def __init__(self, expr: sp.Expr):
        # super().__init__()
        #* Semantic expression: original (sugared) formula for readability.
        self.expr: sp.Expr = expr
        # self.ancestors: Set['Constraint'] = set()
        # #* Numerical identity: -1, Scaled numerial: ≥0, others: None
        # self.rank: int = None
        # self.maxrank: int = None
        #* Use the syntactic model to identify the constraint.
        self.id = hash(sp.srepr(clausify(self.expr)))
        
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
    
    def entails(self, query: str, verbose: bool=True) -> bool:
        """Proof system for entailment checking."""
        query = sp.sympify(query)
        if type(query) == sp.Equivalent:
            #* Needs manual interpretation due to sympy's limitation.
            lhs, rhs = query.args 
            query = [
                (sp.Or(~lhs, rhs), true), #* Sufficiency
                (sp.Or(~rhs, lhs), true), #* Necessity
            ]
        elif type(query) == sp.Implies:
            premise, conclusion = query.args
            query = [
                (query, true), #* Factual
                (premise >> ~conclusion, false), #* Counterfactual
            ]
        # elif type(query) == sp.Equality:
        #     lhs, rhs = query.args
        #     query = [
        #         #! lhs==rhs could be present and the following would be false.
        #         (lhs >= rhs, true), #* A ≥ B
        #         (rhs >= lhs, true), #* B ≥ A
        #     ]
        else:
            query = [(clausify(query), true)]
        
        start = perf_counter()
        sats = [
            bool(satisfiable(
                clausify(sp.Not(sp.Or(~self._model, q)))
            )) & expected
            for q, expected in query
        ]
        #* If the negation is not satisfiable (i.e., there is no interpretation
        #* that KB is true but query is false), then the entailment holds.
        entailed = not any(sats)
        end = perf_counter()
        
        if verbose:
            pprint("Query:", query, sep='\t')
            pprint("Entailed by model: ", entailed, sep=' ')
            log.info(f"Inference time: {end-start:.2f}s")
        return entailed
    
    @staticmethod
    def create(constraints: List[sp.Expr|Constraint] | Set[sp.Expr|Constraint]) -> sp.Expr:
        log.info("Creating model...")
        constraints = list(constraints)
        if type(constraints[0]) == Constraint:
            constraints = [constraint.expr for constraint in constraints]
        
        simplified = []
        for constraint in tqdm(constraints):
            # simplified.append(constraint)
            #! To create syntactic model from semantic constraints, we must desugar them.
            #* Semantic land -> Syntactic land
            simplified.append(clausify(constraint))
        model = sp.And(*simplified)
        log.info(f"Created model of size {len(simplified)}")
        return model
    
    @staticmethod
    def save_constraints(constraints: List[sp.Expr]|Set[sp.Expr], path: str='constraints.rule'):
        assert len(constraints) > 0, "No constraints to save."
        constraints = list(constraints)
        if isinstance(constraints[0], Constraint):
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