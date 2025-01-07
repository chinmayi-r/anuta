import json
import sympy as sp
from sympy.logic.inference import satisfiable
from typing import *
from rich import print as pprint
from tqdm import tqdm
from time import perf_counter
import pickle
from pathlib import Path

from anuta.utils import *


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
        #* Use the syntactic theory to identify the constraint.
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

class Theory(object):
    def __init__(self, path_to_constraints: str):
        self.constraints = Theory.load_constraints(path_to_constraints)
        self._thoery = Theory.create(self.constraints, path_to_constraints)
        
    def interpret(self, constraint: sp.Expr, dataset: str='cidds', reverse=False) -> sp.Expr:
        #TODO: Yield the semantic theory the formula given the mappings in domain knowledge.
        pass
    
    def proves(self, query: str, verbose: bool=True) -> bool:
        """Checks the existence of a proof of query from theory (syntactic consequence).
        
        We use a Fitch proof system, which is sound and complete for propositional logic.
        """
        query = sp.sympify(query)
        if verbose:
            pprint("Query:", query, sep='\t')

        if type(query) == sp.Equivalent:
            #* Biconditional introduction/elimination for Fitch proof system.
            lhs, rhs = query.args 
            query = [
                (clausify(sp.Or(~lhs, rhs)), true), #* Sufficiency
                (clausify(sp.Or(~rhs, lhs)), true), #* Necessity
            ]
        elif type(query) == sp.Implies:
            #* Implies introduction/elimination for Fitch proof system.
            premise, conclusion = query.args
            query = [
                (clausify(query), true), #* Factual
                (clausify(premise >> ~conclusion), false), #* Counterfactual
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
            ~sp.Xor(
                bool(satisfiable(
                    #* Both the theory and query should in clausal form already.
                    # clausify(
                        #* Proof by Resolution rule of Inference: 
                        #*  To determine whether a set of sentences KB logically entails a sentence q, 
                        #*  rewrite KB\/{~q} in clausal form and try to derive the empty clause.
                        sp.And(self._thoery, ~q)
                        #^ Same as below:
                        # sp.Not(sp.Or(~self._model, q))
                    # )
                )),
            #* Compare with expected output using XNOR.
            expected)
            for q, expected in query
        ]
        #* If the negation is not satisfiable (i.e., there is no interpretation
        #* that KB is true but query is false), then the entailment holds.
        entailed = not any(sats)
        end = perf_counter()
        
        if verbose:
            pprint("Entailed by theory:", entailed, sep=' ')
            pprint(f"Inference time:\t{end-start:.2f} s")
        return entailed
    
    @staticmethod
    def create(
        constraints: List[sp.Expr|Constraint] | Set[sp.Expr|Constraint], 
        path: str,
        save=True,
    ) -> sp.Expr:
        log.info("Creating theory...")
        #* Take the last part of the path w/o extension as the theory name.
        modelname = path.split('/')[-1].split('.')[0]
        modelpath = f"theories/{modelname}.pkl"
        # #* Check if the theory is already created.
        # if Path(modelpath).exists():
        #     with open(modelpath, 'rb') as f:
        #         theory: sp.Expr = pickle.load(f)
        #     log.info(f"Theory loaded from {modelpath}")
        #     log.info(f"Theory size {len(theory.args)}")
        #     return theory
        
        constraints = list(constraints)
        if type(constraints[0]) == Constraint:
            constraints = [constraint.expr for constraint in constraints]
        
        simplified = []
        for constraint in tqdm(constraints):
            #! To create syntactic theory from semantic constraints, we must desugar them.
            #* Semantic land -> Syntactic land
            simplified.append(clausify(constraint))
            
        #& A constraint theory (syntactical) is conjucts of clauses consistent with the data.
        theory = sp.simplify_logic(sp.And(*simplified), form='cnf', deep=True)
        log.info(f"Theory size {len(theory.args)}")
        if save:
            Theory.save_thoery(theory, modelpath)
        return theory
    
    @staticmethod
    def save_thoery(theory: sp.Expr, path: str='theories/theory.pkl') -> None:
        with open(f"{path}", 'wb') as f:
            pickle.dump(theory, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Theory saved to {path}")
    
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
                # #! Ignore equality constraints (from prior) as they are too strong.
                # if type(expr) in [sp.Equality]:
                #     continue
                # if not is_purely_or(expr):
                #     constraints.append(expr)
                #     print(f"Loaded {i+1} constraints", end='\r')
                # else:
                #     print(f"Skipping DNF clause: {expr}")
                constraints.append(expr)
                print(f"Loaded # of constraints:\t{i+1}", end='\r')
        log.info(f"Loaded {len(constraints)} constraints from {path}")
        return constraints