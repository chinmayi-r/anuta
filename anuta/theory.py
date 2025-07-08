from enum import Enum
import json
import sympy as sp
from sympy.logic.inference import satisfiable
from typing import *
from rich import print as pprint
from tqdm import tqdm
from time import perf_counter
import pickle
from pathlib import Path
import z3
from IPython.display import display

from anuta.utils import *
from anuta.known import *


class ProofResult(Enum):
    ENTAILMENT = "Entailed"
    CONTRADICATION = "Contradiction"
    UNKNOWN = "Contingency/Unknown"

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
        self.path_to_constraints = path_to_constraints
        self.constraints = self.load_constraints(path_to_constraints)
        # self._theory = self.create(self.constraints, path_to_constraints)
        self._z3theory = self.z3create(self.constraints)
    
    def proves(self, query: str, verbose: bool=True) -> ProofResult:
        """Checks the existence of a proof of query from theory (syntactic consequence).
        
        We use a Fitch proof system, which is sound and complete for propositional logic.
        """
        self._theory = self.create(self.constraints, self.path_to_constraints, save=False)
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
                        sp.And(self._theory, ~q)
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
        
        # queries = []
        # for q, e in query:
        #     if e == true:
        #         queries.append(~q)
        #     else:
        #         queries.append(q)
        # entailed = not satisfiable(sp.And(self._theory, *queries))
        result = ProofResult.ENTAILMENT if entailed else ProofResult.UNKNOWN
        
        #* Need to check if the query is a contradiction.
        if not entailed:
            # queries = []
            # for q, e in query:
            #     if e == true:
            #         queries.append(q)
            #     else:
            #         queries.append(~q)
            # #* Check if the query is a contradiction.
            # sat = satisfiable(sp.And(self._theory, *queries))
            
            sats = [
                ~sp.Xor(
                    bool(satisfiable(sp.And(self._theory, q))),
                    expected,
                )
                for q, expected in query
            ]
            sat = all(sats)
            
            if not sat:
                result = ProofResult.CONTRADICATION
                if verbose:
                    pprint("Counterexample found:")
                    pprint(sat)
            else:
                result = ProofResult.UNKNOWN
        
        end = perf_counter()
        
        if verbose:
            pprint("Entailed by theory:", entailed, sep=' ')
            pprint(f"Inference time:\t{end-start:.2f} s")
        return result
    
    def z3proves(self, query, verbose=True) -> ProofResult:
        """Try to prove the given claim."""
        query = eval(str(sp.sympify(query)), z3evalmap)
        if verbose:
            display(query)
            
        s = z3.Solver()
        s.add(z3.And(
            self._z3theory,
            z3.Not(query)
        ))
        r = s.check()
        if r == z3.unsat:
            #* If the negation of the query is unsatisfiable, then the query is entailed.
            result = ProofResult.ENTAILMENT
        elif r == z3.unknown:
            #* If the solver cannot determine the satisfiability, we consider it unknown/contingency.
            result = ProofResult.UNKNOWN
        elif r == z3.sat:
            #* If the negation of the query is satisfiable, then the query is a contradiction.
            result = ProofResult.CONTRADICATION
            if verbose:
                pprint("Counterexample found:")
                pprint(s.model())
        
        # query = sp.sympify(query)
        # if type(query) == sp.Equivalent:
        #     #* Biconditional introduction/elimination for Fitch proof system.
        #     lhs, rhs = query.args 
        #     query = [
        #         (clausify(sp.Or(~lhs, rhs)), True), #* Sufficiency
        #         (clausify(sp.Or(~rhs, lhs)), True), #* Necessity
        #     ]
        # elif type(query) == sp.Implies:
        #     #* Implies introduction/elimination for Fitch proof system.
        #     premise, conclusion = query.args
        #     query = [
        #         (clausify(query), True), #* Factual
        #         (clausify(premise >> ~conclusion), False), #* Counterfactual
        #     ]
        # # elif type(query) == sp.Equality:
        # #     lhs, rhs = query.args
        # #     query = [
        # #         #! lhs==rhs could be present and the following would be false.
        # #         (lhs >= rhs, true), #* A ≥ B
        # #         (rhs >= lhs, true), #* B ≥ A
        # #     ]
        # else:
        #     query = [(clausify(query), True)]
        
        # checks = []
        # solvers: List[z3.Solver] = []
        # for q, expected in query:
        #     z3q = eval(str(q), z3evalmap)
        #     if verbose: display(query)
            
        #     s = z3.Solver()
        #     s.add(z3.And(
        #         self._z3theory,
        #         z3.Not(z3q)
        #     ))
        #     r = s.check()
        #     if r == z3.unsat:
        #         #* If the negation of the query is unsatisfiable, then the query is entailed.
        #         checks.append(True == expected)
        #     elif r == z3.unknown:
        #         #* If the solver cannot determine the satisfiability, we consider it unknown/contingency.
        #         checks.append(False == expected)
        #     elif r == z3.sat:
        #         #* If the negation of the query is satisfiable, then the query is a contradiction.
        #         checks.append(False == expected)
        #     solvers.append(s)
    
        # #* If all checks are True, then the query is passed.
        # if all(checks):
        #     result = ProofResult.PROVED
        # elif any(checks):
        #     result = ProofResult.UNKNOWN
        # else:
        #     result = ProofResult.CONTRADICATION
        #     if verbose:
        #         pprint("Counterexample found:")
        #         pprint(solvers[0].model())
                    
            #! We DO NOT neeed a second call since z3 already checks for satisfiability.
            # s.reset()
            # s.add(z3.And(
            #     self._z3theory,
            #     query
            # ))
            # r = s.check()
            # if r == z3.unsat:
            #     proved = ProofResult.CONTRADICATION
            #     if verbose:
            #         pprint("Counterexample found:")
            #         pprint(s.model())
            # elif r == z3.sat:
                # result = ProofResult.UNKNOWN

        if verbose: pprint(result)
        return result
        
        
    @staticmethod
    def z3create(constraints: List[sp.Expr]):
        evalmap = z3evalmap
        z3rules = [
            eval(str(sp.sympify(rule)), evalmap) 
            for rule in constraints
        ]
        z3clauses = [z3.simplify(rule) for rule in z3rules]
        z3theory = z3.And(z3clauses)
        return z3theory
    
    @staticmethod
    def create(
        constraints: List[sp.Expr|Constraint] | Set[sp.Expr|Constraint], 
        path: str,
        save=False,
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
    def coalesce(constraints: List[sp.Expr|Constraint], label: str, save=True) -> sp.Expr:
        if isinstance(constraints[0], Constraint):
            rules: List[sp.Expr] = [constraint.expr for constraint in constraints]
        rules: List[sp.Expr] = list(constraints)
        
        coalesced_rules: Dict[str, sp.Expr] = {}
        conflict_count = 0
        conflicts = set()
        for i, rule in tqdm(enumerate(rules), total=len(rules)):
            if isinstance(rule, sp.Implies):
                antecedent, consequent = rule.args
                key = str(clausify(antecedent.copy()))
                if key in coalesced_rules:
                    coalesed_rule = coalesced_rules[key]
                    assert isinstance(coalesed_rule, sp.Implies), f"Expected {i=} {coalesed_rule=} {key=}"
                    assert len(coalesed_rule.args)==2, f"Expected crule {i=} with 2 args: {rule=} {coalesed_rule=} {key=}"
                    cconsequent = coalesed_rule.args[1]
                    #& Coalesce the consequents of the same antecedent.
                    try:
                        new_crule = sp.Implies(antecedent, (cconsequent & consequent))
                    except Exception as e:
                        print(f"Failed to coalesce {i=} {rule=} {coalesed_rule=} {key=}")
                        print(f"{antecedent=}\n{cconsequent=}\n{consequent=}")
                        print(f"{e=}")
                        exit(1)
                    
                    if not isinstance(new_crule, sp.Implies) and sp.Equivalent(new_crule, ~antecedent):
                        conflict_count += 1
                        #& Conflict occurred, e.g., (A => B & C) + (A => ~B) = ~A
                        if isinstance(cconsequent, sp.And):
                            for term in cconsequent.args:
                                if sp.Equivalent(consequent, ~term):
                                    conflict = term
                                    break
                            else:
                                print(f"Coalescing {rule=} {coalesed_rule=} {key=}")
                                print(f"{new_crule=}")
                                raise ValueError(f"Negation of {consequent} NOT found in {cconsequent=}")
                            #* Remove the conflicting term.
                            new_terms = list(cconsequent.args)
                            new_terms.remove(conflict)
                            new_crule = antecedent >> sp.And(*new_terms) 
                            coalesced_rules[key] = new_crule
                        else:
                            #& (A => B) + (A => ~B) = ~A
                            assert sp.Equivalent(consequent, ~cconsequent), (
                                    f"Negation of {consequent} NOT found in {cconsequent=}")
                    # assert isinstance(new_crule, sp.Implies), f"Result {new_crule=} is not an Implication (from {rule=} + {coalesed_rule=})"
                    # if len(new_crule.args)==1 and sp.Equivalent(new_crule, antecedent):
                    #     print(f"Collapsed {new_crule=} {rule=} {coalesed_rule=} {key=}")
                    #     #* Collapsed.
                    #     conflicts.add(key)
                    #     conflict_count += 1
                    # else:
                else:
                    assert len(rule.args)==2, f"Expected rule {i=} with 2 args: {rule=} {key=}"
                    coalesced_rules[key] = rule
            else:
                coalesced_rules[hash(rule)] = rule
        log.info(f"{conflict_count=}")
                
        # coalesced = {k: (v.args[0] >> clausify(v.args[1])) for k,v in coalesced.items() if isinstance(v, sp.Implies)}
        result = []
        for k, rule in tqdm(coalesced_rules.items(), total=len(coalesced_rules)):
            # if k in conflicts:
            #     continue
            if isinstance(rule, sp.Implies):
                # result.append( transform_consequent(rule.args[0] >> clausify(rule.args[1])) )
                result.append( rule.args[0] >> clausify(rule.args[1]) )
            else:
                result.append( clausify(rule) )
        log.info(f"Coalesced {len(result)} constraints by antecedent")
        
        groupby_consequent = {}
        final_result = []
        for rule in tqdm(result):
            if not isinstance(rule, sp.Implies):
                final_result.append(rule)
                continue 
            antecedent, consequent = rule.args
            key = str(consequent)
            if key in groupby_consequent:
                grule = groupby_consequent[key]
                assert len(grule.args)==2, f"Expected {grule=} with 2 args"
                #& Coalesce the antecedents of the same consequent.
                groupby_consequent[key] = ( (grule.args[0] | antecedent) >> consequent )
            else:
                groupby_consequent[key] = rule

        for rule in groupby_consequent.values():
            final_result.append( clausify(rule.args[0]) >> rule.args[1] )
        # final_result = list(groupby_consequent.values())
        log.info(f"Coalesced {len(final_result)} constraints by consequent")
        
        if save:
            Theory.save_constraints(final_result, f"coalesced_{label}.pl")
        return final_result
        
    @staticmethod
    def interpret(rules: List[sp.Expr], dataset: str='cidds', 
                  save_path=None) -> sp.Expr:
        if dataset != 'cidds':
            raise ValueError(f"Dataset not supported: {dataset}")
        #* CIDDs specific conversions.
        def interpret_cidds(varname, varval):
            # varval = int(varval)
            # if any([op in varval for op in ['=', '≠', '∧', '∨', '⇒', '≥', '×']]):
            #     return value
            if not isinstance(varval, int):
                return varval
            
            if 'Ip' in varname:
                value = cidds_ip_conversion.inverse[varval]
            elif 'Flags' in varname:
                value = cidds_flags_conversion.inverse[varval]
            elif 'Proto' in varname:
                value = cidds_proto_conversion.inverse[varval]
            elif 'Pt' in varname:
                value = cidds_port_conversion.inverse[varval]
            else:
                value = varval
            return value

        def eq_str(a, b): return f"({a} = {interpret_cidds(a, b)})"
        def ne_str(a, b): return f"({a} ≠ {interpret_cidds(a, b)})"
        def and_str(*args):
            s = ''
            for arg in args:
                s += f"{arg} ∧ "
            return s[:-3]
        def or_str(*args):
            s = ''
            for arg in args:
                s += f"{arg} v "
            return f"({s[:-3]})"
        def implies_str(a, b): return f"({a}) ⇒ ({b})"
        def sym_str(a): return a
        def int_str(a): return int(a)
        def float_str(n, precision): return f"{float(n):.3f}" #return f"{float(n):.{precision}f}"
        def gt_str(a, b): return f"({a} ≥ {b})"
        def strict_gt_str(a, b): return f"({a} > {b})"
        def lt_str(a, b): return f"({a} ≤ {b})"
        def strict_lt_str(a, b): return f"({a} < {b})"
        def mul_str(a, b): return f"{a}x{b}"

            
        cidds_interpretation = {
            'Equality': eq_str, 'Unequality': ne_str, 'And': and_str, 'Or': or_str, 
            'Implies': implies_str, 'Symbol': sym_str, 'Integer': int_str, 'Mul': mul_str, 
            'GreaterThan': gt_str, 'LessThan': lt_str, 'Float': float_str,
            'StrictGreaterThan': strict_gt_str, 'StrictLessThan': strict_lt_str,
        }
        interpreted = sorted([eval(sp.srepr(rule), cidds_interpretation) for rule in rules])
        if save_path:
            with open(save_path, 'w') as f:
                for rule in interpreted:
                    f.write(rule + '\n')
        return interpreted
    
    @staticmethod
    def save_thoery(theory: sp.Expr, path: str='theories/theory.pkl') -> None:
        with open(f"{path}", 'wb') as f:
            pickle.dump(theory, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Theory saved to {path}")
    
    @staticmethod
    def save_constraints(constraints: List[sp.Expr]|Set[sp.Expr], path: str='constraints.pl'):
        if len(constraints) == 0:
            log.info("No constraints to save.")
            return
        constraints = list(constraints)
        if isinstance(constraints[0], Constraint):
            constraints = [constraint.expr for constraint in constraints]
        
        with open(f"{path}", 'w') as f:
            for constraint in constraints:
                f.write(sp.srepr(constraint) + '\n')

        expressions_str = sorted([str(expr) for expr in constraints])
        with open(f"{path}.json", 'w') as f:
            json.dump(expressions_str, f, indent=4, sort_keys=True)
        log.info(f"Constraints saved to {path}/json")

    @staticmethod
    def load_constraints(path: str='constraints.pl', wrapper=False) -> List[Constraint | sp.Expr]:
        constraints = []
        with open(f"{path}", 'r') as f:
            for i, line in enumerate(f):
                expr = sp.sympify(line.strip()) if not wrapper \
                    else Constraint(sp.sympify(line.strip()))
                
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
    
    @staticmethod
    def load_rules(path: str='rules.pl.json') -> List[sp.Expr]:
        rules = []
        with open(f"{path}", 'r') as f:
            jsonrules = json.load(f)
            for rule in jsonrules:
                rules.append(sp.sympify(rule))
                print(f"Loaded {len(rules)} rules", end='\r')
        log.info(f"Loaded {len(rules)} rules from {path}")
        return rules