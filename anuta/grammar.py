from itertools import combinations
from typing import *
import sympy as sp
from dataclasses import dataclass
from enum import Enum, auto

from utils import log, consecutive_combinations


@dataclass
class Bounds:
    lb: float
    ub: float

class Constant(Enum):
    ASSIGNMENT = auto()
    SCALAR = auto()
    #! Abuse of Enum
    def __init__(self, values: List=None) -> None:
        super().__init__()
        self.values = values

class Kind(Enum):
    NUMERICAL = auto()
    CATEGORICAL = auto()

@dataclass
class Domain:
    kind: Kind
    bounds: Bounds
    values: List[Enum]

class Operator(Enum):
    NOP = 0
    PLUS = auto()
    MAX = auto()
    
    
class Anuta(object):
    def __init__(self, variables: List[str], domains: Dict[str, Domain], 
                 constants: Dict[str, Tuple[Constant, List[int]]]=None, 
                 prior_kb: List[sp.Expr]=[]):
        variables = sp.symbols(' '.join(variables))
        self.variables = {v.name: v for v in variables}
        self.domains = domains
        self.constants = constants
        
        self.prior_kb = prior_kb
        self.initial_kb = []
        self.learned_kb = []
    
    def generate_expressions(self) -> Generator[sp.Expr, None, None]:
        for name, var in self.variables.items():
            if name in self.constants:
                #* If the var has associated constants, don't enumerate its domain.
                match self.constants[name]:
                    case Constant.ASSIGNMENT:
                        for const in self.constants[name].values:
                            #* Var == const
                            yield sp.Eq(var, sp.S(const))
                    case Constant.SCALAR:
                        for const in self.constants[name].values:
                            #* Var x const
                            yield sp.Mul(var, sp.S(const))
                    case _:
                        raise NotImplementedError(f"Unsupported constant: {self.constants[name]}")
            else:
                domain = self.domains[name]
                if domain.kind == Kind.NUMERICAL: continue
                #* Enumerate the domain of a categorical var.
                if domain.kind == Kind.CATEGORICAL:
                    for value in domain.values:
                        #* Var == value
                        yield sp.Eq(var, sp.S(value))
        
    def generate_arity2_constraints(self) -> Generator[sp.Expr, None, None]:
        #* Arity-1 constraints: self.prior_kb
        
        #* Arity-2 constraints:
        expressions = list(self.generate_expressions())
        for i, expr_lhs in enumerate(expressions):
            if type(expr_lhs) == sp.Equality:
                #* Starts from the next expression to avoid self-comparison and ordering.
                for expr_rhs in expressions[i+1:]:
                    # if expr_lhs == expr_rhs: continue
                    # if expr_lhs.args[0] == expr_rhs.args[0]: continue
                    # #! Do NOT avoid (Var==const1) OR (Var==const2)
                    
                    #* Avoid (Var==const1) AND (Var==const2)
                    if type(expr_rhs) == sp.Equality and expr_lhs.args[0] != expr_rhs.args[0]:
                        #* (Var==const1) AND (Var==const2)
                        yield sp.And(expr_lhs, expr_rhs)
                        #* ~(Var==const1) AND (Var==const2)
                        yield sp.And(sp.Not(expr_lhs), expr_rhs)
                        #* ~(Var==const1) AND ~(Var==const2)
                        yield sp.And(sp.Not(expr_lhs), sp.Not(expr_rhs))
                        #^ AND constraints are too restrictive and will be all eliminated.
                        #! But, we might need them for arity-3 constraints.
                        
                        #* (Var==const1) OR (Var==const2)
                        yield sp.Or(expr_lhs, expr_rhs)
                        #* ~(Var==const1) OR (Var==const2)
                        yield sp.Or(sp.Not(expr_lhs), expr_rhs)
                        #* ~(Var==const1) OR ~(Var==const2)
                        yield sp.Or(sp.Not(expr_lhs), sp.Not(expr_rhs))
                
            elif type(expr_lhs) == sp.Mul:
                for name, var in self.variables.items():
                    if self.domains[name].kind == Kind.NUMERICAL:
                        #* (Var x const1) >= Var
                        yield expr_lhs >= var
                        #* (Var x const1) <= Var
                        yield expr_lhs <= var
    
    def interval_filter(self, constraint: sp.Expr) -> bool:
        """Filter constraints using interval arithmetic.

        :param constraint: Constraint to filter.
        :return: False if the constraint is valid, True if the constraint is invalid.
        """
        
        if type(constraint) in [sp.And, sp.Or]:
            #* Do not filter logical constraints.
            return False
        
        lhs, rhs_var = constraint.args
        scalar, lhs_var = lhs.args
        
        match type(constraint):
            case sp.Ge:
                #* s x Amax <= Bmin
                return scalar*self.domains[str(lhs_var)].bounds.ub <= self.domains[str(rhs_var)].bounds.lb
            case sp.Le:
                #* s x Amin >= Bmax
                return scalar*self.domains[str(lhs_var)].bounds.lb >= self.domains[str(rhs_var)].bounds.ub
            case _:
                raise NotImplementedError(f"Unsupported constraint type: {type(constraint)}") 
    
    def generate_arity3_constraints(self, arity2_constraints) -> Generator[sp.Expr, None, None]:
        for expression in self.generate_expressions():
            #* Multiplication expressions can't be premises.
            if type(expression) == sp.Mul: continue
            
            for constraint in arity2_constraints:
                #* Dedupe.
                lhs, rhs = constraint.args
                if (expression == lhs or sp.Not(expression) == lhs or 
                    expression == rhs or sp.Not(expression) == rhs): continue
                #! Omit AND constraints for now.
                # yield sp.And(expression, constraint)
                yield sp.Or(expression, constraint)
                yield sp.Or(sp.Not(expression), constraint)
                yield sp.Or(expression, sp.Not(constraint))
                yield sp.Or(sp.Not(expression), sp.Not(constraint))
        #^ No ≤ or ≥ constraints for now.
    
    def populate_kb(self) -> None:
        num_constraints = 0
        num_trivial = 0
        interval_filtered = 0
        arity2_constraints = []
        for constraint in self.generate_arity2_constraints():
            # log.info(type(constraint))
            if isinstance(constraint, sp.logic.boolalg.BooleanTrue):
                num_trivial += 1
            else:
                if self.interval_filter(constraint):
                    log.info(f"Interval-filtered: {constraint}")
                    interval_filtered += 1
                    continue
                
                arity2_constraints.append(constraint)
                #& Avoid arity-2 AND constraints for now.
                if type(constraint) != sp.And:
                    self.initial_kb.append(constraint)
                
                num_constraints += 1
                if num_constraints % 10_000 == 0:
                    log.info(f"Generated {num_constraints} constraints.")
        log.info(f"Trivial constraints: {num_trivial}")
        log.info(f"Arity-2 constraints: {num_constraints}")
        log.info(f"Interval-filtered constraints: {interval_filtered}")
        
        old_num_constraints = num_constraints
        for constraint in self.generate_arity3_constraints(arity2_constraints):
            self.initial_kb.append(constraint)
            num_constraints += 1
            if num_constraints % 10_000 == 0:
                log.info(f"Generated {num_constraints} constraints.")
        log.info(f"Arity-3 constraints: {num_constraints-old_num_constraints}")
        
        log.info(f"Populated KB with {len(self.initial_kb)} constraints.\n")
        
        #! Too expansive, and the reduction is little (26/149720)
        # log.info(f"Filtering redundant constraints ...")
        # # Nonnegative integers
        # assumptions = [v >= 0 for v in self.variables.values()]
        # cnf = sp.And(*(self.initial_kb + assumptions))
        # simplified_logic = sp.simplify_logic(cnf)
        # #! Too expansive
        # # simplified = simplified_logic.simplify()
        # reduced_constraints = list(simplified_logic.args) \
        #     if isinstance(simplified_logic, sp.And) else [simplified_logic]
        # num_redundant = len(self.initial_kb) - len(reduced_constraints)
        # #* Update KB
        # self.initial_kb = reduced_constraints
        # log.info(f"Removed {num_redundant} redundant constraints.\nFinal KB size: {len(self.initial_kb)}")
        return


class AnutaMilli(object):
    def __init__(self, variables: List[str], bounds: Dict[str, Bounds], constants: Dict[str, int]=None, operators: List[int]=None):
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