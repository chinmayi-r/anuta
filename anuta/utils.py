import logging
import json
from typing import *
import sympy as sp


true = sp.logic.true
false = sp.logic.false

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

def is_purely_or(expr):
    from sympy.logic.boolalg import Or
    """
    Check if a SymPy formula is purely made of `Or` logic (disjunctions of comparison operations).

    Parameters:
    expr: sympy.Expr
        The Boolean expression to check.

    Returns:
    bool
        True if the formula is purely made of `Or` logic with `Eq` or `Ne` comparisons, False otherwise.
    """
    # Check if the expression is a comparison operation (Eq or Ne)
    def is_comparison(sub_expr):
        return isinstance(sub_expr, (sp.Eq, sp.Ne))

    # Main logic: Traverse the tree to ensure it's purely Or logic
    if isinstance(expr, Or):  # Top-level Or
        return all(is_comparison(arg) or is_purely_or(arg) for arg in expr.args)
    return is_comparison(expr)  # Single comparison is valid

def is_pure_dnf(expr):
    from sympy.logic.boolalg import Or, And, Not
    """
    Check if a SymPy formula is purely made of sp.Or logic (DNF).

    Parameters:
    expr: sympy.Expr
        The Boolean expression to check.

    Returns:
    bool
        True if the formula is in DNF, False otherwise.
    """

    # Check if a sub-expression is a valid DNF clause
    def is_dnf_clause(sub_expr):
        # A DNF clause must be a single variable, its negation, or an And operation
        if isinstance(sub_expr, sp.Symbol):
            return True
        elif isinstance(sub_expr, Not) and isinstance(sub_expr.args[0], sp.Symbol):
            return True
        elif isinstance(sub_expr, And):
            # All terms in the And must be symbols or negated symbols
            return all(
                isinstance(arg, sp.Symbol) or 
                (isinstance(arg, Not) and isinstance(arg.args[0], sp.Symbol))
                for arg in sub_expr.args
            )
        return False

    # Main logic: Check if the expression is an Or of DNF clauses
    if isinstance(expr, Or):
        # All arguments of the Or must be valid DNF clauses
        return all(is_dnf_clause(arg) for arg in expr.args)
    elif is_dnf_clause(expr):  # A single clause can itself be valid DNF
        return True
    return False

def consecutive_combinations(lst):
    ccombo = []
    n = len(lst)
    
    #* Start with combinations of size 2 up to size n
    for size in range(2, n + 1):  
        #* Ensure that the combination is consecutive
        for start in range(n - size + 1):  
            ccombo.append(lst[start: start+size])
    
    return ccombo

def to_big_camelcase(string) -> str:
    words = string.split()
    return ''.join(word.capitalize() for word in words)

def save_constraints(constraints: List[sp.Expr], fname: str='constraints'):
    # Convert expressions to strings
    # expressions_str = [str(expr) for expr in constraints]

    with open(f"{fname}.rule", 'w') as f:
        for constraint in constraints:
            f.write(sp.srepr(constraint) + '\n')
        # json.dump(expressions_str, f, indent=4, sort_keys=True)
        # #* Save the variable bounds
    # print(f"Saved to {fname}.json")
    # with open(f"{fname}_bounds.json", 'w') as f:
    #     json.dump({k: (v.lb, v.ub) for k, v in bounds.items()}, f)
    #     print(f"Saved to {fname}_bounds.json")

def load_constraints(fname: str='constraints'):
    constraints = []
    with open(f"{fname}.rule", 'r') as f:
        for line in f:
            constraints.append(sp.sympify(line.strip()))
    return constraints

def clausify(expr: sp.Expr) -> sp.Expr:
    return sp.simplify_logic(expr, form='dnf', deep=True)