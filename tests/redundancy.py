import sys
import sympy as sp

from anuta.theory import Theory
from anuta.utils import clausify


if __name__ == '__main__':
    rulepath = sys.argv[1]
    rules = Theory.load_constraints(rulepath, False)
    print(f"Loaded {len(rules)} rules from {rulepath}")
    
    # rules = set([Constraint(clausify(rule)) for rule in rules])
    # print(f"Reduced form has {len(rules)} rules")
    rules = [clausify(rule) for rule in rules]
    simplified = sp.simplify_logic(sp.And(*rules), form='cnf', deep=True)
    print(f"Converted to CNF with {len(simplified.args)} clauses")