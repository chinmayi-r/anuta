import sys
import sympy as sp

from anuta.theory import Theory
from anuta.utils import clausify


if __name__ == "__main__":
    rulepath1 = sys.argv[1] 
    assert "normal" in rulepath1, "The first argument must be the normal rules file."
    normal_rules = set(Theory.load_constraints(rulepath1, True))
    print(f"Loaded {len(normal_rules)} rules from {rulepath1}")
    
    rulepath2 = sys.argv[2]
    assert "attack" in rulepath2, "The second argument must be the attack rules file."
    attack_rules = set(Theory.load_constraints(rulepath2, True))
    print(f"Loaded {len(attack_rules)} rules from {rulepath2}")
    
    overlap = normal_rules & attack_rules
    print(f"\tOverlap: {len(overlap)}")
    difference = attack_rules - normal_rules
    print(f"\tDifference: {len(difference)}")
    

	