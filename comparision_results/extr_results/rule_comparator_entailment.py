import json
import sympy as sp
from sympy import symbols, And, Or, Not, Implies, Eq, Ne, simplify
from sympy.logic.boolalg import to_cnf, to_dnf
from sympy.parsing.sympy_parser import parse_expr
import z3
import re
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
from rich import print as pprint
import pickle

from anuta.utils import *
from anuta.known import *

class ProofResult(Enum):
    ENTAILMENT = "Entailed"
    CONTRADICTION = "Contradiction"
    UNKNOWN = "Non-Entailed/Unknown"

class Constraint:
    def __init__(self, expr: sp.Expr):
        self.expr: sp.Expr = expr
        self.id = hash(sp.srepr(clausify(self.expr)))
        
    def __hash__(self) -> int:
        return self.id
    
    def __eq__(self, another: 'Constraint') -> bool:
        assert isinstance(another, Constraint), "Can only compare with another 'Constraint'."
        return self.id == another.id
    
    def __repr__(self) -> str:
        return f"Constraint: {self.expr}"

def clausify(expr: sp.Expr) -> sp.Expr:
    """Convert expression to DNF form for better logical reasoning"""
    return sp.simplify_logic(expr, form='dnf', deep=True)

# Z3 evaluation mapping
z3evalmap = {
    'Bytes': z3.Real('Bytes'),
    'Packets': z3.Real('Packets'),
    'Duration': z3.Real('Duration'),
    'DstIpAddr': z3.Real('DstIpAddr'),
    'SrcIpAddr': z3.Real('SrcIpAddr'),
    'DstPt': z3.Real('DstPt'),
    'SrcPt': z3.Real('SrcPt'),
    'Proto': z3.Real('Proto'),
    'Flags': z3.Real('Flags'),
    'And': z3.And,
    'Or': z3.Or,
    'Not': z3.Not,
    'Implies': z3.Implies,
    'Eq': lambda x, y: x == y,
    'Ne': lambda x, y: x != y,
    'GreaterThan': lambda x, y: x > y,
    'LessThan': lambda x, y: x < y,
    'StrictGreaterThan': lambda x, y: x > y,
    'StrictLessThan': lambda x, y: x < y,
    'Add': lambda x, y: x + y,
    'Mul': lambda x, y: x * y,
    'Integer': lambda x: x,
    'Float': lambda x: x,
    'Rational': lambda x: float(x),
}

class FixedRuleComparator:
    def __init__(self):
        # Define domain variables
        self.Bytes, self.Packets, self.Duration = symbols('Bytes Packets Duration', real=True)
        self.DstIpAddr, self.SrcIpAddr = symbols('DstIpAddr SrcIpAddr', real=True)
        self.DstPt, self.SrcPt, self.Proto, self.Flags = symbols('DstPt SrcPt Proto Flags', real=True)
        
        # Z3 solver instances
        self.z3_solver = z3.Solver()
        
    def load_constraints(self, path: str, wrapper=False) -> List[Constraint | sp.Expr]:
        """Load constraints from a file - aligned with second code"""
        constraints = []
        with open(f"{path}", 'r') as f:
            for i, line in enumerate(f):
                expr = sp.sympify(line.strip()) if not wrapper \
                    else Constraint(sp.sympify(line.strip()))
                
                constraints.append(expr)
                print(f"Loaded # of constraints:\t{i+1}", end='\r')
        if ANUTA_AVAILABLE:
            log.info(f"Loaded {len(constraints)} constraints from {path}")
        else:
            print(f"Loaded {len(constraints)} constraints from {path}")
        return constraints
    
    def load_rules(self, path: str) -> List[sp.Expr]:
        """Load rules from JSON file - aligned with second code"""
        rules = []
        with open(f"{path}", 'r') as f:
            jsonrules = json.load(f)
            for rule in jsonrules:
                rules.append(sp.sympify(rule))
                print(f"Loaded {len(rules)} rules", end='\r')
        if ANUTA_AVAILABLE:
            log.info(f"Loaded {len(rules)} rules from {path}")
        else:
            print(f"Loaded {len(rules)} rules from {path}")
        return rules
    
    def load_rules_from_file(self, file_path: str, max_rules: Optional[int] = None) -> List[str]:
        """Load rules from a JSON file with optional limit"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different possible JSON structures
            if isinstance(data, list):
                rules = data
            elif isinstance(data, dict):
                rules = []
                for key in ['rules', 'constraints', 'predicates', 'conditions']:
                    if key in data:
                        rules = data[key]
                        break
                else:
                    for value in data.values():
                        if isinstance(value, list):
                            rules.extend(value)
                        elif isinstance(value, str):
                            rules.append(value)
            else:
                print(f"Unexpected JSON structure in {file_path}")
                return []
            
            if max_rules is not None and max_rules > 0:
                rules = rules[:max_rules]
                print(f"Limited to first {len(rules)} rules from {file_path}")
            
            return rules
                
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []
    
    def parse_rule(self, rule_str: str, file_path: Optional[str] = None, 
                   line_number: Optional[int] = None) -> Optional[sp.Expr]:
        """Parse a string rule into SymPy expression"""
        # Clean up the string
        rule_str = rule_str.strip().strip('"').strip("'")
        rule_str = re.sub(r'\s+', ' ', rule_str)
        
        # Replace logical operators with SymPy equivalents
        rule_str = rule_str.replace('&', ' & ')
        rule_str = rule_str.replace('|', ' | ')
        
        # Handle functions
        rule_str = re.sub(r'Ne\(([^,]+),\s*([^)]+)\)', r'Ne(\1, \2)', rule_str)
        rule_str = re.sub(r'Eq\(([^,]+),\s*([^)]+)\)', r'Eq(\1, \2)', rule_str)
        
        # Handle Implies function
        def replace_implies(match):
            content = match.group(1)
            paren_count = 0
            comma_pos = -1
            for i, char in enumerate(content):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == ',' and paren_count == 0:
                    comma_pos = i
                    break
            
            if comma_pos == -1:
                return match.group(0)
            
            condition = content[:comma_pos].strip()
            conclusion = content[comma_pos+1:].strip()
            return f'Implies({condition}, {conclusion})'
        
        while 'Implies(' in rule_str:
            old_rule = rule_str
            rule_str = re.sub(r'Implies\(([^)]+(?:\([^)]*\)[^)]*)*)\)', replace_implies, rule_str, count=1)
            if rule_str == old_rule:
                break
        
        try:
            local_dict = {
                'Bytes': self.Bytes, 'Packets': self.Packets, 'Duration': self.Duration,
                'DstIpAddr': self.DstIpAddr, 'SrcIpAddr': self.SrcIpAddr,
                'DstPt': self.DstPt, 'SrcPt': self.SrcPt, 'Proto': self.Proto, 'Flags': self.Flags,
                'Implies': Implies, 'Eq': Eq, 'Ne': Ne, 'And': And, 'Or': Or, 'Not': Not
            }
            
            expr = parse_expr(rule_str, local_dict=local_dict, transformations='all')
            return expr
        except Exception as e:
            print(f"Error parsing rule in {file_path or 'unknown file'}, line {line_number or 'unknown'}")
            print(f"Rule: {rule_str}")
            print(f"Error: {e}")
            return None
    
    def sympy_to_z3(self, expr: sp.Expr) -> z3.ExprRef:
        """Convert SymPy expression to Z3 expression"""
        try:
            return eval(str(expr), z3evalmap)
        except Exception as e:
            print(f"Error converting to Z3: {expr}, Error: {e}")
            return None
    
    @staticmethod
    def z3create(constraints: List[sp.Expr]):
        """Create Z3 theory from constraints - aligned with second code"""
        evalmap = z3evalmap
        z3rules = [
            eval(str(sp.sympify(rule)), evalmap) 
            for rule in constraints
        ]
        z3clauses = [z3.simplify(rule) for rule in z3rules]
        z3theory = z3.And(z3clauses)
        return z3theory
    
    def create_normal_theory(self, normal_rules: List[sp.Expr]) -> z3.ExprRef:
        """Create a unified theory from normal rules using AND (all rules must hold)"""
        print("Creating normal theory...")
        
        # Convert all rules to DNF for better logical reasoning (matching second code)
        dnf_rules = []
        for rule in tqdm(normal_rules, desc="Converting to DNF"):
            try:
                dnf_rule = clausify(rule)
                dnf_rules.append(dnf_rule)
            except Exception as e:
                print(f"Error converting rule to DNF: {rule}, Error: {e}")
                continue
        
        # Use the z3create method from second code
        z3_theory = self.z3create(dnf_rules)
        return z3_theory
    
    @staticmethod
    def create(
        constraints: List[sp.Expr|Constraint] | Set[sp.Expr|Constraint], 
        path: str,
        save=False,
    ) -> sp.Expr:
        """Create theory from constraints - aligned with second code"""
        if ANUTA_AVAILABLE:
            log.info("Creating theory...")
        else:
            print("Creating theory...")
        
        # Take the last part of the path w/o extension as the theory name.
        modelname = path.split('/')[-1].split('.')[0]
        modelpath = f"theories/{modelname}.pkl"
        
        constraints = list(constraints)
        if type(constraints[0]) == Constraint:
            constraints = [constraint.expr for constraint in constraints]
        
        simplified = []
        for constraint in tqdm(constraints):
            # To create syntactic theory from semantic constraints, we must desugar them.
            # Semantic land -> Syntactic land
            simplified.append(clausify(constraint))
            
        # A constraint theory (syntactical) is conjucts of clauses consistent with the data.
        theory = sp.simplify_logic(sp.And(*simplified), form='cnf', deep=True)
        
        if ANUTA_AVAILABLE:
            log.info(f"Theory size {len(theory.args)}")
        else:
            print(f"Theory size {len(theory.args)}")
        
        if save:
            FixedRuleComparator.save_theory(theory, modelpath)
        return theory
    
    def z3proves(self, theory: z3.ExprRef, query, verbose=True) -> ProofResult:
        """Try to prove the given claim - aligned with second code"""
        # query = eval(str(sp.sympify(query)), z3evalmap)
        try:
            query_str = str(query)  # Capture the raw query before sympify
            query = eval(str(sp.sympify(query_str)), z3evalmap)  # <-- Fix: sympify the string first
        except Exception as e:
            print(f"ERROR on query: {query}")  # <-- Log the problematic rule
            raise

        if verbose:
            print(query)
            
        s = z3.Solver()
        s.add(z3.And(
            theory,
            z3.Not(query)
        ))
        r = s.check()
        if r == z3.unsat:
            # If the negation of the query is unsatisfiable, then the query is entailed.
            result = ProofResult.ENTAILMENT
        elif r == z3.unknown:
            # If the solver cannot determine the satisfiability, we consider it unknown/contingency.
            result = ProofResult.UNKNOWN
        elif r == z3.sat:
            # If the negation of the query is satisfiable, then the query is a contradiction.
            result = ProofResult.CONTRADICTION
            if verbose:
                pprint("Counterexample found:")
                pprint(s.model())

        if verbose: 
            pprint(result)
        return result
    
    def z3_entails(self, theory: z3.ExprRef, query: z3.ExprRef) -> ProofResult:
        """Check if theory entails query using Z3"""
        return self.z3proves(theory, query, verbose=False)
    
    def z3_contradicts(self, theory: z3.ExprRef, query: z3.ExprRef) -> bool:
        """Check if theory contradicts query (theory AND query is unsatisfiable)"""
        solver = z3.Solver()
        solver.add(z3.And(theory, query))
        
        result = solver.check()
        return result == z3.unsat
    
    @staticmethod
    def save_theory(theory: sp.Expr, path: str='theories/theory.pkl') -> None:
        """Save theory to file - aligned with second code"""
        with open(f"{path}", 'wb') as f:
            pickle.dump(theory, f, protocol=pickle.HIGHEST_PROTOCOL)
        if ANUTA_AVAILABLE:
            log.info(f"Theory saved to {path}")
        else:
            print(f"Theory saved to {path}")
    
    @staticmethod
    def save_constraints(constraints: List[sp.Expr]|Set[sp.Expr], path: str='constraints.pl'):
        """Save constraints to file - aligned with second code"""
        if len(constraints) == 0:
            if ANUTA_AVAILABLE:
                log.info("No constraints to save.")
            else:
                print("No constraints to save.")
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
        
        if ANUTA_AVAILABLE:
            log.info(f"Constraints saved to {path}/json")
        else:
            print(f"Constraints saved to {path}/json")
    
    def find_weak_normal_rules(
        self, 
        attack_rule: sp.Expr, 
        normal_rules: List[Tuple[int, str, sp.Expr]]
    ) -> List[int]:
        """Identifies normal rules that fail to block an independent attack rule."""
        weak_rules = []
        z3_attack = self.sympy_to_z3(attack_rule)
        
        for idx, _, normal_expr in normal_rules:  # (idx, original_rule, parsed_expr)
            z3_normal = self.sympy_to_z3(normal_expr)
            
            # Skip if rules don't share variables (e.g., one uses Proto, another uses SrcPt)
            if not self._rules_share_variables(attack_rule, normal_expr):
                continue
            
            # Check if normal_rule + attack_rule is satisfiable (no contradiction)
            if not self.z3_contradicts(z3_normal, z3_attack):
                weak_rules.append(idx)
        
        return weak_rules

    def _rules_share_variables(self, rule1: sp.Expr, rule2: sp.Expr) -> bool:
        """Check if two rules have overlapping variables (e.g., both use 'Proto')."""
        vars1 = set(rule1.free_symbols)
        vars2 = set(rule2.free_symbols)
        return len(vars1 & vars2) > 0

    def find_non_entailed_rules(self, normal_theory: z3.ExprRef, 
                               attack_rules: List[Tuple[int, str, sp.Expr]]) -> List[Tuple[int, str, sp.Expr]]:
        """Find attack rules that are not entailed by normal theory"""
        non_entailed = []
        
        print("Finding non-entailed attack rules...")
        for idx, orig_rule, parsed_rule in tqdm(attack_rules, desc="Checking entailment"):
            z3_rule = self.sympy_to_z3(parsed_rule)
            if z3_rule is None:
                continue
                
            result = self.z3_entails(normal_theory, z3_rule)
            if result != ProofResult.ENTAILMENT:
                non_entailed.append((idx, orig_rule, parsed_rule))
        
        return non_entailed
    
    def find_conflicting_normal_rules(self, attack_rule: sp.Expr, 
                                    normal_rules: List[Tuple[int, str, sp.Expr]]) -> List[int]:
        """Find normal rules that directly contradict the attack rule"""
        conflicting_rules = []
        z3_attack_rule = self.sympy_to_z3(attack_rule)
        
        if z3_attack_rule is None:
            return conflicting_rules
        
        print(f"Finding rules that contradict attack rule...")
        for idx, orig_rule, parsed_rule in tqdm(normal_rules, desc="Checking conflicts"):
            z3_normal_rule = self.sympy_to_z3(parsed_rule)
            if z3_normal_rule is None:
                continue
            
            # Check if normal rule contradicts attack rule
            if self.z3_contradicts(z3_normal_rule, z3_attack_rule):
                conflicting_rules.append(idx)
        
        return conflicting_rules
    
    def find_minimal_entailing_subset(self, attack_rule: sp.Expr, 
                                    normal_rules: List[Tuple[int, str, sp.Expr]]) -> Optional[List[int]]:
        """Find minimal subset of normal rules that entails the attack rule (if any)"""
        z3_attack_rule = self.sympy_to_z3(attack_rule)
        
        if z3_attack_rule is None:
            return None
        
        # First check if any single rule entails the attack rule
        for idx, orig_rule, parsed_rule in normal_rules:
            z3_normal_rule = self.sympy_to_z3(parsed_rule)
            if z3_normal_rule is None:
                continue
            
            result = self.z3_entails(z3_normal_rule, z3_attack_rule)
            if result == ProofResult.ENTAILMENT:
                return [idx]
        
        # If no single rule entails it, try combinations of increasing size
        # This is computationally expensive, so we limit to small combinations
        from itertools import combinations
        
        normal_rules_with_z3 = []
        for idx, orig_rule, parsed_rule in normal_rules:
            z3_normal_rule = self.sympy_to_z3(parsed_rule)
            if z3_normal_rule is not None:
                normal_rules_with_z3.append((idx, z3_normal_rule))
        
        # Try combinations of size 2, 3, etc. (limited to avoid exponential explosion)
        max_combination_size = min(5, len(normal_rules_with_z3))
        
        for size in range(2, max_combination_size + 1):
            for combo in combinations(normal_rules_with_z3, size):
                indices = [idx for idx, _ in combo]
                z3_rules = [rule for _, rule in combo]
                
                combined_theory = z3.And(*z3_rules)
                result = self.z3_entails(combined_theory, z3_attack_rule)
                
                if result == ProofResult.ENTAILMENT:
                    return indices
        
        return None
    
    def compare_rule_files_fixed(self, normal_file: str, attack_file: str, 
                               max_rules: Optional[int] = None) -> Dict:
        """Fixed comparison of two rule files"""
        print(f"Loading normal rules from: {normal_file}")
        normal_rules_str = self.load_rules_from_file(normal_file, max_rules)
        print(f"Loaded {len(normal_rules_str)} normal rules")
        
        print(f"Loading attack rules from: {attack_file}")
        attack_rules_str = self.load_rules_from_file(attack_file, max_rules)
        print(f"Loaded {len(attack_rules_str)} attack rules")
        
        # Parse all rules
        parsed_normal_rules = []
        parsed_attack_rules = []
        
        print("Parsing normal rules...")
        for i, rule in enumerate(tqdm(normal_rules_str, desc="Parsing normal")):
            parsed = self.parse_rule(rule, file_path=normal_file, line_number=i+1)
            if parsed is not None:
                parsed_normal_rules.append((i, rule, parsed))
        
        print("Parsing attack rules...")
        for i, rule in enumerate(tqdm(attack_rules_str, desc="Parsing attack")):
            parsed = self.parse_rule(rule, file_path=attack_file, line_number=i+1)
            if parsed is not None:
                parsed_attack_rules.append((i, rule, parsed))
        
        # Create normal theory (all normal rules must hold)
        normal_theory = self.create_normal_theory([rule for _, _, rule in parsed_normal_rules])
        
        # Analyze each attack rule
        analysis = {
        'entailed_attack_rules': [],
        'non_entailed_attack_rules': [],
        'contradicted_attack_rules': []
    }
        
        print("Analyzing attack rules...")
        for idx, orig_rule, parsed_rule in tqdm(parsed_attack_rules, desc="Analyzing attack rules"):
            print(f"\nProcessing attack rule #{idx}: {orig_rule}")  # <-- Add this line
            z3_rule = self.sympy_to_z3(parsed_rule)
            if z3_rule is None:
                continue
            
            # Check entailment
            entailment_result = self.z3_entails(normal_theory, z3_rule)
            
            if entailment_result == ProofResult.ENTAILMENT:
                # Attack rule is entailed by normal rules
                minimal_subset = self.find_minimal_entailing_subset(parsed_rule, parsed_normal_rules)
                analysis['entailed_attack_rules'].append({
                    'attack_rule_idx': idx,
                    'attack_rule': orig_rule,
                    'minimal_entailing_normal_rules': minimal_subset,
                    'entailing_normal_rules_text': [parsed_normal_rules[i][1] for i in minimal_subset] if minimal_subset else []
                })
            
            elif self.z3_contradicts(normal_theory, z3_rule):
                # Attack rule contradicts normal rules
                conflicting_rules = self.find_conflicting_normal_rules(parsed_rule, parsed_normal_rules)
                analysis['contradicted_attack_rules'].append({
                    'attack_rule_idx': idx,
                    'attack_rule': orig_rule,
                    'conflicting_normal_rules': conflicting_rules,
                    'conflicting_normal_rules_text': [parsed_normal_rules[i][1] for i in conflicting_rules if i < len(parsed_normal_rules)]
                })
            
            else:
                # Attack rule is neither entailed nor contradicted
                weak_rules = self.find_weak_normal_rules(parsed_rule, parsed_normal_rules)
                analysis['non_entailed_attack_rules'].append({
                    'attack_rule_idx': idx,
                    'attack_rule': orig_rule,
                    "weak_normal_rules": weak_rules,  # New field
                    "weak_normal_rules_text": [
                        parsed_normal_rules[i][1] for i in weak_rules
                    ] if weak_rules else [],
                })
        
        entailed_count = len(analysis['entailed_attack_rules'])
        contradicted_count = len(analysis['contradicted_attack_rules'])
        non_entailed_rules = analysis['non_entailed_attack_rules']
        non_entailed_count = len(non_entailed_rules)
        non_entailed_with_weak = sum(1 for rule in non_entailed_rules if rule['weak_normal_rules'])
        non_entailed_without_weak = non_entailed_count - non_entailed_with_weak

        analysis['stats'] = {
            'total_normal_rules': len(parsed_normal_rules),
            'total_attack_rules': len(parsed_attack_rules),
            'entailed_count': entailed_count,
            'contradicted_count': contradicted_count,
            'non_entailed_count': non_entailed_count,
            'non_entailed_with_weak_rules': non_entailed_with_weak,
            'non_entailed_without_weak_rules': non_entailed_without_weak
        }
        
        return analysis
    
    def save_analysis(self, analysis: Dict, output_file: str):
        """Save analysis results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    def print_summary(self, analysis: Dict):
        """Print summary of analysis results"""
        print("\n" + "="*50)
        print("FIXED ANALYSIS SUMMARY")
        print("="*50)
        stats = analysis['stats']
        print(f"Total normal rules: {stats['total_normal_rules']}")
        print(f"Total attack rules: {stats['total_attack_rules']}")
        print(f"Entailed attack rules: {stats['entailed_count']}")
        print(f"Contradicted attack rules: {stats['contradicted_count']}")
        print(f"Non-entailed attack rules: {stats['non_entailed_count']}")
        print(f"  ├── With weak normal rules: {stats['non_entailed_with_weak_rules']}")
        print(f"  └── Without weak normal rules: {stats['non_entailed_without_weak_rules']}")
        
        if analysis['entailed_attack_rules']:
            print(f"\nExample entailed attack rule:")
            rule = analysis['entailed_attack_rules'][0]
            print(f"  Attack rule[{rule['attack_rule_idx']}]: {rule['attack_rule'][:100]}...")
            print(f"  Entailed by normal rules: {rule['minimal_entailing_normal_rules']}")
        
        if analysis['contradicted_attack_rules']:
            print(f"\nExample contradicted attack rule:")
            rule = analysis['contradicted_attack_rules'][0]
            print(f"  Attack rule[{rule['attack_rule_idx']}]: {rule['attack_rule'][:100]}...")
            print(f"  Conflicts with normal rules: {rule['conflicting_normal_rules']}")
        
        if analysis['non_entailed_attack_rules']:
            print(f"\nExample independent attack rule:")
            rule = analysis['non_entailed_attack_rules'][0]
            print(f"  Attack rule[{rule['attack_rule_idx']}]: {rule['attack_rule'][:100]}...")

def main():
    """Main function to run the fixed comparison"""
    parser = argparse.ArgumentParser(
        description='Compare two rule files with fixed logical analysis using Z3 SMT solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  # Compare files with fixed logic
  python fixed_rule_comparator.py normal.json attack.json
  
  # Compare first 100 rules
  python fixed_rule_comparator.py normal.json attack.json -n 100
        '''
    )
    
    parser.add_argument('normal_file', help='Path to normal rules JSON file')
    parser.add_argument('attack_file', help='Path to attack rules JSON file')
    parser.add_argument('--max-rules', '-n', type=int, default=None,
                        help='Maximum number of rules to process from each file (default: all)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.normal_file).exists():
        print(f"Error: Normal file '{args.normal_file}' not found")
        return
    
    if not Path(args.attack_file).exists():
        print(f"Error: Attack file '{args.attack_file}' not found")
        return
    
    # Generate output filename
    normal_stem = Path(args.normal_file).stem
    attack_stem = Path(args.attack_file).stem
    rules_suffix = f"{args.max_rules}" if args.max_rules else "full"
    output_file = f"{normal_stem}_vs_{attack_stem}_fixed_n{rules_suffix}.json"
    
    print("Starting fixed rule comparison analysis...")
    start_time = time.time()
    
    # Run comparison
    comparator = FixedRuleComparator()
    analysis = comparator.compare_rule_files_fixed(args.normal_file, args.attack_file, args.max_rules)
    
    end_time = time.time()
    print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds")
    
    if analysis is not None:
        comparator.save_analysis(analysis, output_file)
        comparator.print_summary(analysis)

if __name__ == "__main__":
    main()