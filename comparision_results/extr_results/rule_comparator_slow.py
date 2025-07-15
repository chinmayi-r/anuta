import json
import sympy as sp
from sympy import symbols, And, Or, Not, Implies, Eq, Ne, simplify
from sympy.logic.boolalg import to_cnf, to_dnf
from sympy.parsing.sympy_parser import parse_expr
import re
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import time

class RuleFileComparator:
    def __init__(self):
        # Define your domain variables
        self.Bytes, self.Packets, self.Duration = symbols('Bytes Packets Duration')
        self.DstIpAddr, self.SrcIpAddr = symbols('DstIpAddr SrcIpAddr')
        self.DstPt, self.SrcPt, self.Proto, self.Flags = symbols('DstPt SrcPt Proto Flags')
        
    def load_rules_from_file(self, file_path, max_rules=None):
        """Load rules from a JSON file with optional limit"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different possible JSON structures
            if isinstance(data, list):
                rules = data
            elif isinstance(data, dict):
                # Look for common keys that might contain rules
                rules = []
                for key in ['rules', 'constraints', 'predicates', 'conditions']:
                    if key in data:
                        rules = data[key]
                        break
                else:
                    # If no standard key found, try to get all string values
                    for value in data.values():
                        if isinstance(value, list):
                            rules.extend(value)
                        elif isinstance(value, str):
                            rules.append(value)
            else:
                print(f"Unexpected JSON structure in {file_path}")
                return []
            
            # Limit the number of rules if specified
            if max_rules is not None and max_rules > 0:
                rules = rules[:max_rules]
                print(f"Limited to first {len(rules)} rules from {file_path}")
            
            return rules
                
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []
    
    def parse_rule(self, rule_str, file_path=None, line_number=None):
        """Parse a string rule into SymPy expression"""
        # Clean up the string
        rule_str = rule_str.strip().strip('"').strip("'")
        
        # Remove extra spaces
        rule_str = re.sub(r'\s+', ' ', rule_str)
        
        # Replace logical operators with SymPy equivalents
        rule_str = rule_str.replace('&', ' & ')
        rule_str = rule_str.replace('|', ' | ')
        
        # Handle Ne (not equal) function
        rule_str = re.sub(r'Ne\(([^,]+),\s*([^)]+)\)', r'Ne(\1, \2)', rule_str)
        
        # Handle Eq function
        rule_str = re.sub(r'Eq\(([^,]+),\s*([^)]+)\)', r'Eq(\1, \2)', rule_str)
        
        # Handle Implies function - be more robust with nested parentheses
        def replace_implies(match):
            content = match.group(1)
            # Find the comma that separates condition from conclusion
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
                return match.group(0)  # Return original if we can't parse
            
            condition = content[:comma_pos].strip()
            conclusion = content[comma_pos+1:].strip()
            return f'Implies({condition}, {conclusion})'
        
        # Use a more robust regex for Implies that handles nested parentheses
        while 'Implies(' in rule_str:
            old_rule = rule_str
            rule_str = re.sub(r'Implies\(([^)]+(?:\([^)]*\)[^)]*)*)\)', replace_implies, rule_str, count=1)
            if rule_str == old_rule:  # No change, avoid infinite loop
                break
        
        try:
            # Parse the expression with local dict to avoid conflicts
            local_dict = {
                'Bytes': self.Bytes,
                'Packets': self.Packets,
                'Duration': self.Duration,
                'DstIpAddr': self.DstIpAddr,
                'SrcIpAddr': self.SrcIpAddr,
                'DstPt': self.DstPt,
                'SrcPt': self.SrcPt,
                'Proto': self.Proto,
                'Flags': self.Flags,
                'Implies': Implies,
                'Eq': Eq,
                'Ne': Ne,
                'And': And,
                'Or': Or,
                'Not': Not
            }
            
            expr = parse_expr(rule_str, local_dict=local_dict, transformations='all')
            return expr
        except Exception as e:
            print(f"    Error parsing rule in {file_path or 'unknown file'}, line {line_number or 'unknown'}")
            print(f"    Rule: {rule_str}")
            print(f"    Error: {e}")
            return None
    
    def normalize_rule(self, rule):
        """Normalize a rule to canonical form"""
        try:
            # Convert implications to equivalent form: Implies(A, B) -> Or(Not(A), B)
            rule = rule.replace(Implies, lambda x, y: Or(Not(x), y))
            
            # Simplify the expression
            rule = simplify(rule)
            
            return rule
        except Exception as e:
            print(f"Error normalizing rule: {rule}, Error: {e}")
            return rule
    
    def rules_equivalent(self, rule1, rule2):
        """Check if two rules are logically equivalent"""
        try:
            # Quick string comparison first
            if str(rule1) == str(rule2):
                return True
                
            # Normalize both rules
            norm1 = self.normalize_rule(rule1)
            norm2 = self.normalize_rule(rule2)
            
            # Quick comparison after normalization
            if str(norm1) == str(norm2):
                return True
            
            # Check if they're equivalent by checking if (A XOR B) is unsatisfiable
            # XOR is true when rules differ
            xor_expr = Or(And(norm1, Not(norm2)), And(Not(norm1), norm2))
            simplified_xor = simplify(xor_expr)
            
            # If the XOR simplifies to False, they're equivalent
            return simplified_xor == False
        except Exception as e:
            # If comparison fails, assume not equivalent
            return False
    
    def subsumes(self, rule1, rule2):
        """Check if rule1 subsumes rule2 (rule1 implies rule2)"""
        try:
            # Quick string comparison first
            if str(rule1) == str(rule2):
                return True
                
            norm1 = self.normalize_rule(rule1)
            norm2 = self.normalize_rule(rule2)
            
            # Quick comparison after normalization
            if str(norm1) == str(norm2):
                return True
            
            # rule1 subsumes rule2 if (rule1 & ~rule2) is unsatisfiable
            # i.e., whenever rule1 is true, rule2 must also be true
            implication = And(norm1, Not(norm2))
            simplified = simplify(implication)
            
            return simplified == False
        except Exception as e:
            # If comparison fails, assume no subsumption
            return False
    
    def compare_rule_files(self, normal_file, attack_file, max_rules=None):
        """Compare two rule files and identify differences"""
        print(f"Loading normal rules from: {normal_file}")
        normal_rules = self.load_rules_from_file(normal_file, max_rules)
        print(f"Loaded {len(normal_rules)} normal rules")
        
        print(f"Loading attack rules from: {attack_file}")
        attack_rules = self.load_rules_from_file(attack_file, max_rules)
        print(f"Loaded {len(attack_rules)} attack rules")
        
        # Parse all rules
        parsed_normal_rules = []
        parsed_attack_rules = []
        
        print("\nParsing normal rules...")
        for i, rule in enumerate(tqdm(normal_rules, desc="Parsing normal")):
            parsed = self.parse_rule(rule, file_path=normal_file, line_number=i+1)
            if parsed is not None:
                parsed_normal_rules.append((i, rule, parsed))
            else:
                print(f"Failed to parse normal rule {i+1}: {rule}")
        
        print("Parsing attack rules...")
        for i, rule in enumerate(tqdm(attack_rules, desc="Parsing attack")):
            parsed = self.parse_rule(rule, file_path=attack_file, line_number=i+1)
            if parsed is not None:
                parsed_attack_rules.append((i, rule, parsed))
            else:
                print(f"Failed to parse attack rule {i+1}: {rule}")
        
        # Find differences
        analysis = {
            'rules_only_in_normal': [],
            'rules_only_in_attack': [],
            'weakened_rules': [],  # Rules that became less restrictive in attack
            'strengthened_rules': [],  # Rules that became more restrictive in attack
            'equivalent_rules': []
        }
        
        total_comparisons = len(parsed_normal_rules) * len(parsed_attack_rules)
        print(f"\nAnalyzing differences...")
        print(f"Total comparisons to perform: {total_comparisons:,}")
        
        if total_comparisons > 100000:
            print("WARNING: This might take a very long time. Consider using --max-rules to limit analysis")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Analysis cancelled.")
                return None
        
        start_time = time.time()
        
        # Create a set to track which attack rules have been matched
        matched_attack_rules = set()
        
        # Check each normal rule with progress bar
        print("Checking normal rules against attack rules...")
        for i, orig_normal, parsed_normal in tqdm(parsed_normal_rules, desc="Normal rules"):
            found_equivalent = False
            found_subsumer = False
            found_subsumed = False
            
            for j, orig_attack, parsed_attack_rule in parsed_attack_rules:
                # Skip if this attack rule already matched with another normal rule
                if j in matched_attack_rules:
                    continue
                    
                if self.rules_equivalent(parsed_normal, parsed_attack_rule):
                    analysis['equivalent_rules'].append({
                        'normal_idx': i,
                        'attack_idx': j,
                        'normal_rule': orig_normal,
                        'attack_rule': orig_attack
                    })
                    found_equivalent = True
                    matched_attack_rules.add(j)  # Mark this attack rule as matched
                    break  # Early termination - found equivalent rule
                elif self.subsumes(parsed_normal, parsed_attack_rule):
                    # Normal rule subsumes attack rule (attack is more restrictive)
                    analysis['strengthened_rules'].append({
                        'normal_idx': i,
                        'attack_idx': j,
                        'normal_rule': orig_normal,
                        'attack_rule': orig_attack,
                        'description': 'Attack rule is more restrictive than normal rule'
                    })
                    found_subsumed = True
                    matched_attack_rules.add(j)  # Mark this attack rule as matched
                    break  # Early termination - found subsumption relationship
                elif self.subsumes(parsed_attack_rule, parsed_normal):
                    # Attack rule subsumes normal rule (attack is less restrictive)
                    analysis['weakened_rules'].append({
                        'normal_idx': i,
                        'attack_idx': j,
                        'normal_rule': orig_normal,
                        'attack_rule': orig_attack,
                        'description': 'Attack rule is less restrictive than normal rule'
                    })
                    found_subsumer = True
                    matched_attack_rules.add(j)  # Mark this attack rule as matched
                    break  # Early termination - found subsumption relationship
            
            if not (found_equivalent or found_subsumer or found_subsumed):
                analysis['rules_only_in_normal'].append({
                    'idx': i,
                    'rule': orig_normal,
                    'parsed': parsed_normal
                })
        
        # Check remaining unmatched attack rules
        print("Checking remaining attack rules...")
        unmatched_attack_rules = [(j, orig_attack, parsed_attack_rule) 
                                 for j, orig_attack, parsed_attack_rule in parsed_attack_rules 
                                 if j not in matched_attack_rules]
        
        for j, orig_attack, parsed_attack_rule in tqdm(unmatched_attack_rules, desc="Unmatched attack rules"):
            analysis['rules_only_in_attack'].append({
                'idx': j,
                'rule': orig_attack,
                'parsed': parsed_attack_rule
            })
        
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
        
        return analysis
    
    def save_analysis(self, analysis, output_file):
        """Save analysis results to JSON file in compact format"""
        # Create compact representation with indices and rules
        compact_analysis = {
            'summary': {
                'rules_only_in_normal': len(analysis['rules_only_in_normal']),
                'rules_only_in_attack': len(analysis['rules_only_in_attack']),
                'weakened_rules': len(analysis['weakened_rules']),
                'strengthened_rules': len(analysis['strengthened_rules']),
                'equivalent_rules': len(analysis['equivalent_rules'])
            },
            'differences': {
                'only_normal': {
                    'indices': [r['idx'] for r in analysis['rules_only_in_normal']],
                    'rules': [r['rule'] for r in analysis['rules_only_in_normal']]
                },
                'only_attack': {
                    'indices': [r['idx'] for r in analysis['rules_only_in_attack']],
                    'rules': [r['rule'] for r in analysis['rules_only_in_attack']]
                },
                'weakened': {
                    'indices': [[r['normal_idx'], r['attack_idx']] for r in analysis['weakened_rules']],
                    'rules': [[r['normal_rule'], r['attack_rule']] for r in analysis['weakened_rules']]
                },
                'strengthened': {
                    'indices': [[r['normal_idx'], r['attack_idx']] for r in analysis['strengthened_rules']],
                    'rules': [[r['normal_rule'], r['attack_rule']] for r in analysis['strengthened_rules']]
                },
                'equivalent': {
                    'indices': [[r['normal_idx'], r['attack_idx']] for r in analysis['equivalent_rules']],
                    'rules': [[r['normal_rule'], r['attack_rule']] for r in analysis['equivalent_rules']]
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(compact_analysis, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    def print_summary(self, analysis):
        """Print brief summary of analysis results"""
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Rules only in normal: {len(analysis['rules_only_in_normal'])}")
        print(f"Rules only in attack: {len(analysis['rules_only_in_attack'])}")
        print(f"Weakened rules: {len(analysis['weakened_rules'])}")
        print(f"Strengthened rules: {len(analysis['strengthened_rules'])}")
        print(f"Equivalent rules: {len(analysis['equivalent_rules'])}")
        
        # Show a few examples if they exist
        if analysis['weakened_rules']:
            print(f"\nExample weakened rule:")
            r = analysis['weakened_rules'][0]
            print(f"  Normal[{r['normal_idx']}] -> Attack[{r['attack_idx']}]")
            
        if analysis['strengthened_rules']:
            print(f"\nExample strengthened rule:")
            r = analysis['strengthened_rules'][0]
            print(f"  Normal[{r['normal_idx']}] -> Attack[{r['attack_idx']}]")

def main():
    """Main function to run the comparison"""
    parser = argparse.ArgumentParser(
        description='Compare two rule files and identify logical differences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  # Compare full files (very long for large files)
  python rule_comparator.py normal.json attack.json
  
  # Compare first 50 rules
  python rule_comparator.py normal.json attack.json -n 50
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
    output_file = f"{normal_stem}_vs_{attack_stem}_n{rules_suffix}.json"
    
    
    # Show recommended usage for large files
    if args.max_rules is None:
        print("Note: For faster analysis, use --max-rules to limit the number of rules processed")
        print()
    
    # Run comparison
    print("Starting rule comparison analysis...")
    comparator = RuleFileComparator()
    analysis = comparator.compare_rule_files(args.normal_file, args.attack_file, args.max_rules)
    
    if analysis is not None:
        comparator.save_analysis(analysis, output_file)
        comparator.print_summary(analysis)

if __name__ == "__main__":
    main()