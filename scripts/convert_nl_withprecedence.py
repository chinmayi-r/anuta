import re
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Set

class LogicToNaturalLanguageConverter:
    def __init__(self):
        # Field mappings for better readability
        self.field_mappings = {
            'DstIpAddr': 'destination IP address',
            'SrcIpAddr': 'source IP address',
            'DstPt': 'destination port',
            'SrcPt': 'source port',
            'Proto': 'protocol',
            'Flags': 'flags',
            'Bytes': 'bytes',
            'Duration': 'duration',
            'Packets': 'packets'
        }
        
        # Value mappings for better readability with proper articles and grammar
        self.value_mappings = {
            'dns': 'a DNS server',
            'private_p2p': 'a private peer-to-peer address',
            'private_broadcast': 'a private broadcast address',
            'public_p2p': 'a public peer-to-peer address',
            'hasflags': 'set',
            'noflags': 'not set',
            'uninterested': 'dynamic',
            'any': 'any address',
            'ICMP': 'ICMP',
            'TCP': 'TCP',
            'UDP': 'UDP',
            'IGMP': 'IGMP'
        }
        
        # Special handling for certain field-value combinations
        self.special_phrases = {
            ('flags', 'hasflags'): 'flags are set',
            ('flags', 'noflags'): 'flags are not set',
            ('flags', 'uninterested'): 'flags can be any configuration'
        }
        
        # Operator mappings for natural language
        self.operator_mappings = {
            '=': 'is',
            '≤': 'is less than or equal to',
            '≥': 'is greater than or equal to',
            '<': 'is less than',
            '>': 'is greater than',
            '≠': 'is not equal to'
        }

    def tokenize(self, statement: str) -> List[str]:
        """Tokenize the logical statement into components."""
        # Remove extra whitespace and normalize
        statement = re.sub(r'\s+', ' ', statement.strip())
        
        # Replace all variants of 'v' with '∨' (including standalone and parenthesized cases)
        statement = re.sub(r'(?<!\w)v(?!\w)', '∨', statement)
        
        # Split by operators while keeping them (including comparison operators)
        tokens = re.findall(
            r'[()]|∧|∨|⇒|≤|≥|<|>|=|≠|[^()∧∨⇒≤≥<>=≠\s]+', 
            statement
        )
        return [token.strip() for token in tokens if token.strip()]

    def preprocess_statement(self, statement: str) -> str:
        """Enhanced preprocessing to handle ungrouped OR conditions"""
        # Normalize whitespace first
        statement = re.sub(r'\s+', ' ', statement.strip())
        
        # Convert all 'v' to OR symbols, handling various formats
        statement = re.sub(r'(?<!\w)v(?!\w)', '∨', statement)
        
        # Handle cases where OR conditions aren't properly grouped
        # Pattern: (A ∧ B ∨ C ∨ D) -> (A ∧ (B ∨ C ∨ D))
        statement = re.sub(
            r'\(([^()]+)\s*∧\s*([^()]+)\s*∨\s*([^()]+)\)',
            r'(\1 ∧ (\2 ∨ \3))',
            statement
        )
        
        # Ensure spaces around operators
        statement = re.sub(r'([()∧∨⇒≤≥<>=≠])', r' \1 ', statement)
        
        # Remove duplicate spaces
        statement = re.sub(r'\s+', ' ', statement).strip()
        
        return statement
    
    def parse_predicate(self, tokens: List[str], start: int) -> Tuple[Dict, int]:
        """Parse a single predicate (field operator value)."""
        if start + 2 >= len(tokens):
            raise ValueError(
                f"Invalid predicate at position {start}: "
                f"Not enough tokens remaining for a complete predicate. "
                f"Remaining tokens: {tokens[start:]}"
            )
        
        field = tokens[start]
        operator = tokens[start + 1]
        value = tokens[start + 2]
        
        if operator not in ['=', '≤', '≥', '<', '>', '≠']:
            raise ValueError(
                f"Invalid predicate at position {start}: "
                f"Expected comparison operator but got '{operator}'. "
                f"Context: {tokens[max(0, start-2):start+3]}"
            )
        
        return {
            'type': 'predicate',
            'field': field,
            'operator': operator,
            'value': value
        }, start + 3

    def parse_term(self, tokens: List[str], start: int) -> Tuple[Dict, int]:
        """Parse a term (can be a predicate or parenthesized expression)."""
        if start >= len(tokens):
            raise ValueError("Unexpected end of input while parsing term")
            
        if tokens[start] == '(':
            # Parse parenthesized expression - use same precedence parsing within parens
            expr, end = self.parse_expression(tokens, start + 1)
            if end >= len(tokens) or tokens[end] != ')':
                raise ValueError(
                    "Missing closing parenthesis. "
                    f"Current position: {end}, tokens: {tokens[end:end+5]}"
                )
            return expr, end + 1
        else:
            # Parse predicate
            return self.parse_predicate(tokens, start)
    
    def parse_expression(self, tokens: List[str], start: int) -> Tuple[Dict, int]:
        """Parse a logical expression with operators, handling precedence properly."""
        # Parse implication (⇒) - lowest precedence
        left, pos = self.parse_or_and_expression(tokens, start)
        
        if pos < len(tokens) and tokens[pos] == '⇒':
            right, pos = self.parse_expression(tokens, pos + 1)
            left = {'type': 'binary_op', 'operator': '⇒', 'left': left, 'right': right}
        
        return left, pos
    
    def parse_or_and_expression(self, tokens: List[str], start: int) -> Tuple[Dict, int]:
        """Parse OR and AND expressions with equal precedence (left-to-right)."""
        left, pos = self.parse_term(tokens, start)
        
        while pos < len(tokens) and tokens[pos] in ['∨', '∧']:
            op = tokens[pos]
            right, pos = self.parse_term(tokens, pos + 1)
            left = {'type': 'binary_op', 'operator': op, 'left': left, 'right': right}
        
        return left, pos
    
    def _walk_tree(self, node):
        """Helper to walk the AST recursively"""
        yield node
        if node['type'] == 'binary_op':
            yield from self._walk_tree(node['left'])
            yield from self._walk_tree(node['right'])

    def validate_structure(self, parsed):
        """Check for common structural issues with minimal output"""
        if (parsed['type'] == 'binary_op' and 
            parsed['operator'] == '⇒' and
            parsed['left']['type'] == 'binary_op' and
            parsed['left']['operator'] == '∨'):
            
            print("VALIDATION: Detected OR at top level of condition")
            print("Structure:", json.dumps(parsed['left'], indent=2))
            print("This may indicate missing parentheses around OR conditions")

    def parse_statement(self, statement: str) -> Dict:
        """Parse with silent validation"""
        tokens = self.tokenize(statement)
        if not tokens:
            raise ValueError("Empty statement")
        
        expr, _ = self.parse_expression(tokens, 0)
        return expr

    def debug_line(self, line_num, statement):
        """Minimal debug output for specific lines"""
        print(f"\nDEBUG LINE {line_num}:")
        preprocessed = self.preprocess_statement(statement)
        parsed = self.parse_statement(preprocessed)
        
        # Only show validation if issues found
        if (parsed['type'] == 'binary_op' and 
            parsed['operator'] == '⇒' and
            parsed['left']['type'] == 'binary_op' and
            parsed['left']['operator'] == '∨'):
            
            print("INPUT:", statement)
            self.validate_structure(parsed)
            print("CONVERTED:", self.convert_to_natural_language(statement))
    
    def format_value_set(self, values: List[str], field: str) -> str:
        """Format a set of values for natural language output."""
        formatted_values = []
        for value in values:
            if value == 'uninterested':
                if 'port' in field.lower():
                    formatted_values.append('dynamic')
                else:
                    formatted_values.append('dynamic')
            else:
                # For numeric values, keep them as numbers
                if value.isdigit():
                    formatted_values.append(value)
                else:
                    formatted_values.append(self.value_mappings.get(value, str(value)))
        
        if len(formatted_values) == 1:
            return formatted_values[0]
        elif len(formatted_values) == 2:
            return f"{{{formatted_values[0]} or {formatted_values[1]}}}"
        else:
            return "{" + ", ".join(formatted_values[:-1]) + f", or {formatted_values[-1]}" + "}"
    
    def predicate_to_text(self, predicate: Dict) -> str:
        """Convert a predicate to natural language."""
        field = predicate['field']
        operator = predicate['operator']
        value = predicate['value']
        
        field_text = self.field_mappings.get(field, field)
        
        # Check for special phrase combinations
        field_lower = field_text.lower()
        special_key = (field_lower, value)
        if special_key in self.special_phrases:
            return self.special_phrases[special_key]
        
        # Handle "uninterested" values specially
        if value == 'uninterested':
            if 'port' in field_text.lower():
                return f"the {field_text} can be dynamic"
            else:
                return f"the {field_text} can be dynamic"
        
        # Get value text and operator text
        value_text = self.value_mappings.get(value, value)
        operator_text = self.operator_mappings.get(operator, operator)
        
        return f"the {field_text} {operator_text} {value_text}"
    
    def collect_or_predicates(self, expr: Dict) -> List[Dict]:
        """Recursively collect all predicates connected by OR operators."""
        if expr['type'] == 'predicate':
            return [expr]
        elif expr['type'] == 'binary_op' and expr['operator'] == '∨':
            left_preds = self.collect_or_predicates(expr['left'])
            right_preds = self.collect_or_predicates(expr['right'])
            return left_preds + right_preds
        else:
            return []
    
    def can_group_or_chain(self, expr: Dict) -> Tuple[bool, str, str, List[str]]:
        """Check if expression is a chain of OR predicates for the same field with same operator."""
        predicates = self.collect_or_predicates(expr)
        
        if len(predicates) > 1:
            # Check if all predicates are for same field with same operator
            first_field = predicates[0]['field']
            first_op = predicates[0]['operator']
            
            if all(p['field'] == first_field and p['operator'] == first_op for p in predicates):
                values = [p['value'] for p in predicates]
                return True, first_field, first_op, values
        
        return False, "", "", []

    def collect_and_components(self, expr: Dict) -> List[Dict]:
        """Collect all components connected by AND operators at the top level."""
        components = []
        
        def collect_components(node):
            if node['type'] == 'binary_op' and node['operator'] == '∧':
                collect_components(node['left'])
                collect_components(node['right'])
            else:
                components.append(node)
        
        collect_components(expr)
        return components

    def expression_to_text(self, expr: Dict, parent_op: str = None) -> str:
        if expr['type'] == 'predicate':
            return self.predicate_to_text(expr)
        elif expr['type'] == 'binary_op':
            if expr['operator'] == '⇒':
                left_text = self.expression_to_text(expr['left'])
                right_text = self.expression_to_text(expr['right'])
                return f"if {left_text}, then {right_text}"
            elif expr['operator'] == '∧':
                # Process left and right separately
                left_text = self.expression_to_text(expr['left'], expr['operator'])
                right_text = self.expression_to_text(expr['right'], expr['operator'])
                
                # Special handling for OR chains on the right side
                if (expr['right']['type'] == 'binary_op' and 
                    expr['right']['operator'] == '∨'):
                    can_group, field, op, values = self.can_group_or_chain(expr['right'])
                    if can_group and op == '=':
                        field_text = self.field_mappings.get(field, field)
                        value_set = self.format_value_set(values, field)
                        return f"{left_text} and the {field_text} is in {value_set}"
                
                return f"{left_text} and {right_text}"
            elif expr['operator'] == '∨':
                can_group, field, op, values = self.can_group_or_chain(expr)
                if can_group and op == '=':
                    field_text = self.field_mappings.get(field, field)
                    value_set = self.format_value_set(values, field)
                    return f"the {field_text} is in {value_set}"
                left_text = self.expression_to_text(expr['left'], expr['operator'])
                right_text = self.expression_to_text(expr['right'], expr['operator'])
                return f"{left_text} or {right_text}"
        return str(expr)
    
    def convert_to_natural_language(self, statement: str) -> str:
        """Convert a logical statement to natural language."""
        try:
            # Preprocess the statement first
            statement = self.preprocess_statement(statement)
            
            parsed = self.parse_statement(statement)
            return self.expression_to_text(parsed).capitalize() + "."
        except Exception as e:
            return f"Error parsing statement: {e}\nOriginal statement: {statement}"

    def debug_test(self):
        """Debug specific test case with cleaner output"""
        test = "((Bytes > 982280) ∧ ((SrcPt = 0) ∨ (SrcPt = 137) ∨ (SrcPt = 138) ∨ (SrcPt = uninterested))) ⇒ ((DstIpAddr = private_p2p))"
        
        print("\n" + "="*80)
        print("DEBUG TEST CASE")
        print("="*80)
        
        print("\n[1] Original statement:")
        print(f"'{test}'")
        
        print("\n[2] After preprocessing:")
        preprocessed = self.preprocess_statement(test)
        print(f"'{preprocessed}'")
        
        print("\n[3] Parsed structure:")
        parsed = self.parse_statement(preprocessed)
        print(json.dumps(parsed, indent=2))
        
        print("\n[4] Left side analysis:")
        left_side = parsed['left']
        components = self.collect_and_components(left_side)
        for i, comp in enumerate(components):
            can_group, field, op, values = self.can_group_or_chain(comp)
            print(f"Component {i+1}:")
            print(f"- Type: {comp['type']}")
            if comp['type'] == 'predicate':
                print(f"- Predicate: {self.predicate_to_text(comp)}")
            else:
                print(f"- Operator: {comp['operator']}")
                print(f"- Can group as OR chain: {can_group}")
                if can_group:
                    print(f"- Field: {field}")
                    print(f"- Values: {values}")
        
        print("\n[5] Final conversion:")
        result = self.convert_to_natural_language(test)
        print(result)

    def test_converter(self):
        """Run test cases with cleaner output"""
        test_cases = [
            "((Bytes > 982280) ∧ ((SrcPt = 0) ∨ (SrcPt = 137) ∨ (SrcPt = 138) ∨ (SrcPt = uninterested))) ⇒ ((DstIpAddr = private_p2p))",
            "((Bytes > 982280) ∧ (SrcIpAddr = private_p2p) ∨ (SrcIpAddr = any)) ⇒ ((DstIpAddr = private_p2p))",
            "(Bytes > 1000) ∧ (SrcPt = 80) ∧ (DstPt = 443)",
            "(SrcIpAddr = dns) ∨ (SrcIpAddr = private_p2p) ∨ (DstIpAddr = any)"
        ]
        
        print("\n" + "="*80)
        print("TEST SUITE")
        print("="*80)
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n[TEST CASE {i}]")
            print("-"*40)
            print("Input:\n", test)
            
            try:
                preprocessed = self.preprocess_statement(test)
                parsed = self.parse_statement(preprocessed)
                result = self.convert_to_natural_language(test)
                
                print("\nConversion steps:")
                print("1. Preprocessed:", preprocessed)
                print("2. Parsed structure:")
                print(json.dumps(parsed, indent=2))
                print("3. Final output:", result)
            except Exception as e:
                print(f"ERROR in test case {i}: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    converter = LogicToNaturalLanguageConverter()
    
    # Create output filenames
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_natural_language.txt"
    error_file = f"{base_name}_errors.txt"
    
    # Statistics
    start_time = datetime.now()
    processed_count = 0
    error_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile, \
             open(error_file, 'w', encoding='utf-8') as errfile:

            print(f"\nStarting processing at {start_time}")
            print(f"Input: {input_file}")
            print(f"Output: {output_file}")
            print(f"Errors: {error_file}")

            for line_num, line in enumerate(infile, 1):
                statement = line.strip()
                
                if not statement:
                    outfile.write("\n")
                    continue
                
                try:
                    # Special debugging for line 49
                    if line_num == 49:
                        print("\n" + "="*80)
                        print(f"DEBUGGING LINE {line_num}")
                        print("="*80)
                        
                        print("\n[1] Raw input:")
                        print(repr(statement))
                        
                        preprocessed = converter.preprocess_statement(statement)
                        print("\n[2] After preprocessing:")
                        print(repr(preprocessed))
                        
                        parsed = converter.parse_statement(preprocessed)
                        print("\n[3] Parsed structure:")
                        print(json.dumps(parsed, indent=2))
                        
                        # Explicit validation for debug line
                        print("\n[4] Validation results:")
                        converter.validate_structure(parsed)  # <-- Explicit validation
                        
                        if parsed['type'] == 'binary_op' and parsed['operator'] == '⇒':
                            print("\n[5] Implication analysis:")
                            print("Left side (condition):")
                            print(json.dumps(parsed['left'], indent=2))
                            print("\nRight side (conclusion):")
                            print(json.dumps(parsed['right'], indent=2))
                    
                    # Normal processing
                    natural_language = converter.convert_to_natural_language(statement)
                    processed_count += 1
                    outfile.write(f"{natural_language}\n")

                except Exception as e:
                    error_count += 1
                    error_msg = f"Line {line_num}: {str(e)}"
                    errfile.write(f"Line {line_num}: {statement}\nError: {error_msg}\n\n")
                    outfile.write(f"ERROR: {error_msg}\n")
                    
                    if error_count <= 3:
                        print(f"\nFirst error example (line {line_num}):")
                        print("Input:", statement)
                        print("Error:", e)

            # Print summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*80)
            print("PROCESSING SUMMARY")
            print("="*80)
            print(f"Started:  {start_time}")
            print(f"Finished: {end_time}")
            print(f"Duration: {duration}")
            print(f"Lines processed: {processed_count}")
            print(f"Errors encountered: {error_count}")
            
            if error_count == 0:
                print("\nAll statements converted successfully!")
                if os.path.exists(error_file) and os.path.getsize(error_file) == 0:
                    os.remove(error_file)
            else:
                print(f"\nConversion completed with {error_count} errors")
                print(f"See {error_file} for details")

            print(f"\nOutput saved to: {output_file}")

    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    converter = LogicToNaturalLanguageConverter()
    
    # Run debug tests first
    print("\n" + "="*80)
    print("INITIAL DEBUGGING")
    print("="*80)
    converter.debug_test()
    converter.test_converter()
    
    # Then process the main file
    print("\n" + "="*80)
    print("PROCESSING INPUT FILE")
    print("="*80)
    main()