import re
from typing import List, Dict, Tuple

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
            'Bytes': 'bytes'
        }
        
        # Value mappings for better readability with proper articles and grammar
        self.value_mappings = {
            'dns': 'a DNS server',
            'private_p2p': 'a private peer-to-peer address',
            'private_broadcast': 'a private broadcast address',
            'public_p2p': 'a public peer-to-peer address',
            'hasflags': 'set',
            'noflags': 'not set',
            'uninterested': 'any dynamic port',  # Better translation for ports
            'ICMP': 'ICMP',
            'TCP': 'TCP',
            'UDP': 'UDP'
        }
        
        # Context-specific mappings for uninterested values
        self.context_specific_mappings = {
            'port': {  # For SrcPt, DstPt fields
                'uninterested': 'any dynamic port'
            },
            'ip': {  # For IP address fields
                'uninterested': 'any IP address'
            },
            'protocol': {  # For Proto field
                'uninterested': 'any protocol'
            },
            'default': {  # Fallback for other fields
                'uninterested': 'any value'
            }
        }
        
        # Special handling for certain field-value combinations
        self.special_phrases = {
            ('flags', 'hasflags'): 'flags are set',
            ('flags', 'noflags'): 'flags are not set',
            ('flags', 'uninterested'): 'flags can be any value'
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
    
    def get_field_context(self, field: str) -> str:
        """Determine the context type for a field to use appropriate mappings."""
        field_lower = field.lower()
        if 'pt' in field_lower or 'port' in field_lower:
            return 'port'
        elif 'addr' in field_lower or 'ip' in field_lower:
            return 'ip'
        elif 'proto' in field_lower:
            return 'protocol'
        else:
            return 'default'
    
    def get_contextual_value(self, field: str, value: str) -> str:
        """Get the contextually appropriate translation for a value."""
        if value == 'uninterested':
            context = self.get_field_context(field)
            return self.context_specific_mappings[context].get(value, 'any value')
        return self.value_mappings.get(value, value)
    
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
        """Preprocess the statement to handle common formatting issues."""
        # Normalize whitespace
        statement = re.sub(r'\s+', ' ', statement.strip())
        
        # Fix common operator variations
        statement = statement.replace(' v ', ' ∨ ')
        statement = statement.replace('->', '⇒')
        statement = statement.replace('=>', '⇒')
        statement = statement.replace('&', '∧')
        statement = statement.replace('|', '∨')
        
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
            # Parse parenthesized expression
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
        """Parse a logical expression with operators, handling precedence."""
        # Parse implication (⇒) - lowest precedence
        left, pos = self.parse_or_expression(tokens, start)
        
        if pos < len(tokens) and tokens[pos] == '⇒':
            right, pos = self.parse_expression(tokens, pos + 1)
            left = {'type': 'binary_op', 'operator': '⇒', 'left': left, 'right': right}
        
        return left, pos
    
    def parse_or_expression(self, tokens: List[str], start: int) -> Tuple[Dict, int]:
        """Parse OR expressions (∨) - middle precedence."""
        left, pos = self.parse_and_expression(tokens, start)
        
        while pos < len(tokens) and tokens[pos] == '∨':
            right, pos = self.parse_and_expression(tokens, pos + 1)
            left = {'type': 'binary_op', 'operator': '∨', 'left': left, 'right': right}
        
        return left, pos
    
    def parse_and_expression(self, tokens: List[str], start: int) -> Tuple[Dict, int]:
        """Parse AND expressions (∧) - highest precedence."""
        left, pos = self.parse_term(tokens, start)
        
        while pos < len(tokens) and tokens[pos] == '∧':
            right, pos = self.parse_term(tokens, pos + 1)
            left = {'type': 'binary_op', 'operator': '∧', 'left': left, 'right': right}
        
        return left, pos
    
    def parse_statement(self, statement: str) -> Dict:
        """Parse a complete logical statement."""
        tokens = self.tokenize(statement)
        if not tokens:
            raise ValueError("Empty statement")
        
        expr, _ = self.parse_expression(tokens, 0)
        return expr
    
    def predicate_to_text(self, predicate: Dict) -> str:
        """Convert a predicate to natural language."""
        field = predicate['field']
        operator = predicate['operator']
        value = predicate['value']
        
        field_text = self.field_mappings.get(field, field)
        
        # Check for special phrase combinations first
        field_lower = field_text.lower()
        special_key = (field_lower, value)
        if special_key in self.special_phrases:
            return self.special_phrases[special_key]
        
        # Get contextually appropriate value text
        value_text = self.get_contextual_value(field, value)
        operator_text = self.operator_mappings.get(operator, operator)
        
        # Handle "uninterested" values with better context
        if value == 'uninterested':
            if operator == '=':
                return f"the {field_text} is {value_text}"
            else:
                return f"the {field_text} {operator_text} {value_text}"
        
        # Use "the" for most fields
        return f"the {field_text} {operator_text} {value_text}"
    
    def expression_to_text(self, expr: Dict) -> str:
        """Convert a logical expression to natural language."""
        if expr['type'] == 'predicate':
            return self.predicate_to_text(expr)
        elif expr['type'] == 'binary_op':
            left_text = self.expression_to_text(expr['left'])
            right_text = self.expression_to_text(expr['right'])
            
            if expr['operator'] == '∧':
                return f"{left_text} and {right_text}"
            elif expr['operator'] == '∨':
                return f"{left_text} or {right_text}"
            elif expr['operator'] == '⇒':
                return f"if {left_text}, then {right_text}"
        
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

def main():
    import sys
    import os
    
    # Check if input file is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.txt>")
        print("The input file should contain one logical statement per line.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    converter = LogicToNaturalLanguageConverter()
    
    # Create output filenames
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_natural_language.txt"
    error_file = f"{base_name}_errors.txt"
    
    error_count = 0
    total_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile, \
             open(error_file, 'w', encoding='utf-8') as errfile:
            
            # Read all lines to maintain line count
            all_lines = infile.readlines()
            total_lines = len(all_lines)
            
            for line_num, line in enumerate(all_lines, 1):
                statement = line.strip()
                
                # Handle empty lines - write empty line to output
                if not statement:
                    outfile.write("\n")
                    continue
                
                # Convert to natural language
                try:
                    natural_language = converter.convert_to_natural_language(statement)
                    
                    # Check if it's an error message
                    if natural_language.startswith("Error parsing statement:"):
                        error_count += 1
                        print(f"Line {line_num}: {natural_language}")
                        errfile.write(f"Line {line_num}: {statement}\n")
                        errfile.write(f"Error: {natural_language}\n\n")
                        # Write error indicator to output file
                        outfile.write(f"ERROR: {natural_language}\n")
                    else:
                        # Write successful conversion to output file
                        outfile.write(f"{natural_language}\n")
                        
                except Exception as e:
                    error_count += 1
                    error_msg = f"Unexpected error: {str(e)}"
                    print(f"Line {line_num}: {error_msg}")
                    errfile.write(f"Line {line_num}: {statement}\n")
                    errfile.write(f"Error: {error_msg}\n\n")
                    # Write error indicator to output file
                    outfile.write(f"ERROR: {error_msg}\n")
        
        # Print summary
        if error_count == 0:
            print("All statements converted successfully!")
            # Remove empty error file if no errors
            if os.path.exists(error_file):
                os.remove(error_file)
        else:
            print(f"Conversion completed with {error_count} errors out of {total_lines} lines.")
            print(f"Errors saved to: {error_file}")
        
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()