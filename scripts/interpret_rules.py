from anuta.theory import Theory


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python interpret_rules.py <rules_file>")
        sys.exit(1)

    rules_file = sys.argv[1]
    rules_path = rules_file.split('.')[0]
    
    th = Theory(rules_file)
    interpreted_rules = Theory.interpret(th.constraints, save_path=f"{rules_path}.txt")
    print(f"Interpreted rules saved to {rules_path}.txt")
    
    exit(0)