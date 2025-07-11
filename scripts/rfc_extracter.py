import os
import re
import json
from collections import defaultdict

output_path = "rfc_facts.json"

# Pattern to capture lines referencing ports and protocols
PORT_PATTERN = re.compile(r"port (\d+)[\s:,-]+.*?(UDP|TCP)", re.IGNORECASE)
PROTOCOL_PATTERN = re.compile(r"(UDP|TCP|IP)[\s,:-]+.*?(protocol|transport)", re.IGNORECASE)

def is_garbage_line(line):
    # Count occurrences of table characters
    table_chars = ['|', '+', '-', '=', '*']
    count = sum(line.count(c) for c in table_chars)
    # Heuristic: if many such chars, it's probably a table or diagram line
    if count > 10 or len(line.strip()) < 10:
        return True
    return False


def extract_sentences_with_keywords(text, keywords):
    # Split text into sentences by period (.)
    sentences = re.split(r'(?<=\.)\s+', text.replace('\n', ' '))
    cleaned_sentences = [s for s in sentences if not is_garbage_line(s)]
    matches = []
    for sent in cleaned_sentences:
        if any(kw.lower() in sent.lower() for kw in keywords):
            matches.append(sent.strip())
    return matches

def parse_rfc_text(rfc_text):
    """
    Extract port and protocol facts from the text of a single RFC.
    """
    facts = defaultdict(set)

    # Extract sentences mentioning ports
    port_sentences = extract_sentences_with_keywords(rfc_text, ['port', 'udp', 'tcp'])
    for sent in port_sentences:
        # Optionally extract port numbers and protocol names for structured info
        port_match = re.search(r"port (\d+)", sent, re.IGNORECASE)
        proto_match = re.search(r"\b(UDP|TCP|IP)\b", sent, re.IGNORECASE)
        if port_match and proto_match:
            facts['Ports'].add((port_match.group(1), proto_match.group(1).upper(), sent))
        elif proto_match:
            facts['Protocols'].add((proto_match.group(1).upper(), sent))

    return facts

def extract_rfc_facts(directory):
    """
    Process all .txt files in the directory and return extracted facts.
    """
    all_facts = defaultdict(dict)

    for file_name in os.listdir(directory):
        if not file_name.lower().endswith(".txt"):
            continue

        rfc_number = os.path.splitext(file_name)[0]
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            rfc_text = f.read()

        facts = parse_rfc_text(rfc_text)

        for category, entries in facts.items():
            all_facts[category][rfc_number] = sorted(entries)

    return all_facts

if __name__ == "__main__":
    # Example usage
    rfc_dir = "./rfc_texts"  # Make sure this folder contains .txt RFC files
    os.makedirs(rfc_dir, exist_ok=True)

    facts = extract_rfc_facts(rfc_dir)
    
    # Pretty-print results
    for category, rfc_data in facts.items():
        print(f"\n== {category} ==")
        for rfc, entries in rfc_data.items():
            print(f"\nFrom {rfc}.txt:")
            for entry in entries:
                print(" •", entry)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({k: dict(v) for k, v in facts.items()}, f, indent=2)

    print(f"\n Saved extracted RFC facts to {output_path}")
