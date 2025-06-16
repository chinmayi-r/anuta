import re
import os
import csv
import json
import argparse
import time
import requests
from tqdm import tqdm
from openai import AzureOpenAI
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

# Config
sandbox_api_key = os.environ["AI_SANDBOX_KEY"]
sandbox_endpoint = "https://api-ai-sandbox.princeton.edu/"
sandbox_api_version = "2024-02-01"
model_to_be_used = "gpt-4o"
# input_file = "rules.txt"         # Input file: one rule per line
# output_csv = "classified_rules.csv"
max_tokens_per_batch = 12000     # Adjust conservatively below token limit (16k for gpt-4o)
timeout = 60                     # Timeout for API requests (in seconds)
retries = 10_000              # Number of retries for failed requests

# Initialize client (if using AzureOpenAI SDK for other tasks)
client = AzureOpenAI(
    api_key=sandbox_api_key,
    azure_endpoint=sandbox_endpoint,
    api_version=sandbox_api_version
)

# Prepare HTTP headers for direct image/text POST
headers = {
    'api-key': sandbox_api_key,
    'Content-Type': 'application/json'
}

# Prompt template
system_msg = {
    "role": "system",
    "content": "You are a helpful assistant that classifies logic rules extracted from network data."
}

def build_user_message(batch_rules):
    instructions = (
        "You are given a list of logical rules extracted from network measurement data. "
        "For each rule, classify it according to the following:\n\n"
        "- **rtype** (rule type), which must be one of:\n"
        "  • `protocol`: rules derived from network protocol specifications (e.g., TCP, UDP behavior).\n"
        "  • `principle`: rules that arise from general performance, behavioral, or theoretical principles, "
        "like queueing theory or latency constraints.\n"
        "  • `deployment`: rules specific to a particular network's configuration, topology, or operational norms.\n\n"
        "- **meaningful**: a boolean indicating if the rule is semantically meaningful. For example, "
        "`Duration < Bytes` is likely invalid (not meaningful), while `QueueLength > 0 ⇒ PacketsReceived > 0` is valid.\n\n"
        "Return a JSON array. Each element must have this format:\n"
        "{ \"ruleid\": <line number>, \"rtype\": <protocol|principle|deployment>, \"meaningful\": <true|false> }\n\n"
        "Classify these rules:\n"
    )
    for rule in batch_rules:
        instructions += f"{rule['id']}: {rule['text']}\n"
    return {"role": "user", "content": [{"type": "text", "text": instructions}]}

def extract_json_from_response(text):
    """Remove triple backticks and optional 'json' from a markdown code block."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1)
    return text.strip()

def call_api_with_rules(batch):
    data = {
        "messages": [
            system_msg,
            build_user_message(batch)
        ],
        "model": model_to_be_used,
        "max_tokens": max_tokens_per_batch,
        "temperature": 0.0
    }
    
    for attempt in range(1, retries + 1):
        try:
            endpoint = f"{sandbox_endpoint}openai/deployments/{model_to_be_used}/chat/completions?api-version={sandbox_api_version}"
            response = requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=timeout)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            try:
                cleaned = extract_json_from_response(content)
                return json.loads(cleaned)
            except Exception as e:
                print(f"❌ Failed to parse LLM response:\n{content}\nError: {e}")
                return []

        except requests.exceptions.Timeout:
            wait = 1
            print(f"⏳ Timeout on attempt {attempt}. Retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            print(f"❌ Failed on attempt {attempt}: {e}")
            break

    print("❗ Max retries exceeded for this batch.")
    return []

def split_batches(rules, token_limit=max_tokens_per_batch):
    batches = []
    batch, current_len = [], 0
    for rule in rules:
        rule_len = len(rule['text'])  # simple char count proxy for token length
        if current_len + rule_len > token_limit and batch:
            batches.append(batch)
            batch, current_len = [], 0
        batch.append(rule)
        current_len += rule_len
    if batch:
        batches.append(batch)
    return batches

def read_rules(file_path):
    rules = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                rules.append({"id": idx, "text": line.strip()})
    return rules

def save_results_to_csv(results, output_file):
    with open(output_file, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["ruleid", "rtype", "meaningful"])
        writer.writeheader()
        for entry in results:
            writer.writerow(entry)


def classify_batches_in_parallel(batches):
    all_results = []

    core_count = cpu_count()
    print(f"Using {core_count} cores for parallel processing.")
    with ProcessPoolExecutor(max_workers=core_count) as executor:
        # Submit in order; map preserves input-output order
        results = list(tqdm(executor.map(call_api_with_rules, batches), total=len(batches), desc="Classifying rules"))

    # Flatten results while preserving rule order
    for batch_result in results:
        all_results.extend(batch_result)

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Classify logical rules using the Sandbox API.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input file containing logical rules")
    parser.add_argument("--output", "-o", help="Path to the output CSV file (optional)")

    args = parser.parse_args()
    input_file = args.input

    # Default output: <input_name>_llm.csv
    if args.output:
        output_csv = args.output
    else:
        base, _ = os.path.splitext(input_file)
        output_csv = f"{base}_llm.csv"

    print(f"Reading rules from {input_file}...")
    rules = read_rules(input_file)
    batches = split_batches(rules)

    print(f"Processing {len(rules)} rules in {len(batches)} batches...")
    all_results = classify_batches_in_parallel(batches)

    # all_results = []
    # for batch in tqdm(batches, desc="Classifying rules"):
    #     try:
    #         results = call_api_with_rules(batch)
    #         all_results.extend(results)
    #     except Exception as e:
    #         print(f"Batch failed: {e}")

    save_results_to_csv(all_results, output_csv)
    print(f"Saved classification results to {output_csv}")

if __name__ == "__main__":
    main()