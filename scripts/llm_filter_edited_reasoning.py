import re
import os
import csv
import json
import argparse
import time
import requests
import statistics
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
default_max_tokens_per_batch = 12000     # Adjust conservatively below token limit (16k for gpt-4o)
default_rules_per_batch = 100             # Default rule-based batch size
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
        "You are given a list of rules extracted from network measurement data."
        "For each rule, classify it according to the following:\n\n"
        "- **rtype** (rule type), which must be one of:\n"
        "  • `protocol`: rules derived from network protocol specifications (e.g., TCP, UDP behavior).\n"
        "  • `principle`: rules that arise from general performance, behavioral, or theoretical principles, "
        "like queueing theory or latency constraints.\n"
        "  • `deployment`: rules specific to a particular network's configuration, topology, or operational norms.\n\n"
        "- **meaningful**: a boolean indicating if the rule is semantically meaningful. For example, "
        "`Duration < Bytes` is likely invalid (not meaningful), while `QueueLength > 0 ⇒ PacketsReceived > 0` is valid.\n\n"
        "- **explanation**: a brief explanation (1-2 sentences) of why you chose this classification and "
        "meaningfulness assessment. Be specific about what makes it a protocol/principle/deployment rule "
        "and why it is or isn't meaningful.\n\n"
        "Return a JSON array. Each element must have this format:\n"
        "{ \"ruleid\": <line number>, \"rtype\": <protocol|principle|deployment>, \"meaningful\": <true|false>, \"explanation\": \"<brief explanation>\" }\n\n"
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
        "max_tokens": default_max_tokens_per_batch,
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

def split_batches_by_tokens(rules, token_limit=default_max_tokens_per_batch):
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

def split_batches_by_rules(rules, rules_per_batch=default_rules_per_batch):
    """Split rules into batches based on number of rules per batch."""
    batches = []
    for i in range(0, len(rules), rules_per_batch):
        batch = rules[i:i + rules_per_batch]
        batches.append(batch)
    return batches

def split_batches(rules, batch_size=None, batch_mode="rules"):
    """
    Split rules into batches based on specified mode.
    
    Args:
        rules: List of rule dictionaries
        batch_size: Size of batch (number of rules or token limit depending on mode)
        batch_mode: "rules" for rule-based batching, "tokens" for token-based batching
    """
    if batch_mode == "rules":
        batch_size = batch_size or default_rules_per_batch
        return split_batches_by_rules(rules, batch_size)
    else:  # token-based
        batch_size = batch_size or default_max_tokens_per_batch
        return split_batches_by_tokens(rules, batch_size)

def read_rules(file_path):
    rules = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                rules.append({"id": idx, "text": line.strip()})
    return rules

def save_results_to_csv(results, output_file):
    with open(output_file, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["ruleid", "rtype", "meaningful", "explanation"])
        writer.writeheader()
        for entry in results:
            # Ensure all required fields are present
            csv_entry = {
                "ruleid": entry.get("ruleid", ""),
                "rtype": entry.get("rtype", ""),
                "meaningful": entry.get("meaningful", ""),
                "explanation": entry.get("explanation", "")
            }
            writer.writerow(csv_entry)


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

def classify_single_run(rules, batch_size, batch_mode = "rules"):
    """Perform a single classification run."""
    batches = split_batches(rules, batch_size, batch_mode)
    batch_info = f"{len(batches)} batches"
    if batch_mode == "rules":
        batch_info += f" ({batch_size} rules per batch)"
    else:
        batch_info += f" (≤{batch_size} tokens per batch)"
    
    print(f"Processing {len(rules)} rules in {batch_info}...")
    return classify_batches_in_parallel(batches)

def calculate_variance_stats(all_runs_results):
    """Calculate variance statistics across multiple runs."""
    if not all_runs_results:
        return {"error": "No results to analyze"}
    
    # Extract classification consistency
    rtype_distributions = []
    rtype_percentages = []
    meaningful_rates = []
    
    for run_results in all_runs_results:
        if not run_results:  # Skip empty runs
            continue
            
        # Calculate type distribution for this run
        type_counts = {"protocol": 0, "principle": 0, "deployment": 0}
        meaningful_count = 0
        
        for result in run_results:
            rtype = result.get('rtype', '')
            if rtype in type_counts:
                type_counts[rtype] += 1
            if result.get('meaningful', False):
                meaningful_count += 1
        
        total_rules = len(run_results)
        if total_rules > 0:
            rtype_distributions.append(type_counts)
            meaningful_rates.append(meaningful_count / total_rules)
            
            # Calculate percentages for this run
            type_percentages = {
                "protocol": type_counts["protocol"] / total_rules,
                "principle": type_counts["principle"] / total_rules,
                "deployment": type_counts["deployment"] / total_rules
            }
            rtype_percentages.append(type_percentages)
    
    # Calculate statistics
    stats = {
        "num_runs": len(all_runs_results),
        "total_rules_per_run": len(all_runs_results[0]) if all_runs_results else 0,
        "meaningful_classification": {
            "mean_rate": statistics.mean(meaningful_rates) if meaningful_rates else 0,
            "std_rate": statistics.stdev(meaningful_rates) if len(meaningful_rates) > 1 else 0,
            "range": {
                "min": min(meaningful_rates) if meaningful_rates else 0,
                "max": max(meaningful_rates) if meaningful_rates else 0
            }
        },
        "rule_type_classification": {}
    }
    
    # Calculate variance in type distributions (both counts and percentages)
    if rtype_distributions and rtype_percentages:
        for rtype in ["protocol", "principle", "deployment"]:
            # Count-based statistics
            type_counts = [dist[rtype] for dist in rtype_distributions]
            # Percentage-based statistics
            type_percentages = [perc[rtype] for perc in rtype_percentages]
            
            stats["rule_type_classification"][rtype] = {
                "counts": {
                    "mean": statistics.mean(type_counts),
                    "std": statistics.stdev(type_counts) if len(type_counts) > 1 else 0,
                    "range": {"min": min(type_counts), "max": max(type_counts)}
                },
                "percentages": {
                    "mean": statistics.mean(type_percentages),
                    "std": statistics.stdev(type_percentages) if len(type_percentages) > 1 else 0,
                    "range": {"min": min(type_percentages), "max": max(type_percentages)}
                }
            }
    
    return stats

def save_variance_report(stats, output_file):
    """Save variance analysis to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Classify logical rules using the Sandbox API.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input file containing logical rules")
    parser.add_argument("--output", "-o", help="Path to the output CSV file (optional)")
    parser.add_argument("--batch-size", "-b", type=int, default=default_rules_per_batch,
                       help=f"Batch size - number of rules per batch in 'rules' mode, token limit in 'tokens' mode (default: {default_rules_per_batch} rules)")
    parser.add_argument("--batch-mode", "-m", choices=["rules", "tokens"], default="rules",
                       help="Batching mode: 'rules' (by number of rules) or 'tokens' (by token limit) (default: rules)")
    parser.add_argument("--runs", "-r", type=int, default=1,
                       help="Number of runs for variance estimation (default: 1)")
    parser.add_argument("--variance-output", help="Path to save variance analysis report (JSON)")

    args = parser.parse_args()
    input_file = args.input
    batch_size = args.batch_size
    batch_mode = args.batch_mode
    num_runs = args.runs

    # Default output: <input_name>_llm.csv
    if args.output:
        output_csv = args.output
    else:
        base, _ = os.path.splitext(input_file)
        output_csv = f"{base}_llm.csv"

    print(f"Reading rules from {input_file}...")
    rules = read_rules(input_file)
    
    if batch_mode == "rules":
        print(f"Batch mode: {batch_size} rules per batch")
    else:
        print(f"Batch mode: ≤{batch_size} tokens per batch (character approximation)")
    
    # Store results from all runs
    all_runs_results = []
    
    if num_runs == 1:
        print(f"Performing single classification run...")
        results = classify_single_run(rules, batch_size, batch_mode)
        all_runs_results.append(results)
        
        # Save main results
        save_results_to_csv(results, output_csv)
        print(f"Saved classification results to {output_csv}")
        
    else:
        print(f"Performing {num_runs} runs for variance estimation...")
        for run_idx in range(num_runs):
            print(f"\n=== Run {run_idx + 1}/{num_runs} ===")
            results = classify_single_run(rules, batch_size, batch_mode)
            all_runs_results.append(results)
            
            # Save individual run results
            run_output = f"{os.path.splitext(output_csv)[0]}_run_{run_idx + 1}.csv"
            save_results_to_csv(results, run_output)
            print(f"Saved run {run_idx + 1} results to {run_output}")
        
        # Save aggregated results (using first run as primary)
        if all_runs_results:
            save_results_to_csv(all_runs_results[0], output_csv)
            print(f"Saved primary results to {output_csv}")
    
    # Variance analysis
    if num_runs > 1:
        print(f"\nCalculating variance statistics across {num_runs} runs...")
        variance_stats = calculate_variance_stats(all_runs_results)
        
        # Default variance output file
        variance_output = args.variance_output or f"{os.path.splitext(output_csv)[0]}_variance.json"
        save_variance_report(variance_stats, variance_output)
        print(f"Saved variance analysis to {variance_output}")
        
        # Print summary
        print(f"\nVariance Summary:")
        print(f"  Total rules per run: {variance_stats['total_rules_per_run']}")
        print(f"  Number of runs: {variance_stats['num_runs']}")
        
        # Meaningful classification stats
        meaningful = variance_stats['meaningful_classification']
        print(f"\nMeaningful Classification:")
        print(f"  Average rate: {meaningful['mean_rate']:.3f} ± {meaningful['std_rate']:.3f}")
        print(f"  Range: {meaningful['range']['min']:.3f} - {meaningful['range']['max']:.3f}")
        
        # Rule type classification stats
        print(f"\nRule Type Classification:")
        for rtype in ["protocol", "principle", "deployment"]:
            if rtype in variance_stats['rule_type_classification']:
                rtype_stats = variance_stats['rule_type_classification'][rtype]
                counts = rtype_stats['counts']
                percentages = rtype_stats['percentages']
                
                print(f"  {rtype.capitalize()}:")
                print(f"    Count: {counts['mean']:.1f} ± {counts['std']:.1f} (range: {counts['range']['min']}-{counts['range']['max']})")
                print(f"    Percentage: {percentages['mean']:.1%} ± {percentages['std']:.1%} (range: {percentages['range']['min']:.1%}-{percentages['range']['max']:.1%})")

if __name__ == "__main__":
    main()