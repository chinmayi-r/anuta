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
default_max_tokens_per_batch = 12000     # Default token-based batch size
default_rules_per_batch = 10             # Default rule-based batch size
timeout = 60                             # Timeout for API requests (in seconds)
retries = 10_000                         # Number of retries for failed requests

# Initialize client
client = AzureOpenAI(
    api_key=sandbox_api_key,
    azure_endpoint=sandbox_endpoint,
    api_version=sandbox_api_version
)

# Prepare HTTP headers for direct POST
headers = {
    'api-key': sandbox_api_key,
    'Content-Type': 'application/json'
}

# Prompt template
system_msg = {
    "role": "system",
    "content": (
        "You are a network protocol expert familiar with RFC documents. "
        "Use your knowledge of TCP/IP (RFC 793), UDP (RFC 768), and other "
        "networking standards to classify rules. Common protocol behaviors:\n"
        "- TCP flags: SYN, ACK, FIN, RST combinations\n"
        "- Standard ports: 80 (HTTP), 443 (HTTPS), 53 (DNS), etc.\n"
        "- IP address ranges: private (10.0.0.0/8), multicast, etc."
    )
}

def build_user_message(batch_rules):
    instructions = (
        "You are given a list of logical rules extracted from network measurement data. "
        "For each rule, classify it carefully according to the following criteria:\n\n"
        "- **rtype** (rule type):\n"
        "  • `protocol`: ONLY if clearly derived from network protocol specs (e.g., TCP flags, UDP behavior).\n"
        "  • `principle`: ONLY for general performance/behavioral principles (queueing theory, latency constraints).\n"
        "  • `deployment`: ONLY if clearly specific to a particular network's configuration.\n\n"
        "- **meaningful**: Be STRICT. Only mark as true if the rule makes clear logical sense. "
        "Reject rules with nonsensical comparisons or invalid logic structures.\n\n"
        "IMPORTANT: When in doubt, classify as 'principle' and mark as not meaningful. "
        "False negatives are preferable to false positives.\n\n"
        "Return a JSON array with this exact format for each rule:\n"
        "{ \"ruleid\": <line number>, \"rtype\": <protocol|principle|deployment>, \"meaningful\": <true|false> }\n\n"
        "Rules to classify (be critical!):\n"
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
    """Split rules into batches based on character count approximation."""
    batches = []
    batch, current_len = [], 0
    for rule in rules:
        rule_len = len(rule['text'])
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
    """Split rules into batches based on specified mode."""
    if batch_mode == "rules":
        batch_size = batch_size or default_rules_per_batch
        return split_batches_by_rules(rules, batch_size)
    else:  # token-based
        batch_size = batch_size or default_max_tokens_per_batch
        return split_batches_by_tokens(rules, batch_size)

def load_ground_truth_rules(json_file):
    """Load and extract all rules from ground truth JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    rules = []
    rule_id = 0
    
    for entry in ground_truth_data:
        if 'queries' in entry:
            for query in entry['queries']:
                rules.append({
                    'id': rule_id,
                    'text': query.strip(),
                    'original_entry_description': entry.get('description', '')
                })
                rule_id += 1
    
    return rules

def classify_batches_in_parallel(batches):
    """Process batches in parallel and return flattened results."""
    all_results = []

    core_count = cpu_count()
    print(f"Using {core_count} cores for parallel processing.")
    with ProcessPoolExecutor(max_workers=core_count) as executor:
        results = list(tqdm(executor.map(call_api_with_rules, batches), total=len(batches), desc="Classifying rules"))

    # Flatten results while preserving rule order
    for batch_result in results:
        all_results.extend(batch_result)

    return all_results

def classify_single_run(rules, batch_size, batch_mode="rules"):
    """Perform a single classification run."""
    batches = split_batches(rules, batch_size, batch_mode)
    batch_info = f"{len(batches)} batches"
    if batch_mode == "rules":
        batch_info += f" ({batch_size} rules per batch)"
    else:
        batch_info += f" (≤{batch_size} tokens per batch)"
    
    print(f"Processing {len(rules)} rules in {batch_info}...")
    return classify_batches_in_parallel(batches)

def calculate_accuracy_stats(results, expected_meaningful=True):
    """Calculate accuracy statistics assuming all ground truth rules should be meaningful."""
    if not results:
        return {"error": "No results to analyze"}
    
    total_rules = len(results)
    correct_meaningful = sum(1 for r in results if r.get('meaningful') == expected_meaningful)
    accuracy = correct_meaningful / total_rules if total_rules > 0 else 0
    
    # Type distribution
    type_counts = {"protocol": 0, "principle": 0, "deployment": 0, "unknown": 0}
    for result in results:
        rtype = result.get('rtype', 'unknown')
        if rtype in type_counts:
            type_counts[rtype] += 1
        else:
            type_counts['unknown'] += 1
    
    # Incorrectly classified rules
    incorrect_rules = []
    for result in results:
        if result.get('meaningful') != expected_meaningful:
            incorrect_rules.append({
                'rule_id': result.get('ruleid'),
                'classified_as_meaningful': result.get('meaningful'),
                'rule_type': result.get('rtype')
            })
    
    return {
        "total_rules": total_rules,
        "correct_meaningful_classifications": correct_meaningful,
        "accuracy": accuracy,
        "accuracy_percentage": accuracy * 100,
        "type_distribution": type_counts,
        "incorrect_classifications": incorrect_rules,
        "num_incorrect": len(incorrect_rules)
    }

def calculate_variance_stats(all_runs_results, expected_meaningful=True):
    """Calculate variance statistics across multiple runs."""
    if not all_runs_results:
        return {"error": "No results to analyze"}
    
    accuracies = []
    type_distributions = []
    
    for run_results in all_runs_results:
        if not run_results:
            continue
        
        # Calculate accuracy for this run
        stats = calculate_accuracy_stats(run_results, expected_meaningful)
        accuracies.append(stats['accuracy'])
        type_distributions.append(stats['type_distribution'])
    
    # Calculate statistics
    variance_stats = {
        "num_runs": len(all_runs_results),
        "accuracy_stats": {
            "mean": statistics.mean(accuracies) if accuracies else 0,
            "std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            "min": min(accuracies) if accuracies else 0,
            "max": max(accuracies) if accuracies else 0,
            "range": max(accuracies) - min(accuracies) if accuracies else 0
        },
        "individual_run_accuracies": [acc * 100 for acc in accuracies],
        "type_distribution_variance": {}
    }
    
    # Calculate variance in type distributions
    if type_distributions:
        for rtype in ["protocol", "principle", "deployment"]:
            type_counts = [dist.get(rtype, 0) for dist in type_distributions]
            variance_stats["type_distribution_variance"][rtype] = {
                "mean": statistics.mean(type_counts),
                "std": statistics.stdev(type_counts) if len(type_counts) > 1 else 0,
                "range": {"min": min(type_counts), "max": max(type_counts)}
            }
    
    return variance_stats

def save_results_to_csv(results, output_file):
    """Save classification results to CSV."""
    with open(output_file, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["ruleid", "rtype", "meaningful"])
        writer.writeheader()
        for entry in results:
            writer.writerow(entry)

def save_accuracy_report(stats, output_file):
    """Save accuracy analysis to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

def save_variance_report(stats, output_file):
    """Save variance analysis to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Validate model accuracy using ground truth rules (all assumed meaningful).")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--output", "-o", help="Path to the output CSV file (optional)")
    parser.add_argument("--batch-size", "-b", type=int, default=default_max_tokens_per_batch,
                       help=f"Batch size - number of rules per batch in 'rules' mode, token limit in 'tokens' mode (default: {default_rules_per_batch} rules)")
    parser.add_argument("--batch-mode", "-m", choices=["rules", "tokens"], default="rules",
                       help="Batching mode: 'rules' (by number of rules) or 'tokens' (by token limit) (default: rules)")
    parser.add_argument("--runs", "-r", type=int, default=1,
                       help="Number of runs for variance estimation (default: 1)")
    parser.add_argument("--accuracy-output", help="Path to save accuracy analysis report (JSON)")
    parser.add_argument("--variance-output", help="Path to save variance analysis report (JSON)")

    args = parser.parse_args()
    ground_truth_file = args.ground_truth
    batch_size = args.batch_size
    batch_mode = args.batch_mode
    num_runs = args.runs

    # Default output files
    base_name = os.path.splitext(os.path.basename(ground_truth_file))[0]
    if args.output:
        output_csv = args.output
    else:
        output_csv = f"{base_name}_validation_results.csv"

    print(f"Loading ground truth rules from {ground_truth_file}...")
    rules = load_ground_truth_rules(ground_truth_file)
    print(f"Loaded {len(rules)} rules from ground truth (all assumed meaningful)")
    
    if batch_mode == "rules":
        print(f"Batch mode: {batch_size} rules per batch")
    else:
        print(f"Batch mode: ≤{batch_size} tokens per batch (character approximation)")
    
    # Store results from all runs
    all_runs_results = []
    
    if num_runs == 1:
        print(f"Performing single validation run...")
        results = classify_single_run(rules, batch_size, batch_mode)
        all_runs_results.append(results)
        
        # Calculate accuracy
        accuracy_stats = calculate_accuracy_stats(results)
        
        # Save main results
        save_results_to_csv(results, output_csv)
        print(f"Saved classification results to {output_csv}")
        
        # Print accuracy summary
        print(f"\n=== ACCURACY RESULTS ===")
        print(f"Total rules: {accuracy_stats['total_rules']}")
        print(f"Correctly classified as meaningful: {accuracy_stats['correct_meaningful_classifications']}")
        print(f"Accuracy: {accuracy_stats['accuracy_percentage']:.2f}%")
        print(f"Incorrectly classified: {accuracy_stats['num_incorrect']}")
        
        if accuracy_stats['incorrect_classifications']:
            print(f"\nIncorrectly classified rule IDs: {[r['rule_id'] for r in accuracy_stats['incorrect_classifications']]}")
        
        print(f"\nRule type distribution:")
        for rtype, count in accuracy_stats['type_distribution'].items():
            percentage = (count / accuracy_stats['total_rules']) * 100 if accuracy_stats['total_rules'] > 0 else 0
            print(f"  {rtype}: {count} ({percentage:.1f}%)")
            
    else:
        print(f"Performing {num_runs} runs for variance estimation...")
        run_accuracies = []
        
        for run_idx in range(num_runs):
            print(f"\n=== Run {run_idx + 1}/{num_runs} ===")
            results = classify_single_run(rules, batch_size, batch_mode)
            all_runs_results.append(results)
            
            # Calculate accuracy for this run
            accuracy_stats = calculate_accuracy_stats(results)
            run_accuracies.append(accuracy_stats['accuracy_percentage'])
            
            # Save individual run results
            run_output = f"{os.path.splitext(output_csv)[0]}_run_{run_idx + 1}.csv"
            save_results_to_csv(results, run_output)
            print(f"Run {run_idx + 1} accuracy: {accuracy_stats['accuracy_percentage']:.2f}%")
            print(f"Saved run {run_idx + 1} results to {run_output}")
        
        # Save primary results (using first run)
        if all_runs_results:
            save_results_to_csv(all_runs_results[0], output_csv)
            print(f"Saved primary results to {output_csv}")
        
        # Calculate variance statistics
        variance_stats = calculate_variance_stats(all_runs_results)
        
        print(f"\n=== VARIANCE ANALYSIS ===")
        print(f"Accuracy across {num_runs} runs:")
        print(f"  Mean: {variance_stats['accuracy_stats']['mean'] * 100:.2f}%")
        print(f"  Std Dev: {variance_stats['accuracy_stats']['std'] * 100:.2f}%")
        print(f"  Range: {variance_stats['accuracy_stats']['min'] * 100:.2f}% - {variance_stats['accuracy_stats']['max'] * 100:.2f}%")
        print(f"  Individual runs: {[f'{acc:.1f}%' for acc in variance_stats['individual_run_accuracies']]}")
    
    # Save detailed reports
    if num_runs == 1:
        accuracy_output = args.accuracy_output or f"{os.path.splitext(output_csv)[0]}_accuracy.json"
        save_accuracy_report(accuracy_stats, accuracy_output)
        print(f"Saved detailed accuracy report to {accuracy_output}")
    else:
        variance_output = args.variance_output or f"{os.path.splitext(output_csv)[0]}_variance.json"
        save_variance_report(variance_stats, variance_output)
        print(f"Saved variance analysis to {variance_output}")

if __name__ == "__main__":
    main()