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

# Available models in Princeton AI Sandbox
AVAILABLE_MODELS = [
    "o3-mini",
    "gpt-4o-mini", 
    "gpt-4o",
    "gpt-35-turbo-16k",
    "Mistral-Small",
    "Meta-Llama-3-1-8B-Instruct",
    "Meta-Llama-3-1-70B-Instruct"
]

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

# Token usage tracking
token_usage_stats = {}

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
    return {"role": "user", "content": instructions}

def extract_json_from_response(text):
    """Remove triple backticks and optional 'json' from a markdown code block."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1)
    return text.strip()

def call_api_with_rules(batch_and_model):
    """Call API with rules and track token usage."""
    batch, model_name = batch_and_model
    
    messages = [system_msg, build_user_message(batch)]
    
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=default_max_tokens_per_batch,
                temperature=0.0,
                top_p=0.01
            )
            
            content = response.choices[0].message.content
            
            # Track token usage
            if response.usage:
                if model_name not in token_usage_stats:
                    token_usage_stats[model_name] = {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'requests': 0
                    }
                
                token_usage_stats[model_name]['prompt_tokens'] += response.usage.prompt_tokens
                token_usage_stats[model_name]['completion_tokens'] += response.usage.completion_tokens
                token_usage_stats[model_name]['total_tokens'] += response.usage.total_tokens
                token_usage_stats[model_name]['requests'] += 1

            try:
                cleaned = extract_json_from_response(content)
                return json.loads(cleaned)
            except Exception as e:
                print(f"❌ Failed to parse LLM response for {model_name}:\n{content}\nError: {e}")
                return []

        except Exception as e:
            print(f"❌ Failed on attempt {attempt} for {model_name}: {e}")
            if attempt < retries:
                wait = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                print(f"⏳ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                break

    print(f"❗ Max retries exceeded for {model_name} batch.")
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

def classify_batches_in_parallel(batches, model_name):
    """Process batches in parallel and return flattened results."""
    all_results = []
    
    # Prepare batches with model name
    batches_with_model = [(batch, model_name) for batch in batches]

    core_count = cpu_count()
    print(f"Using {core_count} cores for parallel processing with {model_name}.")
    with ProcessPoolExecutor(max_workers=core_count) as executor:
        results = list(tqdm(executor.map(call_api_with_rules, batches_with_model), 
                           total=len(batches), desc=f"Classifying rules ({model_name})"))

    # Flatten results while preserving rule order
    for batch_result in results:
        all_results.extend(batch_result)

    return all_results

def classify_single_run(rules, model_name, batch_size, batch_mode="rules"):
    """Perform a single classification run with specified model."""
    batches = split_batches(rules, batch_size, batch_mode)
    batch_info = f"{len(batches)} batches"
    if batch_mode == "rules":
        batch_info += f" ({batch_size} rules per batch)"
    else:
        batch_info += f" (≤{batch_size} tokens per batch)"
    
    print(f"Processing {len(rules)} rules in {batch_info} using {model_name}...")
    return classify_batches_in_parallel(batches, model_name)

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

def calculate_model_variance_stats(model_runs_results, expected_meaningful=True):
    """Calculate variance statistics for a single model across multiple runs."""
    if not model_runs_results:
        return {"error": "No results to analyze"}
    
    accuracies = []
    type_distributions = []
    
    for run_results in model_runs_results:
        if not run_results:
            continue
        
        # Calculate accuracy for this run
        stats = calculate_accuracy_stats(run_results, expected_meaningful)
        accuracies.append(stats['accuracy'])
        type_distributions.append(stats['type_distribution'])
    
    # Calculate statistics
    variance_stats = {
        "num_runs": len(model_runs_results),
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

def save_results_to_csv(results, output_file, model_name):
    """Save classification results to CSV."""
    with open(output_file, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["ruleid", "rtype", "meaningful", "model"])
        writer.writeheader()
        for entry in results:
            entry_with_model = entry.copy()
            entry_with_model["model"] = model_name
            writer.writerow(entry_with_model)

def save_model_comparison_results(all_model_results, output_file):
    """Save results from all models to a single CSV for comparison."""
    with open(output_file, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "run", "ruleid", "rtype", "meaningful"])
        writer.writeheader()
        
        for model_name, model_data in all_model_results.items():
            for run_idx, run_results in enumerate(model_data['runs']):
                for entry in run_results:
                    writer.writerow({
                        "model": model_name,
                        "run": run_idx + 1,
                        "ruleid": entry.get('ruleid'),
                        "rtype": entry.get('rtype'),
                        "meaningful": entry.get('meaningful')
                    })

def save_comprehensive_report(all_model_results, token_usage, output_file):
    """Save comprehensive analysis report including all models, variance, and token usage."""
    report = {
        "models_tested": list(all_model_results.keys()),
        "model_performance": {},
        "token_usage": token_usage,
        "cross_model_comparison": {}
    }
    
    # Calculate performance stats for each model
    for model_name, model_data in all_model_results.items():
        model_variance_stats = calculate_model_variance_stats(model_data['runs'])
        report["model_performance"][model_name] = model_variance_stats
    
    # Cross-model comparison
    model_means = {}
    for model_name, model_data in all_model_results.items():
        if model_name in report["model_performance"]:
            model_means[model_name] = report["model_performance"][model_name]["accuracy_stats"]["mean"]
    
    if model_means:
        report["cross_model_comparison"] = {
            "best_model": max(model_means, key=model_means.get),
            "worst_model": min(model_means, key=model_means.get),
            "model_accuracy_ranking": sorted(model_means.items(), key=lambda x: x[1], reverse=True),
            "accuracy_range": {
                "min": min(model_means.values()),
                "max": max(model_means.values()),
                "spread": max(model_means.values()) - min(model_means.values())
            }
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Multi-model LLM validation using ground truth rules.")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--models", "-m", nargs='+', choices=AVAILABLE_MODELS, 
                       default=AVAILABLE_MODELS, help="Models to test (default: all available)")
    parser.add_argument("--output-dir", "-o", default="./results", help="Output directory for results")
    parser.add_argument("--batch-size", "-b", type=int, default=default_rules_per_batch,
                       help=f"Batch size (default: {default_max_tokens_per_batch})")
    parser.add_argument("--batch-mode", choices=["rules", "tokens"], default="rules",
                       help="Batching mode: 'rules' or 'tokens' (default: rules)")
    parser.add_argument("--runs", "-r", type=int, default=3,
                       help="Number of runs per model for variance estimation (default: 3)")

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ground truth rules
    print(f"Loading ground truth rules from {args.ground_truth}...")
    rules = load_ground_truth_rules(args.ground_truth)
    print(f"Loaded {len(rules)} rules from ground truth (all assumed meaningful)")
    
    # Initialize results storage
    all_model_results = {}
    
    # Test each model
    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*50}")
        
        model_runs = []
        
        for run_idx in range(args.runs):
            print(f"\n--- Run {run_idx + 1}/{args.runs} for {model_name} ---")
            
            try:
                results = classify_single_run(rules, model_name, args.batch_size, args.batch_mode)
                model_runs.append(results)
                
                # Calculate and display accuracy for this run
                accuracy_stats = calculate_accuracy_stats(results)
                print(f"Run {run_idx + 1} accuracy: {accuracy_stats['accuracy_percentage']:.2f}%")
                
                # Save individual run results
                run_output = os.path.join(args.output_dir, f"{model_name}_run_{run_idx + 1}.csv")
                save_results_to_csv(results, run_output, model_name)
                
            except Exception as e:
                print(f"❌ Error in run {run_idx + 1} for {model_name}: {e}")
                model_runs.append([])  # Add empty results to maintain run count
        
        # Store model results
        all_model_results[model_name] = {
            'runs': model_runs,
            'variance_stats': calculate_model_variance_stats(model_runs)
        }
        
        # Print model summary
        if model_runs and any(model_runs):
            variance_stats = all_model_results[model_name]['variance_stats']
            print(f"\n{model_name} Summary ({args.runs} runs):")
            print(f"  Mean Accuracy: {variance_stats['accuracy_stats']['mean'] * 100:.2f}%")
            print(f"  Std Dev: {variance_stats['accuracy_stats']['std'] * 100:.2f}%")
            print(f"  Range: {variance_stats['accuracy_stats']['min'] * 100:.2f}% - {variance_stats['accuracy_stats']['max'] * 100:.2f}%")
    
    # Save comprehensive results
    comparison_output = os.path.join(args.output_dir, "model_comparison_results.csv")
    save_model_comparison_results(all_model_results, comparison_output)
    print(f"\nSaved model comparison results to {comparison_output}")
    
    # Save comprehensive report
    report_output = os.path.join(args.output_dir, "comprehensive_report.json")
    save_comprehensive_report(all_model_results, token_usage_stats, report_output)
    print(f"Saved comprehensive report to {report_output}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - MODEL COMPARISON")
    print(f"{'='*60}")
    
    model_summaries = []
    for model_name, model_data in all_model_results.items():
        if model_data['runs'] and any(model_data['runs']):
            stats = model_data['variance_stats']['accuracy_stats']
            model_summaries.append((model_name, stats['mean'], stats['std']))
    
    # Sort by mean accuracy
    model_summaries.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<30} {'Mean Accuracy':<15} {'Std Dev':<10}")
    print("-" * 60)
    for model_name, mean_acc, std_acc in model_summaries:
        print(f"{model_name:<30} {mean_acc*100:>10.2f}% {std_acc*100:>10.2f}%")
    
    # Print token usage summary
    if token_usage_stats:
        print(f"\n{'='*60}")
        print("TOKEN USAGE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Total Tokens':<15} {'Requests':<10} {'Avg/Request':<12}")
        print("-" * 70)
        for model_name, usage in token_usage_stats.items():
            avg_tokens = usage['total_tokens'] / usage['requests'] if usage['requests'] > 0 else 0
            print(f"{model_name:<30} {usage['total_tokens']:>10,} {usage['requests']:>10} {avg_tokens:>10.1f}")

if __name__ == "__main__":
    main()