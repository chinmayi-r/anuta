#!/usr/bin/env python3
"""
Prompt Evolution Performance Analyzer

This script analyzes the performance differences across 4 validation scripts:
1. Original prompt
2. Critical prompt  
3. Original prompt with RFC docs
4. Critical prompt with RFC docs

It generates a line chart showing accuracy and variance trends across prompt evolution.
"""

import os
import json
import glob
import argparse
import statistics
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from pathlib import Path

def ensure_results_dir(results_dir):
    """Create results directory if it doesn't exist."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    return results_dir

def get_expected_files(results_dir, runs=5):
    """Get expected result file paths for each prompt version."""
    files = {
        "Original": {
            "variance": os.path.join(results_dir, "cidds_queries_validation_variance.json"),
            "accuracy": os.path.join(results_dir, "cidds_queries_validation_accuracy.json")
        },
        "Critical": {
            "variance": os.path.join(results_dir, "cidds_queries_validation_crit_variance.json"),
            "accuracy": os.path.join(results_dir, "cidds_queries_validation_crit_accuracy.json")
        },
        "Original+RFC": {
            "variance": os.path.join(results_dir, "cidds_queries_validation_orirfc_variance.json"),
            "accuracy": os.path.join(results_dir, "cidds_queries_validation_orirfc_accuracy.json")
        },
        "Critical+RFC": {
            "variance": os.path.join(results_dir, "cidds_queries_validation_critrfc_variance.json"),
            "accuracy": os.path.join(results_dir, "cidds_queries_validation_critrfc_accuracy.json")
        }
    }
    return files

def check_file_has_sufficient_runs(filepath, required_runs):
    """Check if a variance file has sufficient runs."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'num_runs' in data:
            return data['num_runs'] >= required_runs
        elif 'individual_run_accuracies' in data:
            return len(data['individual_run_accuracies']) >= required_runs
        else:
            return False
    except:
        return False

def extract_accuracy_from_file(filepath):
    """Extract accuracy statistics from a JSON result file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if 'accuracy_stats' in data:
            # Variance report format
            return {
                'mean_accuracy': data['accuracy_stats']['mean'] * 100,
                'std_accuracy': data['accuracy_stats']['std'] * 100,
                'accuracies': data.get('individual_run_accuracies', []),
                'num_runs': data.get('num_runs', len(data.get('individual_run_accuracies', [])))
            }
        elif 'accuracy_percentage' in data:
            # Single run accuracy report format
            return {
                'mean_accuracy': data['accuracy_percentage'],
                'std_accuracy': 0.0,
                'accuracies': [data['accuracy_percentage']],
                'num_runs': 1
            }
        else:
            print(f"Unrecognized format in {filepath}")
            return None
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def run_validation_script(script_path, ground_truth_file, rfc_facts_file, results_dir, runs, version_name):
    """Run a validation script and save results to the specified directory."""
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return False
        
    print(f"Running validation for {version_name} ({runs} runs)...")
    
    # Construct output filenames
    base_name = os.path.splitext(os.path.basename(ground_truth_file))[0]
    
    if version_name == "Original":
        output_prefix = f"{base_name}_validation"
    elif version_name == "Critical":
        output_prefix = f"{base_name}_validation_crit"
    elif version_name == "Original+RFC":
        output_prefix = f"{base_name}_validation_orirfc"
    elif version_name == "Critical+RFC":
        output_prefix = f"{base_name}_validation_critrfc"
    
    output_csv = os.path.join(results_dir, f"{output_prefix}.csv")
    
    # Construct command
    cmd_parts = [
        "python", script_path,
        "--ground-truth", ground_truth_file,
        "--runs", str(runs),
        "--output", output_csv
    ]
    
    if "RFC" in version_name and rfc_facts_file and os.path.exists(rfc_facts_file):
        cmd_parts.extend(["--rfc-facts", rfc_facts_file])
    
    # Run the script
    try:
        result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"Completed {version_name}")
            return True
        else:
            print(f"Failed {version_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout for {version_name}")
        return False
    except Exception as e:
        print(f"Error running {version_name}: {e}")
        return False

def run_all_validations(ground_truth_file, rfc_facts_file, results_dir, runs=5):
    """Run all validation scripts with specified number of runs."""
    scripts = {
        "Original": "scripts/llm_filter_edited_validation.py",
        "Critical": "scripts/llm_filter_edited_validation_crit.py", 
        "Original+RFC": "scripts/llm_filter_edited_validation_orirfc.py",
        "Critical+RFC": "scripts/llm_filter_edited_validation_critrfc.py"
    }
    
    expected_files = get_expected_files(results_dir, runs)
    
    for version, script_path in scripts.items():
        # Check if we need to run this validation
        variance_file = expected_files[version]["variance"]
        need_to_run = True
        
        if os.path.exists(variance_file):
            if check_file_has_sufficient_runs(variance_file, runs):
                print(f"Sufficient results already exist for {version} ({runs} runs)")
                need_to_run = False
            else:
                print(f"Insufficient runs for {version}, re-running with {runs} runs")
        
        if need_to_run:
            success = run_validation_script(
                script_path, ground_truth_file, rfc_facts_file, 
                results_dir, runs, version
            )
            if not success:
                print(f"Failed to run {version}")

def load_results_from_directory(results_dir, runs=5):
    """Load results from the specified directory."""
    expected_files = get_expected_files(results_dir, runs)
    results = {}
    
    for version, file_paths in expected_files.items():
        # Try variance file first, then accuracy file
        result = None
        for file_type in ["variance", "accuracy"]:
            filepath = file_paths[file_type]
            if os.path.exists(filepath):
                result = extract_accuracy_from_file(filepath)
                if result:
                    print(f"Loaded {version} from {file_type} file ({result['num_runs']} runs)")
                    break
        
        if result:
            results[version] = result
        else:
            print(f"No valid results found for {version}")
    
    return results

def create_prompt_evolution_chart(results, output_file="results/prompt_evolution_performance.png"):
    """Create line chart showing prompt evolution performance."""
    
    # Define the order of prompt versions
    version_order = ["Original", "Critical", "Original+RFC", "Critical+RFC"]
    
    # Filter and order results
    ordered_results = {}
    for version in version_order:
        if version in results:
            ordered_results[version] = results[version]
    
    if len(ordered_results) < 2:
        print("Need at least 2 prompt versions to create evolution chart")
        return None
    
    # Prepare data
    versions = list(ordered_results.keys())
    mean_accuracies = [ordered_results[v]['mean_accuracy'] for v in versions]
    std_accuracies = [ordered_results[v]['std_accuracy'] for v in versions]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    x_positions = range(len(versions))
    
    # Plot mean accuracy line
    plt.errorbar(x_positions, mean_accuracies, yerr=std_accuracies, 
                marker='o', linewidth=2.5, markersize=8, capsize=5, capthick=2,
                label='Mean Accuracy ± Std Dev', color='#2E86AB')
    
    # Plot standard deviation as a separate line
    plt.plot(x_positions, std_accuracies, 
            marker='s', linewidth=2, markersize=6, linestyle='--',
            label='Standard Deviation', color='#A23B72')
    
    # Customize the plot
    plt.xlabel('Prompt Evolution', fontsize=12, fontweight='bold')
    plt.ylabel('Performance (%)', fontsize=12, fontweight='bold')
    plt.title('Prompt Evolution Performance\nAccuracy vs Consistency Trade-off', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    plt.xticks(x_positions, versions, rotation=45, ha='right')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Add annotations for key insights
    if len(mean_accuracies) >= 2:
        accuracy_change = mean_accuracies[-1] - mean_accuracies[0]
        consistency_change = std_accuracies[0] - std_accuracies[-1]  # Lower std is better
        
        plt.figtext(0.02, 0.02, 
                   f"Accuracy Change: {accuracy_change:+.1f}%\n"
                   f"Consistency Improvement: {consistency_change:+.1f}% std",
                   fontsize=10, ha='left', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved prompt evolution chart to: {output_file}")
    
    return plt

def print_detailed_analysis(results):
    """Print detailed analysis of the prompt evolution."""
    print(f"\n=== DETAILED ANALYSIS ===")
    
    version_order = ["Original", "Critical", "Original+RFC", "Critical+RFC"]
    ordered_results = {v: results[v] for v in version_order if v in results}
    
    if len(ordered_results) < 2:
        return
    
    versions = list(ordered_results.keys())
    
    # Trade-off analysis
    print(f"\nACCURACY PROGRESSION:")
    base_accuracy = ordered_results[versions[0]]['mean_accuracy']
    for version in versions:
        accuracy = ordered_results[version]['mean_accuracy']
        change = accuracy - base_accuracy
        runs = ordered_results[version]['num_runs']
        print(f"  {version:15}: {accuracy:5.1f}% ({change:+5.1f}%) [{runs} runs]")
    
    print(f"\nCONSISTENCY ANALYSIS:")
    for version in versions:
        std = ordered_results[version]['std_accuracy']
        runs = ordered_results[version]['num_runs']
        print(f"  {version:15}: ±{std:4.1f}% std dev ({runs} runs)")
    
    # Best performer
    best_accuracy = max(ordered_results.values(), key=lambda x: x['mean_accuracy'])
    best_consistency = min(ordered_results.values(), key=lambda x: x['std_accuracy'])
    
    best_acc_version = [v for v, r in ordered_results.items() if r['mean_accuracy'] == best_accuracy['mean_accuracy']][0]
    best_cons_version = [v for v, r in ordered_results.items() if r['std_accuracy'] == best_consistency['std_accuracy']][0]
    
    print(f"\nBEST PERFORMERS:")
    print(f"  Highest Accuracy: {best_acc_version} ({best_accuracy['mean_accuracy']:.1f}%)")
    print(f"  Most Consistent:  {best_cons_version} (±{best_consistency['std_accuracy']:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze prompt evolution performance across validation scripts")
    parser.add_argument("--results-dir", "-d", default="results", 
                       help="Directory for result files (default: results)")
    parser.add_argument("--output", "-o", default="results/prompt_evolution_performance.png",
                       help="Output filename for the chart")
    parser.add_argument("--ground-truth", "-g", required=True,
                       help="Ground truth file (required)")
    parser.add_argument("--rfc-facts", "-f", default="rfc_facts.json",
                       help="RFC facts file (default: rfc_facts.json)")
    parser.add_argument("--runs", "-r", type=int, default=5,
                       help="Number of runs for validations (default: 5)")
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    results_dir = ensure_results_dir(args.results_dir)
    
    print(f"Using results directory: {results_dir}")
    print(f"Target runs: {args.runs}")
    
    # Run all validations (will skip if sufficient results exist)
    run_all_validations(args.ground_truth, args.rfc_facts, results_dir, args.runs)
    
    # Load results
    results = load_results_from_directory(results_dir, args.runs)
    
    if not results:
        print("No valid results found. Check that validation scripts ran successfully.")
        return
    
    print(f"Loaded results for {len(results)} prompt versions")
    
    # Create the chart
    plt_obj = create_prompt_evolution_chart(results, args.output)
    if plt_obj is None:
        return
    
    # Print detailed analysis
    print_detailed_analysis(results)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()