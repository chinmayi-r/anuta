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

# Create results directory
RESULTS_DIR = "rfc_results"

def ensure_results_dir():
    """Ensure the results directory exists."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR

def find_result_files(results_dir):
    """Find result files for each prompt version in the specified directory."""
    patterns = {
        "Original": "cidds_queries_validation_results_*.json",
        "Critical": "cidds_queries_validation_crit_*.json", 
        "Original+RFC": "cidds_queries_validation_orirfc_*.json",
        "Critical+RFC": "cidds_queries_validation_critrfc_*.json"
    }
    
    found_files = {}
    for version, pattern in patterns.items():
        files = glob.glob(os.path.join(results_dir, pattern))
        if files:
            found_files[version] = files
        else:
            print(f"No files found for {version} with pattern: {pattern}")
    
    return found_files

def check_file_runs(filepath, required_runs):
    """Check if a file has sufficient runs."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Count runs in the file
        if 'individual_run_accuracies' in data:
            file_runs = len(data['individual_run_accuracies'])
        elif 'accuracy_percentage' in data:
            file_runs = 1
        else:
            file_runs = 0
        
        return file_runs >= required_runs, file_runs
        
    except Exception as e:
        print(f"Error checking {filepath}: {e}")
        return False, 0

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
                'accuracies': data.get('individual_run_accuracies', [])
            }
        elif 'accuracy_percentage' in data:
            # Single run accuracy report format
            return {
                'mean_accuracy': data['accuracy_percentage'],
                'std_accuracy': 0.0,
                'accuracies': [data['accuracy_percentage']]
            }
        else:
            print(f"Unrecognized format in {filepath}")
            return None
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def load_results_from_files(result_files):
    """Load and aggregate results from all found files."""
    results = {}
    
    for version, files in result_files.items():
        version_results = []
        
        for filepath in files:
            result = extract_accuracy_from_file(filepath)
            if result:
                version_results.append(result)
        
        if version_results:
            # Aggregate multiple files if they exist
            all_accuracies = []
            for result in version_results:
                all_accuracies.extend(result['accuracies'])
            
            if all_accuracies:
                results[version] = {
                    'mean_accuracy': statistics.mean(all_accuracies),
                    'std_accuracy': statistics.stdev(all_accuracies) if len(all_accuracies) > 1 else 0.0,
                    'num_runs': len(all_accuracies),
                    'raw_accuracies': all_accuracies
                }
        
        if version not in results:
            print(f"No valid results found for {version}")
    
    return results

def run_validation(script_path, ground_truth_file, rfc_facts_file, runs, output_prefix):
    """Run a single validation script."""
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return False
        
    print(f"Running validation: {script_path}")
    
    # Construct command
    cmd_parts = [
        "python", script_path,
        "--ground-truth", ground_truth_file,
        "--runs", str(runs),
        "--output", os.path.join(RESULTS_DIR, f"{output_prefix}.csv"),
        "--variance-output", os.path.join(RESULTS_DIR, f"{output_prefix}_variance.json")
    ]
    
    if "rfc" in script_path.lower() and rfc_facts_file and os.path.exists(rfc_facts_file):
        cmd_parts.extend(["--rfc-facts", rfc_facts_file])
    
    # Run the script
    try:
        result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"Completed: {script_path}")
            return True
        else:
            print(f"Failed {script_path}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout for {script_path}")
        return False
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return False

def run_all_validations(ground_truth_file, rfc_facts_file, runs, force_rerun=False):
    """Run validation scripts for all prompt versions."""
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(ground_truth_file))[0]
    
    scripts_and_outputs = [
        ("scripts/llm_filter_edited_validation.py", f"{base_name}_validation_results"),
        ("scripts/llm_filter_edited_validation_crit.py", f"{base_name}_validation_crit"),
        ("scripts/llm_filter_edited_validation_orirfc.py", f"{base_name}_validation_orirfc"),
        ("scripts/llm_filter_edited_validation_critrfc.py", f"{base_name}_validation_critrfc")
    ]
    
    results_dir = ensure_results_dir()
    
    for script_path, output_prefix in scripts_and_outputs:
        variance_file = os.path.join(results_dir, f"{output_prefix}_variance.json")
        
        # Check if we need to run this validation
        should_run = force_rerun
        
        if not should_run and os.path.exists(variance_file):
            # Check if existing file has enough runs
            has_enough_runs, existing_runs = check_file_runs(variance_file, runs)
            if not has_enough_runs:
                print(f"Existing file has {existing_runs} runs, need {runs}. Re-running...")
                should_run = True
            else:
                print(f"Using existing results for {output_prefix} ({existing_runs} runs)")
        else:
            should_run = True
        
        if should_run:
            success = run_validation(script_path, ground_truth_file, rfc_facts_file, runs, output_prefix)
            if not success:
                print(f"Failed to run {script_path}")

def create_prompt_evolution_chart(results, output_file="rfc_results/prompt_evolution_performance.png"):
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
        return
    
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
    
    # Save the plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved prompt evolution chart to: {output_file}")
    
    # Show summary statistics
    print(f"\n=== PROMPT EVOLUTION SUMMARY ===")
    for version in versions:
        result = ordered_results[version]
        print(f"{version:15}: {result['mean_accuracy']:5.1f}% (±{result['std_accuracy']:4.1f}%) [{result['num_runs']} runs]")
    
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
        print(f"  {version:15}: {accuracy:5.1f}% ({change:+5.1f}%)")
    
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
    parser.add_argument("--ground-truth", "-g", required=True,
                       help="Ground truth JSON file")
    parser.add_argument("--rfc-facts", "-f", default="rfc_facts.json",
                       help="RFC facts file (default: rfc_facts.json)")
    parser.add_argument("--runs", "-r", type=int, default=5,
                       help="Number of runs for validations (default: 5)")
    parser.add_argument("--output", "-o", default="rfc_results/prompt_evolution_performance.png",
                       help="Output filename for the chart (default: rfc_results/prompt_evolution_performance.png)")
    parser.add_argument("--force-rerun", action="store_true",
                       help="Force re-run all validations even if results exist")
    parser.add_argument("--use-existing", action="store_true",
                       help="Use existing results even if they have fewer runs than requested")
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    results_dir = ensure_results_dir()
    print(f"Using results directory: {results_dir}")
    
    # Run all validations (only if needed)
    if not args.use_existing:
        print(f"Running validations with {args.runs} runs each...")
        run_all_validations(args.ground_truth, args.rfc_facts, args.runs, args.force_rerun)
    else:
        print("Using existing results regardless of run count...")
    
    # Search for result files
    print("Searching for result files...")
    result_files = find_result_files(results_dir)
    
    # Load results
    results = load_results_from_files(result_files)
    
    if not results:
        print("No valid results found. Check that validation scripts completed successfully.")
        return
    
    print(f"Loaded results for {len(results)} prompt versions")
    
    # Create the chart
    create_prompt_evolution_chart(results, args.output)
    
    # Print detailed analysis
    print_detailed_analysis(results)
    
    plt.show()

if __name__ == "__main__":
    main()