import os
import json
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuration
INPUT_FILES = [
    "rules/new/dt_cidds_1000000.txt",
    "rules/new/fpgrowth_cidds_all.txt", 
    "rules/new/hmine_cidds_all.txt",
    "rules/new/lgbm_cidds_all.txt",
    "rules/new/xgb_cidds_all.txt"
]

SCRIPTS = {
    "original": "scripts/llm_filter_edited.py",
    "critical": "scripts/llm_filter_edited copy.py"
}

# Extract readable names from file paths
def get_input_name(filepath):
    """Extract a clean name from the input file path"""
    basename = os.path.basename(filepath)
    name = basename.replace("_cidds_all.txt", "").replace("_cidds_1000000.txt", "")
    return name.replace("_", " ").title()

def run_analysis(script_path, input_file, output_dir):
    """Run the LLM filter script and return the variance report path"""
    input_name = os.path.basename(input_file).replace(".txt", "")
    script_name = "original" if "copy" not in script_path else "critical"
    
    variance_output = os.path.join(output_dir, f"{input_name}_{script_name}_variance.json")
    
    # Construct command
    cmd = [
        "python", script_path,
        "-i", input_file,
        "-m", "tokens",
        "-r", "3",
        "--variance-output", variance_output
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Completed {script_name} analysis for {input_name}")
        return variance_output
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name} on {input_name}: {e}")
        print(f"stderr: {e.stderr}")
        return None

def load_variance_data(filepath):
    """Load variance statistics from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_metrics(variance_data):
    """Extract key metrics from variance data"""
    if not variance_data:
        return {}
    
    metrics = {}
    
    # Meaningful classification metrics
    meaningful = variance_data.get('meaningful_classification', {})
    metrics['meaningful_mean'] = meaningful.get('mean_rate', 0)
    metrics['meaningful_std'] = meaningful.get('std_rate', 0)
    
    # Rule type classification metrics
    rule_types = variance_data.get('rule_type_classification', {})
    for rtype in ['protocol', 'principle', 'deployment']:
        if rtype in rule_types:
            percentages = rule_types[rtype].get('percentages', {})
            metrics[f'{rtype}_mean'] = percentages.get('mean', 0)
            metrics[f'{rtype}_std'] = percentages.get('std', 0)
    
    return metrics

def create_comparison_plots(results_df):
    """Create comprehensive comparison plots with error bars"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM Filter: Original vs Critical Prompt Comparison', fontsize=16, fontweight='bold')
    
    # Define colors for each input type
    input_types = results_df['input_name'].unique()
    x_pos = np.arange(len(input_types))
    width = 0.35
    
    # Helper function to extract means and stds
    def get_means_and_stds(metric_name):
        original_means = []
        original_stds = []
        critical_means = []
        critical_stds = []
        
        for inp in input_types:
            # Original data
            orig_data = results_df[(results_df['input_name'] == inp) & 
                                 (results_df['prompt_type'] == 'original')]
            if len(orig_data) > 0:
                original_means.append(orig_data[f'{metric_name}_mean'].iloc[0])
                original_stds.append(orig_data[f'{metric_name}_std'].iloc[0])
            else:
                original_means.append(0)
                original_stds.append(0)
            
            # Critical data
            crit_data = results_df[(results_df['input_name'] == inp) & 
                                 (results_df['prompt_type'] == 'critical')]
            if len(crit_data) > 0:
                critical_means.append(crit_data[f'{metric_name}_mean'].iloc[0])
                critical_stds.append(crit_data[f'{metric_name}_std'].iloc[0])
            else:
                critical_means.append(0)
                critical_stds.append(0)
                
        return original_means, original_stds, critical_means, critical_stds
    
    # 1. Meaningful Classification Rate
    ax1 = axes[0, 0]
    orig_means, orig_stds, crit_means, crit_stds = get_means_and_stds('meaningful')
    
    bars1 = ax1.bar(x_pos - width/2, orig_means, width, label='Original', alpha=0.8, 
                   yerr=orig_stds, capsize=5)
    bars2 = ax1.bar(x_pos + width/2, crit_means, width, label='Critical', alpha=0.8,
                   yerr=crit_stds, capsize=5)
    
    ax1.set_xlabel('Input Type')
    ax1.set_ylabel('Meaningful Classification Rate')
    ax1.set_title('Meaningful Classification Rate Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(input_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rule Type Distribution - Protocol
    ax2 = axes[0, 1]
    orig_means, orig_stds, crit_means, crit_stds = get_means_and_stds('protocol')
    
    bars1 = ax2.bar(x_pos - width/2, orig_means, width, label='Original', alpha=0.8,
                   yerr=orig_stds, capsize=5)
    bars2 = ax2.bar(x_pos + width/2, crit_means, width, label='Critical', alpha=0.8,
                   yerr=crit_stds, capsize=5)
    
    ax2.set_xlabel('Input Type')
    ax2.set_ylabel('Protocol Classification Rate')
    ax2.set_title('Protocol Rules Classification Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(input_types, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rule Type Distribution - Principle
    ax3 = axes[1, 0]
    orig_means, orig_stds, crit_means, crit_stds = get_means_and_stds('principle')
    
    bars1 = ax3.bar(x_pos - width/2, orig_means, width, label='Original', alpha=0.8,
                   yerr=orig_stds, capsize=5)
    bars2 = ax3.bar(x_pos + width/2, crit_means, width, label='Critical', alpha=0.8,
                   yerr=crit_stds, capsize=5)
    
    ax3.set_xlabel('Input Type')
    ax3.set_ylabel('Principle Classification Rate')
    ax3.set_title('Principle Rules Classification Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(input_types, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Rule Type Distribution - Deployment
    ax4 = axes[1, 1]
    orig_means, orig_stds, crit_means, crit_stds = get_means_and_stds('deployment')
    
    bars1 = ax4.bar(x_pos - width/2, orig_means, width, label='Original', alpha=0.8,
                   yerr=orig_stds, capsize=5)
    bars2 = ax4.bar(x_pos + width/2, crit_means, width, label='Critical', alpha=0.8,
                   yerr=crit_stds, capsize=5)
    
    ax4.set_xlabel('Input Type')
    ax4.set_ylabel('Deployment Classification Rate')
    ax4.set_title('Deployment Rules Classification Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(input_types, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def load_existing_results(output_dir):
    """Load existing results from CSV if available"""
    results_csv = os.path.join(output_dir, "comparison_results.csv")
    if os.path.exists(results_csv):
        print(f"📂 Loading existing results from {results_csv}")
        return pd.read_csv(results_csv)
    return None

def check_existing_variance_files(output_dir):
    """Check which variance files already exist"""
    existing_files = {}
    for input_file in INPUT_FILES:
        input_name = os.path.basename(input_file).replace(".txt", "")
        for prompt_type in ["original", "critical"]:
            variance_file = os.path.join(output_dir, f"{input_name}_{prompt_type}_variance.json")
            if os.path.exists(variance_file):
                existing_files[(input_file, prompt_type)] = variance_file
    return existing_files

def main():
    # Create output directory
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting LLM Filter Comparison Analysis...")
    
    # First, try to load existing results
    existing_results_df = load_existing_results(output_dir)
    
    if existing_results_df is not None:
        print("✅ Found existing results! Skipping API calls.")
        results_df = existing_results_df
    else:
        print(f"Will analyze {len(INPUT_FILES)} input files with {len(SCRIPTS)} different prompts")
        
        # Check for existing variance files
        existing_files = check_existing_variance_files(output_dir)
        if existing_files:
            print(f"📂 Found {len(existing_files)} existing variance files")
        
        # Run analyses only for missing files
        all_results = []
        
        for input_file in INPUT_FILES:
            if not os.path.exists(input_file):
                print(f"⚠️  Input file not found: {input_file}")
                continue
                
            input_name = get_input_name(input_file)
            print(f"\n--- Processing {input_name} ---")
            
            for prompt_type, script_path in SCRIPTS.items():
                if not os.path.exists(script_path):
                    print(f"⚠️  Script not found: {script_path}")
                    continue
                
                # Check if we already have results for this combination
                if (input_file, prompt_type) in existing_files:
                    print(f"✅ Using existing {prompt_type} results for {input_name}")
                    variance_file = existing_files[(input_file, prompt_type)]
                else:
                    # Run analysis
                    variance_file = run_analysis(script_path, input_file, output_dir)
                
                if variance_file and os.path.exists(variance_file):
                    # Load and extract metrics
                    variance_data = load_variance_data(variance_file)
                    metrics = extract_metrics(variance_data)
                    
                    # Store results
                    result = {
                        'input_file': input_file,
                        'input_name': input_name,
                        'prompt_type': prompt_type,
                        **metrics
                    }
                    all_results.append(result)
        
        if not all_results:
            print("❌ No results collected. Check file paths and permissions.")
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results to CSV
        results_csv = os.path.join(output_dir, "comparison_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\n📊 Results saved to: {results_csv}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Check if we have both original and critical results
    prompt_types = results_df['prompt_type'].unique()
    if 'original' not in prompt_types or 'critical' not in prompt_types:
        print("⚠️  Need both 'original' and 'critical' results for comparison.")
        print(f"Available prompt types: {list(prompt_types)}")
        # Still create visualization with available data
    
    # Main comparison plot with error bars
    fig = create_comparison_plots(results_df)
    fig.savefig(os.path.join(output_dir, "prompt_comparison_with_errorbars.png"), 
                dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print("="*50)
    
    for input_name in results_df['input_name'].unique():
        print(f"\n{input_name}:")
        input_data = results_df[results_df['input_name'] == input_name]
        
        original_data = input_data[input_data['prompt_type'] == 'original']
        critical_data = input_data[input_data['prompt_type'] == 'critical']
        
        if len(original_data) > 0 and len(critical_data) > 0:
            print(f"  Meaningful Rate: {original_data['meaningful_mean'].iloc[0]:.3f}±{original_data['meaningful_std'].iloc[0]:.3f} → " +
                  f"{critical_data['meaningful_mean'].iloc[0]:.3f}±{critical_data['meaningful_std'].iloc[0]:.3f} " +
                  f"(Δ{critical_data['meaningful_mean'].iloc[0] - original_data['meaningful_mean'].iloc[0]:+.3f})")
            print(f"  Protocol Rate:   {original_data['protocol_mean'].iloc[0]:.3f}±{original_data['protocol_std'].iloc[0]:.3f} → " + 
                  f"{critical_data['protocol_mean'].iloc[0]:.3f}±{critical_data['protocol_std'].iloc[0]:.3f} " +
                  f"(Δ{critical_data['protocol_mean'].iloc[0] - original_data['protocol_mean'].iloc[0]:+.3f})")
            print(f"  Principle Rate:  {original_data['principle_mean'].iloc[0]:.3f}±{original_data['principle_std'].iloc[0]:.3f} → " +
                  f"{critical_data['principle_mean'].iloc[0]:.3f}±{critical_data['principle_std'].iloc[0]:.3f} " +
                  f"(Δ{critical_data['principle_mean'].iloc[0] - original_data['principle_mean'].iloc[0]:+.3f})")
            print(f"  Deployment Rate: {original_data['deployment_mean'].iloc[0]:.3f}±{original_data['deployment_std'].iloc[0]:.3f} → " +
                  f"{critical_data['deployment_mean'].iloc[0]:.3f}±{critical_data['deployment_std'].iloc[0]:.3f} " +
                  f"(Δ{critical_data['deployment_mean'].iloc[0] - original_data['deployment_mean'].iloc[0]:+.3f})")
        elif len(original_data) > 0:
            print(f"  [Original only] Meaningful: {original_data['meaningful_mean'].iloc[0]:.3f}±{original_data['meaningful_std'].iloc[0]:.3f}")
        elif len(critical_data) > 0:
            print(f"  [Critical only] Meaningful: {critical_data['meaningful_mean'].iloc[0]:.3f}±{critical_data['meaningful_std'].iloc[0]:.3f}")
    
    print(f"\n✅ Analysis complete! Check the '{output_dir}' directory for all outputs.")

if __name__ == "__main__":
    main()