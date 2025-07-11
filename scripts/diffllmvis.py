import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from matplotlib.patches import Rectangle

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_comprehensive_report(report_file):
    """Load the comprehensive report JSON file."""
    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_comparison_csv(csv_file):
    """Load the model comparison CSV file."""
    return pd.read_csv(csv_file)

def create_accuracy_comparison_plot(report_data, output_dir):
    """Create bar plot comparing mean accuracy across models with error bars."""
    models = []
    mean_accuracies = []
    std_devs = []
    
    for model_name, performance in report_data['model_performance'].items():
        if 'accuracy_stats' in performance and performance['accuracy_stats']['mean'] > 0:
            models.append(model_name)
            mean_accuracies.append(performance['accuracy_stats']['mean'] * 100)
            std_devs.append(performance['accuracy_stats']['std'] * 100)
    
    # Sort by mean accuracy
    sorted_data = sorted(zip(models, mean_accuracies, std_devs), key=lambda x: x[1], reverse=True)
    models, mean_accuracies, std_devs = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot with error bars
    bars = ax.bar(models, mean_accuracies, yerr=std_devs, capsize=5, 
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Color bars based on performance
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('LLM Model Accuracy Comparison\n(with Standard Deviation)', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, mean_acc, std_dev) in enumerate(zip(bars, mean_accuracies, std_devs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_dev + 0.5,
                f'{mean_acc:.1f}%\n±{std_dev:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_variance_comparison_plot(report_data, output_dir):
    """Create plot showing accuracy variance across models."""
    models = []
    means = []
    mins = []
    maxes = []
    stds = []
    
    for model_name, performance in report_data['model_performance'].items():
        if 'accuracy_stats' in performance and performance['accuracy_stats']['mean'] > 0:
            stats = performance['accuracy_stats']
            models.append(model_name)
            means.append(stats['mean'] * 100)
            mins.append(stats['min'] * 100)
            maxes.append(stats['max'] * 100)
            stds.append(stats['std'] * 100)
    
    # Sort by mean accuracy
    sorted_data = sorted(zip(models, means, mins, maxes, stds), key=lambda x: x[1], reverse=True)
    models, means, mins, maxes, stds = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(models))
    
    # Plot error bars showing min-max range
    ax.errorbar(x_pos, means, yerr=[np.array(means) - np.array(mins), 
                                   np.array(maxes) - np.array(means)], 
                fmt='o', capsize=5, capthick=2, markersize=8, linewidth=2)
    
    # Add mean points
    ax.scatter(x_pos, means, s=100, c='red', zorder=5, label='Mean Accuracy')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Accuracy Variance\n(Error bars show min-max range)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mean_val, std_val) in enumerate(zip(means, stds)):
        ax.text(i, mean_val + 1, f'{mean_val:.1f}%\n(σ={std_val:.1f}%)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_variance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_token_usage_plot(report_data, output_dir):
    """Create plot showing token usage across models."""
    if 'token_usage' not in report_data or not report_data['token_usage']:
        print("No token usage data available for plotting.")
        return
    
    models = []
    total_tokens = []
    avg_tokens = []
    requests = []
    
    for model_name, usage in report_data['token_usage'].items():
        models.append(model_name)
        total_tokens.append(usage['total_tokens'])
        requests.append(usage['requests'])
        avg_tokens.append(usage['total_tokens'] / usage['requests'] if usage['requests'] > 0 else 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Total tokens used
    bars1 = ax1.bar(models, total_tokens, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Total Tokens Used', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_title('Total Token Usage by Model', fontsize=14, fontweight='bold')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, total in zip(bars1, total_tokens):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{total:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Average tokens per request
    bars2 = ax2.bar(models, avg_tokens, alpha=0.8, color='orange', edgecolor='black')
    ax2.set_ylabel('Average Tokens per Request', fontsize=12)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_title('Average Token Usage per Request', fontsize=14, fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, avg in zip(bars2, avg_tokens):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,