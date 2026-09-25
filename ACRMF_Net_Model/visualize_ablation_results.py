"""
Visualize Ablation Study Results
=================================

Creates comparison plots and tables from ablation study results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_ablation_results(output_dir="experiments/ablation_study"):
    """Load all ablation study results"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Directory not found: {output_path}")
        return None
    
    summary_file = output_path / "ablation_summary.json"
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    return None


def create_comparison_table(results):
    """Create a comparison table of all experiments"""
    
    data = []
    for exp in results['experiments']:
        data.append({
            'Experiment': exp['experiment'],
            'REN': 'Yes' if exp['components']['REN'] else 'No',
            'AWG': 'Yes' if exp['components']['AWG'] else 'No',
            'CEN': 'Yes' if exp['components']['CEN'] else 'No',
            'Test Accuracy (%)': f"{exp['test_accuracy']:.2f}",
            'Test F1': f"{exp['test_f1']:.4f}",
            'Test Precision': f"{exp['test_precision']:.4f}",
            'Test Recall': f"{exp['test_recall']:.4f}"
        })
    
    df = pd.DataFrame(data)
    return df


def plot_ablation_comparison(results, output_dir="experiments/ablation_study"):
    """Create comparison plots"""
    
    experiments = results['experiments']
    
    # Extract data
    names = [exp['experiment'] for exp in experiments]
    test_acc = [exp['test_accuracy'] for exp in experiments]
    test_f1 = [exp['test_f1'] * 100 for exp in experiments]  # Scale to %
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Test Accuracy
    colors = ['green' if 'Full' in name else 'orange' if 'Without' in name and '&' in name else 'blue' if 'Without' in name else 'red' for name in names]
    
    ax1.barh(range(len(names)), test_acc, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels([name[:35] + '...' if len(name) > 35 else name for name in names], fontsize=9)
    ax1.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Ablation Study - Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(test_acc):
        ax1.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=9)
    
    # Plot 2: Test F1 Score
    ax2.barh(range(len(names)), test_f1, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([name[:35] + '...' if len(name) > 35 else name for name in names], fontsize=9)
    ax2.set_xlabel('Test F1 Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Ablation Study - Test F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(test_f1):
        ax2.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path / 'ablation_comparison.png'}")
    
    plt.show()


def plot_component_impact(results, output_dir="experiments/ablation_study"):
    """Analyze the impact of each component"""
    
    experiments = results['experiments']
    
    # Find full model and variants
    full_model = next(exp for exp in experiments if exp['components']['REN'] and 
                     exp['components']['AWG'] and exp['components']['CEN'])
    
    # Calculate impact of removing each component
    impacts = {}
    
    for exp in experiments:
        comp = exp['components']
        if sum(comp.values()) == 2:  # Two components active
            missing = [k for k, v in comp.items() if not v][0]
            impacts[f'Without {missing}'] = full_model['test_f1'] - exp['test_f1']
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = list(impacts.keys())
    impact_values = [impacts[c] * 100 for c in components]  # Convert to percentage points
    
    colors = ['red' if v > 0 else 'green' for v in impact_values]
    
    ax.bar(components, impact_values, color=colors, alpha=0.7)
    ax.set_ylabel('Performance Drop (percentage points)', fontsize=12, fontweight='bold')
    ax.set_title('Component Impact Analysis\n(Performance drop when component is removed)', 
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(impact_values):
        ax.text(i, v + 0.5 if v > 0 else v - 0.5, f'{v:.2f}pp', 
               ha='center', va='bottom' if v > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    plt.savefig(output_path / 'component_impact.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path / 'component_impact.png'}")
    
    plt.show()


def main():
    print("="*80)
    print("ABLATION STUDY VISUALIZATION")
    print("="*80)
    
    # Load results
    results = load_ablation_results()
    
    if results is None:
        print("No ablation study results found. Run ablation_study.py first.")
        return
    
    print(f"\nLoaded {len(results['experiments'])} experiments")
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    df = create_comparison_table(results)
    print(df.to_string(index=False))
    
    # Save table to CSV
    output_path = Path("experiments/ablation_study")
    df.to_csv(output_path / 'ablation_comparison.csv', index=False)
    print(f"\nTable saved to: {output_path / 'ablation_comparison.csv'}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_ablation_comparison(results)
    plot_component_impact(results)
    
    print("\nVisualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
