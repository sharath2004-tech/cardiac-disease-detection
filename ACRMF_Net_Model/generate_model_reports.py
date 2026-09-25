"""
Comprehensive Model Performance Report Generator
=================================================
Generates detailed PDF/HTML reports for multiple models with:
- Step-by-step training analysis
- Performance metrics and visualizations
- Comparison between models
- All plots and charts included

Usage:
    py generate_model_reports.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score, 
                             accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import torch

sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*80)
print("COMPREHENSIVE MODEL PERFORMANCE REPORT GENERATOR")
print("="*80)


class ModelReportGenerator:
    """Generate comprehensive reports for trained models"""
    
    def __init__(self, report_dir='reports'):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_report_for_model(self, model_name, model_path, data_path='cleaned_data'):
        """Generate complete report for a single model"""
        
        print(f"\n{'='*80}")
        print(f"GENERATING REPORT FOR: {model_name}")
        print(f"{'='*80}\n")
        
        # Create model-specific directory
        model_report_dir = self.report_dir / f"{model_name}_{self.timestamp}"
        model_report_dir.mkdir(exist_ok=True)
        
        report_data = {
            'model_name': model_name,
            'timestamp': self.timestamp,
            'plots': [],
            'metrics': {}
        }
        
        # Step 1: Load Model and Data
        print("Step 1: Loading model and data...")
        model_info = self._load_model_info(model_path)
        data_info = self._load_data_info(data_path)
        report_data['model_info'] = model_info
        report_data['data_info'] = data_info
        
        # Step 2: Analyze Training History
        print("Step 2: Analyzing training history...")
        if model_info and 'history' in model_info:
            plot_path = self._plot_training_history(model_info['history'], model_report_dir)
            report_data['plots'].append(('Training History', plot_path))
        
        # Step 3: Performance Metrics
        print("Step 3: Computing performance metrics...")
        if model_info and 'metrics' in model_info:
            metrics = model_info['metrics']
            report_data['metrics'] = metrics
            
            # Plot confusion matrix only if it exists
            if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
                plot_path = self._plot_confusion_matrix(metrics, model_report_dir)
                if plot_path:
                    report_data['plots'].append(('Confusion Matrix', plot_path))
            
            # Plot per-class metrics
            plot_path = self._plot_per_class_metrics(metrics, model_report_dir)
            if plot_path:
                report_data['plots'].append(('Per-Class Performance', plot_path))
        
        # Step 4: ROC Curves
        print("Step 4: Generating ROC curves...")
        if model_info and 'roc_data' in model_info:
            plot_path = self._plot_roc_curves(model_info['roc_data'], model_report_dir)
            report_data['plots'].append(('ROC Curves', plot_path))
        
        # Step 5: Learning Rate Analysis
        print("Step 5: Analyzing learning rate schedule...")
        if model_info and 'lr_schedule' in model_info:
            plot_path = self._plot_lr_schedule(model_info['lr_schedule'], model_report_dir)
            report_data['plots'].append(('Learning Rate Schedule', plot_path))
        
        # Step 6: Class Distribution
        print("Step 6: Analyzing class distribution...")
        if data_info and 'labels' in data_info:
            plot_path = self._plot_class_distribution(data_info['labels'], model_report_dir)
            report_data['plots'].append(('Class Distribution', plot_path))
        
        # Step 7: Generate Summary Report
        print("Step 7: Generating summary report...")
        self._generate_summary_report(report_data, model_report_dir)
        
        # Step 8: Generate HTML Report
        print("Step 8: Generating HTML report...")
        html_path = self._generate_html_report(report_data, model_report_dir)
        
        print(f"\n[OK] Report generated: {model_report_dir}/")
        print(f"     HTML Report: {html_path}")
        
        return report_data
    
    def _load_model_info(self, model_path):
        """Load model checkpoint and extract information"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"  [WARN] Model not found: {model_path}")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            val_metrics = checkpoint.get('val_metrics', {})
            history = checkpoint.get('history', {})
            
            # Extract per-class accuracy from val_metrics
            per_class_acc = {}
            if 'per_class_accuracy' in val_metrics:
                per_class_acc_list = val_metrics['per_class_accuracy']
                per_class_acc = {i: float(acc)/100 for i, acc in enumerate(per_class_acc_list)}
            
            # Extract confusion matrix if available
            confusion_matrix = None
            if 'confusion_matrix' in val_metrics:
                confusion_matrix = val_metrics['confusion_matrix']
            
            info = {
                'path': str(model_path),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'train_acc': checkpoint.get('train_acc', 0),
                'history': history,
                'metrics': {
                    'val_acc': val_metrics.get('accuracy', 0) / 100,  # Convert from percentage
                    'val_f1': val_metrics.get('macro_f1', 0),
                    'val_weighted_f1': val_metrics.get('weighted_f1', 0),
                    'val_precision': val_metrics.get('macro_precision', 0),
                    'val_recall': val_metrics.get('macro_recall', 0),
                    'per_class_acc': per_class_acc,
                    'per_class_f1': val_metrics.get('per_class_f1', []),
                    'per_class_precision': val_metrics.get('per_class_precision', []),
                    'per_class_recall': val_metrics.get('per_class_recall', []),
                    'confusion_matrix': confusion_matrix,
                }
            }
            
            print(f"  [OK] Model loaded: Epoch {info['epoch']}, Val Acc: {info['metrics']['val_acc']*100:.2f}%")
            return info
            
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            return None
    
    def _load_data_info(self, data_path):
        """Load dataset information"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            print(f"  [WARN] Data directory not found: {data_path}")
            return None
        
        try:
            labels = np.load(data_path / 'labels_cleaned.npy')
            
            with open(data_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            info = {
                'num_samples': len(labels),
                'num_classes': metadata['num_classes'],
                'labels': labels,
                'metadata': metadata
            }
            
            print(f"  [OK] Data loaded: {info['num_samples']} samples, {info['num_classes']} classes")
            return info
            
        except Exception as e:
            print(f"  [ERROR] Failed to load data: {e}")
            return None
    
    def _plot_training_history(self, history, output_dir):
        """Plot training history curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Convert val_acc from percentage to decimal if needed
        train_acc = history.get('train_acc', [])
        val_acc = history.get('val_acc', [])
        
        # Detect if values are percentages (>1) or decimals (<1)
        if train_acc and train_acc[0] > 1:
            train_acc = [a for a in train_acc]  # Keep as percentage
        else:
            train_acc = [a*100 for a in train_acc]  # Convert to percentage
            
        if val_acc and val_acc[0] > 1:
            val_acc = [a for a in val_acc]  # Keep as percentage
        else:
            val_acc = [a*100 for a in val_acc]  # Convert to percentage
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        # Loss
        ax = axes[0, 0]
        if 'train_loss' in history:
            ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[0, 1]
        if train_acc:
            ax.plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
        if val_acc:
            ax.plot(epochs, val_acc, 'r-', label='Val', linewidth=2)
        ax.axhline(y=90, color='g', linestyle='--', label='90% Target')
        ax.axhline(y=80, color='orange', linestyle='--', label='80% Target')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 Score
        ax = axes[0, 2]
        train_f1 = history.get('train_f1', [])
        val_f1 = history.get('val_macro_f1', history.get('val_f1', []))
        
        if train_f1:
            ax.plot(epochs, train_f1, 'b-', label='Train', linewidth=2)
        if val_f1:
            ax.plot(epochs, val_f1, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning curve analysis
        ax = axes[1, 0]
        if train_acc and val_acc:
            gap = [t - v for t, v in zip(train_acc, val_acc)]
            ax.plot(epochs, gap, 'purple', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--')
            ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% Warning')
            ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% Danger')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gap (%)')
            ax.set_title('Overfitting Monitor (Train - Val)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Best metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        
        best_val_acc = max(val_acc) if val_acc else 0
        best_val_f1 = max(val_f1) if val_f1 else 0
        final_train_acc = train_acc[-1] if train_acc else 0
        final_val_acc = val_acc[-1] if val_acc else 0
        
        summary = f"""
        TRAINING SUMMARY
        
        Total Epochs: {len(epochs)}
        
        Best Val Accuracy: {best_val_acc:.2f}%
        Best Val F1: {best_val_f1:.4f}
        
        Final Train Acc: {final_train_acc:.2f}%
        Final Val Acc: {final_val_acc:.2f}%
        
        Overfitting Gap: {final_train_acc - final_val_acc:.2f}%
        
        Status: {"[OK]" if best_val_acc >= 90 else "[WARN]" if best_val_acc >= 80 else "[FAIL]"}
        """
        
        ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Convergence analysis
        ax = axes[1, 2]
        if 'val_loss' in history and len(history['val_loss']) > 10:
            val_loss = history['val_loss']
            # Moving average
            window = 5
            moving_avg = np.convolve(val_loss, np.ones(window)/window, mode='valid')
            ax.plot(range(len(val_loss)), val_loss, 'b-', alpha=0.3, label='Val Loss')
            ax.plot(range(window-1, len(val_loss)), moving_avg, 'r-', linewidth=2, label='Moving Avg')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Convergence Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Training history plot saved")
        return save_path
    
    def _plot_confusion_matrix(self, metrics, output_dir):
        """Plot confusion matrix"""
        if 'confusion_matrix' not in metrics:
            return None
        
        cm = np.array(metrics['confusion_matrix'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Raw counts
        ax = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix (Counts)')
        
        # Normalized by row (Recall)
        ax = axes[1]
        cm_recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_recall, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax,
                   vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix (Recall %)')
        
        # Normalized by column (Precision)
        ax = axes[2]
        cm_precision = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        sns.heatmap(cm_precision, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax,
                   vmin=0, vmax=1, cbar_kws={'label': 'Precision'})
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix (Precision %)')
        
        plt.tight_layout()
        save_path = output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Confusion matrix plot saved")
        return save_path
    
    def _plot_per_class_metrics(self, metrics, output_dir):
        """Plot per-class performance metrics"""
        per_class_acc = metrics.get('per_class_acc', {})
        
        if not per_class_acc:
            return None
        
        num_classes = len(per_class_acc)
        classes = sorted(per_class_acc.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
        
        # Accuracy
        ax = axes[0, 0]
        accs = [per_class_acc[cls] * 100 for cls in classes]
        colors_bar = ['green' if a >= 80 else 'orange' if a >= 70 else 'red' for a in accs]
        bars = ax.bar(classes, accs, color=colors_bar, alpha=0.7, edgecolor='black')
        ax.axhline(y=90, color='green', linestyle='--', label='90% Target', linewidth=2)
        ax.axhline(y=80, color='orange', linestyle='--', label='80% Threshold', linewidth=2)
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Per-Class Accuracy')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Comparison with target
        ax = axes[0, 1]
        meets_80 = sum(1 for a in accs if a >= 80)
        meets_90 = sum(1 for a in accs if a >= 90)
        below_80 = sum(1 for a in accs if a < 80)
        
        categories = ['≥90%\n(Excellent)', '80-90%\n(Good)', '<80%\n(Needs Work)']
        counts = [meets_90, meets_80 - meets_90, below_80]
        colors_pie = ['green', 'orange', 'red']
        
        wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%d',
                                           colors=colors_pie, startangle=90)
        ax.set_title(f'Class Performance Distribution\n(Total: {num_classes} classes)')
        
        # Statistics table
        ax = axes[1, 0]
        ax.axis('off')
        
        stats_text = f"""
        PER-CLASS STATISTICS
        
        Total Classes: {num_classes}
        
        Excellent (≥90%): {meets_90} ({meets_90/num_classes*100:.1f}%)
        Good (80-90%): {meets_80 - meets_90} ({(meets_80-meets_90)/num_classes*100:.1f}%)
        Needs Work (<80%): {below_80} ({below_80/num_classes*100:.1f}%)
        
        Average Accuracy: {np.mean(accs):.2f}%
        Best Class: Class {classes[np.argmax(accs)]} ({max(accs):.2f}%)
        Worst Class: Class {classes[np.argmin(accs)]} ({min(accs):.2f}%)
        
        Standard Deviation: {np.std(accs):.2f}%
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Sorted performance
        ax = axes[1, 1]
        sorted_indices = np.argsort(accs)[::-1]
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_accs = [accs[i] for i in sorted_indices]
        sorted_colors = [colors_bar[i] for i in sorted_indices]
        
        bars = ax.barh(range(len(sorted_classes)), sorted_accs, color=sorted_colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_classes)))
        ax.set_yticklabels([f'Class {c}' for c in sorted_classes])
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('Classes Ranked by Performance')
        ax.axvline(x=80, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(x=90, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlim([0, 105])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = output_dir / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Per-class metrics plot saved")
        return save_path
    
    def _plot_roc_curves(self, roc_data, output_dir):
        """Plot ROC curves"""
        # Placeholder - would need actual prediction probabilities
        print(f"  [SKIP] ROC curves (requires prediction probabilities)")
        return None
    
    def _plot_lr_schedule(self, lr_schedule, output_dir):
        """Plot learning rate schedule"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(lr_schedule, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / 'lr_schedule.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Learning rate schedule plot saved")
        return save_path
    
    def _plot_class_distribution(self, labels, output_dir):
        """Plot class distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        percentages = (counts / total) * 100
        
        # Bar chart
        ax = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        bars = ax.bar(unique, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Class')
        ax.set_ylabel('Sample Count')
        ax.set_title('Class Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                   f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax = axes[1]
        ax.pie(counts, labels=[f'Class {c}' for c in unique], autopct='%1.1f%%',
              colors=colors, startangle=90)
        ax.set_title(f'Class Distribution\n(Total: {total} samples)')
        
        plt.tight_layout()
        save_path = output_dir / 'class_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Class distribution plot saved")
        return save_path
    
    def _generate_summary_report(self, report_data, output_dir):
        """Generate text summary report"""
        summary_path = output_dir / 'summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"MODEL PERFORMANCE REPORT: {report_data['model_name']}\n")
            f.write(f"Generated: {report_data['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            # Model info
            if report_data.get('model_info'):
                f.write("MODEL INFORMATION\n")
                f.write("-"*80 + "\n")
                info = report_data['model_info']
                f.write(f"Model Path: {info.get('path', 'N/A')}\n")
                f.write(f"Training Epoch: {info.get('epoch', 'N/A')}\n")
                
                if 'metrics' in info:
                    metrics = info['metrics']
                    f.write(f"Validation Accuracy: {metrics.get('val_acc', 0)*100:.2f}%\n")
                    f.write(f"Validation F1 Score: {metrics.get('val_f1', 0):.4f}\n")
                    
                    if 'per_class_acc' in metrics:
                        f.write("\nPer-Class Accuracy:\n")
                        for cls, acc in sorted(metrics['per_class_acc'].items()):
                            status = "[OK]" if acc >= 0.8 else "[WARN]" if acc >= 0.7 else "[FAIL]"
                            f.write(f"  {status} Class {cls}: {acc*100:.2f}%\n")
                
                f.write("\n")
            
            # Data info
            if report_data.get('data_info'):
                f.write("DATASET INFORMATION\n")
                f.write("-"*80 + "\n")
                info = report_data['data_info']
                f.write(f"Total Samples: {info.get('num_samples', 'N/A')}\n")
                f.write(f"Number of Classes: {info.get('num_classes', 'N/A')}\n")
                f.write("\n")
            
            # Plots generated
            f.write("VISUALIZATIONS GENERATED\n")
            f.write("-"*80 + "\n")
            for title, path in report_data.get('plots', []):
                f.write(f"  - {title}: {path.name}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"  [OK] Summary report saved")
        return summary_path
    
    def _generate_html_report(self, report_data, output_dir):
        """Generate HTML report with embedded images"""
        html_path = output_dir / 'report.html'
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Report: {report_data['model_name']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .status-ok {{
            color: #27ae60;
        }}
        .status-warn {{
            color: #f39c12;
        }}
        .status-fail {{
            color: #e74c3c;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>[OK] Model Performance Report: {report_data['model_name']}</h1>
    <p class="timestamp">Generated: {report_data['timestamp']}</p>
    
    <div class="section">
        <h2>Executive Summary</h2>
"""
        
        # Add metrics if available
        if report_data.get('model_info') and 'metrics' in report_data['model_info']:
            metrics = report_data['model_info']['metrics']
            val_acc = metrics.get('val_acc', 0) * 100
            val_f1 = metrics.get('val_f1', 0)
            
            status_class = 'status-ok' if val_acc >= 90 else 'status-warn' if val_acc >= 80 else 'status-fail'
            
            html_content += f"""
        <div class="metric">
            <div class="metric-label">Validation Accuracy</div>
            <div class="metric-value {status_class}">{val_acc:.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Validation F1 Score</div>
            <div class="metric-value">{val_f1:.4f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Training Epoch</div>
            <div class="metric-value">{report_data['model_info'].get('epoch', 'N/A')}</div>
        </div>
            """
        
        html_content += """
    </div>
    """
        
        # Add model information
        if report_data.get('model_info'):
            html_content += """
    <div class="section">
        <h2>Model Information</h2>
            """
            
            info = report_data['model_info']
            html_content += f"""
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Model Path</td><td>{info.get('path', 'N/A')}</td></tr>
            <tr><td>Training Epoch</td><td>{info.get('epoch', 'N/A')}</td></tr>
            """
            
            if 'metrics' in info and 'per_class_acc' in info['metrics']:
                html_content += """
        </table>
        <h3>Per-Class Accuracy</h3>
        <table>
            <tr><th>Class</th><th>Accuracy</th><th>Status</th></tr>
                """
                for cls, acc in sorted(info['metrics']['per_class_acc'].items()):
                    acc_pct = acc * 100
                    status = "[OK]" if acc >= 0.8 else "[WARN]" if acc >= 0.7 else "[FAIL]"
                    status_class = 'status-ok' if acc >= 0.8 else 'status-warn' if acc >= 0.7 else 'status-fail'
                    html_content += f"""
            <tr>
                <td>Class {cls}</td>
                <td>{acc_pct:.2f}%</td>
                <td class="{status_class}">{status}</td>
            </tr>
                    """
            
            html_content += """
        </table>
    </div>
            """
        
        # Add visualizations
        if report_data.get('plots'):
            html_content += """
    <div class="section">
        <h2>Visualizations</h2>
            """
            
            for title, plot_path in report_data['plots']:
                html_content += f"""
        <h3>{title}</h3>
        <img src="{plot_path.name}" alt="{title}">
                """
            
            html_content += """
    </div>
            """
        
        # Add data information
        if report_data.get('data_info'):
            info = report_data['data_info']
            html_content += f"""
    <div class="section">
        <h2>Dataset Information</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Total Samples</td><td>{info.get('num_samples', 'N/A')}</td></tr>
            <tr><td>Number of Classes</td><td>{info.get('num_classes', 'N/A')}</td></tr>
        </table>
    </div>
            """
        
        html_content += """
    <div class="section">
        <h2>Report Information</h2>
        <p>This report was automatically generated by the Model Report Generator.</p>
        <p>For questions or issues, please refer to the documentation.</p>
    </div>
</body>
</html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  [OK] HTML report saved")
        return html_path
    
    def compare_models(self, model_reports):
        """Generate comparison report for multiple models"""
        print(f"\n{'='*80}")
        print("GENERATING MODEL COMPARISON REPORT")
        print(f"{'='*80}\n")
        
        comparison_dir = self.report_dir / f"comparison_{self.timestamp}"
        comparison_dir.mkdir(exist_ok=True)
        
        # Extract metrics for comparison
        model_names = []
        val_accs = []
        val_f1s = []
        
        for report in model_reports:
            model_names.append(report['model_name'])
            if report.get('model_info') and 'metrics' in report['model_info']:
                metrics = report['model_info']['metrics']
                val_accs.append(metrics.get('val_acc', 0) * 100)
                val_f1s.append(metrics.get('val_f1', 0))
            else:
                val_accs.append(0)
                val_f1s.append(0)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy comparison
        ax = axes[0]
        colors = ['green' if a >= 90 else 'orange' if a >= 80 else 'red' for a in val_accs]
        bars = ax.bar(range(len(model_names)), val_accs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.axhline(y=90, color='green', linestyle='--', label='90% Target', linewidth=2)
        ax.axhline(y=80, color='orange', linestyle='--', label='80% Threshold', linewidth=2)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, val_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # F1 comparison
        ax = axes[1]
        bars = ax.bar(range(len(model_names)), val_f1s, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('F1 Score')
        ax.set_title('Model F1 Score Comparison')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, f1 in zip(bars, val_f1s):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{f1:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = comparison_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Comparison plot saved: {save_path}")
        
        # Generate comparison HTML
        self._generate_comparison_html(model_reports, comparison_dir, save_path)
        
        return comparison_dir
    
    def _generate_comparison_html(self, model_reports, output_dir, comparison_plot):
        """Generate HTML comparison report"""
        html_path = output_dir / 'comparison_report.html'
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best {{
            background-color: #d5f4e6;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <h1>[OK] Model Comparison Report</h1>
    <p>Generated: {self.timestamp}</p>
    
    <div class="section">
        <h2>Performance Comparison</h2>
        <img src="{comparison_plot.name}" alt="Model Comparison">
    </div>
    
    <div class="section">
        <h2>Detailed Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Validation Accuracy</th>
                <th>Validation F1</th>
                <th>Epoch</th>
            </tr>
        """
        
        # Find best model
        best_acc_idx = -1
        best_acc = 0
        
        for idx, report in enumerate(model_reports):
            if report.get('model_info') and 'metrics' in report['model_info']:
                acc = report['model_info']['metrics'].get('val_acc', 0)
                if acc > best_acc:
                    best_acc = acc
                    best_acc_idx = idx
        
        # Add table rows
        for idx, report in enumerate(model_reports):
            row_class = 'best' if idx == best_acc_idx else ''
            model_name = report['model_name']
            
            if report.get('model_info') and 'metrics' in report['model_info']:
                metrics = report['model_info']['metrics']
                val_acc = metrics.get('val_acc', 0) * 100
                val_f1 = metrics.get('val_f1', 0)
                epoch = report['model_info'].get('epoch', 'N/A')
            else:
                val_acc = 0
                val_f1 = 0
                epoch = 'N/A'
            
            html_content += f"""
            <tr class="{row_class}">
                <td>{model_name}</td>
                <td>{val_acc:.2f}%</td>
                <td>{val_f1:.4f}</td>
                <td>{epoch}</td>
            </tr>
            """
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Links to Individual Reports</h2>
        <ul>
        """
        
        for report in model_reports:
            model_name = report['model_name']
            html_content += f"""
            <li><a href="../{model_name}_{self.timestamp}/report.html">{model_name} Report</a></li>
            """
        
        html_content += """
        </ul>
    </div>
</body>
</html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  [OK] Comparison HTML saved: {html_path}")


def main():
    """Main execution"""
    generator = ModelReportGenerator()
    
    # Define models to analyze (using actual trained models)
    models = [
        {
            'name': 'Exp1_SmoothedInvFreq',
            'path': 'experiments/Exp1_SmoothedInvFreq/best_model.pth',
            'description': 'Smoothed Inverse Frequency class weighting'
        },
        {
            'name': 'Exp2_EffectiveNum',
            'path': 'experiments/Exp2_EffectiveNum_0.9999/best_model.pth',
            'description': 'Effective Number weighting (beta=0.9999)'
        }
    ]
    
    print("\nModels to analyze:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']}: {model['description']}")
    
    # Generate reports for each model
    model_reports = []
    
    for model in models:
        try:
            report = generator.generate_report_for_model(
                model_name=model['name'],
                model_path=model['path']
            )
            model_reports.append(report)
        except Exception as e:
            print(f"\n[ERROR] Failed to generate report for {model['name']}: {e}")
    
    # Generate comparison report if multiple models
    if len(model_reports) > 1:
        try:
            comparison_dir = generator.compare_models(model_reports)
            print(f"\n[OK] Comparison report: {comparison_dir}/comparison_report.html")
        except Exception as e:
            print(f"\n[ERROR] Failed to generate comparison report: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE!")
    print("="*80)
    print(f"\nReports saved in: {generator.report_dir}/")
    print("\nGenerated reports:")
    for report in model_reports:
        model_name = report['model_name']
        print(f"  - {model_name}_{generator.timestamp}/report.html")
    
    if len(model_reports) > 1:
        print(f"  - comparison_{generator.timestamp}/comparison_report.html")
    
    print("\nOpen the HTML files in your browser to view the reports.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
