#!/usr/bin/env python3
"""evaluate.py - Evaluate trained BetaDogma model on test set with enhanced metrics for sparse labels."""

import torch
from pathlib import Path
import numpy as np
import yaml
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    accuracy_score,
    precision_recall_curve,
    auc,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import sparse
import json
import os

from train import Config, BetaDogmaLightning, BetaDogmaDataset
from transformers import AutoTokenizer


def plot_curves(targets, preds, task_name, output_dir):
    """Plot precision-recall and ROC curves with enhanced visualization."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics
    precision, recall, _ = precision_recall_curve(targets, preds)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (maximizing F1 score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = (preds >= (np.sort(preds)[::-1][optimal_idx])).astype(int)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Precision-Recall curve
    pr_display = PrecisionRecallDisplay(
        precision=precision, 
        recall=recall,
        average_precision=pr_auc
    )
    pr_display.plot(ax=ax1, name=f'{task_name} (AP = {pr_auc:.4f})')
    ax1.set_title(f'Precision-Recall Curve - {task_name}')
    ax1.grid(True)
    
    # Add optimal threshold point
    ax1.plot(recall[optimal_idx], precision[optimal_idx], 'ro', 
             label=f'Optimal threshold (F1={f1_scores[optimal_idx]:.2f})')
    ax1.legend()
    
    # Plot ROC curve
    roc_display = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        estimator_name=task_name
    )
    roc_display.plot(ax=ax2, name=f'{task_name} (AUC = {roc_auc:.4f}')
    ax2.set_title(f'ROC Curve - {task_name}')
    ax2.grid(True)
    
    # Add diagonal line for reference
    ax2.plot([0, 1], [0, 1], 'k--')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / f'curves_{task_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to file
    metrics = {
        'task': task_name,
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(np.sort(preds)[::-1][optimal_idx])
    }
    
    with open(output_dir / f'metrics_{task_name}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def evaluate_model(checkpoint_path: str, test_files: list, config: Config = None):
    """Evaluate a trained model on test data with enhanced metrics and visualizations for sparse labels.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        test_files: List of test file paths
        config: Optional Config object. If None, loads from config.yaml
    """
    if config is None:
        # Load default config from YAML
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create Config object
        config = Config()
        # Update config with values from YAML
        config.max_seq_len = config_dict['model']['max_seq_len']
        config.batch_size = config_dict['training']['batch_size']
        config.num_workers = config_dict['training']['num_workers']
        config.encoder_name = config_dict['model']['name']
    print(f"Loading model from {checkpoint_path}")
    model = BetaDogmaLightning.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model.freeze()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name, trust_remote_code=True)
    
    print(f"Loading test data...")
    test_dataset = BetaDogmaDataset(test_files, tokenizer, config.max_seq_len, mode="test")
    
    print(f"Evaluating on {len(test_dataset)} examples...")
    
    # Create output directories
    output_dir = Path('output/evaluation')
    plots_dir = output_dir / 'plots'
    metrics_dir = output_dir / 'metrics'
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize storage for metrics
    metrics = {
        task: {
            'preds': [],
            'targets': [],
            'loss': [],
            'f1_scores': [],
            'mcc_scores': [],
            'balanced_acc': []
        } for task in ['donor', 'acceptor', 'tss', 'polya']
    }
    
    # For storing individual example results for analysis
    example_results = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            try:
                batch = test_dataset[i]
                
                # Move to device and handle missing attention_mask
                input_ids = batch['input_ids'].unsqueeze(0).to(device)
                attention_mask = batch.get('attention_mask')
                
                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(0).to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids)
                
                # Store results for each task
                for task in ['donor', 'acceptor', 'tss', 'polya']:
                    # Get predictions and targets
                    preds = torch.sigmoid(outputs[task]).cpu().numpy().flatten()
                    targets = batch[task].numpy().flatten()
                    
                    # If we have attention_mask, use it for masking
                    if 'attention_mask' in batch:
                        mask = batch['attention_mask'].numpy().astype(bool)
                        preds = preds[mask]
                        targets = targets[mask]
                    
                    # Store for metrics calculation
                    metrics[task]['preds'].extend(preds)
                    metrics[task]['targets'].extend(targets)
                    
                    # Calculate binary cross-entropy loss
                    loss = torch.nn.functional.binary_cross_entropy(
                        torch.tensor(preds, dtype=torch.float32),
                        torch.tensor(targets, dtype=torch.float32)
                    ).item()
                    metrics[task]['loss'].append(loss)
                
                # Store example results for analysis
                example_results.append({
                    'input_length': len(batch['input_ids']),
                    'has_variant': batch['has_variant'].item(),
                    'is_pathogenic': batch['is_pathogenic'].item()
                })
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
    
    # Calculate and print metrics
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Dictionary to store all metrics for final summary
    all_metrics = {}
    
    for task in ['donor', 'acceptor', 'tss', 'polya']:
        preds = np.array(metrics[task]['preds'])
        targets = np.array(metrics[task]['targets'])
        
        if len(preds) == 0:
            print(f"\n{task.upper()}: No predictions to evaluate")
            continue
        
        # Calculate class distribution
        n_pos = int(targets.sum())
        n_total = len(targets)
        pos_ratio = n_pos / n_total if n_total > 0 else 0
        
        # Initialize metrics dict for this task
        task_metrics = {
            'n_pos': n_pos,
            'n_total': n_total,
            'pos_ratio': pos_ratio,
            'avg_loss': float(np.mean(metrics[task]['loss']))
        }
        
        # Only calculate metrics if we have both positive and negative examples
        if 0 < n_pos < n_total:
            # Calculate threshold-dependent metrics
            binary_preds = (preds > 0.5).astype(int)
            
            # Basic metrics
            task_metrics.update({
                'accuracy': float(accuracy_score(targets, binary_preds)),
                'f1': float(f1_score(targets, binary_preds, zero_division=0)),
                'mcc': float(matthews_corrcoef(targets, binary_preds)),
                'balanced_accuracy': float(balanced_accuracy_score(targets, binary_preds)),
                'precision': float(average_precision_score(targets, binary_preds, pos_label=1)),
                'recall': float(recall_score(targets, binary_preds, zero_division=0))
            })
            
            # Threshold-independent metrics
            task_metrics.update({
                'roc_auc': float(roc_auc_score(targets, preds)),
                'average_precision': float(average_precision_score(targets, preds))
            })
            
            # Find optimal threshold (maximizing F1 score)
            precision, recall, thresholds = precision_recall_curve(targets, preds)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            task_metrics['optimal_threshold'] = float(optimal_threshold)
            task_metrics['optimal_f1'] = float(f1_scores[optimal_idx])
            
            # Generate and save plots
            plot_metrics = plot_curves(targets, preds, task, plots_dir)
            task_metrics.update(plot_metrics)
            
        else:
            print(f"\n{task.upper()}: Not enough positive samples for full evaluation")
            print(f"  Positive samples: {n_pos}/{n_total} ({pos_ratio*100:.4f}%)")
            continue
        
        # Store metrics
        all_metrics[task] = task_metrics
        
        # Print metrics
        print(f"\n{task.upper()}:")
        print(f"  Loss: {task_metrics['avg_loss']:.4f}")
        print(f"  Pos/Neg: {n_pos:,}/{n_total:,} ({pos_ratio*100:.4f}%)")
        print("\n  Classification Metrics:")
        print(f"    Accuracy: {task_metrics['accuracy']:.4f}")
        print(f"    F1 Score: {task_metrics['f1']:.4f}")
        print(f"    MCC: {task_metrics['mcc']:.4f}")
        print(f"    Balanced Accuracy: {task_metrics['balanced_accuracy']:.4f}")
        print("\n  Threshold-independent Metrics:")
        print(f"    ROC AUC: {task_metrics['roc_auc']:.4f}")
        print(f"    Avg Precision: {task_metrics['average_precision']:.4f}")
        print(f"    Optimal Threshold: {task_metrics['optimal_threshold']:.4f} (F1={task_metrics['optimal_f1']:.4f})")
    
    # Save all metrics to file
    metrics_file = metrics_dir / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    if example_results:
        total_examples = len(example_results)
        has_variant = sum(1 for x in example_results if x['has_variant'])
        is_pathogenic = sum(1 for x in example_results if x.get('is_pathogenic', False))
        avg_length = np.mean([x['input_length'] for x in example_results])
        
        print(f"Total examples: {total_examples}")
        print(f"Examples with variants: {has_variant} ({has_variant/total_examples*100:.1f}%)")
        print(f"Pathogenic variants: {is_pathogenic} ({is_pathogenic/max(1, has_variant)*100:.1f}% of variants)")
        print(f"Average sequence length: {avg_length:.1f} bp")
    
    # Print summary of metrics across all tasks
    if all_metrics:
        print("\n" + "="*80)
        print("SUMMARY METRICS ACROSS TASKS")
        print("="*80)
        
        # Calculate average metrics
        avg_metrics = {
            'f1': np.mean([m['f1'] for m in all_metrics.values() if 'f1' in m]),
            'roc_auc': np.mean([m['roc_auc'] for m in all_metrics.values() if 'roc_auc' in m]),
            'avg_precision': np.mean([m['average_precision'] for m in all_metrics.values() if 'average_precision' in m]),
            'mcc': np.mean([m['mcc'] for m in all_metrics.values() if 'mcc' in m])
        }
        
        print(f"\nAverage Metrics:")
        print(f"  F1 Score: {avg_metrics['f1']:.4f}")
        print(f"  ROC AUC: {avg_metrics['roc_auc']:.4f}")
        print(f"  Avg Precision: {avg_metrics['avg_precision']:.4f}")
        print(f"  MCC: {avg_metrics['mcc']:.4f}")
    
    print(f"\nEvaluation complete. Results saved to:")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Plots: {plots_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate BetaDogma model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-files', type=str, nargs='+', required=True,
                        help='Test data files (parquet)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config.yaml in project root)')
    
    args = parser.parse_args()
    
    # Initialize config (will load from YAML if not provided)
    config = None
    if args.config:
        config_path = Path(args.config)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create Config object and update with YAML values
        config = Config()
        config.max_seq_len = config_dict['model']['max_seq_len']
        config.batch_size = config_dict['training']['batch_size']
        config.num_workers = config_dict['training']['num_workers']
        config.encoder_name = config_dict['model']['name']
    
    # Run evaluation
    print(f"Found {len(args.test_files)} test files")
    evaluate_model(args.checkpoint, args.test_files, config)