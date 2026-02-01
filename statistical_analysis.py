#!/usr/bin/env python3
"""
Statistical Significance Testing for Agentic AI Trust-Aware Orchestration Study
Performs McNemar's test, bootstrap confidence intervals, and paired t-tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def load_predictions(case_path):
    """Load prediction data from CSV files"""
    if 'Case1' in case_path:
        df = pd.read_csv(case_path)
        return {
            'true_labels': df['Category'].values,
            'qwen_pred': df['Qwen'].values,
            'gpt_pred': df['GPT'].values,
            'orch_pred': df['Orchestrator'].values,
            'qwen_conf': df['Qwen_confidence'].values,
            'gpt_conf': df['GPT_confidence'].values,
            'orch_conf': df['Orchestrator_confidence'].values
        }
    elif 'Case2' in case_path:
        df = pd.read_csv(case_path)
        return {
            'true_labels': df['Category'].values,
            'qwen_pred': df['Qwen'].values,
            'gpt_pred': df['GPT'].values,
            'orch_pred': df['Orchestrator'].values,
            'qwen_conf': df['Qwen_confidence'].values,
            'gpt_conf': df['GPT_confidence'].values,
            'orch_conf': df['Orchestrator_confidence'].values
        }
    elif 'Case3' in case_path:
        df = pd.read_csv(case_path)
        return {
            'true_labels': df['Category'].values,
            'qwen_pred': df['Qwen'].values,
            'gpt_pred': df['GPT'].values,
            'orch_pred': df['Orchestrator'].values,
            'qwen_conf': df['Qwen_confidence'].values,
            'gpt_conf': df['GPT_confidence'].values,
            'orch_conf': df['Orchestrator_confidence'].values
        }

def mcnemar_test(pred1, pred2, true_labels):
    """
    Perform McNemar's test for paired nominal data
    Tests if two classifiers have significantly different error rates
    """
    # Create contingency table
    # [both correct, pred1 correct pred2 wrong, pred1 wrong pred2 correct, both wrong]
    both_correct = np.sum((pred1 == true_labels) & (pred2 == true_labels))
    pred1_correct_pred2_wrong = np.sum((pred1 == true_labels) & (pred2 != true_labels))
    pred1_wrong_pred2_correct = np.sum((pred1 != true_labels) & (pred2 == true_labels))
    both_wrong = np.sum((pred1 != true_labels) & (pred2 != true_labels))
    
    # McNemar's test statistic
    b = pred1_correct_pred2_wrong
    c = pred1_wrong_pred2_correct
    
    # Use continuity correction for small samples
    if b + c < 25:
        statistic = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
    else:
        statistic = (b - c)**2 / (b + c) if (b + c) > 0 else 0
    
    # p-value from chi-square distribution with 1 df
    p_value = 1 - chi2.cdf(statistic, 1)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'contingency': {
            'both_correct': both_correct,
            'only_pred1_correct': pred1_correct_pred2_wrong,
            'only_pred2_correct': pred1_wrong_pred2_correct,
            'both_wrong': both_wrong
        }
    }

def bootstrap_confidence_interval(predictions, true_labels, n_iterations=10000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for accuracy
    """
    n_samples = len(predictions)
    accuracies = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_pred = predictions[indices]
        sample_true = true_labels[indices]
        
        # Calculate accuracy for this bootstrap sample
        accuracy = np.mean(sample_pred == sample_true)
        accuracies.append(accuracy)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(accuracies, lower_percentile)
    ci_upper = np.percentile(accuracies, upper_percentile)
    
    # Point estimate
    accuracy = np.mean(predictions == true_labels)
    
    return {
        'accuracy': accuracy,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_iterations': n_iterations
    }

def paired_t_test(metric1, metric2):
    """
    Perform paired t-test for continuous metrics (e.g., confidence scores)
    """
    # Calculate differences
    differences = metric1 - metric2
    
    # Perform t-test
    t_statistic, p_value = stats.ttest_rel(metric1, metric2)
    
    # Calculate effect size (Cohen's d for paired samples)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'cohens_d': cohens_d
    }

def calculate_precision_recall_f1(predictions, true_labels):
    """Calculate precision, recall, F1 per class and macro average"""
    classes = np.unique(true_labels)
    metrics = {}
    
    for cls in classes:
        tp = np.sum((predictions == cls) & (true_labels == cls))
        fp = np.sum((predictions == cls) & (true_labels != cls))
        fn = np.sum((predictions != cls) & (true_labels == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Macro averages
    macro_precision = np.mean([m['precision'] for m in metrics.values()])
    macro_recall = np.mean([m['recall'] for m in metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    
    return {
        'per_class': metrics,
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }
    }

def main():
    # Define paths
    base_path = Path('/Users/rkonstadinos/PycharmProjects/Trust-Paper/Results')
    case1_path = base_path / 'Case1' / 'test_set_zero_shot_classification.csv'
    case2_path = base_path / 'Case2' / 'test_set_few_shot_classification.csv'
    case3_path = base_path / 'Case3' / 'predictions.csv'
    
    print("="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("Agentic AI with Trust-Aware Orchestration")
    print("="*80)
    print()
    
    # Load all datasets
    print("Loading datasets...")
    exp1 = load_predictions(str(case1_path))
    exp2 = load_predictions(str(case2_path))
    exp3 = load_predictions(str(case3_path))
    
    n_samples = len(exp1['true_labels'])
    print(f"Number of test samples: {n_samples}")
    print()
    
    # ========================================================================
    # 1. BOOTSTRAP CONFIDENCE INTERVALS FOR ACCURACY
    # ========================================================================
    print("="*80)
    print("1. BOOTSTRAP CONFIDENCE INTERVALS (95% CI, 10,000 iterations)")
    print("="*80)
    print()
    
    results = {}
    
    for exp_name, exp_data in [('Exp I', exp1), ('Exp II', exp2), ('Exp III', exp3)]:
        print(f"\n{exp_name} (Zero-shot)" if 'I' in exp_name else f"\n{exp_name} (Fine-tuned)" if 'II' in exp_name else f"\n{exp_name} (Trust+RAG)")
        print("-" * 40)
        
        exp_results = {}
        
        for agent_name, predictions in [('Qwen', exp_data['qwen_pred']), 
                                        ('GPT', exp_data['gpt_pred']), 
                                        ('Orchestrator', exp_data['orch_pred'])]:
            ci = bootstrap_confidence_interval(predictions, exp_data['true_labels'])
            exp_results[agent_name] = ci
            
            print(f"{agent_name:15s}: {ci['accuracy']:.4f} (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
        
        results[exp_name] = exp_results
    
    # ========================================================================
    # 2. MCNEMAR'S TEST - Pairwise Comparisons
    # ========================================================================
    print("\n")
    print("="*80)
    print("2. McNEMAR'S TEST - Pairwise Classifier Comparisons")
    print("="*80)
    print()
    
    mcnemar_results = {}
    
    # Exp I vs Exp II
    print("\nExperiment I vs. Experiment II")
    print("-" * 40)
    
    comparisons = [
        ('Qwen (Exp I vs II)', exp1['qwen_pred'], exp2['qwen_pred'], exp1['true_labels']),
        ('GPT (Exp I vs II)', exp1['gpt_pred'], exp2['gpt_pred'], exp1['true_labels']),
        ('Orch (Exp I vs II)', exp1['orch_pred'], exp2['orch_pred'], exp1['true_labels'])
    ]
    
    for name, pred1, pred2, true_labels in comparisons:
        result = mcnemar_test(pred1, pred2, true_labels)
        mcnemar_results[name] = result
        
        print(f"\n{name}:")
        print(f"  McNemar statistic: {result['statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f} {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}")
        print(f"  Contingency table:")
        print(f"    Both correct: {result['contingency']['both_correct']}")
        print(f"    Only first correct: {result['contingency']['only_pred1_correct']}")
        print(f"    Only second correct: {result['contingency']['only_pred2_correct']}")
        print(f"    Both wrong: {result['contingency']['both_wrong']}")
    
    # Exp I vs Exp III
    print("\n\nExperiment I vs. Experiment III")
    print("-" * 40)
    
    comparisons = [
        ('Qwen (Exp I vs III)', exp1['qwen_pred'], exp3['qwen_pred'], exp1['true_labels']),
        ('GPT (Exp I vs III)', exp1['gpt_pred'], exp3['gpt_pred'], exp1['true_labels']),
        ('Orch (Exp I vs III)', exp1['orch_pred'], exp3['orch_pred'], exp1['true_labels'])
    ]
    
    for name, pred1, pred2, true_labels in comparisons:
        result = mcnemar_test(pred1, pred2, true_labels)
        mcnemar_results[name] = result
        
        print(f"\n{name}:")
        print(f"  McNemar statistic: {result['statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f} {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}")
        print(f"  Contingency table:")
        print(f"    Both correct: {result['contingency']['both_correct']}")
        print(f"    Only first correct: {result['contingency']['only_pred1_correct']}")
        print(f"    Only second correct: {result['contingency']['only_pred2_correct']}")
        print(f"    Both wrong: {result['contingency']['both_wrong']}")
    
    # Exp II vs Exp III
    print("\n\nExperiment II vs. Experiment III")
    print("-" * 40)
    
    comparisons = [
        ('Qwen (Exp II vs III)', exp2['qwen_pred'], exp3['qwen_pred'], exp2['true_labels']),
        ('GPT (Exp II vs III)', exp2['gpt_pred'], exp3['gpt_pred'], exp2['true_labels']),
        ('Orch (Exp II vs III)', exp2['orch_pred'], exp3['orch_pred'], exp2['true_labels'])
    ]
    
    for name, pred1, pred2, true_labels in comparisons:
        result = mcnemar_test(pred1, pred2, true_labels)
        mcnemar_results[name] = result
        
        print(f"\n{name}:")
        print(f"  McNemar statistic: {result['statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f} {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}")
        print(f"  Contingency table:")
        print(f"    Both correct: {result['contingency']['both_correct']}")
        print(f"    Only first correct: {result['contingency']['only_pred1_correct']}")
        print(f"    Only second correct: {result['contingency']['only_pred2_correct']}")
        print(f"    Both wrong: {result['contingency']['both_wrong']}")
    
    # ========================================================================
    # 3. PAIRED T-TESTS - Confidence Scores
    # ========================================================================
    print("\n")
    print("="*80)
    print("3. PAIRED T-TESTS - Confidence Score Comparisons")
    print("="*80)
    print()
    
    ttest_results = {}
    
    # Exp I vs Exp II - Confidence
    print("\nExperiment I vs. Experiment II (Confidence Scores)")
    print("-" * 40)
    
    for agent_name in ['Qwen', 'GPT', 'Orchestrator']:
        conf1 = exp1[f'{agent_name.lower()}_conf'] if agent_name != 'Orchestrator' else exp1['orch_conf']
        conf2 = exp2[f'{agent_name.lower()}_conf'] if agent_name != 'Orchestrator' else exp2['orch_conf']
        
        result = paired_t_test(conf1, conf2)
        ttest_results[f'{agent_name} (Exp I vs II)'] = result
        
        print(f"\n{agent_name}:")
        print(f"  Mean Exp I: {np.mean(conf1):.4f}")
        print(f"  Mean Exp II: {np.mean(conf2):.4f}")
        print(f"  Mean difference: {result['mean_difference']:.4f}")
        print(f"  t-statistic: {result['t_statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f} {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}")
        print(f"  Cohen's d: {result['cohens_d']:.4f}")
    
    # Exp I vs Exp III - Confidence
    print("\n\nExperiment I vs. Experiment III (Confidence Scores)")
    print("-" * 40)
    
    for agent_name in ['Qwen', 'GPT', 'Orchestrator']:
        conf1 = exp1[f'{agent_name.lower()}_conf'] if agent_name != 'Orchestrator' else exp1['orch_conf']
        conf3 = exp3[f'{agent_name.lower()}_conf'] if agent_name != 'Orchestrator' else exp3['orch_conf']
        
        result = paired_t_test(conf1, conf3)
        ttest_results[f'{agent_name} (Exp I vs III)'] = result
        
        print(f"\n{agent_name}:")
        print(f"  Mean Exp I: {np.mean(conf1):.4f}")
        print(f"  Mean Exp III: {np.mean(conf3):.4f}")
        print(f"  Mean difference: {result['mean_difference']:.4f}")
        print(f"  t-statistic: {result['t_statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f} {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}")
        print(f"  Cohen's d: {result['cohens_d']:.4f}")
    
    # ========================================================================
    # 4. SUMMARY TABLE
    # ========================================================================
    print("\n")
    print("="*80)
    print("4. SUMMARY TABLE - All Results")
    print("="*80)
    print()
    
    print("Accuracy with 95% Confidence Intervals:")
    print("-" * 80)
    print(f"{'Agent':<15s} {'Exp I':<30s} {'Exp II':<30s} {'Exp III':<30s}")
    print("-" * 80)
    
    for agent in ['Qwen', 'GPT', 'Orchestrator']:
        exp1_ci = results['Exp I'][agent]
        exp2_ci = results['Exp II'][agent]
        exp3_ci = results['Exp III'][agent]
        
        print(f"{agent:<15s} "
              f"{exp1_ci['accuracy']:.4f} [{exp1_ci['ci_lower']:.4f}, {exp1_ci['ci_upper']:.4f}]  "
              f"{exp2_ci['accuracy']:.4f} [{exp2_ci['ci_lower']:.4f}, {exp2_ci['ci_upper']:.4f}]  "
              f"{exp3_ci['accuracy']:.4f} [{exp3_ci['ci_lower']:.4f}, {exp3_ci['ci_upper']:.4f}]")
    
    # Save results to JSON
    output = {
        'confidence_intervals': results,
        'mcnemar_tests': {k: {**v, 'contingency': v['contingency']} for k, v in mcnemar_results.items()},
        'paired_ttests': ttest_results
    }
    
    output_path = base_path / 'statistical_analysis_results.json'
    with open(output_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json.dump(output, f, indent=2, default=convert)
    
    print(f"\n\nResults saved to: {output_path}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
