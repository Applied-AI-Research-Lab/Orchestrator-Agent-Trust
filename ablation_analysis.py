"""
Ablation Analysis: Isolating Component Contributions
=====================================================

This script re-analyzes Experiment III data to isolate the individual contributions of:
1. Trust-aware orchestration (without RAG)
2. RAG re-evaluation (without trust metrics)
3. Combined system (trust + RAG)

By strategically re-processing existing data with different arbitration strategies,
we can perform ablation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

def load_experiment_data():
    """Load data from all three experiments"""
    print("Loading experiment data...")
    
    # Experiment I: Zero-shot baseline
    exp1 = pd.read_csv('Results/Case1/test_set_zero_shot_classification.csv')
    
    # Experiment II: Fine-tuned
    exp2 = pd.read_csv('Results/Case2/test_set_few_shot_classification.csv')
    
    # Experiment III: Trust-aware + RAG
    exp3 = pd.read_csv('Results/Case3/predictions.csv')
    
    return exp1, exp2, exp3

def calculate_trust_score(ece, ocr, confidence):
    """
    Calculate trust score using the formula from Algorithm 1:
    τ = (1 / (1 + ECE)) × (1 - OCR) × confidence
    """
    return (1 / (1 + ece)) * (1 - ocr) * confidence

def ablation_trust_only(exp3_data, trust_profiles):
    """
    Ablation Component 2: Trust-aware orchestration WITHOUT RAG
    
    Use initial predictions from Experiment III (before re-evaluation)
    Apply trust-aware arbitration based on ECE, OCR, and confidence
    """
    print("\n=== Ablation Component 2: Trust Metrics Only (No RAG) ===")
    
    predictions = []
    correct = 0
    
    for idx, row in exp3_data.iterrows():
        # Extract initial predictions (before re-evaluation)
        # Qwen_1 and GPT_1 are the initial zero-shot predictions
        qwen_pred = row['Qwen_1']
        qwen_conf = row['Qwen_1_confidence']
        gpt_pred = row['GPT_1']
        gpt_conf = row['GPT_1_confidence']
        true_label = row['Category']
        
        # Calculate trust scores using pre-computed trust profiles
        qwen_trust = calculate_trust_score(
            trust_profiles['qwen']['ece'],
            trust_profiles['qwen']['ocr'],
            qwen_conf
        )
        gpt_trust = calculate_trust_score(
            trust_profiles['gpt']['ece'],
            trust_profiles['gpt']['ocr'],
            gpt_conf
        )
        
        # Apply trust-aware arbitration (simplified from Algorithm 1)
        # If agents agree and both have high trust, accept consensus
        if qwen_pred == gpt_pred and min(qwen_trust, gpt_trust) > 0.75:
            final_pred = qwen_pred
        # Otherwise, select agent with highest trust score
        else:
            final_pred = qwen_pred if qwen_trust > gpt_trust else gpt_pred
        
        predictions.append({
            'image': row.get('Image', row.get('image', idx)),
            'true_label': true_label,
            'final_prediction': final_pred,
            'qwen_pred': qwen_pred,
            'qwen_conf': qwen_conf,
            'qwen_trust': qwen_trust,
            'gpt_pred': gpt_pred,
            'gpt_conf': gpt_conf,
            'gpt_trust': gpt_trust
        })
        
        if final_pred == true_label:
            correct += 1
    
    accuracy = correct / len(predictions)
    print(f"Accuracy (Trust Only): {accuracy:.4f} ({correct}/{len(predictions)})")
    
    return predictions, accuracy

def ablation_rag_only(exp3_data):
    """
    Ablation Component 3: RAG re-evaluation WITHOUT trust metrics
    
    Use re-evaluated predictions from Experiment III (after RAG)
    Apply simple confidence-based arbitration (like Experiment I)
    Ignore trust profiles
    """
    print("\n=== Ablation Component 3: RAG Only (No Trust Metrics) ===")
    
    predictions = []
    correct = 0
    
    for idx, row in exp3_data.iterrows():
        # Extract re-evaluated predictions (after RAG)
        # Qwen and GPT (without suffix) are post-reevaluation predictions
        qwen_reevaluated = row['Qwen']
        qwen_conf_reevaluated = row['Qwen_confidence']
        
        gpt_reevaluated = row['GPT']
        gpt_conf_reevaluated = row['GPT_confidence']
        
        true_label = row['Category']
        
        # Apply simple confidence-based arbitration (like Experiment I)
        # If agents agree, accept consensus
        if qwen_reevaluated == gpt_reevaluated:
            final_pred = qwen_reevaluated
        # Otherwise, select agent with higher confidence
        else:
            final_pred = qwen_reevaluated if qwen_conf_reevaluated > gpt_conf_reevaluated else gpt_reevaluated
        
        predictions.append({
            'image': row.get('Image', row.get('image', idx)),
            'true_label': true_label,
            'final_prediction': final_pred,
            'qwen_pred': qwen_reevaluated,
            'qwen_conf': qwen_conf_reevaluated,
            'gpt_pred': gpt_reevaluated,
            'gpt_conf': gpt_conf_reevaluated
        })
        
        if final_pred == true_label:
            correct += 1
    
    accuracy = correct / len(predictions)
    print(f"Accuracy (RAG Only): {accuracy:.4f} ({correct}/{len(predictions)})")
    
    return predictions, accuracy

def calculate_component_contributions(exp1_acc, trust_only_acc, rag_only_acc, exp3_acc):
    """
    Calculate individual component contributions and synergy effects
    """
    print("\n=== Component Contribution Analysis ===")
    
    # Marginal contributions
    trust_contribution = trust_only_acc - exp1_acc
    rag_contribution = rag_only_acc - exp1_acc
    
    # Expected additive contribution
    expected_additive = exp1_acc + trust_contribution + rag_contribution
    
    # Actual combined performance
    actual_combined = exp3_acc
    
    # Synergy (can be positive or negative)
    synergy = actual_combined - expected_additive
    
    print(f"\nBaseline (Exp I - Confidence-aware only): {exp1_acc:.4f}")
    print(f"\nMarginal Contributions:")
    print(f"  Trust Metrics:  +{trust_contribution:.4f} ({trust_contribution/exp1_acc*100:+.1f}%)")
    print(f"  RAG Re-eval:    +{rag_contribution:.4f} ({rag_contribution/exp1_acc*100:+.1f}%)")
    print(f"\nExpected (Additive): {expected_additive:.4f}")
    print(f"Actual (Combined):   {actual_combined:.4f}")
    print(f"\nSynergy Effect: {synergy:.4f} ({'positive' if synergy > 0 else 'negative'})")
    
    if synergy > 0:
        print(f"  → Trust metrics and RAG exhibit POSITIVE synergy")
        print(f"  → Combined system exceeds sum of individual contributions")
    else:
        print(f"  → Slight negative interaction effect observed")
        print(f"  → May indicate redundancy or interference between components")
    
    return {
        'baseline': exp1_acc,
        'trust_only': trust_only_acc,
        'rag_only': rag_only_acc,
        'combined': exp3_acc,
        'trust_contribution': trust_contribution,
        'rag_contribution': rag_contribution,
        'expected_additive': expected_additive,
        'synergy': synergy
    }

def create_ablation_table(results):
    """Create formatted ablation results table"""
    print("\n" + "="*70)
    print("ABLATION STUDY: COMPONENT-WISE PERFORMANCE")
    print("="*70)
    
    configurations = [
        ("Baseline: Zero-shot + Confidence-aware Orchestration", 
         results['baseline'], 
         "Experiment I"),
        ("+ Trust Metrics (ECE, OCR, CG)", 
         results['trust_only'], 
         f"Δ = +{results['trust_contribution']:.4f}"),
        ("+ RAG Re-evaluation", 
         results['rag_only'], 
         f"Δ = +{results['rag_contribution']:.4f}"),
        ("Full System: Trust + RAG", 
         results['combined'], 
         f"Δ = +{results['combined']-results['baseline']:.4f}")
    ]
    
    print(f"\n{'Configuration':<50} {'Accuracy':>10} {'Contribution':>15}")
    print("-"*70)
    for config, acc, contrib in configurations:
        print(f"{config:<50} {acc:>10.4f} {contrib:>15}")
    
    print("-"*70)
    print(f"\n{'Expected (Additive Model)':<50} {results['expected_additive']:>10.4f}")
    print(f"{'Actual (Combined System)':<50} {results['combined']:>10.4f}")
    print(f"{'Synergy Effect':<50} {results['synergy']:>10.4f}")
    print("="*70)

def main():
    """Main execution function"""
    print("="*70)
    print("ABLATION ANALYSIS: ISOLATING COMPONENT CONTRIBUTIONS")
    print("="*70)
    
    # Load data
    exp1, exp2, exp3 = load_experiment_data()
    
    # Trust profiles from Experiment III (Table in Appendix)
    trust_profiles = {
        'qwen': {
            'ece': 0.453,
            'ocr': 0.508,
            'cg': 0.009,
            'ccc': 0.120
        },
        'gpt': {
            'ece': 0.293,
            'ocr': 0.416,
            'cg': 0.014,
            'ccc': 0.361
        }
    }
    
    # Baseline accuracy (Experiment I)
    exp1_orchestrator_correct = (exp1['Orchestrator'] == exp1['Category']).sum()
    exp1_accuracy = exp1_orchestrator_correct / len(exp1)
    print(f"\nBaseline (Exp I): {exp1_accuracy:.4f} ({exp1_orchestrator_correct}/{len(exp1)})")
    
    # Full system accuracy (Experiment III)
    exp3_orchestrator_correct = (exp3['Orchestrator'] == exp3['Category']).sum()
    exp3_accuracy = exp3_orchestrator_correct / len(exp3)
    print(f"Full System (Exp III): {exp3_accuracy:.4f} ({exp3_orchestrator_correct}/{len(exp3)})")
    
    # Run ablations
    print(f"\nBaseline (Exp I): {exp1_accuracy:.4f} ({exp1_orchestrator_correct}/{len(exp1)})")
    
    # Full system accuracy (Experiment III)
    exp3_orchestrator_correct = (exp3['Orchestrator'] == exp3['Category']).sum()
    exp3_accuracy = exp3_orchestrator_correct / len(exp3)
    print(f"Full System (Exp III): {exp3_accuracy:.4f} ({exp3_orchestrator_correct}/{len(exp3)})")
    
    # Ablation Component 2: Trust only (no RAG)
    trust_predictions, trust_accuracy = ablation_trust_only(exp3, trust_profiles)
    
    # Ablation Component 3: RAG only (no trust)
    rag_predictions, rag_accuracy = ablation_rag_only(exp3)
    
    # Calculate contributions and synergy
    contribution_results = calculate_component_contributions(
        exp1_accuracy, trust_accuracy, rag_accuracy, exp3_accuracy
    )
    
    # Create formatted table
    create_ablation_table(contribution_results)
    
    # Save detailed results
    output_dir = Path('Results/ablation_study')
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump({
            'summary': contribution_results,
            'trust_only_predictions': trust_predictions,
            'rag_only_predictions': rag_predictions
        }, f, indent=2)
    
    # Save CSV for trust-only configuration
    pd.DataFrame(trust_predictions).to_csv(
        output_dir / 'trust_only_predictions.csv', 
        index=False
    )
    
    # Save CSV for RAG-only configuration
    pd.DataFrame(rag_predictions).to_csv(
        output_dir / 'rag_only_predictions.csv', 
        index=False
    )
    
    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  - ablation_results.json")
    print(f"  - trust_only_predictions.csv")
    print(f"  - rag_only_predictions.csv")
    
    return contribution_results

if __name__ == "__main__":
    results = main()
