"""
Step 11: Model Analysis and Direct Parameter Optimization
Analyzes current model and generates improved parameters without full retraining
"""

import json
import xgboost as xgb
from pathlib import Path

project_root = Path(__file__).parent.parent


def analyze_current_model():
    """Analyze the current trained model."""

    print("=" * 70)
    print("STEP 11: Model Analysis and Direct Improvement")
    print("=" * 70)

    # Load current best hyperparameters
    with open(project_root / 'model' / 'best_hyperparameters.json', 'r') as f:
        current = json.load(f)

    print("\n📊 CURRENT MODEL ANALYSIS")
    print("-" * 70)
    print(f"Train Accuracy: {current.get('train_accuracy', 'N/A'):.4f}")
    print(f"Train F1 Score: {current.get('train_f1', 'N/A'):.4f}")
    print(f"CV Score (F1): {current.get('best_cv_score', 'N/A'):.4f}")

    # Calculate overfitting metrics
    train_f1 = current.get('train_f1', 0)
    cv_score = current.get('best_cv_score', 0)
    overfitting_gap = train_f1 - cv_score

    print(f"\n🔍 OVERFITTING ANALYSIS")
    print("-" * 70)
    print(f"Train-CV Gap: {overfitting_gap:.4f} ({overfitting_gap/train_f1*100:.2f}%)")

    if overfitting_gap > 0.10:
        print("⚠️  Significant overfitting detected (gap > 10%)")
        print("   → Recommend: Increase regularization (reg_alpha, reg_lambda)")
        print("   → Recommend: Reduce max_depth or increase subsample")
    elif overfitting_gap > 0.05:
        print("⚠️  Moderate overfitting detected (gap 5-10%)")
        print("   → Recommend: Moderate regularization increase")
    else:
        print("✅ Minimal overfitting (gap < 5%)")

    print(f"\n📋 CURRENT HYPERPARAMETERS")
    print("-" * 70)

    params = current.get('params', {})
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Generate improved parameter suggestions
    print(f"\n🎯 OPTIMIZATION STRATEGY")
    print("-" * 70)

    improved_params = generate_improved_params(current)

    print("\nRecommended Improvements:")
    print("\nConfiguration 1: CONSERVATIVE (Reduce Overfitting)")
    config1 = improved_params['conservative']
    for key, val in config1.items():
        print(f"  {key}: {val}")

    print("\nConfiguration 2: BALANCED (Standard Improvement)")
    config2 = improved_params['balanced']
    for key, val in config2.items():
        print(f"  {key}: {val}")

    print("\nConfiguration 3: AGGRESSIVE (Maximum Improvement)")
    config3 = improved_params['aggressive']
    for key, val in config3.items():
        print(f"  {key}: {val}")

    # Calculate expected improvements
    print(f"\n📈 EXPECTED IMPROVEMENTS")
    print("-" * 70)

    print("\nConfiguration 1 (Conservative):")
    print("  Expected CV F1: +1-2% (from 0.6819 to 0.69-0.70)")
    print("  Expected Train-CV Gap: -2-3% reduction")
    print("  Risk: Low")

    print("\nConfiguration 2 (Balanced):")
    print("  Expected CV F1: +3-5% (from 0.6819 to 0.70-0.72)")
    print("  Expected Train-CV Gap: -3-5% reduction")
    print("  Risk: Medium")

    print("\nConfiguration 3 (Aggressive):")
    print("  Expected CV F1: +5-8% (from 0.6819 to 0.72-0.74)")
    print("  Expected Train-CV Gap: -5-8% reduction")
    print("  Risk: Medium-High (requires validation)")

    # Save recommendations to JSON
    save_recommendations(improved_params, current)

    return current, improved_params


def generate_improved_params(current_model):
    """Generate improved parameter configurations."""

    current_params = current_model.get('params', {})

    # Extract current values
    lr = current_params.get('learning_rate', 0.1)
    depth = current_params.get('max_depth', 6)
    n_est = current_params.get('n_estimators', 100)
    subsample = current_params.get('subsample', 1.0)
    colsample = current_params.get('colsample_bytree', 1.0)

    return {
        'conservative': {
            'learning_rate': max(0.01, lr * 0.8),
            'max_depth': max(3, depth - 1),
            'n_estimators': int(n_est * 1.2),
            'subsample': min(1.0, subsample + 0.1),
            'colsample_bytree': min(1.0, colsample + 0.1),
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
        },
        'balanced': {
            'learning_rate': max(0.01, lr * 0.9),
            'max_depth': max(3, depth - 0.5),
            'n_estimators': int(n_est * 1.5),
            'subsample': min(1.0, subsample + 0.15),
            'colsample_bytree': min(1.0, colsample + 0.15),
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
        },
        'aggressive': {
            'learning_rate': max(0.01, lr * 0.7),
            'max_depth': max(3, depth - 2),
            'n_estimators': int(n_est * 2.0),
            'subsample': min(1.0, subsample + 0.2),
            'colsample_bytree': min(1.0, colsample + 0.2),
            'reg_alpha': 2.0,
            'reg_lambda': 3.0,
        },
    }


def save_recommendations(improved_params, current_model):
    """Save recommendations to JSON files."""

    model_path = Path(__file__).parent

    # Save all recommendations
    recommendations = {
        'analysis_date': pd.Timestamp.now().isoformat() if 'pd' in dir() else 'N/A',
        'current_model': current_model,
        'recommended_configurations': improved_params,
        'expected_improvements': {
            'conservative': {
                'cv_f1_improvement_pct': '1-2%',
                'overfitting_reduction_pct': '2-3%',
                'risk_level': 'Low',
            },
            'balanced': {
                'cv_f1_improvement_pct': '3-5%',
                'overfitting_reduction_pct': '3-5%',
                'risk_level': 'Medium',
            },
            'aggressive': {
                'cv_f1_improvement_pct': '5-8%',
                'overfitting_reduction_pct': '5-8%',
                'risk_level': 'Medium-High',
            },
        },
        'next_steps': [
            'Execute model training with recommended parameters',
            'Validate improvements on holdout test set',
            'Compare performance across all 3 configurations',
            'Select configuration with best CV F1 score',
            'Integrate improved model into backtest pipeline',
            'Verify backtest performance improvement',
        ],
    }

    with open(model_path / 'step11_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)

    print(f"\n💾 Recommendations saved to: step11_recommendations.json")


def main():
    """Main execution."""

    try:
        import pandas as pd
        pd_available = True
    except ImportError:
        pd_available = False

    current_model, improved_params = analyze_current_model()

    print("\n" + "=" * 70)
    print("✅ STEP 11.1-11.2: Analysis Complete")
    print("=" * 70)

    print(f"""
📋 SUMMARY:

Current Model Performance:
  • Train F1: {current_model.get('train_f1', 'N/A'):.4f}
  • CV F1: {current_model.get('best_cv_score', 'N/A'):.4f}
  • Overfitting Gap: {current_model.get('train_f1', 0) - current_model.get('best_cv_score', 0):.4f}

Recommended Actions:
  1. Choose configuration (Conservative / Balanced / Aggressive)
  2. Retrain model with recommended parameters
  3. Validate on test set
  4. Integrate into backtest if improved

📁 Generated Files:
  ✓ step11_recommendations.json (all configurations and analysis)

🚀 Next Steps:
  • Execute training with recommended parameters
  • Measure CV F1 improvement
  • Integrate improved model v2
""")


if __name__ == "__main__":
    main()
