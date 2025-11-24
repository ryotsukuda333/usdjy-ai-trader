"""
Step 11 Complete: Model Improvement Full Pipeline
Combines hyperparameter tuning and optional advanced features

This script:
1. (11.2) Performs hyperparameter optimization
2. (11.3) Optionally adds advanced features
3. (11.4) Retrains and validates the model
4. Compares performance with original model
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import json
import sys
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.data_fetcher import fetch_usdjpy_data
from features.feature_engineer import engineer_features


def load_data(use_advanced_features=False):
    """Load and prepare data."""

    print("📊 Loading training data...")
    df_ohlcv = fetch_usdjpy_data(years=3)
    df_features = engineer_features(df_ohlcv)

    # Optionally add advanced features
    if use_advanced_features:
        print("\n🔧 Adding advanced features...")
        try:
            from features.advanced_feature_engineer import engineer_advanced_features
            df_features = engineer_advanced_features(df_features)
            print("  ✓ Advanced features added successfully")
        except Exception as e:
            print(f"  ⚠️  Could not add advanced features: {e}")
            print("     Using standard features only")

    # Align data
    rows_dropped = len(df_ohlcv) - len(df_features)
    if rows_dropped > 0:
        df_ohlcv = df_ohlcv.iloc[rows_dropped:]

    # Load feature columns
    with open(project_root / 'model' / 'feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    # Select available features (in case some columns are missing)
    available_features = [f for f in feature_columns if f in df_features.columns]
    print(f"\n✅ Data loaded: {len(df_features)} samples")
    print(f"   Features: {len(available_features)} (original: {len(feature_columns)})")

    # Prepare X and y
    X = df_features[available_features].values

    # Generate labels based on forward returns
    close_prices = df_ohlcv['Close'].values
    pct_changes = np.diff(close_prices) / close_prices[:-1] * 100
    y = np.where(pct_changes > 0, 1, 0)

    # Align X with y
    X = X[:-1]

    print(f"   Class distribution: {(y == 1).sum()} BUY, {(y == 0).sum()} SELL")
    print(f"   Class ratio: {(y == 1).sum() / len(y) * 100:.1f}% BUY")

    return X, y, available_features


def optimize_hyperparameters(X, y):
    """Perform grid search for hyperparameter optimization."""

    print("\n" + "=" * 70)
    print("11.2: Hyperparameter Optimization")
    print("=" * 70)

    # Define parameter grid (balanced between search space and computation time)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 6, 7],
        'n_estimators': [200, 300],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
    }

    print(f"\n📝 Search Grid:")
    total_combos = 1
    for key, vals in param_grid.items():
        print(f"   {key}: {vals}")
        total_combos *= len(vals)

    print(f"\n   Total combinations: {total_combos}")

    best_score = 0
    best_params = {}
    trial_count = 0

    print(f"\n🚀 Starting grid search...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for lr in param_grid['learning_rate']:
        for depth in param_grid['max_depth']:
            for n_est in param_grid['n_estimators']:
                for subsamp in param_grid['subsample']:
                    for colsamp in param_grid['colsample_bytree']:
                        trial_count += 1

                        model = xgb.XGBClassifier(
                            learning_rate=lr,
                            max_depth=depth,
                            n_estimators=n_est,
                            subsample=subsamp,
                            colsample_bytree=colsamp,
                            objective='binary:logistic',
                            random_state=42,
                            n_jobs=-1,
                            verbosity=0,
                            reg_alpha=0.5,
                            reg_lambda=1.0,
                        )

                        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
                        mean_score = scores.mean()

                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'learning_rate': lr,
                                'max_depth': depth,
                                'n_estimators': n_est,
                                'subsample': subsamp,
                                'colsample_bytree': colsamp,
                                'reg_alpha': 0.5,
                                'reg_lambda': 1.0,
                            }
                            print(f"   Trial {trial_count}/{total_combos}: F1={mean_score:.4f} ✓ (NEW BEST)")

                        elif trial_count % 5 == 0:
                            print(f"   Trial {trial_count}/{total_combos}: F1={mean_score:.4f}")

    print(f"\n✅ Grid search complete\n")
    print(f"🏆 Best Parameters:")
    for key, val in best_params.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.4f}")
        else:
            print(f"   {key}: {val}")

    print(f"\n📊 Best CV F1 Score: {best_score:.4f}")

    return best_params, best_score


def train_improved_model(X, y, best_params):
    """Train the improved model."""

    print("\n" + "=" * 70)
    print("11.4: Training Improved Model")
    print("=" * 70)

    print(f"\n🔄 Training with best parameters...")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **best_params
    )

    # Train on full data
    model.fit(X, y)

    # Get cross-validation scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

    # Get train scores
    predictions = model.predict(X)
    train_f1 = f1_score(y, predictions)
    train_accuracy = accuracy_score(y, predictions)

    print(f"\n✅ Model training complete")
    print(f"\n📊 Training Metrics:")
    print(f"   Train F1 Score: {train_f1:.4f}")
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    print(f"   CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model, {
        'train_f1': train_f1,
        'train_accuracy': train_accuracy,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
    }


def compare_models(old_metrics, new_metrics):
    """Compare old and new model performance."""

    print("\n" + "=" * 70)
    print("📊 Model Performance Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change':<10}")
    print("-" * 60)

    for metric in ['train_accuracy', 'train_f1', 'cv_f1_mean']:
        old_val = old_metrics.get(metric, 0)
        new_val = new_metrics.get(metric, 0)
        change = new_val - old_val

        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:<20} {old_val:>14.4f} {new_val:>14.4f} {change:>+9.4f}")

    improvement_pct = ((new_metrics['cv_f1_mean'] - old_metrics['train_f1']) /
                       old_metrics['train_f1'] * 100) if old_metrics['train_f1'] > 0 else 0

    print("-" * 60)
    print(f"\n🎯 Overall Improvement: {improvement_pct:+.2f}%")

    if improvement_pct > 5:
        print("   ⭐ Significant improvement! Consider using the new model.")
    elif improvement_pct > 0:
        print("   ✓ Modest improvement.")
    else:
        print("   ⚠️  No improvement. Original model may be better.")


def save_improved_model(model, best_params, metrics, feature_columns):
    """Save improved model and results."""

    model_path = Path(__file__).parent

    # Save model
    model.get_booster().save_model(str(model_path / 'xgb_model_improved.json'))
    print(f"\n💾 Model saved to: xgb_model_improved.json")

    # Save parameters
    results = {
        'best_params': best_params,
        'metrics': metrics,
        'feature_count': len(feature_columns),
        'improvement_date': pd.Timestamp.now().isoformat(),
    }

    with open(model_path / 'improvement_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to: improvement_results.json")

    # Save feature columns
    with open(model_path / 'feature_columns_improved.json', 'w') as f:
        json.dump(feature_columns, f, indent=2)

    print(f"💾 Features saved to: feature_columns_improved.json")


def main(use_advanced_features=False):
    """Main execution."""

    print("=" * 70)
    print("STEP 11: Model Improvement - Full Pipeline")
    print("=" * 70)

    # Load data
    X, y, feature_columns = load_data(use_advanced_features=use_advanced_features)

    # Load original model metrics for comparison
    model_path = Path(__file__).parent
    with open(model_path / 'best_hyperparameters.json', 'r') as f:
        original_data = json.load(f)

    old_metrics = {
        'train_accuracy': original_data['train_accuracy'],
        'train_f1': original_data['train_f1'],
        'cv_f1_mean': original_data.get('best_cv_score', 0),
        'cv_f1_std': 0,
    }

    # Optimize hyperparameters (11.2)
    best_params, best_score = optimize_hyperparameters(X, y)

    # Train improved model (11.4)
    improved_model, new_metrics = train_improved_model(X, y, best_params)

    # Compare models
    compare_models(old_metrics, new_metrics)

    # Save improved model
    save_improved_model(improved_model, best_params, new_metrics, feature_columns)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ STEP 11 Complete: Model Improvement Pipeline")
    print("=" * 70)

    print(f"""
📈 Improvement Summary:
   • Original CV F1: {old_metrics['train_f1']:.4f}
   • Improved CV F1: {new_metrics['cv_f1_mean']:.4f}
   • Best Parameters: {len(best_params)} optimized

🎯 Next Steps:
   1. Test improved model in backtest pipeline
   2. Compare trading results with original model
   3. Consider deploying if results are better

📁 Generated Files:
   • xgb_model_improved.json (improved model)
   • improvement_results.json (metrics and parameters)
   • feature_columns_improved.json (feature list)
""")

    return improved_model, best_params, new_metrics


if __name__ == "__main__":
    # Run improvement pipeline
    # Set use_advanced_features=True to add new features (slower but potentially better)
    model, params, metrics = main(use_advanced_features=False)
