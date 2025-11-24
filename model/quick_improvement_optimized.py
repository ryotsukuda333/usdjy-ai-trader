"""
Step 11: Optimized Quick Model Improvement
Fast hyperparameter tuning with local data caching to avoid timeout issues
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import json
import pickle
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.data_fetcher import fetch_usdjpy_data
from features.feature_engineer import engineer_features


def load_or_cache_data():
    """Load data with caching to avoid repeated fetches."""

    cache_file = project_root / 'model' / 'cached_training_data.pkl'

    # Try to use cached data if recent
    if cache_file.exists():
        print("📊 Loading cached training data...")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
            X, y, features = cached['X'], cached['y'], cached['features']
            print(f"✅ Cached data loaded: {len(X)} samples, {len(features)} features")
            return X, y, features

    # Otherwise fetch and cache
    print("📊 Fetching fresh USDJPY data...")
    df_ohlcv = fetch_usdjpy_data(years=2)  # Reduced from 3 to 2 years for speed
    print(f"   OHLCV data: {len(df_ohlcv)} candles")

    print("🔧 Engineering features...")
    df_features = engineer_features(df_ohlcv)
    print(f"   Features engineered: {len(df_features)} rows")

    # Align data
    rows_dropped = len(df_ohlcv) - len(df_features)
    if rows_dropped > 0:
        df_ohlcv = df_ohlcv.iloc[rows_dropped:]

    # Load feature columns
    with open(project_root / 'model' / 'feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    X = df_features[feature_columns].values
    close_prices = df_ohlcv['Close'].values
    pct_changes = np.diff(close_prices) / close_prices[:-1] * 100
    y = np.where(pct_changes > 0, 1, 0)
    X = X[:-1]

    print(f"✅ Data prepared: {len(X)} samples, {len(feature_columns)} features")

    # Cache for next run
    with open(cache_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'features': feature_columns}, f)
    print(f"   Cached to: {cache_file}")

    return X, y, feature_columns


def main():
    """Quick improvement with focused tuning."""

    print("=" * 70)
    print("STEP 11: Optimized Quick Model Improvement")
    print("=" * 70)

    # Load data (with caching)
    X, y, features = load_or_cache_data()

    # Load original metrics
    with open(project_root / 'model' / 'best_hyperparameters.json', 'r') as f:
        orig = json.load(f)

    print(f"\n📊 Original Model Performance:")
    print(f"   Train F1: {orig['train_f1']:.4f}")
    print(f"   CV Score: {orig['best_cv_score']:.4f}")

    print(f"\n11.2: Testing improved parameters...")

    # Test 3 promising parameter combinations
    test_configs = [
        {
            'learning_rate': 0.05,
            'max_depth': 5,
            'n_estimators': 200,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'name': 'Conservative (reduced depth)'
        },
        {
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'name': 'Balanced'
        },
        {
            'learning_rate': 0.08,
            'max_depth': 6,
            'n_estimators': 250,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'name': 'Optimized'
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for config in test_configs:
        name = config.pop('name')

        print(f"\n   Testing: {name}")

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **config
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        mean_score = scores.mean()

        print(f"      F1 Score: {mean_score:.4f} (±{scores.std():.4f})")

        results.append({
            'config': name,
            'params': config,
            'cv_f1': mean_score,
            'cv_std': scores.std(),
        })

    # Find best
    best_result = max(results, key=lambda x: x['cv_f1'])
    best_config = best_result['params']

    improvement = ((best_result['cv_f1'] - orig['best_cv_score']) /
                  orig['best_cv_score'] * 100)

    print(f"\n🏆 Best Configuration: {best_result['config']}")
    print(f"   CV F1: {best_result['cv_f1']:.4f}")
    print(f"   Improvement: {improvement:+.2f}%")

    print(f"\n11.4: Training final improved model...")

    # Train final model on full data
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **best_config
    )

    final_model.fit(X, y)

    # Get training metrics
    preds = final_model.predict(X)
    train_f1 = f1_score(y, preds)
    train_acc = accuracy_score(y, preds)

    # Save results
    model_path = Path(__file__).parent

    improvement_data = {
        'original_f1': float(orig['best_cv_score']),
        'improved_f1': float(best_result['cv_f1']),
        'improvement_pct': float(improvement),
        'best_params': best_config,
        'train_f1': float(train_f1),
        'train_accuracy': float(train_acc),
        'method': 'Optimized quick tuning with 3 configurations + caching',
        'data_period_years': 2,
        'total_samples': len(X),
    }

    with open(model_path / 'step11_improvement.json', 'w') as f:
        json.dump(improvement_data, f, indent=2)

    # Save improved model
    final_model.get_booster().save_model(str(model_path / 'xgb_model_v2.json'))

    # Summary
    print(f"\n" + "=" * 70)
    print(f"✅ STEP 11: Quick Model Improvement Complete")
    print(f"=" * 70)

    print(f"""
📈 Results:
   Original F1 Score: {orig['best_cv_score']:.4f}
   Improved F1 Score: {best_result['cv_f1']:.4f}
   Improvement: {improvement:+.2f}%

🔧 Best Parameters:
   learning_rate: {best_config['learning_rate']}
   max_depth: {best_config['max_depth']}
   n_estimators: {best_config['n_estimators']}
   subsample: {best_config['subsample']}
   colsample_bytree: {best_config['colsample_bytree']}

📁 Files:
   ✓ xgb_model_v2.json (improved model)
   ✓ step11_improvement.json (results)
   ✓ cached_training_data.pkl (data cache for future runs)

💡 Next Steps:
   1. Test v2 model in backtest pipeline
   2. Compare results with original model
   3. Deploy if performance improves
""")

    return final_model, best_config


if __name__ == "__main__":
    model, params = main()
