"""
Advanced Hyperparameter Tuning for XGBoost Model
Uses Bayesian Optimization (via hyperopt) for efficient hyperparameter search
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json
import time
from pathlib import Path

# Check if hyperopt is available, if not use basic grid search
try:
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("⚠️  hyperopt not available, using basic grid search instead")


def load_training_data():
    """Load and prepare training data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from features.data_fetcher import fetch_usdjpy_data
    from features.feature_engineer import engineer_features

    print("📊 Loading training data...")
    df_ohlcv = fetch_usdjpy_data(years=3)
    df_features = engineer_features(df_ohlcv)

    # Align data
    rows_dropped = len(df_ohlcv) - len(df_features)
    if rows_dropped > 0:
        df_ohlcv = df_ohlcv.iloc[rows_dropped:]

    # Load feature columns
    with open(Path(__file__).parent / 'feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    # Prepare X and y
    X = df_features[feature_columns].values

    # Generate labels based on forward returns
    close_prices = df_ohlcv['Close'].values
    pct_changes = np.diff(close_prices) / close_prices[:-1] * 100
    y = np.where(pct_changes > 0, 1, 0)  # 1 for buy (positive return), 0 for sell

    # Align X with y
    X = X[:-1]  # Remove last row to match y length

    print(f"✅ Data loaded: {len(X)} samples, {len(feature_columns)} features")
    print(f"   Class distribution: {(y == 1).sum()} BUY, {(y == 0).sum()} SELL")
    print(f"   Class ratio: {(y == 1).sum() / len(y) * 100:.1f}% BUY")

    return X, y, feature_columns


def objective_bayesian(params):
    """Objective function for Bayesian optimization."""

    # Round integer parameters
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    # Ensure reasonable parameter ranges
    params['learning_rate'] = np.clip(params['learning_rate'], 0.001, 0.5)
    params['subsample'] = np.clip(params['subsample'], 0.5, 1.0)
    params['colsample_bytree'] = np.clip(params['colsample_bytree'], 0.5, 1.0)

    print(f"\n🔧 Testing parameters:")
    for key, val in params.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.4f}")
        else:
            print(f"   {key}: {val}")

    # Create model with current parameters
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **params
    )

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

    mean_score = scores.mean()
    std_score = scores.std()

    print(f"   F1 Score: {mean_score:.4f} (+/- {std_score:.4f})")

    # Return loss (we want to maximize F1, so return negative)
    return {'loss': -mean_score, 'status': STATUS_OK, 'scores': scores}


def bayesian_optimization(X, y):
    """Perform Bayesian optimization using hyperopt."""

    print("\n" + "=" * 70)
    print("🔍 STEP 11.2: Bayesian Optimization for Hyperparameter Tuning")
    print("=" * 70)

    global X_train, y_train
    X_train = X
    y_train = y

    # Define search space
    search_space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.1)),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
        'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(10.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.001), np.log(10.0)),
    }

    print(f"\n📝 Search Space:")
    print(f"   learning_rate: [0.001, 0.1] (log scale)")
    print(f"   max_depth: [3, 10]")
    print(f"   n_estimators: [100, 500]")
    print(f"   subsample: [0.5, 1.0]")
    print(f"   colsample_bytree: [0.5, 1.0]")
    print(f"   reg_alpha, reg_lambda: [0.001, 10.0] (log scale)")

    print(f"\n🚀 Starting Bayesian optimization (20 iterations)...")
    start_time = time.time()

    trials = Trials()
    best = fmin(
        fn=objective_bayesian,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials,
        verbose=0
    )

    elapsed_time = time.time() - start_time

    # Round best parameters
    best['max_depth'] = int(best['max_depth'])
    best['n_estimators'] = int(best['n_estimators'])
    best['min_child_weight'] = int(best['min_child_weight'])

    print(f"\n✅ Optimization complete ({elapsed_time:.1f}s)\n")
    print(f"🏆 Best Parameters:")
    for key, val in best.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.4f}")
        else:
            print(f"   {key}: {val}")

    # Get best trial
    best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_trial_idx]
    best_loss = best_trial['result']['loss']
    best_scores = best_trial['result']['scores']

    print(f"\n📊 Best Trial Performance:")
    print(f"   F1 Score: {-best_loss:.4f}")
    print(f"   Mean F1: {best_scores.mean():.4f} (+/- {best_scores.std():.4f})")

    return best, -best_loss, trials


def grid_search_alternative(X, y):
    """Simple grid search as alternative to Bayesian optimization."""

    print("\n" + "=" * 70)
    print("🔍 STEP 11.2: Grid Search for Hyperparameter Tuning")
    print("=" * 70)

    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 5, 6, 7],
        'n_estimators': [200, 300, 400],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    print(f"\n📝 Search Grid:")
    for key, vals in param_grid.items():
        print(f"   {key}: {vals}")

    best_score = 0
    best_params = {}
    trial_count = 0
    total_trials = 1
    for v in param_grid.values():
        total_trials *= len(v)

    print(f"\n🚀 Starting grid search ({total_trials} combinations)...")
    start_time = time.time()

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
                            verbosity=0
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
                                'colsample_bytree': colsamp
                            }
                            print(f"   Trial {trial_count}/{total_trials}: F1={mean_score:.4f} ✓ (NEW BEST)")
                        elif trial_count % 10 == 0:
                            print(f"   Trial {trial_count}/{total_trials}: F1={mean_score:.4f}")

    elapsed_time = time.time() - start_time

    print(f"\n✅ Grid search complete ({elapsed_time:.1f}s)\n")
    print(f"🏆 Best Parameters:")
    for key, val in best_params.items():
        print(f"   {key}: {val:.4f}" if isinstance(val, float) else f"   {key}: {val}")

    print(f"\n📊 Best Trial Performance:")
    print(f"   F1 Score: {best_score:.4f}")

    return best_params, best_score, None


def train_final_model(X, y, best_params):
    """Train final model with best parameters."""

    print("\n" + "=" * 70)
    print("🚀 Training final model with best parameters")
    print("=" * 70)

    # Add default parameters
    final_params = {
        'objective': 'binary:logistic',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'scale_pos_weight': 1,  # Can be adjusted for class imbalance
    }
    final_params.update(best_params)

    model = xgb.XGBClassifier(**final_params)

    # Train on full data
    print(f"\n🔄 Training on full dataset ({len(X)} samples)...")
    model.fit(X, y)

    # Cross-validation score on final model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

    print(f"\n✅ Model trained successfully")
    print(f"   CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   Training time: Complete")

    return model, cv_scores


def main():
    """Main execution function."""

    # Load data
    X, y, feature_columns = load_training_data()

    # Perform optimization
    if HYPEROPT_AVAILABLE:
        best_params, best_score, trials = bayesian_optimization(X, y)
    else:
        best_params, best_score, trials = grid_search_alternative(X, y)

    # Train final model
    model, cv_scores = train_final_model(X, y, best_params)

    # Save results
    results = {
        'best_params': best_params,
        'best_score': float(best_score),
        'cv_scores': {
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'scores': [float(s) for s in cv_scores]
        },
        'optimization_method': 'bayesian' if HYPEROPT_AVAILABLE else 'grid_search'
    }

    output_path = Path(__file__).parent / 'advanced_tuning_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
    print(f"\n" + "=" * 70)
    print(f"✅ Step 11.2 Complete: Advanced Hyperparameter Tuning")
    print(f"=" * 70)
    print(f"\nBest F1 Score: {best_score:.4f}")
    print(f"Optimization Method: {'Bayesian (hyperopt)' if HYPEROPT_AVAILABLE else 'Grid Search'}")

    return model, best_params, results


if __name__ == "__main__":
    model, best_params, results = main()
