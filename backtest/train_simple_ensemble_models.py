"""Train Simplified Ensemble Models - Phase 5-C

Trains XGBoost, Random Forest, Gradient Boosting, and Neural Network models on 2-year USDJPY data.
Uses scikit-learn models instead of LightGBM/CatBoost to avoid external dependencies.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from features.multi_timeframe_fetcher import MultiTimeframeFetcher
from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from utils.data_labeler import label_data


class SimpleEnsembleTrainer:
    """Train simplified ensemble with scikit-learn compatible models."""

    def __init__(self, random_state: int = 42):
        """Initialize trainer."""
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.scaler_nn = None

    def prepare_data(self, df_features, test_size=0.2, val_size=0.2):
        """Prepare data splits."""
        # Use all columns except target (including OHLCV)
        feature_cols = [c for c in df_features.columns if c != 'target']
        X = df_features[feature_cols].values
        y = df_features['target'].values

        # Time-series split
        split_idx = int(len(X) * (1 - test_size))
        X_tv, X_test = X[:split_idx], X[split_idx:]
        y_tv, y_test = y[:split_idx], y[split_idx:]

        split_idx2 = int(len(X_tv) * (1 - val_size))
        X_train, X_val = X_tv[:split_idx2], X_tv[split_idx2:]
        y_train, y_val = y_tv[:split_idx2], y_tv[split_idx2:]

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"  Target: {np.bincount(y_train)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost."""
        print("\n[XGBoost]")
        start = time.time()

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print(f"  ✓ AUC: {auc:.4f} ({time.time()-start:.1f}s)")
        return model, {'auc': auc}

    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest."""
        print("\n[Random Forest]")
        start = time.time()

        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=self.random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print(f"  ✓ AUC: {auc:.4f} ({time.time()-start:.1f}s)")
        return model, {'auc': auc}

    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting."""
        print("\n[Gradient Boosting]")
        start = time.time()

        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=7,
            subsample=0.8, random_state=self.random_state
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print(f"  ✓ AUC: {auc:.4f} ({time.time()-start:.1f}s)")
        return model, {'auc': auc}

    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train Neural Network."""
        print("\n[Neural Network]")
        start = time.time()

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        self.scaler_nn = scaler

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', learning_rate_init=0.001, batch_size=32,
            max_iter=500, early_stopping=True, validation_fraction=0.2,
            random_state=self.random_state
        )
        model.fit(X_train_s, y_train)

        y_proba = model.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print(f"  ✓ AUC: {auc:.4f} ({time.time()-start:.1f}s)")
        return model, {'auc': auc}

    def calculate_weights(self, metrics):
        """Calculate ensemble weights."""
        aucs = {name: m['auc'] for name, m in metrics.items()}
        total = sum(aucs.values())
        weights = {name: auc / total for name, auc in aucs.items()}
        return weights

    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble."""
        pred = {}
        pred['xgboost'] = self.models['xgboost'].predict_proba(X_test)[:, 1]
        pred['random_forest'] = self.models['random_forest'].predict_proba(X_test)[:, 1]
        pred['gradient_boosting'] = self.models['gradient_boosting'].predict_proba(X_test)[:, 1]

        X_test_s = self.scaler_nn.transform(X_test)
        pred['neural_network'] = self.models['neural_network'].predict_proba(X_test_s)[:, 1]

        # Weighted ensemble
        ens_proba = np.zeros_like(pred['xgboost'])
        for name, weight in self.weights.items():
            ens_proba += pred[name] * weight

        ens_pred = (ens_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, ens_proba)
        acc = accuracy_score(y_test, ens_pred)
        f1 = f1_score(y_test, ens_pred, zero_division=0)

        return {'auc': auc, 'accuracy': acc, 'f1': f1}

    def train(self, df_features):
        """Train full ensemble."""
        print("=" * 80)
        print("SIMPLIFIED ENSEMBLE TRAINING")
        print("=" * 80)

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df_features)

        print("\n[Training Models]")
        metrics = {}

        self.models['xgboost'], metrics['xgboost'] = self.train_xgboost(
            X_train, y_train, X_val, y_val)
        self.models['random_forest'], metrics['random_forest'] = self.train_random_forest(
            X_train, y_train, X_val, y_val)
        self.models['gradient_boosting'], metrics['gradient_boosting'] = self.train_gradient_boosting(
            X_train, y_train, X_val, y_val)
        self.models['neural_network'], metrics['neural_network'] = self.train_neural_network(
            X_train, y_train, X_val, y_val)

        print("\n[Ensemble Weights]")
        self.weights = self.calculate_weights(metrics)
        for name, w in self.weights.items():
            print(f"  {name:20s}: {w:.4f}")

        print("\n[Ensemble Evaluation]")
        test_metrics = self.evaluate_ensemble(X_test, y_test)
        print(f"  Test AUC: {test_metrics['auc']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")

        return {
            'validation_metrics': metrics,
            'test_metrics': test_metrics,
            'ensemble_weights': self.weights
        }

    def save(self, output_dir="model/ensemble_models"):
        """Save models."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        with open(output_path / "xgboost_model.pkl", "wb") as f:
            pickle.dump(self.models['xgboost'], f)
        with open(output_path / "random_forest_model.pkl", "wb") as f:
            pickle.dump(self.models['random_forest'], f)
        with open(output_path / "gradient_boosting_model.pkl", "wb") as f:
            pickle.dump(self.models['gradient_boosting'], f)
        with open(output_path / "neural_network_model.pkl", "wb") as f:
            pickle.dump(self.models['neural_network'], f)
        with open(output_path / "nn_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler_nn, f)
        with open(output_path / "ensemble_weights.json", "w") as f:
            json.dump(self.weights, f, indent=2)

        print(f"\n✓ Models saved to {output_path}")
        return str(output_path)


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("SIMPLIFIED ENSEMBLE TRAINING - PHASE 5-C")
    print("=" * 80)

    try:
        # Fetch data
        print("\n[1/4] Fetching data...")
        fetcher = MultiTimeframeFetcher()
        data_dict = fetcher.fetch_and_resample(years=2)
        print(f"✓ 1D: {len(data_dict['1d'])}, 5m: {len(data_dict['5m'])} bars")

        # Engineer features
        print("\n[2/4] Engineering features...")
        engineer = MultiTimeframeFeatureEngineer()
        features_dict = engineer.engineer_all_timeframes(data_dict)
        df_1d = features_dict['1d'].copy()
        print(f"✓ {len(df_1d)} rows × {len(df_1d.columns)} columns")

        # Label data
        print("\n[3/4] Labeling data...")
        df_1d = label_data(df_1d)
        print(f"✓ {len(df_1d)} training samples")

        # Train ensemble
        print("\n[4/4] Training ensemble...")
        trainer = SimpleEnsembleTrainer()
        results = trainer.train(df_1d)

        # Save
        trainer.save()

        # Save results
        Path("backtest").mkdir(exist_ok=True)
        with open("backtest/ENSEMBLE_TRAINING_RESULTS.json", "w") as f:
            json.dump({
                'validation_metrics': {k: v for k, v in results['validation_metrics'].items()},
                'test_metrics': results['test_metrics'],
                'ensemble_weights': results['ensemble_weights']
            }, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
