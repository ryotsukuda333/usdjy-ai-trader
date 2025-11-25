"""Ensemble Learning Model Trainer for Phase 5-C

Implements training of multiple complementary models (XGBoost, LightGBM, CatBoost, Neural Network)
and creates an ensemble system for improved prediction robustness.

Ensemble Strategy:
- Base Learners: XGBoost, LightGBM, CatBoost, Simple Neural Network
- Meta-Learner: Weighted averaging based on validation performance
- Output: Ensemble probability predictions combining all base learners
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
import warnings

warnings.filterwarnings('ignore')

# Optional: LightGBM and CatBoost (not required for basic ensemble)
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None


class EnsembleTrainer:
    """Train and manage ensemble of multiple models."""

    def __init__(self, data_dir: str = "model", random_state: int = 42):
        """Initialize ensemble trainer."""
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self.val_metrics = {}

    def prepare_data(self, df_features: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
        """Prepare train/validation/test split for ensemble training."""
        print("\n[Data Preparation]")

        X = df_features.drop('target', axis=1).values
        y = df_features['target'].values

        print(f"  Total samples: {len(X)}")

        # Time-series aware split
        split_idx = int(len(X) * (1 - test_size))
        X_trainval, X_test = X[:split_idx], X[split_idx:]
        y_trainval, y_test = y[:split_idx], y[split_idx:]

        val_split_idx = int(len(X_trainval) * (1 - val_size))
        X_train, X_val = X_trainval[:val_split_idx], X_trainval[val_split_idx:]
        y_train, y_val = y_trainval[:val_split_idx], y_trainval[val_split_idx:]

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"  Target distribution (train): {np.bincount(y_train)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        print("\n[Training XGBoost]")
        start = time.time()

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=-1, tree_method='hist'
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20, verbose=False)

        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_val),
            'f1': f1_score(y_val, y_pred_val, zero_division=0),
            'auc': roc_auc_score(y_val, y_proba_val),
            'time': time.time() - start
        }

        print(f"  ✓ XGBoost: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        return model, metrics

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model."""
        print("\n[Training LightGBM]")
        start = time.time()

        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=-1, verbose=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20, verbose=False)

        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_val),
            'f1': f1_score(y_val, y_pred_val, zero_division=0),
            'auc': roc_auc_score(y_val, y_proba_val),
            'time': time.time() - start
        }

        print(f"  ✓ LightGBM: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        return model, metrics

    def train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model."""
        print("\n[Training CatBoost]")
        start = time.time()

        model = cb.CatBoostClassifier(
            iterations=200, depth=7, learning_rate=0.05,
            subsample=0.8, random_state=self.random_state,
            verbose=0, thread_count=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20, verbose=False)

        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_val),
            'f1': f1_score(y_val, y_pred_val, zero_division=0),
            'auc': roc_auc_score(y_val, y_proba_val),
            'time': time.time() - start
        }

        print(f"  ✓ CatBoost: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        return model, metrics

    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network model."""
        print("\n[Training Neural Network]")
        start = time.time()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['neural_network'] = scaler

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', learning_rate_init=0.001, batch_size=32,
            max_iter=500, early_stopping=True, validation_fraction=0.2,
            random_state=self.random_state
        )

        model.fit(X_train_scaled, y_train)

        y_pred_val = model.predict(X_val_scaled)
        y_proba_val = model.predict_proba(X_val_scaled)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_val),
            'f1': f1_score(y_val, y_pred_val, zero_division=0),
            'auc': roc_auc_score(y_val, y_proba_val),
            'time': time.time() - start
        }

        print(f"  ✓ Neural Network: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        return model, metrics

    def calculate_ensemble_weights(self, metrics_dict):
        """Calculate weights for ensemble based on validation AUC."""
        print("\n[Ensemble Weights Calculation]")

        auc_scores = {name: metrics['auc'] for name, metrics in metrics_dict.items()}
        total_auc = sum(auc_scores.values())
        weights = {name: auc / total_auc for name, auc in auc_scores.items()}

        print("  Model Weights (based on validation AUC):")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:20s}: {weight:.4f}")

        return weights

    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble on test set."""
        print("\n[Ensemble Test Evaluation]")

        predictions = {}
        predictions['xgboost'] = self.models['xgboost'].predict_proba(X_test)[:, 1]
        predictions['lightgbm'] = self.models['lightgbm'].predict_proba(X_test)[:, 1]
        predictions['catboost'] = self.models['catboost'].predict_proba(X_test)[:, 1]

        X_test_scaled = self.scalers['neural_network'].transform(X_test)
        predictions['neural_network'] = self.models['neural_network'].predict_proba(X_test_scaled)[:, 1]

        ensemble_proba = np.zeros_like(predictions['xgboost'])
        for model_name, weight in self.weights.items():
            ensemble_proba += predictions[model_name] * weight

        ensemble_pred = (ensemble_proba >= 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'f1': f1_score(y_test, ensemble_pred, zero_division=0),
            'auc': roc_auc_score(y_test, ensemble_proba),
            'ensemble_prob_mean': ensemble_proba.mean(),
            'ensemble_prob_std': ensemble_proba.std()
        }

        print(f"  Ensemble Test: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        return metrics

    def train_ensemble(self, df_features: pd.DataFrame):
        """Train complete ensemble."""
        print("=" * 80)
        print("ENSEMBLE MODEL TRAINING PIPELINE")
        print("=" * 80)

        start_total = time.time()

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df_features)

        val_metrics = {}

        self.models['xgboost'], val_metrics['xgboost'] = self.train_xgboost(
            X_train, y_train, X_val, y_val)

        self.models['lightgbm'], val_metrics['lightgbm'] = self.train_lightgbm(
            X_train, y_train, X_val, y_val)

        self.models['catboost'], val_metrics['catboost'] = self.train_catboost(
            X_train, y_train, X_val, y_val)

        self.models['neural_network'], val_metrics['neural_network'] = self.train_neural_network(
            X_train, y_train, X_val, y_val)

        self.weights = self.calculate_ensemble_weights(val_metrics)

        test_metrics = self.evaluate_ensemble(X_test, y_test)

        total_time = time.time() - start_total

        results = {
            'status': 'success',
            'total_training_time': total_time,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'ensemble_weights': self.weights,
            'summary': {
                'num_train_samples': len(X_train),
                'num_val_samples': len(X_val),
                'num_test_samples': len(X_test),
                'ensemble_auc': test_metrics['auc'],
                'ensemble_accuracy': test_metrics['accuracy'],
                'training_time_seconds': total_time
            }
        }

        print(f"\n{'=' * 80}")
        print(f"ENSEMBLE TRAINING COMPLETE - Total time: {total_time:.2f}s")
        print(f"Ensemble AUC (test): {test_metrics['auc']:.4f}")
        print(f"{'=' * 80}")

        return results

    def save_ensemble(self, output_dir: str = None):
        """Save ensemble models to disk."""
        if output_dir is None:
            output_dir = str(self.data_dir)

        output_path = Path(output_dir) / "ensemble_models"
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"\nSaving ensemble models to {output_path}...")

        with open(output_path / "xgboost_model.pkl", "wb") as f:
            pickle.dump(self.models['xgboost'], f)

        with open(output_path / "lightgbm_model.pkl", "wb") as f:
            pickle.dump(self.models['lightgbm'], f)

        with open(output_path / "catboost_model.pkl", "wb") as f:
            pickle.dump(self.models['catboost'], f)

        with open(output_path / "neural_network_model.pkl", "wb") as f:
            pickle.dump(self.models['neural_network'], f)

        with open(output_path / "nn_scaler.pkl", "wb") as f:
            pickle.dump(self.scalers['neural_network'], f)

        with open(output_path / "ensemble_weights.json", "w") as f:
            json.dump(self.weights, f, indent=2)

        print(f"✓ Ensemble models saved to {output_path}")
        return str(output_path)

    @staticmethod
    def load_ensemble(ensemble_path: str):
        """Load pre-trained ensemble from disk."""
        ensemble_path = Path(ensemble_path)

        trainer = EnsembleTrainer()

        with open(ensemble_path / "xgboost_model.pkl", "rb") as f:
            trainer.models['xgboost'] = pickle.load(f)

        with open(ensemble_path / "lightgbm_model.pkl", "rb") as f:
            trainer.models['lightgbm'] = pickle.load(f)

        with open(ensemble_path / "catboost_model.pkl", "rb") as f:
            trainer.models['catboost'] = pickle.load(f)

        with open(ensemble_path / "neural_network_model.pkl", "rb") as f:
            trainer.models['neural_network'] = pickle.load(f)

        with open(ensemble_path / "nn_scaler.pkl", "rb") as f:
            trainer.scalers['neural_network'] = pickle.load(f)

        with open(ensemble_path / "ensemble_weights.json", "r") as f:
            trainer.weights = json.load(f)

        print(f"✓ Ensemble loaded from {ensemble_path}")
        return trainer

    def predict_ensemble(self, X: np.ndarray):
        """Generate ensemble predictions on new data."""
        predictions = {}

        predictions['xgboost'] = self.models['xgboost'].predict_proba(X)[:, 1]
        predictions['lightgbm'] = self.models['lightgbm'].predict_proba(X)[:, 1]
        predictions['catboost'] = self.models['catboost'].predict_proba(X)[:, 1]

        X_scaled = self.scalers['neural_network'].transform(X)
        predictions['neural_network'] = self.models['neural_network'].predict_proba(X_scaled)[:, 1]

        ensemble_proba = np.zeros_like(predictions['xgboost'])
        for model_name, weight in self.weights.items():
            ensemble_proba += predictions[model_name] * weight

        return ensemble_proba
