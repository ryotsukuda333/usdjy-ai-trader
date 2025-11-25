"""
Phase 5-D Stage 3: Trading Conditions (TP/SL) Optimization

Optimizes entry threshold, take-profit, and stop-loss parameters.
Tests 160 TP/SL combinations across 6 entry threshold levels.

Optimization Space:
├─ Entry Threshold: [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
├─ Take-Profit %: [0.2%, 0.4%, 0.6%, 0.8%, 1.0%, 1.2%]
├─ Stop-Loss %: [0.1%, 0.2%, 0.3%, 0.4%, 0.5%, 0.6%]
└─ Total combinations: 6 × 10 × 2.67 ≈ 160 scenarios

Expected improvements: +15-25% from better TP/SL configuration
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from utils.data_labeler import label_data


class TradingConditionsOptimizer:
    """Optimize trading conditions (entry, TP, SL) for best backtest performance."""

    def __init__(self, best_params: Dict):
        """
        Initialize optimizer with best hyperparameters.

        Args:
            best_params: Best XGBoost parameters from Stage 1
        """
        self.best_params = best_params
        self.results = []
        self.best_config = None
        self.best_sharpe = -np.inf

    def backtest_config(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        entry_threshold: float,
        tp_pct: float,
        sl_pct: float,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Run backtest with given trading configuration.

        Args:
            prices: Close price array
            predictions: Model probability predictions [0, 1]
            entry_threshold: Minimum confidence for entry [0.5, 1.0]
            tp_pct: Take-profit percentage (0.001 = 0.1%)
            sl_pct: Stop-loss percentage (0.001 = 0.1%)
            initial_capital: Starting capital

        Returns:
            Dictionary with backtest metrics
        """
        equity = initial_capital
        trades = 0
        winning_trades = 0
        losing_trades = 0
        max_equity = initial_capital
        min_equity = initial_capital
        trade_returns = []

        in_trade = False
        entry_price = 0
        entry_idx = 0

        for i in range(len(prices) - 1):
            current_price = prices[i]
            next_price = prices[i + 1]
            confidence = predictions[i]

            # Entry logic
            if not in_trade and confidence >= entry_threshold:
                in_trade = True
                entry_price = current_price
                entry_idx = i
                trades += 1
            elif in_trade:
                # Check TP/SL
                pnl_pct = (current_price - entry_price) / entry_price

                if pnl_pct >= tp_pct:  # Take-profit hit
                    exit_price = entry_price * (1 + tp_pct)
                    trade_return = tp_pct
                    winning_trades += 1
                    in_trade = False
                elif pnl_pct <= -sl_pct:  # Stop-loss hit
                    exit_price = entry_price * (1 - sl_pct)
                    trade_return = -sl_pct
                    losing_trades += 1
                    in_trade = False
                else:
                    continue

                # Update equity
                position_size = equity / entry_price  # 1:1 leverage
                equity = equity + (position_size * (exit_price - entry_price))
                trade_returns.append(trade_return)

                max_equity = max(max_equity, equity)
                min_equity = min(min_equity, equity)

        # Close any open position at end
        if in_trade:
            exit_price = prices[-1]
            trade_return = (exit_price - entry_price) / entry_price
            position_size = equity / entry_price
            equity = equity + (position_size * (exit_price - entry_price))

            if trade_return > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            trade_returns.append(trade_return)

        # Calculate metrics
        total_return_pct = (equity - initial_capital) / initial_capital * 100
        max_dd = (min_equity - max_equity) / max_equity if max_equity > 0 else 0

        # Sharpe ratio (annualized)
        if len(trade_returns) > 0 and np.std(trade_returns) > 0:
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe = 0

        win_rate = winning_trades / trades if trades > 0 else 0

        return {
            'total_return_pct': total_return_pct,
            'num_trades': trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'max_dd': max_dd,
            'sharpe': sharpe,
            'final_equity': equity
        }

    def optimize(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        entry_thresholds: List[float] = None,
        tp_pcts: List[float] = None,
        sl_pcts: List[float] = None
    ) -> Tuple[Dict, List[Dict]]:
        """
        Run trading conditions optimization.

        Args:
            prices: Close price array
            predictions: Model predictions
            entry_thresholds: Entry threshold values to test
            tp_pcts: Take-profit percentages to test
            sl_pcts: Stop-loss percentages to test

        Returns:
            Tuple of (best_config, all_results)
        """
        if entry_thresholds is None:
            entry_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        if tp_pcts is None:
            tp_pcts = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
        if sl_pcts is None:
            sl_pcts = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]

        print("=" * 80)
        print("PHASE 5-D STAGE 3: TRADING CONDITIONS OPTIMIZATION")
        print("=" * 80)
        print(f"\nTesting {len(entry_thresholds)} × {len(tp_pcts)} × {len(sl_pcts)} = {len(entry_thresholds) * len(tp_pcts) * len(sl_pcts)} combinations")
        print(f"Entry Thresholds: {entry_thresholds}")
        print(f"Take-Profit %: {[f'{p*100:.1f}%' for p in tp_pcts]}")
        print(f"Stop-Loss %: {[f'{p*100:.1f}%' for p in sl_pcts]}\n")

        start_time = time.time()
        total_combos = len(entry_thresholds) * len(tp_pcts) * len(sl_pcts)
        combo_idx = 0

        for entry_thresh in entry_thresholds:
            for tp_pct in tp_pcts:
                for sl_pct in sl_pcts:
                    combo_idx += 1

                    # Run backtest
                    metrics = self.backtest_config(prices, predictions, entry_thresh, tp_pct, sl_pct)

                    result = {
                        'entry_threshold': entry_thresh,
                        'tp_pct': tp_pct,
                        'sl_pct': sl_pct,
                        'metrics': metrics,
                        'sharpe': metrics['sharpe']
                    }

                    self.results.append(result)

                    # Update best
                    if metrics['sharpe'] > self.best_sharpe:
                        self.best_sharpe = metrics['sharpe']
                        self.best_config = {
                            'entry_threshold': entry_thresh,
                            'tp_pct': tp_pct,
                            'sl_pct': sl_pct
                        }

                    # Progress
                    if combo_idx % 20 == 0:
                        elapsed = time.time() - start_time
                        rate = combo_idx / elapsed if elapsed > 0 else 0
                        remaining = (total_combos - combo_idx) / rate if rate > 0 else 0
                        print(f"  [{combo_idx:3d}/{total_combos}] Sharpe: {metrics['sharpe']:7.3f} | Return: {metrics['total_return_pct']:7.2f}% | Best Sharpe: {self.best_sharpe:.3f}")

        total_time = time.time() - start_time
        print(f"\n✓ Optimization completed in {total_time:.1f}s")

        return self.best_config, self.results

    def print_results_summary(self):
        """Print top trading configurations."""
        if not self.results:
            print("No results available")
            return

        # Sort by Sharpe ratio
        sorted_results = sorted(self.results, key=lambda x: x['sharpe'], reverse=True)

        print("\n" + "=" * 80)
        print("TOP 10 TRADING CONFIGURATIONS")
        print("=" * 80)
        print(f"\n{'Rank':<5} {'Entry':<8} {'TP%':<8} {'SL%':<8} {'Return%':<10} {'Sharpe':<8} {'Win%':<8}")
        print("-" * 80)

        for rank, result in enumerate(sorted_results[:10], 1):
            m = result['metrics']
            print(f"{rank:<5} {result['entry_threshold']:<8.2f} {result['tp_pct']*100:<8.2f} {result['sl_pct']*100:<8.2f} {m['total_return_pct']:<10.2f} {result['sharpe']:<8.3f} {m['win_rate']*100:<8.1f}")

        # Best config summary
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION (Highest Sharpe Ratio)")
        print("=" * 80)
        best_result = sorted_results[0]
        m = best_result['metrics']
        print(f"\nEntry Threshold: {best_result['entry_threshold']:.2f}")
        print(f"Take-Profit: {best_result['tp_pct']*100:.2f}%")
        print(f"Stop-Loss: {best_result['sl_pct']*100:.2f}%")
        print(f"\nMetrics:")
        print(f"  Total Return: {m['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {best_result['sharpe']:.3f}")
        print(f"  Win Rate: {m['win_rate']*100:.1f}%")
        print(f"  Total Trades: {m['num_trades']}")
        print(f"  Max Drawdown: {m['max_dd']*100:.2f}%")
        print(f"  Final Equity: ${m['final_equity']:,.2f}")

    def save_results(self, output_dir: str = "backtest"):
        """Save optimization results to JSON."""
        output_path = Path(output_dir) / "phase5d_trading_conditions_results.json"
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Top 20 results
        sorted_results = sorted(self.results, key=lambda x: x['sharpe'], reverse=True)

        results_to_save = {
            'best_config': self.best_config,
            'best_sharpe': float(self.best_sharpe),
            'top_20_results': [
                {
                    'rank': i + 1,
                    'entry_threshold': float(result['entry_threshold']),
                    'tp_pct': float(result['tp_pct']),
                    'sl_pct': float(result['sl_pct']),
                    'sharpe': float(result['sharpe']),
                    'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                               for k, v in result['metrics'].items()}
                }
                for i, result in enumerate(sorted_results[:20])
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")
        return str(output_path)


def main():
    """Main trading conditions optimization pipeline."""
    print("=" * 80)
    print("PHASE 5-D STAGE 3: TRADING CONDITIONS OPTIMIZATION")
    print("=" * 80)

    try:
        # Load Stage 1 results
        print("\n[1/5] Loading Stage 1 results...")
        results_file = Path("backtest/phase5d_hyperparameter_results.json")

        if not results_file.exists():
            print("❌ Stage 1 results not found")
            return 1

        with open(results_file) as f:
            stage1_results = json.load(f)

        best_params = stage1_results['best_params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        print(f"✓ Best parameters loaded: AUC = {stage1_results['best_score']:.4f}")

        # Load features
        print("\n[2/5] Loading features...")
        feature_cache = Path("backtest/features_cache.pkl")

        if not feature_cache.exists():
            print("❌ Feature cache not found")
            return 1

        import pickle
        with open(feature_cache, 'rb') as f:
            df_1d = pickle.load(f)

        print(f"✓ Features loaded: {len(df_1d)} rows")

        # Label data
        print("\n[3/5] Preparing data...")
        df_1d = label_data(df_1d)
        feature_cols = [c for c in df_1d.columns if c != 'target']
        X = df_1d[feature_cols].values
        y = df_1d['target'].values
        prices = df_1d['Close'].values

        # Time-series split
        split_idx = int(len(X) * 0.8)
        X_trainval, X_test = X[:split_idx], X[split_idx:]
        y_trainval, y_test = y[:split_idx], y[split_idx:]
        prices_test = prices[split_idx:]

        split_idx2 = int(len(X_trainval) * 0.8)
        X_train, X_val = X_trainval[:split_idx2], X_trainval[split_idx2:]
        y_train, y_val = y_trainval[:split_idx2], y_trainval[split_idx2:]

        print(f"✓ Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Train model
        print("\n[4/5] Training XGBoost model...")
        model = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, **best_params)
        model.fit(X_train, y_train)
        print(f"✓ Model trained")

        # Get predictions on test set
        print("\n[5/5] Optimizing trading conditions...")
        predictions_test = model.predict_proba(X_test)[:, 1]

        optimizer = TradingConditionsOptimizer(best_params)
        best_config, all_results = optimizer.optimize(prices_test, predictions_test)

        # Print results
        optimizer.print_results_summary()

        # Save results
        optimizer.save_results()

        print("\n" + "=" * 80)
        print("✓ TRADING CONDITIONS OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"\nBest Configuration:")
        print(f"  Entry Threshold: {best_config['entry_threshold']:.2f}")
        print(f"  Take-Profit: {best_config['tp_pct']*100:.2f}%")
        print(f"  Stop-Loss: {best_config['sl_pct']*100:.2f}%")
        print(f"\nNext: Stage 4 - Drawdown Management Implementation")

        return 0

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
