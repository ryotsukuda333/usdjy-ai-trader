"""
Phase 5-D Stage 5: Final Integration & Report Generation

Combines all optimized configurations from Stages 1-4 and runs comprehensive backtest.
Generates detailed performance report comparing against Phase 5-B baseline.

Configuration Integration:
├─ Stage 1: XGBoost parameters (AUC: 0.5770)
├─ Stage 2: Feature importance analysis
├─ Stage 3: Trading conditions (Entry: 0.70, TP: 1.20%, SL: 0.30%)
└─ Stage 4: Drawdown management (Trailing: 2.0%, Daily Limit: 1.0%)

Expected: Improve upon Phase 5-B (+293.83% baseline)
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from utils.data_labeler import label_data


class FinalIntegrationBacktester:
    """Comprehensive backtest with all optimizations integrated."""

    def __init__(self):
        self.results = {}
        self.all_configs = []

    def load_all_optimized_configs(self) -> Dict:
        """Load optimal configurations from all stages."""
        configs = {}

        # Stage 1: Hyperparameters
        with open("backtest/phase5d_hyperparameter_results.json") as f:
            stage1 = json.load(f)
        configs['hyperparams'] = stage1['best_params']
        configs['stage1_auc'] = stage1['best_score']

        # Stage 3: Trading conditions
        with open("backtest/phase5d_trading_conditions_results.json") as f:
            stage3 = json.load(f)
        configs['trading'] = stage3['best_config']

        # Stage 4: Drawdown management
        with open("backtest/phase5d_drawdown_management_results.json") as f:
            stage4 = json.load(f)
        configs['drawdown'] = stage4['best_config']

        return configs

    def run_final_backtest(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        trading_config: Dict,
        drawdown_config: Dict,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Run final comprehensive backtest with all optimizations.

        Args:
            prices: Close price array
            predictions: Model predictions
            trading_config: Trading conditions (entry, TP, SL)
            drawdown_config: Drawdown management (trailing stop, daily limit)
            initial_capital: Starting capital

        Returns:
            Dictionary with comprehensive backtest metrics
        """
        equity = initial_capital
        max_equity = initial_capital
        min_equity = initial_capital
        trades = 0
        winning_trades = 0
        losing_trades = 0
        trade_returns = []
        trade_details = []

        # Drawdown management tracking
        daily_loss = 0

        # Calculate volatility for drawdown management
        returns = np.diff(prices) / prices[:-1]
        rolling_std = pd.Series(returns).rolling(20).std().values
        vol = np.zeros(len(prices))
        vol[1:1+len(rolling_std)] = rolling_std
        vol[0] = rolling_std[0] if len(rolling_std) > 0 else 0
        vol_median = np.median(vol[vol > 0])

        in_trade = False
        entry_price = 0
        entry_idx = 0
        highest_price = 0
        entry_threshold = trading_config['entry_threshold']
        tp_pct = trading_config['tp_pct']
        sl_pct = trading_config['sl_pct']
        trailing_stop_pct = drawdown_config['trailing_stop_pct']
        daily_loss_limit = drawdown_config['daily_loss_limit_pct']

        for i in range(len(prices) - 1):
            current_price = prices[i]
            confidence = predictions[i]

            # Daily loss limit check
            if daily_loss < -daily_loss_limit:
                continue

            # Entry logic
            if not in_trade and confidence >= entry_threshold:
                # Volatility-based position sizing
                vol_factor = vol_median / (vol[i] + 1e-6)
                vol_factor = np.clip(vol_factor, 0.5, 2.0)

                in_trade = True
                entry_price = current_price
                highest_price = current_price
                entry_idx = i
                trades += 1

            elif in_trade:
                highest_price = max(highest_price, current_price)
                pnl_pct = (current_price - entry_price) / entry_price
                trailing_stop = highest_price * (1 - trailing_stop_pct)

                # Check exit conditions
                exit_price = None
                trade_return = None
                exit_reason = None

                if current_price >= entry_price * (1 + tp_pct):
                    exit_price = entry_price * (1 + tp_pct)
                    trade_return = tp_pct
                    exit_reason = "TP"
                    winning_trades += 1
                elif current_price <= entry_price * (1 - sl_pct):
                    exit_price = entry_price * (1 - sl_pct)
                    trade_return = -sl_pct
                    exit_reason = "SL"
                    losing_trades += 1
                elif current_price <= trailing_stop:
                    exit_price = trailing_stop
                    trade_return = (trailing_stop - entry_price) / entry_price
                    exit_reason = "TSL"
                    losing_trades += 1

                if exit_price is not None:
                    position_size = equity / entry_price
                    equity = equity + (position_size * (exit_price - entry_price))
                    trade_returns.append(trade_return)
                    daily_loss += trade_return * 100
                    max_equity = max(max_equity, equity)
                    min_equity = min(min_equity, equity)

                    trade_details.append({
                        'trade_num': trades,
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': float(entry_price),
                        'exit_price': float(exit_price),
                        'return_pct': float(trade_return * 100),
                        'exit_reason': exit_reason,
                        'equity': float(equity)
                    })

                    in_trade = False

        # Close any open position
        if in_trade:
            exit_price = prices[-1]
            trade_return = (exit_price - entry_price) / entry_price
            position_size = equity / entry_price
            equity = equity + (position_size * (exit_price - entry_price))
            trade_returns.append(trade_return)

            if trade_return > 0:
                winning_trades += 1
            else:
                losing_trades += 1

            trade_details.append({
                'trade_num': trades,
                'entry_idx': entry_idx,
                'exit_idx': len(prices) - 1,
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'return_pct': float(trade_return * 100),
                'exit_reason': 'CLOSE',
                'equity': float(equity)
            })

        # Calculate metrics
        total_return_pct = (equity - initial_capital) / initial_capital * 100
        max_dd = (min_equity - max_equity) / max_equity if max_equity > 0 else 0
        max_dd_pct = max_dd * 100

        win_rate = winning_trades / trades if trades > 0 else 0

        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe = 0

        avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
        avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
        profit_factor = (winning_trades * avg_win) / (losing_trades * abs(avg_loss)) if losing_trades > 0 and avg_loss != 0 else 0

        return {
            'total_return_pct': total_return_pct,
            'num_trades': trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'max_dd_pct': max_dd_pct,
            'sharpe': sharpe,
            'final_equity': equity,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_factor': profit_factor,
            'trade_details': trade_details
        }

    def generate_report(self, configs: Dict, metrics: Dict) -> str:
        """Generate comprehensive final report."""
        report = []
        report.append("=" * 80)
        report.append("PHASE 5-D FINAL INTEGRATION & COMPREHENSIVE BACKTEST REPORT")
        report.append("=" * 80)

        # Configuration summary
        report.append("\n[STAGE 1] HYPERPARAMETER OPTIMIZATION")
        report.append("-" * 80)
        report.append(f"Best AUC Score: {configs.get('stage1_auc', 'N/A'):.4f}")
        report.append("\nBest Parameters:")
        for k, v in configs['hyperparams'].items():
            report.append(f"  {k:25s}: {v}")

        # Trading conditions
        report.append("\n[STAGE 3] TRADING CONDITIONS")
        report.append("-" * 80)
        report.append(f"Entry Threshold: {configs['trading']['entry_threshold']:.2f}")
        report.append(f"Take-Profit: {configs['trading']['tp_pct']*100:.2f}%")
        report.append(f"Stop-Loss: {configs['trading']['sl_pct']*100:.2f}%")

        # Drawdown management
        report.append("\n[STAGE 4] DRAWDOWN MANAGEMENT")
        report.append("-" * 80)
        report.append(f"Trailing Stop: {configs['drawdown']['trailing_stop_pct']*100:.2f}%")
        report.append(f"Daily Loss Limit: {configs['drawdown']['daily_loss_limit_pct']:.1f}%")

        # Final performance
        report.append("\n[FINAL PERFORMANCE]")
        report.append("=" * 80)
        report.append(f"\nTotal Return: {metrics['total_return_pct']:.2f}%")
        report.append(f"Final Equity: ${metrics['final_equity']:,.2f}")
        report.append(f"Maximum Drawdown: {metrics['max_dd_pct']:.2f}%")
        report.append(f"\nTrade Statistics:")
        report.append(f"  Total Trades: {metrics['num_trades']}")
        report.append(f"  Winning Trades: {metrics['winning_trades']}")
        report.append(f"  Losing Trades: {metrics['losing_trades']}")
        report.append(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        report.append(f"\nQuality Metrics:")
        report.append(f"  Average Win: {metrics['avg_win_pct']:.2f}%")
        report.append(f"  Average Loss: {metrics['avg_loss_pct']:.2f}%")
        report.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")

        # Trade details
        if metrics['trade_details']:
            report.append("\n[TRADE DETAILS]")
            report.append("=" * 80)
            report.append(f"{'Num':<4} {'Bars':<6} {'Entry':<10} {'Exit':<10} {'Return%':<10} {'Reason':<6} {'Equity':<12}")
            report.append("-" * 80)
            for t in metrics['trade_details']:
                bars = t['exit_idx'] - t['entry_idx']
                report.append(
                    f"{t['trade_num']:<4} {bars:<6} "
                    f"{t['entry_price']:<10.4f} {t['exit_price']:<10.4f} "
                    f"{t['return_pct']:<10.2f} {t['exit_reason']:<6} "
                    f"${t['equity']:<11,.2f}"
                )

        # Comparison vs Phase 5-B
        report.append("\n[COMPARISON vs PHASE 5-B BASELINE]")
        report.append("=" * 80)
        phase5b_return = 293.83
        improvement = metrics['total_return_pct'] - phase5b_return
        improvement_pct = (improvement / phase5b_return * 100) if phase5b_return != 0 else 0
        report.append(f"Phase 5-B Return: {phase5b_return:.2f}%")
        report.append(f"Phase 5-D Return: {metrics['total_return_pct']:.2f}%")
        report.append(f"Difference: {improvement:+.2f}% ({improvement_pct:+.1f}%)")

        report.append("\n" + "=" * 80)
        report.append("✓ PHASE 5-D OPTIMIZATION COMPLETE")
        report.append("=" * 80)

        return "\n".join(report)

    def save_final_report(self, report_text: str, metrics: Dict, configs: Dict, output_dir: str = "backtest"):
        """Save comprehensive final report."""
        # Markdown report
        report_path = Path(output_dir) / "PHASE5D_FINAL_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report_text)

        # JSON results
        results_path = Path(output_dir) / "phase5d_final_results.json"
        results_to_save = {
            'final_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                            for k, v in metrics.items() if k != 'trade_details'},
            'num_trades': int(metrics['num_trades']),
            'winning_trades': int(metrics['winning_trades']),
            'losing_trades': int(metrics['losing_trades']),
            'configurations': {
                'hyperparameters': {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v)
                                   for k, v in configs['hyperparams'].items()},
                'trading_conditions': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                      for k, v in configs['trading'].items()},
                'drawdown_management': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                       for k, v in configs['drawdown'].items()}
            }
        }

        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"✓ Report saved to {report_path}")
        print(f"✓ Results saved to {results_path}")


def main():
    """Main final integration pipeline."""
    print("=" * 80)
    print("PHASE 5-D STAGE 5: FINAL INTEGRATION & REPORT GENERATION")
    print("=" * 80)

    try:
        # Load all configurations
        print("\n[1/4] Loading optimized configurations from all stages...")
        backtester = FinalIntegrationBacktester()
        configs = backtester.load_all_optimized_configs()
        print(f"✓ All configurations loaded")

        # Load data and prepare
        print("\n[2/4] Loading data and preparing for final backtest...")
        feature_cache = Path("backtest/features_cache.pkl")
        import pickle
        with open(feature_cache, 'rb') as f:
            df_1d = pickle.load(f)

        df_1d = label_data(df_1d)
        feature_cols = [c for c in df_1d.columns if c != 'target']
        X = df_1d[feature_cols].values
        y = df_1d['target'].values
        prices = df_1d['Close'].values

        # Split data (same as stages)
        split_idx = int(len(X) * 0.8)
        X_trainval, X_test = X[:split_idx], X[split_idx:]
        y_trainval, y_test = y[:split_idx], y[split_idx:]
        prices_test = prices[split_idx:]

        split_idx2 = int(len(X_trainval) * 0.8)
        X_train, X_val = X_trainval[:split_idx2], X_trainval[split_idx2:]
        y_train, y_val = y_trainval[:split_idx2], y_trainval[split_idx2:]

        print(f"✓ Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Train final model
        print("\n[3/4] Training final XGBoost model with best parameters...")
        best_params = configs['hyperparams'].copy()
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])

        model = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, **best_params)
        model.fit(X_train, y_train)

        # Get predictions on test set
        predictions_test = model.predict_proba(X_test)[:, 1]
        print(f"✓ Model trained and predictions generated")

        # Run final comprehensive backtest
        print("\n[4/4] Running final comprehensive backtest...")
        metrics = backtester.run_final_backtest(
            prices_test, predictions_test,
            configs['trading'], configs['drawdown']
        )

        # Generate report
        report = backtester.generate_report(configs, metrics)
        print(report)

        # Save results
        backtester.save_final_report(report, metrics, configs)

        print("\n" + "=" * 80)
        print("✓ PHASE 5-D COMPLETE")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ Final integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
