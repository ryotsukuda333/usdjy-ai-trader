"""Ensemble Model Backtest - Phase 5-C

Runs backtest using ensemble model predictions integrated with multi-timeframe signals.
Compares ensemble performance against Phase 5-A baseline (single XGBoost).
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from features.multi_timeframe_fetcher import MultiTimeframeFetcher
from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from model.ensemble_signal_generator import EnsembleSignalGenerator
from trader.dynamic_risk_manager import DynamicRiskManager


class EnsembleBacktester:
    """Backtest ensemble model with multi-timeframe strategy."""

    def __init__(self, initial_equity: float = 100000, ensemble_threshold: float = 0.55):
        """Initialize backtest engine."""
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.ensemble_threshold = ensemble_threshold

        # Trading state
        self.position = 0  # 0: no position, > 0: long, < 0: short
        self.entry_price = 0
        self.entry_date = None

        # Results tracking
        self.trades = []
        self.equity_curve = [initial_equity]
        self.dates = []

        # Risk management
        self.risk_manager = DynamicRiskManager()

    def process_bar(
        self,
        datetime_val: pd.Timestamp,
        price: float,
        signal: float,
        confidence: float,
        stop_loss_pct: float = 0.30,
        take_profit_pct: float = 0.60
    ) -> bool:
        """Process single bar and generate trade signals.

        Args:
            datetime_val: Bar datetime
            price: Current price
            signal: Trading signal (-1, 0, 1)
            confidence: Signal confidence
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage

        Returns:
            True if trade executed, False otherwise
        """
        trade_executed = False

        # Exit logic: close position if opposite signal
        if self.position != 0 and signal != 0 and signal != np.sign(self.position):
            pnl = (price - self.entry_price) * self.position
            pnl_pct = (pnl / (self.entry_price * abs(self.position))) * 100

            trade = {
                'entry_date': self.entry_date,
                'entry_price': self.entry_price,
                'exit_date': datetime_val,
                'exit_price': price,
                'position': self.position,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'confidence': confidence
            }
            self.trades.append(trade)
            self.current_equity += pnl

            self.position = 0
            self.entry_price = 0
            self.entry_date = None
            trade_executed = True

        # Entry logic: open new position if strong signal
        if self.position == 0 and signal != 0 and confidence >= self.ensemble_threshold:
            self.position = np.sign(signal) * 1
            self.entry_price = price
            self.entry_date = datetime_val
            trade_executed = True

        return trade_executed

    def run_backtest(
        self,
        df_1d: pd.DataFrame,
        df_5m: pd.DataFrame,
        signal_generator: EnsembleSignalGenerator
    ) -> Dict:
        """Run complete backtest.

        Args:
            df_1d: 1D OHLCV data
            df_5m: 5m OHLCV data
            signal_generator: Ensemble signal generator

        Returns:
            Dict of backtest results
        """
        print("=" * 80)
        print("ENSEMBLE BACKTEST - PHASE 5-C")
        print("=" * 80)

        start_time = time.time()

        # Generate signals
        print("\n[1/3] Generating ensemble signals...")
        signals, confidences, metadata = signal_generator.generate_signals(df_1d, df_5m)
        print(f"✓ Signals generated: {len(signals)} bars")
        print(f"  Buy signals: {(signals > 0).sum()}")
        print(f"  Sell signals: {(signals < 0).sum()}")

        # Run backtest
        print(f"\n[2/3] Running backtest ({len(df_5m)} bars)...")
        for i in range(len(df_5m)):
            datetime_val = df_5m.index[i]
            price = df_5m.iloc[i]['Close']
            signal = signals[i]
            confidence = confidences[i]

            self.process_bar(datetime_val, price, signal, confidence)
            self.equity_curve.append(self.current_equity)
            self.dates.append(datetime_val)

            # Progress indicator
            if (i + 1) % 50000 == 0:
                print(f"  {i+1:,}/{len(df_5m):,} bars processed...")

        # Close final position
        if self.position != 0:
            final_price = df_5m.iloc[-1]['Close']
            pnl = (final_price - self.entry_price) * self.position
            self.trades.append({
                'entry_date': self.entry_date,
                'entry_price': self.entry_price,
                'exit_date': df_5m.index[-1],
                'exit_price': final_price,
                'position': self.position,
                'pnl': pnl,
                'pnl_pct': (pnl / (self.entry_price * abs(self.position))) * 100
            })
            self.current_equity += pnl

        print(f"✓ Backtest complete in {time.time() - start_time:.2f}s")

        # Calculate metrics
        print(f"\n[3/3] Calculating metrics...")
        results = self._calculate_metrics()
        print(f"✓ Metrics calculated")

        return results

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest metrics."""
        total_return = ((self.current_equity - self.initial_equity) / self.initial_equity) * 100

        if not self.trades:
            return {
                'total_return': 0,
                'num_trades': 0,
                'final_equity': self.current_equity
            }

        df_trades = pd.DataFrame(self.trades)

        winning_trades = (df_trades['pnl'] > 0).sum()
        losing_trades = (df_trades['pnl'] <= 0).sum()
        win_rate = (winning_trades / len(df_trades) * 100) if len(df_trades) > 0 else 0

        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0

        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else 0

        metrics = {
            'total_return_pct': total_return,
            'num_trades': len(df_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_equity': self.current_equity,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }

        return metrics


def main():
    """Main backtest pipeline."""
    print("=" * 80)
    print("ENSEMBLE BACKTEST PIPELINE - PHASE 5-C")
    print("=" * 80)

    try:
        # Fetch data
        print("\n[1/5] Fetching multi-timeframe data...")
        fetcher = MultiTimeframeFetcher()
        data_dict = fetcher.fetch_and_resample(years=2)
        print(f"✓ Data fetched: 1D={len(data_dict['1d'])}, 5m={len(data_dict['5m'])} bars")

        # Engineer features
        print("\n[2/5] Engineering features...")
        engineer = MultiTimeframeFeatureEngineer()
        features_dict = engineer.engineer_all_timeframes(data_dict)
        df_1d = features_dict['1d']
        df_5m = features_dict['5m']
        print(f"✓ Features engineered: 1D={len(df_1d)}, 5m={len(df_5m)} bars")

        # Load ensemble
        print("\n[3/5] Loading ensemble models...")
        try:
            signal_generator = EnsembleSignalGenerator(
                ensemble_path="model/ensemble_models",
                confidence_threshold=0.55
            )
            print("✓ Ensemble models loaded")
        except FileNotFoundError:
            print("✗ Ensemble models not found. Please train first with: train_ensemble_models.py")
            return None

        # Run backtest
        print("\n[4/5] Running backtest...")
        backtest = EnsembleBacktester(initial_equity=100000)
        metrics = backtest.run_backtest(df_1d, df_5m, signal_generator)

        # Save results
        print("\n[5/5] Saving results...")
        results_path = Path("backtest") / "ENSEMBLE_BACKTEST_RESULTS.json"
        with open(results_path, "w") as f:
            json.dump({
                k: v for k, v in metrics.items() if k not in ['equity_curve', 'trades']
            }, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Return:         {metrics['total_return_pct']:+.2f}%")
        print(f"Trades:         {metrics['num_trades']}")
        print(f"Win Rate:       {metrics['win_rate_pct']:.1f}%")
        print(f"Final Equity:   ${metrics['final_equity']:,.2f}")
        print(f"Profit Factor:  {metrics['profit_factor']:.2f}")

        print(f"\n✓ Results saved to {results_path}")

        return metrics

    except Exception as e:
        print(f"\n✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results else 1)
