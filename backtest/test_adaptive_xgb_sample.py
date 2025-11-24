"""Test Adaptive XGBoost with Sampled Data

Tests with every 10th bar to verify logic without full computation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.multi_timeframe_fetcher import MultiTimeframeFetcher
from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from model.adaptive_xgb_signal_generator import AdaptiveXGBSignalGenerator

import warnings
warnings.filterwarnings('ignore')


class SampledAdaptiveXGBBacktester:
    """Backtest with sampled data (every 10th bar)."""

    def __init__(self, initial_capital: float = 100000.0,
                 risk_per_trade: float = 0.01,
                 take_profit_pct: float = 0.01,
                 stop_loss_pct: float = 0.005,
                 sample_rate: int = 10):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.sample_rate = sample_rate

        self.signal_generator = AdaptiveXGBSignalGenerator()
        self.equity = initial_capital
        self.trades = []

    def backtest(self, data_dict: Dict, features_dict: Dict) -> Dict:
        """Run backtest on sampled data."""

        df_5m = data_dict['5m'].copy()
        df_5m_features = features_dict.get('5m', pd.DataFrame()).copy()
        df_1d_features = features_dict.get('1d', pd.DataFrame()).copy()

        if len(df_5m) == 0:
            return {}

        print(f"\n{'='*80}")
        print(f"SAMPLED ADAPTIVE XGBOOST BACKTEST (Every {self.sample_rate}th bar)")
        print(f"{'='*80}")
        print(f"Total bars: {len(df_5m):,}")
        print(f"Sampled bars: {len(df_5m) // self.sample_rate:,}")
        print(f"Processing...")

        position_size = 0
        entry_price = 0.0
        entry_bar = 0
        processed = 0

        # Sample every N bars
        for idx in range(0, len(df_5m), self.sample_rate):
            processed += 1
            if processed % 500 == 0:
                print(f"  Processed {processed} samples...")

            current_price = df_5m.iloc[idx]['Close']

            # Get features
            df_5m_features_slice = df_5m_features.iloc[:idx+1]
            if len(df_5m_features_slice) == 0:
                continue

            try:
                signal_result = self.signal_generator.generate_adaptive_signal(
                    df_1d_features,
                    df_5m_features_slice,
                    bars_in_trade=max(0, idx - entry_bar) if position_size > 0 else 0
                )
            except Exception as e:
                print(f"⚠️ Signal error at bar {idx}: {e}")
                continue

            # Exit logic
            if position_size > 0:
                should_exit = False
                exit_reason = ""

                # Stop loss
                if signal_result.signal == 1 and current_price <= entry_price * (1 - self.stop_loss_pct):
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                elif signal_result.signal == -1 and current_price >= entry_price * (1 + self.stop_loss_pct):
                    should_exit = True
                    exit_reason = "STOP_LOSS"

                # Take profit
                if signal_result.signal == 1 and current_price >= entry_price * (1 + self.take_profit_pct):
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                elif signal_result.signal == -1 and current_price <= entry_price * (1 - self.take_profit_pct):
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"

                if should_exit:
                    profit = position_size * (current_price - entry_price)
                    self.equity += profit
                    self.trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': idx,
                        'bars_held': idx - entry_bar,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_size': position_size,
                        'profit': profit,
                        'profit_pct': (profit / entry_price) * 100 if entry_price > 0 else 0,
                        'exit_reason': exit_reason
                    })
                    position_size = 0

            # Entry logic
            if position_size == 0 and signal_result.should_execute and signal_result.signal != 0:
                entry_price = current_price
                position_size = max(100, int(self.equity * self.risk_per_trade / (current_price * self.stop_loss_pct)))
                entry_bar = idx

        print(f"✓ Backtest complete! ({processed} samples processed)")

        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict:
        """Calculate metrics."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'final_equity': self.initial_capital
            }

        trades_df = pd.DataFrame(self.trades)
        num_trades = len(trades_df)
        winning = len(trades_df[trades_df['profit'] > 0])
        losing = len(trades_df[trades_df['profit'] < 0])

        return {
            'total_return': ((self.equity - self.initial_capital) / self.initial_capital) * 100,
            'num_trades': num_trades,
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': winning / num_trades * 100 if num_trades > 0 else 0,
            'profit_factor': (trades_df[trades_df['profit'] > 0]['profit'].sum() /
                            abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
                            if len(trades_df[trades_df['profit'] < 0]) > 0 else 0),
            'final_equity': self.equity
        }


def run_sampled():
    """Main execution."""
    print("\n" + "="*80)
    print("SAMPLED ADAPTIVE XGBOOST BACKTEST")
    print("="*80)

    # Fetch data
    print("\n[1/3] Fetching data...")
    start = datetime.now()
    fetcher = MultiTimeframeFetcher()
    data_dict = fetcher.fetch_and_resample(years=2)
    fetch_time = (datetime.now() - start).total_seconds()
    print(f"✓ Data fetched in {fetch_time:.1f}s")
    print(f"  5m bars: {len(data_dict['5m']):,}")

    # Engineer features
    print("\n[2/3] Engineering features...")
    start = datetime.now()
    engineer = MultiTimeframeFeatureEngineer()
    features_dict = engineer.engineer_all_timeframes(data_dict)
    eng_time = (datetime.now() - start).total_seconds()
    print(f"✓ Features engineered in {eng_time:.1f}s")

    # Backtest (every 10th bar)
    print("\n[3/3] Running backtest...")
    start = datetime.now()
    backtester = SampledAdaptiveXGBBacktester(sample_rate=10)
    metrics = backtester.backtest(data_dict, features_dict)
    backtest_time = (datetime.now() - start).total_seconds()
    print(f"✓ Backtest complete in {backtest_time:.1f}s")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS (SAMPLED - Every 10th bar)")
    print("="*80)
    print(f"Total Return:     {metrics['total_return']:+.2f}%")
    print(f"Num Trades:       {metrics['num_trades']}")
    print(f"Win Rate:         {metrics['win_rate']:.1f}%")
    print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"Final Equity:     ${metrics['final_equity']:,.0f}")
    print("="*80)

    # Comparison
    print("\nCOMPARISON WITH PHASE 5-A")
    print("-"*80)
    print(f"{'Metric':<25} {'Phase 5-A':<20} {'Adaptive XGB (Sampled)':<25}")
    print("-"*80)
    print(f"{'Total Return':<25} {2.00:>8.2f}% {metrics['total_return']:>18.2f}%")
    print(f"{'Num Trades':<25} {7:>8} {metrics['num_trades']:>18}")
    print(f"{'Win Rate':<25} {57.1:>8.1f}% {metrics['win_rate']:>18.1f}%")
    print("="*80)

    # Save results
    output_dir = Path(__file__).parent
    with open(output_dir / 'ADAPTIVE_XGB_SAMPLED_RESULTS.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'note': 'Sampled every 10th bar for performance testing'
        }, f, indent=2)
    print(f"\n✓ Results saved to ADAPTIVE_XGB_SAMPLED_RESULTS.json")

    return metrics


if __name__ == "__main__":
    try:
        metrics = run_sampled()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
