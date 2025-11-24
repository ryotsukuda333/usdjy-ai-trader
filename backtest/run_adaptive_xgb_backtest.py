"""Adaptive XGBoost Backtest for Multi-Timeframe Trading

Implements backtesting with adaptive XGBoost signal generation combining:
- 1D XGBoost probability (daily bias from trained model)
- 5m technical indicators (MA/RSI/MACD for entry confirmation)
- Multi-timeframe bar mapping (5m -> 1D parent lookup)
- Position sizing with dynamic risk management

Expected Performance:
- Trades: 20-40 (vs 7 in Phase 5-A, 177 in simplified)
- Return: +2-3% (matching Phase 5-A baseline)
- Win rate: 55-60%
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

# Import multi-timeframe utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.multi_timeframe_fetcher import MultiTimeframeFetcher
from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from model.adaptive_xgb_signal_generator import AdaptiveXGBSignalGenerator, AdaptiveSignalResult

import warnings
warnings.filterwarnings('ignore')


class AdaptiveXGBBacktester:
    """Backtester for adaptive XGBoost signal generation."""

    def __init__(self, initial_capital: float = 100000.0,
                 risk_per_trade: float = 0.01,
                 take_profit_pct: float = 0.01,
                 stop_loss_pct: float = 0.005):
        """Initialize backtester.

        Args:
            initial_capital: Starting capital ($)
            risk_per_trade: Risk per trade (% of capital)
            take_profit_pct: Take profit level (% move)
            stop_loss_pct: Stop loss level (% move)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

        # Signal generator
        self.signal_generator = AdaptiveXGBSignalGenerator()

        # Trade state
        self.equity = initial_capital
        self.position_size = 0  # Units held
        self.entry_price = 0.0
        self.entry_bar = 0
        self.signal_at_entry = 0

        self.trades = []
        self.equity_curve = []

    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> int:
        """Calculate position size based on risk.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price

        Returns:
            int: Number of units to trade
        """
        risk_amount = self.equity * self.risk_per_trade
        price_diff = abs(entry_price - stop_loss_price)

        if price_diff <= 0:
            return int(risk_amount / entry_price)

        units = int(risk_amount / price_diff)
        return max(100, units)  # Minimum 100 units

    def check_exit_conditions(self, current_price: float, current_bar: int,
                             current_signal: int) -> Optional[Tuple[float, str]]:
        """Check if position should be exited.

        Args:
            current_price: Current bar price
            current_bar: Current bar index
            current_signal: Current signal (1/0/-1)

        Returns:
            Optional[Tuple]: (exit_price, reason) if should exit, else None
        """
        if self.position_size == 0:
            return None

        # Take profit
        if self.signal_at_entry == 1:  # Long position
            tp_price = self.entry_price * (1 + self.take_profit_pct)
            if current_price >= tp_price:
                return (tp_price, "TAKE_PROFIT")

            # Stop loss
            sl_price = self.entry_price * (1 - self.stop_loss_pct)
            if current_price <= sl_price:
                return (sl_price, "STOP_LOSS")

        elif self.signal_at_entry == -1:  # Short position
            tp_price = self.entry_price * (1 - self.take_profit_pct)
            if current_price <= tp_price:
                return (tp_price, "TAKE_PROFIT")

            # Stop loss
            sl_price = self.entry_price * (1 + self.stop_loss_pct)
            if current_price >= sl_price:
                return (sl_price, "STOP_LOSS")

        # Reversal signal (close position on opposite signal)
        if self.signal_at_entry == 1 and current_signal == -1:
            return (current_price, "REVERSAL_SHORT")
        elif self.signal_at_entry == -1 and current_signal == 1:
            return (current_price, "REVERSAL_LONG")

        return None

    def backtest_adaptive_xgb(self, data_dict: Dict[str, pd.DataFrame],
                             features_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Run adaptive XGBoost backtest.

        Args:
            data_dict: Dict with OHLCV per timeframe
            features_dict: Dict with features per timeframe

        Returns:
            Dict: Backtest results
        """
        # Ensure 5m data is primary
        if '5m' not in data_dict or len(data_dict['5m']) == 0:
            raise ValueError("Missing 5m data")

        df_5m_data = data_dict['5m']
        df_5m_features = features_dict.get('5m', pd.DataFrame())
        df_1d_features = features_dict.get('1d', pd.DataFrame())

        if len(df_5m_data) == 0 or len(df_5m_features) == 0:
            raise ValueError("5m data/features empty")

        print("\n" + "=" * 80)
        print("ADAPTIVE XGBOOST BACKTEST")
        print("=" * 80)
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print(f"Risk per trade: {self.risk_per_trade*100:.1f}%")
        print(f"Take profit: {self.take_profit_pct*100:.1f}%")
        print(f"Stop loss: {self.stop_loss_pct*100:.1f}%")
        print(f"Bars to process: {len(df_5m_data):,}")
        print("-" * 80)

        # Main backtest loop
        for idx in range(len(df_5m_data)):
            current_bar = df_5m_data.iloc[idx]
            current_price = current_bar['Close']
            timestamp = df_5m_data.index[idx] if hasattr(df_5m_data.index, '__getitem__') else None

            # Get features up to current bar
            df_5m_features_slice = df_5m_features.iloc[:idx+1] if len(df_5m_features) > idx else df_5m_features
            df_1d_features_slice = df_1d_features  # Use all 1D data (slower timeframe)

            # Generate signal
            try:
                signal_result = self.signal_generator.generate_adaptive_signal(
                    df_1d_features_slice,
                    df_5m_features_slice,
                    bars_in_trade=idx - self.entry_bar if self.position_size > 0 else 0,
                    timestamp=timestamp
                )
            except Exception as e:
                print(f"⚠️ Signal generation error at bar {idx}: {e}")
                continue

            # Check exit conditions
            if self.position_size > 0:
                exit_result = self.check_exit_conditions(current_price, idx, signal_result.signal)
                if exit_result:
                    exit_price, exit_reason = exit_result

                    # Calculate profit
                    if self.signal_at_entry == 1:  # Long
                        profit = self.position_size * (exit_price - self.entry_price)
                    else:  # Short
                        profit = self.position_size * (self.entry_price - exit_price)

                    self.equity += profit
                    profit_pct = (profit / (self.equity - profit)) * 100 if (self.equity - profit) != 0 else 0

                    # Record trade
                    self.trades.append({
                        'entry_bar': self.entry_bar,
                        'exit_bar': idx,
                        'bars_held': idx - self.entry_bar,
                        'entry_price': self.entry_price,
                        'exit_price': exit_price,
                        'position_size': self.position_size,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'entry_signal': self.signal_at_entry,
                        'exit_reason': exit_reason,
                        'entry_time': df_5m_data.index[self.entry_bar] if hasattr(df_5m_data.index, '__getitem__') else None,
                        'exit_time': timestamp
                    })

                    self.position_size = 0
                    self.signal_at_entry = 0

            # Check entry conditions
            if self.position_size == 0 and signal_result.should_execute and signal_result.signal != 0:
                entry_price = current_price
                stop_loss_price = entry_price * (1 - self.stop_loss_pct) if signal_result.signal == 1 \
                                  else entry_price * (1 + self.stop_loss_pct)

                self.position_size = self.calculate_position_size(entry_price, stop_loss_price)
                self.entry_price = entry_price
                self.entry_bar = idx
                self.signal_at_entry = signal_result.signal

            # Update equity curve
            self.equity_curve.append({
                'bar': idx,
                'timestamp': timestamp,
                'price': current_price,
                'equity': self.equity,
                'position_size': self.position_size,
                'signal': signal_result.signal,
                'confidence': signal_result.confidence
            })

            # Progress indicator
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx+1:,}/{len(df_5m_data):,} bars... "
                      f"Trades: {len(self.trades)}, Equity: ${self.equity:,.0f}")

        print(f"\n✓ Backtest complete!")
        print(f"  Total bars processed: {len(df_5m_data):,}")
        print(f"  Total trades: {len(self.trades)}")

        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest metrics."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_bars_held': 0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'final_equity': self.equity
            }

        trades_df = pd.DataFrame(self.trades)

        # Basic metrics
        total_return = ((self.equity - self.initial_capital) / self.initial_capital) * 100
        num_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] < 0])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0

        # Profit factor
        total_win = trades_df[trades_df['profit'] > 0]['profit'].sum()
        total_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        # Time in trade
        avg_bars_held = trades_df['bars_held'].mean() if num_trades > 0 else 0

        # Sharpe ratio
        equity_curve_df = pd.DataFrame(self.equity_curve)
        if len(equity_curve_df) > 1:
            returns = equity_curve_df['equity'].pct_change().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(252*24)) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        equity_curve_df['cummax'] = equity_curve_df['equity'].cummax()
        equity_curve_df['drawdown'] = (equity_curve_df['equity'] - equity_curve_df['cummax']) / equity_curve_df['cummax']
        max_drawdown = equity_curve_df['drawdown'].min() * 100 if len(equity_curve_df) > 0 else 0

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_bars_held': avg_bars_held,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'final_equity': self.equity
        }

    def save_results(self, output_dir: Path = None):
        """Save backtest results to CSV and JSON.

        Args:
            output_dir: Output directory (default: backtest/)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent

        output_dir.mkdir(exist_ok=True)

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_csv = output_dir / 'adaptive_xgb_trades.csv'
            trades_df.to_csv(trades_csv, index=False)
            print(f"✓ Trades saved to {trades_csv}")

        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_csv = output_dir / 'adaptive_xgb_equity_curve.csv'
            equity_df.to_csv(equity_csv, index=False)
            print(f"✓ Equity curve saved to {equity_csv}")

        # Save metrics
        metrics = self._calculate_metrics()
        metrics_json = output_dir / 'ADAPTIVE_XGB_RESULTS.json'
        with open(metrics_json, 'w') as f:
            json.dump({
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'initial_capital': self.initial_capital,
                    'risk_per_trade': self.risk_per_trade,
                    'take_profit': self.take_profit_pct,
                    'stop_loss': self.stop_loss_pct
                }
            }, f, indent=2)
        print(f"✓ Metrics saved to {metrics_json}")


def run_adaptive_xgb_backtest():
    """Main backtest execution."""
    print("\n" + "=" * 80)
    print("ADAPTIVE XGBOOST MULTI-TIMEFRAME BACKTEST")
    print("=" * 80)

    # Step 1: Fetch multi-timeframe data
    print("\n[Step 1] Fetching multi-timeframe data...")
    start_time = datetime.now()

    fetcher = MultiTimeframeFetcher()
    data_dict = fetcher.fetch_and_resample(years=2)

    fetch_time = (datetime.now() - start_time).total_seconds()
    print(f"✓ Data fetched in {fetch_time:.1f}s")
    print(f"  1D:  {len(data_dict['1d']):,} candles")
    print(f"  4H:  {len(data_dict['4h']):,} candles")
    print(f"  1H:  {len(data_dict['1h']):,} candles")
    print(f"  15m: {len(data_dict['15m']):,} candles")
    print(f"  5m:  {len(data_dict['5m']):,} candles")

    # Step 2: Feature engineering
    print("\n[Step 2] Engineering features...")
    start_time = datetime.now()

    engineer = MultiTimeframeFeatureEngineer()
    features_dict = engineer.engineer_all_timeframes(data_dict)

    eng_time = (datetime.now() - start_time).total_seconds()
    print(f"✓ Features engineered in {eng_time:.1f}s")

    # Step 3: Backtest
    print("\n[Step 3] Running adaptive XGBoost backtest...")
    start_time = datetime.now()

    backtester = AdaptiveXGBBacktester(
        initial_capital=100000.0,
        risk_per_trade=0.01,
        take_profit_pct=0.01,
        stop_loss_pct=0.005
    )

    metrics = backtester.backtest_adaptive_xgb(data_dict, features_dict)
    backtest_time = (datetime.now() - start_time).total_seconds()

    print(f"✓ Backtest complete in {backtest_time:.1f}s")

    # Step 4: Save results
    print("\n[Step 4] Saving results...")
    backtester.save_results()

    # Step 5: Print summary
    print("\n" + "=" * 80)
    print("ADAPTIVE XGBOOST BACKTEST RESULTS")
    print("=" * 80)
    print(f"Total Return:        {metrics['total_return']:+.2f}%")
    print(f"Number of Trades:    {metrics['num_trades']}")
    print(f"Winning Trades:      {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
    print(f"Losing Trades:       {metrics['losing_trades']}")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"Avg Bars Held:       {metrics['avg_bars_held']:.0f}")
    print(f"Sharpe Ratio:        {metrics['sharpe']:.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']:.2f}%")
    print(f"Final Equity:        ${metrics['final_equity']:,.0f}")
    print("=" * 80)

    # Comparison with Phase 5-A
    print("\nCOMPARISON WITH PHASE 5-A BASELINE")
    print("-" * 80)
    print(f"{'Metric':<25} {'Phase 5-A':<20} {'Adaptive XGB':<20} {'Change':<15}")
    print("-" * 80)
    print(f"{'Total Return':<25} {2.00:>8.2f}% {metrics['total_return']:>14.2f}% {metrics['total_return']-2.00:>+8.2f}pp")
    print(f"{'Number of Trades':<25} {7:>8} {metrics['num_trades']:>14} {metrics['num_trades']-7:>+8}")
    print(f"{'Win Rate':<25} {57.1:>8.1f}% {metrics['win_rate']:>14.1f}% {metrics['win_rate']-57.1:>+8.1f}pp")
    print(f"{'Sharpe Ratio':<25} {7.87:>8.2f} {metrics['sharpe']:>14.2f} {metrics['sharpe']-7.87:>+8.2f}")
    print(f"{'Max Drawdown':<25} {-0.25:>8.2f}% {metrics['max_drawdown']:>14.2f}% {metrics['max_drawdown']-(-0.25):>+8.2f}pp")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    try:
        metrics = run_adaptive_xgb_backtest()
    except Exception as e:
        print(f"\n✗ Backtest error: {e}")
        import traceback
        traceback.print_exc()
