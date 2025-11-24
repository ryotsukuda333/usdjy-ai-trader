"""Multi-Timeframe Backtest with XGBoost Probability Weighting

Enhanced backtest engine that integrates XGBoost probabilities for improved signal
discrimination and trade execution.

Key Improvements over Technical-Only Confluence:
- Uses XGBoost probability weighting (50% of confidence calculation)
- Lowers execution threshold: 0.55 (vs 0.70 alignment-only)
- Incorporates seasonality and technical strength metrics
- Expected: 30-50 executable signals, 3-8 trades, +2-3% return

Entry Logic:
- 1D BUY signal confirmed by 4H
- XGBoost probability ≥ threshold (via combined confidence)
- Optional: 15m/5m entry confirmation

Exit Logic:
- 1D reversal to SELL
- Take-profit at 1% return
- Stop-loss at 0.5% loss
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.multi_timeframe_fetcher import fetch_multi_timeframe_usdjpy
from features.multi_timeframe_engineer import engineer_features_multi_timeframe
from model.multi_timeframe_signal_generator_xgb import generate_multi_timeframe_signals_xgb


class MultiTimeframeBacktesterXGB:
    """Backtester with XGBoost probability weighting."""

    def __init__(
        self,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.01,  # 1% risk
        take_profit_pct: float = 1.0,
        stop_loss_pct: float = 0.5,
        xgb_threshold: float = 0.55  # Execute threshold with XGBoost
    ):
        """Initialize backtester with XGBoost integration.

        Args:
            initial_capital: Starting capital in USD
            risk_per_trade: Risk per trade as fraction of capital
            take_profit_pct: Take-profit target in percent
            stop_loss_pct: Stop-loss distance in percent
            xgb_threshold: Minimum combined confidence to execute
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.xgb_threshold = xgb_threshold

        # State tracking
        self.current_equity = initial_capital
        self.trades = []
        self.equity_curve = []
        self.position = None  # Current position dict

    def map_5m_to_higher_timeframes(self, idx_5m: int) -> Dict[str, int]:
        """Map 5-minute bar index to higher timeframe indices."""
        return {
            '5m': idx_5m,
            '15m': idx_5m // 3,
            '1h': idx_5m // 12,
            '4h': idx_5m // 48,
            '1d': idx_5m // 288,
        }

    def get_signal_at_index(self,
                           signals_dict: Dict[str, pd.DataFrame],
                           idx_map: Dict[str, int]) -> Tuple[int, float, str]:
        """Get confluence signal at current bar indices.

        Args:
            signals_dict: Dict with signal DataFrames per timeframe
            idx_map: Map of timeframe indices

        Returns:
            Tuple: (signal, confidence, reason)
        """
        try:
            # Get signal from 5m (highest resolution, most recent)
            if '5m' in signals_dict and idx_map['5m'] < len(signals_dict['5m']):
                df_signal = signals_dict['5m']
                idx = idx_map['5m']

                signal = int(df_signal.iloc[idx].get('signal', 0))
                confidence = float(df_signal.iloc[idx].get('confidence', 0.0))
                should_execute = bool(df_signal.iloc[idx].get('should_execute', False))
                reason = str(df_signal.iloc[idx].get('reason', 'No reason'))

                return signal, confidence, reason if should_execute else f"Below threshold: {reason}"

            return 0, 0.0, "No signal available"

        except Exception as e:
            return 0, 0.0, f"Signal error: {str(e)}"

    def calculate_position_size(self,
                               current_price: float,
                               stop_loss_price: float,
                               account_balance: float) -> int:
        """Calculate position size based on risk management.

        Args:
            current_price: Entry price
            stop_loss_price: Stop-loss price
            account_balance: Current account balance

        Returns:
            int: Number of units to trade
        """
        # Calculate loss if stopped out
        stop_loss_amount = abs(current_price - stop_loss_price)

        if stop_loss_amount <= 0:
            return 0

        # Risk amount = account * risk_per_trade
        risk_amount = account_balance * self.risk_per_trade

        # Position size = risk_amount / stop_loss_amount
        position_size = int(risk_amount / stop_loss_amount)

        return max(1, position_size)  # At least 1 unit

    def check_exit_conditions(self,
                             position: Dict,
                             current_price: float,
                             current_signal: int) -> Tuple[bool, str]:
        """Check if current position should be exited.

        Args:
            position: Current position dict
            current_price: Current price
            current_signal: Current signal (1=BUY, -1=SELL, 0=HOLD)

        Returns:
            Tuple: (should_exit, exit_reason)
        """
        entry_price = position['entry_price']
        entry_signal = position['signal']
        position_type = 'LONG' if entry_signal == 1 else 'SHORT'

        # Exit condition 1: Signal reversal
        if entry_signal == 1 and current_signal == -1:
            return True, f"Signal reversal ({position_type} to SHORT)"

        if entry_signal == -1 and current_signal == 1:
            return True, f"Signal reversal ({position_type} to LONG)"

        # Exit condition 2: Take-profit
        if entry_signal == 1:  # Long position
            if current_price >= entry_price * (1 + self.take_profit_pct / 100):
                return True, f"Take-profit (+{self.take_profit_pct}%)"

            # Exit condition 3: Stop-loss
            if current_price <= entry_price * (1 - self.stop_loss_pct / 100):
                return True, f"Stop-loss (-{self.stop_loss_pct}%)"

        else:  # Short position
            if current_price <= entry_price * (1 - self.take_profit_pct / 100):
                return True, f"Take-profit (+{self.take_profit_pct}%)"

            if current_price >= entry_price * (1 + self.stop_loss_pct / 100):
                return True, f"Stop-loss (-{self.stop_loss_pct}%)"

        return False, ""

    def backtest_multi_timeframe_xgb(self,
                                     data_dict: Dict[str, pd.DataFrame],
                                     features_dict: Dict[str, pd.DataFrame],
                                     signals_dict: Dict[str, pd.DataFrame]) -> Tuple[float, Dict]:
        """Execute backtest with XGBoost probability weighting.

        Args:
            data_dict: Dict with OHLCV per timeframe
            features_dict: Dict with features per timeframe
            signals_dict: Dict with signals per timeframe (from XGBoost generator)

        Returns:
            Tuple: (return_pct, metrics_dict)
        """
        print("\n" + "=" * 80)
        print("MULTI-TIMEFRAME BACKTEST WITH XGBoost PROBABILITY WEIGHTING")
        print("=" * 80)

        # Get 5m data (iteration resolution)
        df_5m = data_dict['5m']
        bars_count = len(df_5m)

        print(f"\nBacktest Configuration:")
        print(f"  Initial Capital:      ${self.initial_capital:,.2f}")
        print(f"  Risk per Trade:       {self.risk_per_trade*100:.1f}%")
        print(f"  Take Profit:          {self.take_profit_pct:.2f}%")
        print(f"  Stop Loss:            {self.stop_loss_pct:.2f}%")
        print(f"  XGBoost Threshold:    {self.xgb_threshold:.2f}")
        print(f"\nData:")
        print(f"  5m Bars:              {bars_count:,}")
        print(f"  Date Range:           {df_5m.index[0]} to {df_5m.index[-1]}")

        # Backtest loop
        start_time = time.time()
        signal_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'EXECUTED': 0}

        for idx_5m in range(len(df_5m)):
            if idx_5m % 10000 == 0:
                print(f"  Processing bar {idx_5m:,} / {bars_count:,}...")

            # Get current price
            current_row = df_5m.iloc[idx_5m]
            current_price = current_row['Close']
            current_time = df_5m.index[idx_5m]

            # Map to higher timeframe indices
            idx_map = self.map_5m_to_higher_timeframes(idx_5m)

            # Get signal
            signal, confidence, reason = self.get_signal_at_index(signals_dict, idx_map)

            # Track signal counts
            if signal == 1:
                signal_count['BUY'] += 1
            elif signal == -1:
                signal_count['SELL'] += 1
            else:
                signal_count['HOLD'] += 1

            # Check exit conditions
            if self.position is not None:
                should_exit, exit_reason = self.check_exit_conditions(
                    self.position,
                    current_price,
                    signal
                )

                if should_exit:
                    # Exit trade
                    entry_price = self.position['entry_price']
                    profit_loss = current_price - entry_price if self.position['signal'] == 1 else entry_price - current_price
                    profit_pct = (profit_loss / entry_price) * 100
                    profit_usd = profit_loss * self.position['size']

                    # Update equity
                    self.current_equity += profit_usd

                    # Record trade
                    self.trades.append({
                        'entry_time': self.position['entry_time'],
                        'entry_price': entry_price,
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'signal': self.position['signal'],
                        'size': self.position['size'],
                        'profit': profit_usd,
                        'profit_pct': profit_pct,
                        'bars_held': idx_5m - self.position['entry_idx'],
                        'exit_reason': exit_reason,
                        'confidence_at_entry': self.position['confidence']
                    })

                    print(f"  EXIT: {self.position['signal']:+d} at {current_price:.4f} "
                          f"(P&L: {profit_pct:+.3f}%, Reason: {exit_reason})")

                    self.position = None

            # Check entry conditions
            if self.position is None and signal != 0 and confidence >= self.xgb_threshold:
                # Enter trade
                if signal == 1:
                    stop_loss_price = current_price * (1 - self.stop_loss_pct / 100)
                else:
                    stop_loss_price = current_price * (1 + self.stop_loss_pct / 100)

                size = self.calculate_position_size(current_price, stop_loss_price, self.current_equity)

                if size > 0:
                    signal_count['EXECUTED'] += 1

                    self.position = {
                        'entry_time': current_time,
                        'entry_idx': idx_5m,
                        'entry_price': current_price,
                        'stop_loss': stop_loss_price,
                        'take_profit': current_price * (1 + (self.take_profit_pct if signal == 1 else -self.take_profit_pct) / 100),
                        'signal': signal,
                        'size': size,
                        'confidence': confidence
                    }

                    signal_type = "BUY" if signal == 1 else "SELL"
                    print(f"  ENTRY: {signal_type} at {current_price:.4f} (Conf: {confidence:.3f}, Size: {size})")

            # Record equity
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': self.current_equity,
                'position': 'OPEN' if self.position else 'CLOSED'
            })

        elapsed = time.time() - start_time

        # Calculate metrics
        total_return = self.current_equity - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100

        print(f"\nBacktest Completed in {elapsed:.1f} seconds")
        print(f"\nSignal Summary:")
        print(f"  BUY signals:          {signal_count['BUY']:,}")
        print(f"  SELL signals:         {signal_count['SELL']:,}")
        print(f"  HOLD signals:         {signal_count['HOLD']:,}")
        print(f"  Executable signals:   {signal_count['EXECUTED']} ({100*signal_count['EXECUTED']/bars_count:.2f}%)")

        # Trade statistics
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            win_count = len(df_trades[df_trades['profit'] > 0])
            loss_count = len(df_trades[df_trades['profit'] < 0])
            win_rate = (win_count / len(df_trades)) * 100 if len(df_trades) > 0 else 0

            total_wins = df_trades[df_trades['profit'] > 0]['profit'].sum()
            total_losses = abs(df_trades[df_trades['profit'] < 0]['profit'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

            # Sharpe ratio
            returns = df_trades['profit_pct'].values
            if len(returns) > 1:
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe = 0

            # Max drawdown
            equity_curve = [self.initial_capital]
            for trade in self.trades:
                equity_curve.append(equity_curve[-1] + trade['profit'])

            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (np.array(equity_curve) - running_max) / running_max
            max_dd_pct = np.min(drawdown) * 100 if len(drawdown) > 0 else 0

            metrics = {
                'total_return': return_pct,
                'num_trades': len(self.trades),
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_bars_held': df_trades['bars_held'].mean(),
                'sharpe': sharpe,
                'max_drawdown': max_dd_pct,
                'final_equity': self.current_equity
            }

            print(f"\n" + "=" * 80)
            print("RESULTS: Multi-Timeframe + XGBoost vs Phase 5-A")
            print("=" * 80)
            print(f"\n{'Metric':<30} {'Phase 5-A':<20} {'Multi-TF+XGB':<20}")
            print("-" * 70)
            print(f"{'Total Return':<30} {2.00:>7.2f}% {return_pct:>15.2f}%")
            print(f"{'Trades':<30} {7:>7d} {len(self.trades):>15d}")
            print(f"{'Win Rate':<30} {57.1:>7.1f}% {win_rate:>15.1f}%")
            print(f"{'Sharpe Ratio':<30} {7.87:>7.2f} {sharpe:>15.2f}")
            print(f"{'Max Drawdown':<30} {-0.25:>7.2f}% {max_dd_pct:>15.2f}%")
            print(f"{'Final Equity':<30} ${102000:>17,.0f} ${self.current_equity:>15,.0f}")
            print("\n" + "=" * 80)

        else:
            metrics = {
                'total_return': 0.0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_bars_held': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'final_equity': self.initial_capital
            }

            print(f"\n⚠️ No trades executed (check XGBoost threshold: {self.xgb_threshold})")

        return return_pct, metrics

    def save_results(self, output_dir: Path):
        """Save backtest results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trades
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            trades_file = output_dir / "multi_timeframe_xgb_trades.csv"
            df_trades.to_csv(trades_file, index=False)
            print(f"✓ Trades saved to {trades_file}")

        # Save equity curve
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            equity_file = output_dir / "multi_timeframe_xgb_equity.csv"
            df_equity.to_csv(equity_file, index=False)
            print(f"✓ Equity curve saved to {equity_file}")


def run_multi_timeframe_backtest_xgb(years: int = 2) -> Tuple[float, Dict]:
    """Run complete multi-timeframe backtest with XGBoost.

    Args:
        years: Number of years of data to backtest

    Returns:
        Tuple: (return_pct, metrics)
    """
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME + XGBoost BACKTEST PIPELINE")
    print("=" * 80)

    # Step 1: Fetch data
    print("\n[1/4] Fetching multi-timeframe data...")
    start = time.time()
    data_dict = fetch_multi_timeframe_usdjpy(years=years)
    print(f"✓ Data fetching completed in {time.time() - start:.1f}s")

    # Step 2: Engineer features
    print("\n[2/4] Engineering features...")
    start = time.time()
    features_dict = engineer_features_multi_timeframe(data_dict)
    print(f"✓ Feature engineering completed in {time.time() - start:.1f}s")

    # Step 3: Generate signals with XGBoost
    print("\n[3/4] Generating signals with XGBoost...")
    start = time.time()
    signals_dict = generate_multi_timeframe_signals_xgb(data_dict, features_dict)
    print(f"✓ Signal generation completed in {time.time() - start:.1f}s")

    # Step 4: Run backtest
    print("\n[4/4] Running backtest...")
    start = time.time()
    backtester = MultiTimeframeBacktesterXGB()
    return_pct, metrics = backtester.backtest_multi_timeframe_xgb(data_dict, features_dict, signals_dict)

    # Save results
    output_dir = Path(__file__).parent
    backtester.save_results(output_dir)

    print(f"✓ Backtest completed in {time.time() - start:.1f}s")

    return return_pct, metrics


if __name__ == "__main__":
    return_pct, metrics = run_multi_timeframe_backtest_xgb(years=2)

    # Save metrics
    results = {
        'metrics': metrics,
        'timestamp': str(pd.Timestamp.now()),
        'configuration': {
            'initial_capital': 100000,
            'risk_per_trade': 0.01,
            'xgb_threshold': 0.55
        }
    }

    results_file = Path(__file__).parent / "MULTI_TIMEFRAME_XGB_RESULTS.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")
    print(f"\nFinal Return: {return_pct:+.2f}%")
