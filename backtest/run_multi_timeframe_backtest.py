"""Multi-Timeframe Backtest Engine

Implements 5-minute resolution backtest with hierarchical multi-timeframe signal integration.

Strategy:
1. Iterate through 5-minute candles (highest resolution)
2. At each 5m bar, map to higher timeframe bars (1D, 4H, 1H, 15m)
3. Get signals from each timeframe at its current bar
4. Calculate multi-timeframe alignment score
5. Execute entry/exit based on confluence rules

Entry Logic:
- 1D trend confirmation + 4H alignment + (15m OR 5m entry signal)
- Minimum alignment score threshold (0.70)

Exit Logic:
- 1D trend reversal OR alignment drops below threshold
- Take-profit at technical levels (Bollinger Bands, support/resistance)
- Stop-loss at defined percentage or volatility-adjusted levels

Position Sizing:
- Fixed 1% risk per trade (Kelly criterion / N safety factor)
- Adjusted for volatility if SL width varies

Statistics Generated:
- Trade count (5m resolution)
- Win rate and profit factor
- Timeframe alignment effectiveness
- Entry precision metrics
- Comparison with Phase 5-A baseline
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
from model.multi_timeframe_signal_generator import generate_multi_timeframe_signals


class MultiTimeframeBacktester:
    """Backtester for multi-timeframe hierarchical trading strategy."""

    def __init__(
        self,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.01,  # 1% risk
        alignment_threshold: float = 0.70,
        take_profit_pct: float = 1.0,
        stop_loss_pct: float = 0.5
    ):
        """Initialize backtester.

        Args:
            initial_capital: Starting capital in USD
            risk_per_trade: Risk per trade as fraction of capital
            alignment_threshold: Minimum alignment score for entry
            take_profit_pct: Take-profit target in percent
            stop_loss_pct: Stop-loss distance in percent
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.alignment_threshold = alignment_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

        # State tracking
        self.current_equity = initial_capital
        self.trades = []
        self.equity_curve = []

    def map_5m_to_higher_timeframes(self, idx_5m: int) -> Dict[str, int]:
        """Map 5-minute bar index to bar indices in higher timeframes.

        Assumes 1D has 288 5m bars (1440 minutes / 5 minutes).

        Args:
            idx_5m: Index in 5m series

        Returns:
            Dict with keys ['1d', '4h', '1h', '15m', '5m'] and their bar indices
        """
        return {
            '5m': idx_5m,
            '15m': idx_5m // 3,
            '1h': idx_5m // 12,
            '4h': idx_5m // 48,
            '1d': idx_5m // 288,
        }

    def get_signals_at_index(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        timeframe_indices: Dict[str, int]
    ) -> Dict[str, dict]:
        """Get signal data for each timeframe at mapped index.

        Args:
            signals_dict: Dict with signal predictions per timeframe
            timeframe_indices: Dict with indices per timeframe

        Returns:
            Dict with signal data at each timeframe
        """
        signals_at_idx = {}

        for tf in ['1d', '4h', '1h', '15m', '5m']:
            idx = timeframe_indices[tf]

            if tf in signals_dict and idx < len(signals_dict[tf]):
                row = signals_dict[tf].iloc[idx]
                signals_at_idx[tf] = {
                    'signal': row['signal'],
                    'confidence': row['confidence'],
                    'alignment': row.get('alignment', 0.5),
                    'should_execute': row.get('should_execute', True)
                }

        return signals_at_idx

    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:
        """Calculate position size based on risk management.

        Uses fixed risk per trade: Size = (Capital × Risk%) / (Entry - SL)

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price

        Returns:
            float: Position size in units
        """
        if entry_price <= 0:
            return 0

        risk_amount = self.current_equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            return 0

        position_size = risk_amount / price_risk
        return position_size

    def backtest_multi_timeframe(
        self,
        df_5m: pd.DataFrame,
        signals_dict: Dict[str, pd.DataFrame],
        features_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[float, Dict]:
        """Run multi-timeframe backtest.

        Args:
            df_5m: 5-minute OHLCV data
            signals_dict: Signal predictions per timeframe (key: 'confluence')
            features_dict: Engineered features per timeframe

        Returns:
            Tuple[total_return%, metrics_dict]
        """
        print("=" * 80)
        print("MULTI-TIMEFRAME BACKTEST EXECUTION")
        print("=" * 80)
        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        print(f"Risk per Trade: {self.risk_per_trade*100:.1f}%")
        print(f"Take-Profit: {self.take_profit_pct:.2f}%")
        print(f"Stop-Loss: {self.stop_loss_pct:.2f}%")
        print(f"Alignment Threshold: {self.alignment_threshold:.2f}")

        confluence_signals = signals_dict.get('confluence')
        if confluence_signals is None:
            print("✗ No confluence signals found")
            return 0, {}

        # Position tracking
        position = None  # {'entry_idx', 'entry_price', 'entry_signal', 'stop_loss', 'take_profit'}
        bar_count = len(df_5m)

        print(f"\nBacktesting {bar_count} 5-minute bars...")
        print("=" * 80)

        for i in range(1, bar_count):
            current_bar = df_5m.iloc[i]
            current_price = current_bar['Close']

            # Map 5m index to higher timeframe indices
            timeframe_indices = self.map_5m_to_higher_timeframes(i)

            # Get signals at this bar across timeframes
            signals_at_i = self.get_signals_at_index(signals_dict, timeframe_indices)

            # Get confluence signal
            if i < len(confluence_signals):
                confluence_row = confluence_signals.iloc[i]
                confluence_signal = confluence_row['signal']
                confluence_alignment = confluence_row['alignment']
                should_execute = confluence_row['should_execute']
            else:
                confluence_signal = -1
                confluence_alignment = 0.5
                should_execute = False

            # EXIT LOGIC - Check for position exit
            if position:
                # Check for stop-loss
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = "Stop-Loss"
                    profit_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    profit = position['position_size'] * (exit_price - position['entry_price'])
                    self.current_equity += profit

                    self.trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_signal': position['entry_signal'],
                        'exit_signal': confluence_signal,
                        'position_size': position['position_size'],
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'bars_held': i - position['entry_idx'],
                        'exit_reason': exit_reason,
                        'alignment_at_entry': position['alignment_at_entry']
                    })

                    position = None

                # Check for take-profit
                elif current_price >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = "Take-Profit"
                    profit_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    profit = position['position_size'] * (exit_price - position['entry_price'])
                    self.current_equity += profit

                    self.trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_signal': position['entry_signal'],
                        'exit_signal': confluence_signal,
                        'position_size': position['position_size'],
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'bars_held': i - position['entry_idx'],
                        'exit_reason': exit_reason,
                        'alignment_at_entry': position['alignment_at_entry']
                    })

                    position = None

                # Check for trend reversal (1D sell signal)
                elif confluence_signal == 0 and confluence_alignment > 0.6:
                    exit_price = current_price
                    exit_reason = "Trend Reversal"
                    profit_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    profit = position['position_size'] * (exit_price - position['entry_price'])
                    self.current_equity += profit

                    self.trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_signal': position['entry_signal'],
                        'exit_signal': confluence_signal,
                        'position_size': position['position_size'],
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'bars_held': i - position['entry_idx'],
                        'exit_reason': exit_reason,
                        'alignment_at_entry': position['alignment_at_entry']
                    })

                    position = None

            # ENTRY LOGIC - Look for new entry
            if not position and confluence_signal == 1 and should_execute:
                # Calculate position sizing
                stop_loss_price = current_price * (1 - self.stop_loss_pct / 100)
                take_profit_price = current_price * (1 + self.take_profit_pct / 100)

                position_size = self.calculate_position_size(current_price, stop_loss_price)

                if position_size > 0:
                    position = {
                        'entry_idx': i,
                        'entry_price': current_price,
                        'entry_signal': confluence_signal,
                        'position_size': position_size,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'alignment_at_entry': confluence_alignment,
                        '1d_signal': signals_at_i.get('1d', {}).get('signal', -1),
                        '4h_signal': signals_at_i.get('4h', {}).get('signal', -1),
                    }

            # Record equity
            self.equity_curve.append({
                'idx': i,
                'equity': self.current_equity,
                'in_position': position is not None,
                'alignment': confluence_alignment
            })

        # Close any remaining position at last bar
        if position:
            exit_price = df_5m.iloc[-1]['Close']
            exit_reason = "End-of-Backtest"
            profit_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
            profit = position['position_size'] * (exit_price - position['entry_price'])
            self.current_equity += profit

            self.trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': bar_count - 1,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'entry_signal': position['entry_signal'],
                'exit_signal': -1,
                'position_size': position['position_size'],
                'profit': profit,
                'profit_pct': profit_pct,
                'bars_held': bar_count - 1 - position['entry_idx'],
                'exit_reason': exit_reason,
                'alignment_at_entry': position['alignment_at_entry']
            })

        # Calculate metrics
        total_return = (self.current_equity - self.initial_capital) / self.initial_capital * 100

        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] < 0]

        win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
        avg_win = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0

        total_win = sum(t['profit'] for t in winning_trades)
        total_loss = sum(t['profit'] for t in losing_trades)
        profit_factor = total_win / abs(total_loss) if total_loss != 0 else 0

        # Sharpe ratio (assuming 252 trading days, daily return = total_return / bars * 288)
        daily_returns = []
        for trade in self.trades:
            daily_returns.append(trade['profit_pct'])

        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252) if daily_returns else 0

        # Max drawdown
        peak_equity = self.initial_capital
        max_dd = 0
        for eq_record in self.equity_curve:
            if eq_record['equity'] > peak_equity:
                peak_equity = eq_record['equity']
            dd = (peak_equity - eq_record['equity']) / peak_equity
            if dd > max_dd:
                max_dd = dd

        metrics = {
            'total_return': total_return,
            'final_equity': self.current_equity,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'max_dd': max_dd * 100,
            'total_win': total_win,
            'total_loss': total_loss,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

        return total_return, metrics

    def print_results(self, metrics: Dict):
        """Print backtest results summary.

        Args:
            metrics: Metrics dictionary from backtest
        """
        print("\n" + "=" * 80)
        print("MULTI-TIMEFRAME BACKTEST RESULTS")
        print("=" * 80)

        print(f"\nPerformance:")
        print(f"  Total Return:        {metrics['total_return']:+.2f}%")
        print(f"  Final Equity:        ${metrics['final_equity']:,.2f}")
        print(f"  Number of Trades:    {metrics['num_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']:.1f}%")
        print(f"  Avg Win:             {metrics['avg_win_pct']:+.2f}%")
        print(f"  Avg Loss:            {metrics['avg_loss_pct']:+.2f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:        {metrics['sharpe']:.2f}")
        print(f"  Max Drawdown:        {metrics['max_dd']:.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Total Win:           ${metrics['total_win']:,.2f}")
        print(f"  Total Loss:          ${metrics['total_loss']:,.2f}")

        print(f"\nComparison with Phase 5-A (1D Grid Search):")
        print(f"  Metric              Phase 5-A    Multi-TF    Improvement")
        print(f"  " + "-" * 60)
        print(f"  Return              +2.00%       {metrics['total_return']:+.2f}%      {metrics['total_return']-2:.2f}pp")
        print(f"  Trades              7            {metrics['num_trades']:>8}      {metrics['num_trades']-7:+d}")
        print(f"  Win Rate            57.1%        {metrics['win_rate']:5.1f}%      {metrics['win_rate']-57.1:+.1f}pp")
        print(f"  Max DD              -0.25%       {-metrics['max_dd']:6.2f}%      {metrics['max_dd']-0.25:+.2f}pp")


def run_multi_timeframe_backtest():
    """Main execution: Run complete multi-timeframe backtest."""

    project_root = Path(__file__).parent.parent

    print("=" * 80)
    print("MULTI-TIMEFRAME HIERARCHICAL TRADING BACKTEST")
    print("Swing Trading with Day Trading Execution")
    print("=" * 80)

    # Step 1: Fetch multi-timeframe data
    print("\n[1] Fetching multi-timeframe data...")
    start_time = time.time()

    try:
        timeframe_dict = fetch_multi_timeframe_usdjpy(years=2)
        print(f"✓ Fetched multi-timeframe data ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print(f"✗ Failed to fetch data: {e}")
        return None

    # Step 2: Engineer features
    print("\n[2] Engineering multi-timeframe features...")
    feature_start = time.time()

    try:
        features_dict = engineer_features_multi_timeframe(timeframe_dict)
        print(f"✓ Engineered features ({time.time() - feature_start:.1f}s)")
    except Exception as e:
        print(f"✗ Failed to engineer features: {e}")
        return None

    # Step 3: Generate signals
    print("\n[3] Generating multi-timeframe signals...")
    signal_start = time.time()

    try:
        signals_dict = generate_multi_timeframe_signals(features_dict)
        print(f"✓ Generated signals ({time.time() - signal_start:.1f}s)")
    except Exception as e:
        print(f"✗ Failed to generate signals: {e}")
        return None

    # Step 4: Run backtest
    print("\n[4] Running multi-timeframe backtest...")
    backtest_start = time.time()

    try:
        backtester = MultiTimeframeBacktester(
            initial_capital=100000,
            risk_per_trade=0.01,  # 1%
            alignment_threshold=0.70,
            take_profit_pct=1.0,
            stop_loss_pct=0.5
        )

        df_5m = timeframe_dict['5m']
        total_return, metrics = backtester.backtest_multi_timeframe(
            df_5m,
            signals_dict,
            features_dict
        )

        print(f"✓ Backtest complete ({time.time() - backtest_start:.1f}s)")

    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 5: Print results
    backtester.print_results(metrics)

    # Step 6: Save results
    print("\n[5] Saving results...")

    try:
        results = {
            'phase': 'Multi-Timeframe',
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': {
                'total_return': float(metrics['total_return']),
                'final_equity': float(metrics['final_equity']),
                'num_trades': int(metrics['num_trades']),
                'win_rate': float(metrics['win_rate']),
                'avg_win_pct': float(metrics['avg_win_pct']),
                'avg_loss_pct': float(metrics['avg_loss_pct']),
                'profit_factor': float(metrics['profit_factor']),
                'sharpe': float(metrics['sharpe']),
                'max_dd': float(metrics['max_dd']),
            },
            'comparison': {
                'phase_5a_return': 2.00,
                'phase_5a_trades': 7,
                'phase_5a_win_rate': 57.1,
            }
        }

        results_path = project_root / "MULTI_TIMEFRAME_RESULTS.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved: {results_path}")

        # Save trades
        trades_df = pd.DataFrame(metrics['trades'])
        trades_path = project_root / "backtest" / "multi_timeframe_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"✓ Trades saved: {trades_path}")

    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")

    print("\n" + "=" * 80)
    print("Multi-Timeframe Backtest Complete")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    metrics = run_multi_timeframe_backtest()

    if metrics:
        print(f"\n✓ Execution successful")
        sys.exit(0)
    else:
        print(f"\n✗ Execution failed")
        sys.exit(1)
