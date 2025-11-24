"""Multi-Timeframe Trading Statistics Analysis

Generates comprehensive statistics comparing multi-timeframe approach with Phase 5-A baseline.

Metrics Include:
- Overall performance (return, trades, win rate)
- Timeframe effectiveness (which TF combinations worked best)
- Entry precision (how many bars waited for 15m/5m confirmation)
- Signal alignment analysis
- Risk metrics (max drawdown, Sharpe ratio)
- Trade quality distribution
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class MultiTimeframeStatisticsAnalyzer:
    """Analyze multi-timeframe trading statistics."""

    def __init__(self):
        """Initialize statistics analyzer."""
        self.phase_5a_baseline = {
            'total_return': 2.00,
            'num_trades': 7,
            'win_rate': 57.1,
            'sharpe': 7.87,
            'max_dd': 0.25,
        }

    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """Analyze trade-level statistics.

        Args:
            trades: List of trade dictionaries from backtest

        Returns:
            Dict with comprehensive trade statistics
        """
        if not trades:
            return {
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_bars_held': 0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
            }

        trades_df = pd.DataFrame(trades)

        # Basic counts
        num_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] < 0])

        # Win rate and profit factor
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
        total_win = trades_df[trades_df['profit'] > 0]['profit'].sum()
        total_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        # Time in trade
        avg_bars_held = trades_df['bars_held'].mean()

        # Profitability
        winning_df = trades_df[trades_df['profit'] > 0]
        losing_df = trades_df[trades_df['profit'] < 0]

        avg_win_pct = winning_df['profit_pct'].mean() if len(winning_df) > 0 else 0
        avg_loss_pct = losing_df['profit_pct'].mean() if len(losing_df) > 0 else 0

        return {
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_bars_held': avg_bars_held,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'total_win': total_win,
            'total_loss': total_loss,
        }

    def analyze_timeframe_alignment(self, trades: List[Dict]) -> Dict:
        """Analyze effectiveness of multi-timeframe alignment.

        Args:
            trades: List of trade dictionaries

        Returns:
            Dict with alignment effectiveness metrics
        """
        if not trades:
            return {
                'aligned_trades': 0,
                'alignment_rate': 0.0,
                'aligned_win_rate': 0.0,
                'aligned_avg_profit': 0.0,
            }

        trades_df = pd.DataFrame(trades)

        # Trades with high alignment at entry
        aligned_trades = trades_df[trades_df['alignment_at_entry'] >= 0.70]
        misaligned_trades = trades_df[trades_df['alignment_at_entry'] < 0.70]

        alignment_rate = (len(aligned_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0

        # Performance by alignment
        if len(aligned_trades) > 0:
            aligned_win_rate = (len(aligned_trades[aligned_trades['profit'] > 0]) /
                               len(aligned_trades) * 100)
            aligned_avg_profit = aligned_trades['profit_pct'].mean()
        else:
            aligned_win_rate = 0.0
            aligned_avg_profit = 0.0

        if len(misaligned_trades) > 0:
            misaligned_win_rate = (len(misaligned_trades[misaligned_trades['profit'] > 0]) /
                                  len(misaligned_trades) * 100)
            misaligned_avg_profit = misaligned_trades['profit_pct'].mean()
        else:
            misaligned_win_rate = 0.0
            misaligned_avg_profit = 0.0

        return {
            'alignment_rate': alignment_rate,
            'aligned_trades': len(aligned_trades),
            'misaligned_trades': len(misaligned_trades),
            'aligned_win_rate': aligned_win_rate,
            'misaligned_win_rate': misaligned_win_rate,
            'aligned_avg_profit_pct': aligned_avg_profit,
            'misaligned_avg_profit_pct': misaligned_avg_profit,
        }

    def analyze_signal_combinations(self, trades: List[Dict]) -> Dict:
        """Analyze which timeframe signal combinations worked best.

        Args:
            trades: List of trade dictionaries

        Returns:
            Dict with signal combination effectiveness
        """
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)

        # Analyze 1D/4H signal combinations
        combinations = {}

        # Combination: 1D=BUY, 4H=BUY
        buy_buy = trades_df[(trades_df['1d_signal'] == 1) & (trades_df['4h_signal'] == 1)]
        if len(buy_buy) > 0:
            combinations['1D_BUY_4H_BUY'] = {
                'count': len(buy_buy),
                'win_rate': len(buy_buy[buy_buy['profit'] > 0]) / len(buy_buy) * 100,
                'avg_profit_pct': buy_buy['profit_pct'].mean(),
            }

        # Combination: 1D=BUY, 4H=HOLD
        buy_hold = trades_df[(trades_df['1d_signal'] == 1) & (trades_df['4h_signal'] == -1)]
        if len(buy_hold) > 0:
            combinations['1D_BUY_4H_HOLD'] = {
                'count': len(buy_hold),
                'win_rate': len(buy_hold[buy_hold['profit'] > 0]) / len(buy_hold) * 100,
                'avg_profit_pct': buy_hold['profit_pct'].mean(),
            }

        # Combination: 1D=BUY, 4H=SELL
        buy_sell = trades_df[(trades_df['1d_signal'] == 1) & (trades_df['4h_signal'] == 0)]
        if len(buy_sell) > 0:
            combinations['1D_BUY_4H_SELL'] = {
                'count': len(buy_sell),
                'win_rate': len(buy_sell[buy_sell['profit'] > 0]) / len(buy_sell) * 100,
                'avg_profit_pct': buy_sell['profit_pct'].mean(),
            }

        return combinations

    def generate_comparison_report(
        self,
        multi_tf_metrics: Dict,
        trades: List[Dict]
    ) -> str:
        """Generate comprehensive comparison report.

        Args:
            multi_tf_metrics: Metrics from multi-timeframe backtest
            trades: List of trades

        Returns:
            str: Formatted report
        """
        trade_stats = self.analyze_trades(trades)
        alignment_stats = self.analyze_timeframe_alignment(trades)
        signal_stats = self.analyze_signal_combinations(trades)

        report = []
        report.append("=" * 100)
        report.append("MULTI-TIMEFRAME vs PHASE 5-A COMPARISON REPORT")
        report.append("=" * 100)

        # Performance Comparison
        report.append("\n📊 PERFORMANCE COMPARISON\n")
        report.append(f"{'Metric':<25} {'Phase 5-A':<20} {'Multi-TF':<20} {'Change':<15}")
        report.append("-" * 80)

        metrics_to_compare = [
            ('Total Return', 'total_return', '%'),
            ('Number of Trades', 'num_trades', 'count'),
            ('Win Rate', 'win_rate', '%'),
            ('Sharpe Ratio', 'sharpe', 'ratio'),
            ('Max Drawdown', 'max_dd', '%'),
        ]

        for metric_name, key, unit in metrics_to_compare:
            phase_5a_val = self.phase_5a_baseline.get(key, 0)
            multi_tf_val = multi_tf_metrics.get(key, 0)

            if unit == '%':
                change = multi_tf_val - phase_5a_val
                report.append(f"{metric_name:<25} {phase_5a_val:>8.2f}% {multi_tf_val:>14.2f}% {change:>+8.2f}pp")
            elif unit == 'count':
                change = multi_tf_val - phase_5a_val
                report.append(f"{metric_name:<25} {phase_5a_val:>8.0f}   {multi_tf_val:>14.0f}   {change:>+8.0f}")
            else:
                change = multi_tf_val - phase_5a_val
                report.append(f"{metric_name:<25} {phase_5a_val:>8.2f}   {multi_tf_val:>14.2f}   {change:>+8.2f}")

        # Trade Analysis
        report.append("\n\n📈 TRADE ANALYSIS\n")
        report.append(f"Total Trades:           {trade_stats['num_trades']}")
        report.append(f"Winning Trades:         {trade_stats['winning_trades']} ({trade_stats['win_rate']:.1f}%)")
        report.append(f"Losing Trades:          {trade_stats['losing_trades']}")
        report.append(f"Profit Factor:          {trade_stats['profit_factor']:.2f}")
        report.append(f"Avg Bars Held:          {trade_stats['avg_bars_held']:.0f}")
        report.append(f"Avg Win:                {trade_stats['avg_win_pct']:+.2f}%")
        report.append(f"Avg Loss:               {trade_stats['avg_loss_pct']:+.2f}%")
        report.append(f"Total Win:              ${trade_stats['total_win']:,.2f}")
        report.append(f"Total Loss:             ${trade_stats['total_loss']:,.2f}")

        # Alignment Analysis
        report.append("\n\n🎯 MULTI-TIMEFRAME ALIGNMENT EFFECTIVENESS\n")
        report.append(f"High Alignment Trades:  {alignment_stats['aligned_trades']} "
                     f"({alignment_stats['alignment_rate']:.1f}%)")
        report.append(f"├─ Win Rate:            {alignment_stats['aligned_win_rate']:.1f}%")
        report.append(f"└─ Avg Profit:          {alignment_stats['aligned_avg_profit_pct']:+.2f}%")
        report.append(f"\nLow Alignment Trades:   {alignment_stats['misaligned_trades']} "
                     f"({100-alignment_stats['alignment_rate']:.1f}%)")
        report.append(f"├─ Win Rate:            {alignment_stats['misaligned_win_rate']:.1f}%")
        report.append(f"└─ Avg Profit:          {alignment_stats['misaligned_avg_profit_pct']:+.2f}%")

        # Signal Combinations
        if signal_stats:
            report.append("\n\n🔄 SIGNAL COMBINATION EFFECTIVENESS\n")
            for combo, stats in sorted(signal_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                report.append(f"\n{combo}:")
                report.append(f"  Count:                  {stats['count']}")
                report.append(f"  Win Rate:               {stats['win_rate']:.1f}%")
                report.append(f"  Avg Profit:             {stats['avg_profit_pct']:+.2f}%")

        # Summary
        report.append("\n\n" + "=" * 100)
        report.append("SUMMARY")
        report.append("=" * 100)

        if multi_tf_metrics['total_return'] > 35:
            verdict = "✅ STRONG SUCCESS - Multi-timeframe approach significantly outperforms Phase 5-A"
        elif multi_tf_metrics['total_return'] > 12:
            verdict = "✅ SUCCESS - Multi-timeframe provides meaningful improvement"
        elif multi_tf_metrics['total_return'] > 2:
            verdict = "⚠️ MARGINAL - Multi-timeframe matches or slightly exceeds Phase 5-A"
        else:
            verdict = "❌ INSUFFICIENT - Need Phase 5-C (Ensemble) or further optimization"

        report.append(f"\n{verdict}")

        if multi_tf_metrics['num_trades'] > 50:
            report.append(f"✅ Sufficient trade frequency (>{50} trades)")
        else:
            report.append(f"⚠️ Limited trade frequency ({multi_tf_metrics['num_trades']} trades)")

        if alignment_stats['aligned_win_rate'] > 60:
            report.append(f"✅ High-alignment trades are profitable ({alignment_stats['aligned_win_rate']:.1f}%)")
        else:
            report.append(f"⚠️ Alignment signal effectiveness needs improvement")

        report.append("\n" + "=" * 100)

        return "\n".join(report)


def analyze_multi_timeframe_results(
    metrics: Dict,
    trades: List[Dict],
    output_path: Optional[Path] = None
) -> str:
    """Generate complete analysis report.

    Args:
        metrics: Backtest metrics
        trades: List of trades
        output_path: Optional path to save report

    Returns:
        str: Complete analysis report
    """
    analyzer = MultiTimeframeStatisticsAnalyzer()
    report = analyzer.generate_comparison_report(metrics, trades)

    if output_path:
        output_path.write_text(report)
        print(f"✓ Report saved to {output_path}")

    return report


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    results_path = Path("/home/tsukuda/works/usdjy-ai-trader/MULTI_TIMEFRAME_RESULTS.json")

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        trades_path = Path("/home/tsukuda/works/usdjy-ai-trader/backtest/multi_timeframe_trades.csv")
        if trades_path.exists():
            trades_df = pd.read_csv(trades_path)
            trades = trades_df.to_dict('records')

            report = analyze_multi_timeframe_results(
                results['metrics'],
                trades,
                output_path=Path("/home/tsukuda/works/usdjy-ai-trader/MULTI_TIMEFRAME_ANALYSIS.md")
            )

            print(report)
