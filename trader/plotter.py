"""Plotter module for visualization of trading results.

Generates equity curve, drawdown chart, and performance metrics visualization.

Task: 7.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from utils.errors import VisualizationError


def plot_backtest_results(trades: pd.DataFrame) -> str:
    """Generate visualization of backtest results.

    Creates comprehensive equity curve plot with performance metrics.
    Requirement 6: Visualization of backtest results with equity curve.

    Args:
        trades: DataFrame with trade records
               Required columns: entry_date, entry_price, exit_date, exit_price,
                                return_percent, win_loss, exit_reason

    Returns:
        str: Path to generated PNG file

    Raises:
        VisualizationError: If trades data invalid or visualization fails
    """
    # Validate input
    if trades is None or trades.empty:
        raise VisualizationError(
            error_code="INVALID_INPUT",
            user_message="Trades DataFrame is empty or None",
            technical_message="trades DataFrame must contain trade records"
        )

    required_columns = ['entry_date', 'exit_date', 'return_percent', 'win_loss', 'exit_reason']
    missing_columns = [col for col in required_columns if col not in trades.columns]
    if missing_columns:
        raise VisualizationError(
            error_code="MISSING_COLUMNS",
            user_message=f"Missing required columns: {missing_columns}",
            technical_message=f"Expected: {required_columns}, Got: {list(trades.columns)}"
        )

    try:
        # Calculate metrics
        total_trades = len(trades)
        wins = (trades['win_loss'] == 1).sum()
        losses = (trades['win_loss'] == 0).sum()
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_return = trades['return_percent'].sum()
        avg_return = trades['return_percent'].mean()

        # Calculate cumulative returns (equity curve)
        trades_sorted = trades.sort_values('exit_date').copy()
        trades_sorted['cumulative_return'] = (1 + trades_sorted['return_percent'] / 100).cumprod() - 1
        trades_sorted['cumulative_pnl'] = trades_sorted['cumulative_return'] * 100

        # Calculate drawdown
        cumulative_max = trades_sorted['cumulative_pnl'].cummax()
        drawdown = trades_sorted['cumulative_pnl'] - cumulative_max

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Backtest Results - Equity Curve & Performance', fontsize=16, fontweight='bold')

        # Plot 1: Equity Curve
        ax1 = axes[0]
        ax1.plot(trades_sorted['exit_date'], trades_sorted['cumulative_pnl'], 'b-', linewidth=2, label='Equity Curve')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Drawdown
        ax2 = axes[1]
        colors = ['red' if x < 0 else 'green' for x in drawdown]
        ax2.bar(trades_sorted['exit_date'], drawdown, color=colors, alpha=0.6, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.set_title('Drawdown Chart', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 3: Performance Metrics
        ax3 = axes[2]
        ax3.axis('off')

        metrics_text = f"""
BACKTEST PERFORMANCE METRICS
{'='*50}

Total Trades:        {total_trades}
Winning Trades:      {wins}
Losing Trades:       {losses}
Win Rate:            {win_rate:.2f}%

Total Return:        {total_return:+.2f}%
Average Return/Trade: {avg_return:+.3f}%

Max Drawdown:        {drawdown.min():.2f}%
Final Equity:        {trades_sorted['cumulative_pnl'].iloc[-1] if len(trades_sorted) > 0 else 0:+.2f}%
        """

        ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # Save to PNG
        trader_dir = Path(__file__).parent
        trader_dir.mkdir(exist_ok=True)
        plot_path = trader_dir / 'backtest_equity_curve.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✓ Backtest visualization saved to {plot_path}")
        return str(plot_path)

    except VisualizationError:
        raise
    except Exception as e:
        raise VisualizationError(
            error_code="PLOT_FAILED",
            user_message="Failed to generate backtest visualization",
            technical_message=f"Error creating plot: {str(e)}"
        )
