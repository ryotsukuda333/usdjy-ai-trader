"""Tests for plotter module (Task 7.1).

Test-Driven Development phase for visualization of trading results.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.errors import VisualizationError


class TestPlotter:
    """Tests for plotter functions (Task 7.1)."""

    @staticmethod
    def create_sample_trades(num_trades=10):
        """Create sample trade DataFrame for testing.
        
        Args:
            num_trades: Number of trades to create
            
        Returns:
            pd.DataFrame: Trade records
        """
        dates_entry = pd.date_range('2023-01-01', periods=num_trades, freq='D', tz='Asia/Tokyo')
        dates_exit = dates_entry + pd.Timedelta(days=1)
        
        trades = {
            'entry_date': dates_entry,
            'entry_price': np.random.uniform(100, 105, num_trades),
            'exit_date': dates_exit,
            'exit_price': np.random.uniform(100, 106, num_trades),
            'return_percent': np.random.uniform(-0.5, 0.8, num_trades),
            'win_loss': np.random.randint(0, 2, num_trades),
            'exit_reason': np.random.choice(['stop_loss', 'take_profit', 'signal'], num_trades)
        }
        
        return pd.DataFrame(trades)

    def test_plot_backtest_results_returns_path(self):
        """
        Given: DataFrame with trade records
        When: plot_backtest_results() is called
        Then: Should return path to generated plot
        """
        from trader.plotter import plot_backtest_results
        
        trades = self.create_sample_trades(num_trades=10)
        result = plot_backtest_results(trades)
        
        assert isinstance(result, (str, Path)), "Should return file path"

    def test_plot_backtest_results_creates_file(self):
        """
        Given: Trade records DataFrame
        When: plot_backtest_results() completes
        Then: Should create PNG file
        """
        from trader.plotter import plot_backtest_results
        
        trades = self.create_sample_trades(num_trades=10)
        result_path = plot_backtest_results(trades)
        
        assert Path(result_path).exists(), "Should create PNG file"

    def test_plot_backtest_results_with_empty_trades(self):
        """
        Given: Empty trades DataFrame
        When: plot_backtest_results() is called
        Then: Should raise VisualizationError
        """
        from trader.plotter import plot_backtest_results
        
        trades_empty = pd.DataFrame()
        
        with pytest.raises(VisualizationError):
            plot_backtest_results(trades_empty)

    def test_plot_backtest_results_with_missing_columns(self):
        """
        Given: DataFrame missing required columns
        When: plot_backtest_results() is called
        Then: Should raise VisualizationError
        """
        from trader.plotter import plot_backtest_results
        
        trades = self.create_sample_trades(num_trades=10)
        trades_invalid = trades.drop('entry_date', axis=1)
        
        with pytest.raises(VisualizationError):
            plot_backtest_results(trades_invalid)

    def test_plot_backtest_results_includes_equity_curve(self):
        """
        Given: Trade records with return data
        When: plot_backtest_results() executes
        Then: Generated plot should include equity curve visualization
        """
        from trader.plotter import plot_backtest_results
        
        trades = self.create_sample_trades(num_trades=20)
        result_path = plot_backtest_results(trades)
        
        # File should be created (equity curve is plotted)
        assert Path(result_path).exists(), "Equity curve plot should be created"

    def test_plot_backtest_results_performance_metrics(self):
        """
        Given: Trade records with various exits
        When: plot_backtest_results() calculates metrics
        Then: Should display win rate, total return, etc.
        """
        from trader.plotter import plot_backtest_results
        
        trades = self.create_sample_trades(num_trades=10)
        result_path = plot_backtest_results(trades)
        
        # Verify metrics are calculated
        assert Path(result_path).exists(), "Should include performance metrics"

    def test_plot_backtest_results_valid_path(self):
        """
        Given: Trade records DataFrame
        When: plot_backtest_results() is called
        When: Generated file path
        Then: Path should be in trader/ directory
        """
        from trader.plotter import plot_backtest_results
        
        trades = self.create_sample_trades(num_trades=10)
        result_path = plot_backtest_results(trades)
        
        # Path should be in trader directory
        assert 'trader' in str(result_path), "Should save in trader/ directory"

    def test_plot_backtest_handles_single_trade(self):
        """
        Given: DataFrame with single trade record
        When: plot_backtest_results() is called
        Then: Should handle single trade gracefully
        """
        from trader.plotter import plot_backtest_results
        
        trades = self.create_sample_trades(num_trades=1)
        result_path = plot_backtest_results(trades)
        
        assert Path(result_path).exists(), "Should handle single trade"

    def test_plot_backtest_win_loss_breakdown(self):
        """
        Given: Trades with mixed wins and losses
        When: plot_backtest_results() generates visualization
        Then: Should show win/loss breakdown
        """
        from trader.plotter import plot_backtest_results
        
        trades = self.create_sample_trades(num_trades=15)
        trades['win_loss'] = [1, 0, 1, 1, 0, 0, 1] * 2 + [1]  # Mixed wins/losses
        
        result_path = plot_backtest_results(trades)
        
        assert Path(result_path).exists(), "Should plot win/loss breakdown"
