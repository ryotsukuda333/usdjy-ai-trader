"""
Session-based Trading Analysis for USDJPY

Analyzes trading performance and volatility by trading session:
- Tokyo Session: 09:00-15:00 JST (1% volatility, morning trend)
- London Session: 16:00-23:00 JST (1.5% volatility, transition period)
- New York Session: 00:00-08:00 JST (2% volatility, active trading)

Tasks: 8.1, 8.2, 8.3
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class SessionAnalyzer:
    """Analyzes trading data by time-based sessions."""

    # Session definitions (JST hours)
    SESSIONS = {
        'tokyo': {'start': 9, 'end': 15, 'name': 'Tokyo (09:00-15:00)'},
        'london': {'start': 16, 'end': 23, 'name': 'London (16:00-23:00)'},
        'newyork': {'start': 0, 'end': 8, 'name': 'New York (00:00-08:00)'}
    }

    def __init__(self):
        """Initialize session analyzer."""
        self.session_stats = {}
        self.session_trades = {}

    def assign_session(self, hour: int) -> str:
        """
        Assign a trade to a session based on hour (JST).

        Args:
            hour: Hour in JST (0-23)

        Returns:
            Session name: 'tokyo', 'london', or 'newyork'
        """
        if 9 <= hour <= 15:
            return 'tokyo'
        elif 16 <= hour <= 23:
            return 'london'
        else:  # 0-8
            return 'newyork'

    def analyze_trades_by_session(self,
                                 trades_df: pd.DataFrame) -> Dict:
        """
        Analyze trading statistics by session.

        Args:
            trades_df: DataFrame with trades (must have datetime index or entry_date)

        Returns:
            Dictionary with session statistics
        """
        # Ensure we have a datetime index
        if isinstance(trades_df.index, pd.DatetimeIndex):
            entry_times = trades_df.index
        else:
            entry_times = pd.to_datetime(trades_df['entry_date'])

        # Assign sessions
        sessions = [self.assign_session(ts.hour) for ts in entry_times]
        trades_df = trades_df.copy()
        trades_df['session'] = sessions

        # Calculate statistics by session
        stats = {}
        for session_key in self.SESSIONS.keys():
            session_trades = trades_df[trades_df['session'] == session_key]

            if len(session_trades) == 0:
                stats[session_key] = {
                    'name': self.SESSIONS[session_key]['name'],
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'total_return': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0,
                    'pnl_total': 0.0
                }
            else:
                wins = (session_trades['win_loss'] == 1).sum()
                losses = (session_trades['win_loss'] == 0).sum()
                total = len(session_trades)

                winning_trades = session_trades[session_trades['win_loss'] == 1]['return_percent']
                losing_trades = session_trades[session_trades['win_loss'] == 0]['return_percent']

                stats[session_key] = {
                    'name': self.SESSIONS[session_key]['name'],
                    'trades': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': (wins / total * 100) if total > 0 else 0.0,
                    'avg_return': session_trades['return_percent'].mean(),
                    'total_return': session_trades['return_percent'].sum(),
                    'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0.0,
                    'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0.0,
                    'best_trade': session_trades['return_percent'].max(),
                    'worst_trade': session_trades['return_percent'].min(),
                    'pnl_total': session_trades['pnl_usd'].sum() if 'pnl_usd' in session_trades.columns else 0.0
                }

        self.session_stats = stats
        return stats

    def analyze_volatility_by_session(self,
                                     df_features: pd.DataFrame) -> Dict:
        """
        Analyze volatility statistics by session.

        Args:
            df_features: Features DataFrame with volatility data

        Returns:
            Dictionary with volatility statistics by session
        """
        if 'volatility_20' not in df_features.columns:
            raise ValueError("DataFrame must contain 'volatility_20' column")

        # Get hour from index (assumes datetime index)
        if isinstance(df_features.index, pd.DatetimeIndex):
            hours = df_features.index.hour
        else:
            return {}

        # Assign sessions
        sessions = [self.assign_session(h) for h in hours]

        vol_df = df_features.copy()
        vol_df['session'] = sessions

        # Calculate volatility stats by session
        vol_stats = {}
        for session_key in self.SESSIONS.keys():
            session_vol = vol_df[vol_df['session'] == session_key]['volatility_20']

            if len(session_vol) == 0:
                vol_stats[session_key] = {
                    'name': self.SESSIONS[session_key]['name'],
                    'candles': 0,
                    'avg_volatility': 0.0,
                    'median_volatility': 0.0,
                    'min_volatility': 0.0,
                    'max_volatility': 0.0,
                    'volatility_std': 0.0
                }
            else:
                vol_stats[session_key] = {
                    'name': self.SESSIONS[session_key]['name'],
                    'candles': len(session_vol),
                    'avg_volatility': session_vol.mean(),
                    'median_volatility': session_vol.median(),
                    'min_volatility': session_vol.min(),
                    'max_volatility': session_vol.max(),
                    'volatility_std': session_vol.std()
                }

        return vol_stats

    def get_session_summary_table(self) -> pd.DataFrame:
        """
        Get summary statistics as a formatted DataFrame.

        Returns:
            DataFrame with session statistics
        """
        if not self.session_stats:
            return pd.DataFrame()

        rows = []
        for session_key in ['tokyo', 'london', 'newyork']:
            if session_key in self.session_stats:
                stat = self.session_stats[session_key]
                rows.append({
                    'Session': stat['name'],
                    'Trades': stat['trades'],
                    'Wins': stat['wins'],
                    'Losses': stat['losses'],
                    'Win Rate %': f"{stat['win_rate']:.2f}%",
                    'Avg Return %': f"{stat['avg_return']:+.3f}%",
                    'Total Return %': f"{stat['total_return']:+.2f}%",
                    'Avg Win %': f"{stat['avg_win']:+.3f}%",
                    'Avg Loss %': f"{stat['avg_loss']:+.3f}%",
                    'Best Trade %': f"{stat['best_trade']:+.3f}%",
                    'Worst Trade %': f"{stat['worst_trade']:+.3f}%",
                    'Total PnL $': f"{stat['pnl_total']:+,.0f}"
                })

        return pd.DataFrame(rows)

    def recommend_risk_by_session(self) -> Dict[str, float]:
        """
        Recommend risk percentage by session based on win rate and volatility.

        Returns:
            Dictionary mapping session to recommended risk %
        """
        recommendations = {}

        for session_key in self.SESSIONS.keys():
            if session_key not in self.session_stats:
                recommendations[session_key] = 1.0
                continue

            stat = self.session_stats[session_key]
            trades = stat['trades']

            if trades == 0:
                recommendations[session_key] = 1.0
                continue

            # Risk adjustment based on win rate
            win_rate = stat['win_rate'] / 100

            # Base risk: 5% (optimal from Step 7)
            base_risk = 5.0

            # Adjust by win rate
            # High win rate (>60%): +0.5% risk
            # Medium win rate (55-60%): base risk
            # Low win rate (<55%): -1% risk
            if win_rate > 0.60:
                risk_pct = base_risk + 0.5
            elif win_rate >= 0.55:
                risk_pct = base_risk
            else:
                risk_pct = max(1.0, base_risk - 1.0)

            recommendations[session_key] = risk_pct

        return recommendations


def create_session_analyzer() -> SessionAnalyzer:
    """
    Factory function to create SessionAnalyzer.

    Returns:
        Configured SessionAnalyzer instance
    """
    return SessionAnalyzer()
