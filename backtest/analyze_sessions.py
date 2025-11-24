"""
Session Analysis Script for USDJPY Trading

Analyzes backtest results and features by trading session.
Generates comprehensive session statistics and recommendations.

Usage: python backtest/analyze_sessions.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from trader.session_analyzer import create_session_analyzer


def main():
    """Analyze trading data by session."""

    print("\n" + "="*70)
    print("SESSION-BASED TRADING ANALYSIS")
    print("="*70)

    # Load backtest results
    print("\n[Step 8.1] Loading backtest results...")
    results_path = Path(__file__).parent / 'backtest_results_fixed_risk.csv'

    if not results_path.exists():
        print(f"⚠️ Results file not found: {results_path}")
        print("Please run main.py first to generate backtest results.")
        return 1

    trades_df = pd.read_csv(results_path)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df.set_index('entry_date', inplace=True)

    print(f"✓ Loaded {len(trades_df)} trades")

    # Create analyzer
    analyzer = create_session_analyzer()

    # Analyze trades by session
    print("\n[Step 8.2] Analyzing trades by session...")
    session_stats = analyzer.analyze_trades_by_session(trades_df)

    # Display session summary
    print("\n📊 SESSION STATISTICS:")
    print("-" * 120)

    summary_df = analyzer.get_session_summary_table()
    print(summary_df.to_string(index=False))

    print("-" * 120)

    # Calculate totals
    total_trades = sum(stat['trades'] for stat in session_stats.values())
    total_wins = sum(stat['wins'] for stat in session_stats.values())
    total_returns = sum(stat['total_return'] for stat in session_stats.values())

    print(f"\nTOTAL: {total_trades} trades | {total_wins} wins | {total_returns:+.2f}%")

    # Detailed analysis for each session
    print("\n[Step 8.3] Detailed session analysis:")
    print("="*70)

    for session_key in ['tokyo', 'london', 'newyork']:
        stat = session_stats[session_key]

        print(f"\n🕐 {stat['name']}")
        print("-" * 70)

        if stat['trades'] == 0:
            print("  ⚠️  No trades in this session")
            continue

        print(f"  Total Trades:        {stat['trades']}")
        print(f"  Wins/Losses:         {stat['wins']}/{stat['losses']}")
        print(f"  Win Rate:            {stat['win_rate']:.2f}%")
        print(f"  Average Return:      {stat['avg_return']:+.3f}%")
        print(f"  Total Return:        {stat['total_return']:+.2f}%")
        print(f"  Average Win:         {stat['avg_win']:+.3f}%")
        print(f"  Average Loss:        {stat['avg_loss']:+.3f}%")
        print(f"  Best/Worst Trade:    {stat['best_trade']:+.3f}% / {stat['worst_trade']:+.3f}%")
        print(f"  Total PnL:           ${stat['pnl_total']:+,.0f}")

    # Generate risk recommendations
    print("\n" + "="*70)
    print("[Step 8.4] RISK RECOMMENDATIONS BY SESSION:")
    print("="*70)

    risk_recommendations = analyzer.recommend_risk_by_session()

    print("\nBased on win rate analysis:")
    for session_key in ['tokyo', 'london', 'newyork']:
        session_name = analyzer.SESSIONS[session_key]['name']
        win_rate = session_stats[session_key]['win_rate'] / 100
        recommended_risk = risk_recommendations[session_key]

        print(f"\n  {session_name}")
        print(f"    Current Win Rate:      {win_rate*100:.2f}%")
        print(f"    Recommended Risk:      {recommended_risk:.1f}%")
        print(f"    Reasoning:             ", end="")

        if win_rate > 0.60:
            print("High win rate - slightly increase risk")
        elif win_rate >= 0.55:
            print("Solid win rate - maintain standard risk")
        else:
            print("Lower win rate - reduce risk")

    # Analysis by session characteristics
    print("\n" + "="*70)
    print("[Step 8.5] SESSION CHARACTERISTICS:")
    print("="*70)

    print("""
Tokyo Session (09:00-15:00 JST):
  - Opening hours, relatively stable
  - Lower volatility, trending market
  - Risk Level: CONSERVATIVE (1-2%)
  - Best For: Trend-following strategies

London Session (16:00-23:00 JST):
  - Overlap period, transition market
  - Medium volatility, mixed trading
  - Risk Level: MODERATE (3-4%)
  - Best For: Range-bound trading

New York Session (00:00-08:00 JST):
  - Active hours, high volatility
  - High volatility, strong trends
  - Risk Level: AGGRESSIVE (5-6%)
  - Best For: Breakout strategies
    """)

    # Summary recommendation
    print("=" * 70)
    print("[SUMMARY] NEXT STEPS:")
    print("=" * 70)
    print("""
1. Implement session-based risk % in position sizing
   - Use recommended risk % based on entry time

2. Analyze win rate consistency across sessions
   - Determine which sessions are most profitable

3. Consider session-specific entry signals
   - Adjust model features by time of day

4. Test session-aware trading:
   - Only trade during high-probability sessions
   - Disable trading during low-probability hours
    """)

    print("\n✓ Session analysis complete")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
