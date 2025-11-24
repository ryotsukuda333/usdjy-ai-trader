"""Verify Adaptive XGBoost Logic WITHOUT Loading XGBoost Model

Tests the signal generation logic without XGBoost overhead.
This proves the adaptive logic works, with XGBoost as placeholder.
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

import warnings
warnings.filterwarnings('ignore')


class AdaptiveLogicVerifier:
    """Verify adaptive signal logic without XGBoost."""

    EXECUTE_THRESHOLD = 0.55

    TECHNICAL_WEIGHTS = {
        'ma_crossover': 0.40,
        'rsi': 0.30,
        'macd': 0.30
    }

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.trades = []

    def get_5m_technical_score(self, df_features: pd.DataFrame) -> tuple:
        """Get technical score from 5m features."""
        if df_features.empty:
            return 0, 0.5

        last_row = df_features.iloc[-1]
        price = last_row.get('Close', 0)
        score = 0.0

        # MA crossover
        ma_cols = [c for c in df_features.columns
                  if c.startswith('ma') and not c.endswith('_slope') and '_' not in c]
        if len(ma_cols) >= 2:
            try:
                ma_periods = sorted([int(c[2:]) for c in ma_cols if c[2:].isdigit()])
                if len(ma_periods) >= 2:
                    short_ma = last_row.get(f'ma{ma_periods[0]}', price)
                    long_ma = last_row.get(f'ma{ma_periods[-1]}', price)
                    if short_ma > long_ma:
                        score += self.TECHNICAL_WEIGHTS['ma_crossover']
                    else:
                        score -= self.TECHNICAL_WEIGHTS['ma_crossover']
            except:
                pass

        # RSI
        if 'rsi14' in df_features.columns:
            rsi = last_row.get('rsi14', 50)
            if rsi < 30:
                score += self.TECHNICAL_WEIGHTS['rsi']
            elif rsi > 70:
                score -= self.TECHNICAL_WEIGHTS['rsi']

        # MACD
        if 'macd' in df_features.columns and 'macd_signal' in df_features.columns:
            macd = last_row.get('macd', 0)
            macd_signal = last_row.get('macd_signal', 0)
            if macd > macd_signal:
                score += self.TECHNICAL_WEIGHTS['macd']
            else:
                score -= self.TECHNICAL_WEIGHTS['macd']

        signal = 1 if score > 0.2 else (-1 if score < -0.2 else 0)
        tech_prob = 0.5 + (score / 2.0)
        tech_prob = max(0.0, min(1.0, tech_prob))

        return signal, tech_prob

    def backtest(self, data_dict: Dict, features_dict: Dict) -> Dict:
        """Run backtest."""

        df_5m = data_dict['5m'].copy()
        df_5m_features = features_dict.get('5m', pd.DataFrame()).copy()

        if len(df_5m) == 0:
            return {}

        print(f"\n{'='*80}")
        print("ADAPTIVE LOGIC VERIFICATION (XGBoost = 0.5 placeholder)")
        print(f"{'='*80}")
        print(f"Bars: {len(df_5m):,}")
        print(f"Processing...")

        position_size = 0
        entry_price = 0.0

        for idx in range(len(df_5m)):
            if (idx + 1) % 30000 == 0:
                print(f"  {idx+1:,}/{len(df_5m):,}...")

            current_price = df_5m.iloc[idx]['Close']

            df_slice = df_5m_features.iloc[:min(idx+1, len(df_5m_features))]
            if len(df_slice) == 0:
                continue

            # Get 5m signal
            tech_signal, tech_score = self.get_5m_technical_score(df_slice)

            if tech_signal == 0:
                continue

            # Adaptive confidence = 50% XGBoost (0.5 placeholder) + 50% Technical
            xgb_prob = 0.5  # PLACEHOLDER
            confidence = 0.50 * xgb_prob + 0.50 * tech_score
            should_execute = confidence >= self.EXECUTE_THRESHOLD

            # Exit
            if position_size > 0:
                should_exit = False
                if tech_signal == 1 and current_price <= entry_price * (1 - 0.005):
                    should_exit = True
                elif tech_signal == -1 and current_price >= entry_price * (1 + 0.005):
                    should_exit = True
                elif tech_signal == 1 and current_price >= entry_price * (1 + 0.01):
                    should_exit = True
                elif tech_signal == -1 and current_price <= entry_price * (1 - 0.01):
                    should_exit = True

                if should_exit:
                    profit = position_size * (current_price - entry_price)
                    self.equity += profit
                    self.trades.append({
                        'entry': entry_price,
                        'exit': current_price,
                        'profit': profit,
                        'profit_pct': (profit / entry_price * 100) if entry_price > 0 else 0
                    })
                    position_size = 0

            # Entry
            if position_size == 0 and should_execute:
                entry_price = current_price
                position_size = max(100, int(self.equity * 0.01 / (current_price * 0.005)))

        print(f"✓ Complete!")

        if not self.trades:
            return {'total_return': 0.0, 'num_trades': 0, 'final_equity': self.equity}

        trades_df = pd.DataFrame(self.trades)
        return {
            'total_return': ((self.equity - self.initial_capital) / self.initial_capital) * 100,
            'num_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['profit'] > 0]),
            'win_rate': len(trades_df[trades_df['profit'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'final_equity': self.equity
        }


def run():
    """Main."""
    print("\n" + "="*80)
    print("ADAPTIVE LOGIC VERIFICATION")
    print("="*80)

    # Fetch
    print("\n[1/3] Fetching...")
    fetcher = MultiTimeframeFetcher()
    data_dict = fetcher.fetch_and_resample(years=2)
    print(f"✓ 5m bars: {len(data_dict['5m']):,}")

    # Engineer
    print("\n[2/3] Engineering...")
    engineer = MultiTimeframeFeatureEngineer()
    features_dict = engineer.engineer_all_timeframes(data_dict)
    print(f"✓ Features ready")

    # Backtest
    print("\n[3/3] Backtesting...")
    backtester = AdaptiveLogicVerifier()
    metrics = backtester.backtest(data_dict, features_dict)

    # Results
    print("\n" + "="*80)
    print(f"Return:      {metrics['total_return']:+.2f}%")
    print(f"Trades:      {metrics['num_trades']}")
    print(f"Win Rate:    {metrics['win_rate']:.1f}%")
    print(f"Equity:      ${metrics['final_equity']:,.0f}")
    print("="*80)

    print("\nComparison with Phase 5-A:")
    print(f"  Return:     {metrics['total_return']:+.2f}% vs +2.00%")
    print(f"  Trades:     {metrics['num_trades']} vs 7")
    print(f"  Win Rate:   {metrics['win_rate']:.1f}% vs 57.1%")

    # Save
    with open(Path(__file__).parent / 'ADAPTIVE_LOGIC_VERIFICATION.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved")

    return metrics


if __name__ == "__main__":
    try:
        metrics = run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
