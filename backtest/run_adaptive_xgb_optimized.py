"""Optimized Adaptive XGBoost Backtest with XGBoost Prediction Caching

Optimizes by:
1. Pre-computing all 1D XGBoost predictions at startup (batch predict)
2. Caching results by date
3. Reusing same probability for all 5m bars in same day
4. Expected speedup: 285x (149,185 calls → 519 calls)
5. Expected runtime: 30-40 seconds (vs 120+ seconds)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import json
from datetime import datetime
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.multi_timeframe_fetcher import MultiTimeframeFetcher
from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer

import warnings
warnings.filterwarnings('ignore')


class OptimizedAdaptiveXGBBacktester:
    """Optimized backtest with pre-computed XGBoost cache."""

    EXECUTE_THRESHOLD = 0.55
    TECHNICAL_WEIGHTS = {
        'ma_crossover': 0.40,
        'rsi': 0.30,
        'macd': 0.30
    }

    def __init__(self, initial_capital: float = 100000.0,
                 risk_per_trade: float = 0.01,
                 take_profit_pct: float = 0.01,
                 stop_loss_pct: float = 0.005):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

        self.equity = initial_capital
        self.trades = []
        self.xgb_cache = {}  # Cache: date_str -> probability

        # Load XGBoost model
        project_root = Path(__file__).parent.parent
        model_path = project_root / "model" / "xgb_model.json"
        feature_cols_path = project_root / "model" / "feature_columns.json"

        self.model = self._load_model(model_path)
        self.feature_columns = self._load_feature_columns(feature_cols_path)

    def _load_model(self, model_path: Path):
        """Load XGBoost model."""
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}, using 0.5 placeholder")
            return None
        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            return booster
        except Exception as e:
            print(f"⚠️  Error loading model: {e}, using 0.5 placeholder")
            return None

    def _load_feature_columns(self, columns_path: Path):
        """Load feature column names."""
        if not columns_path.exists():
            print(f"⚠️  Feature columns file not found: {columns_path}")
            return None
        try:
            with open(columns_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading feature columns: {e}")
            return None

    def _compute_xgb_cache(self, df_1d_features: pd.DataFrame) -> Dict:
        """Pre-compute all XGBoost predictions for 1D features (batch mode).

        Returns: {date_str: probability}
        """
        cache = {}

        if self.model is None or self.feature_columns is None:
            print(f"⚠️  XGBoost unavailable, using 0.5 for all dates")
            for idx in range(len(df_1d_features)):
                date_str = pd.Timestamp(df_1d_features.index[idx]).strftime('%Y-%m-%d')
                cache[date_str] = 0.5
            return cache

        try:
            # Align features
            features_aligned = df_1d_features[self.feature_columns].copy()
            features_aligned = features_aligned.fillna(0.0)

            # Batch predict
            dmatrix = xgb.DMatrix(features_aligned)
            predictions = self.model.predict(dmatrix)

            # Cache by date
            for idx in range(len(predictions)):
                date_str = pd.Timestamp(df_1d_features.index[idx]).strftime('%Y-%m-%d')
                cache[date_str] = float(predictions[idx])

            print(f"✓ Pre-computed XGBoost cache: {len(cache)} unique dates")
            return cache

        except Exception as e:
            print(f"⚠️  XGBoost batch predict failed: {e}")
            for idx in range(len(df_1d_features)):
                date_str = pd.Timestamp(df_1d_features.index[idx]).strftime('%Y-%m-%d')
                cache[date_str] = 0.5
            return cache

    def _get_xgb_probability(self, timestamp) -> float:
        """Get cached XGBoost probability for given timestamp."""
        date_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d')
        return self.xgb_cache.get(date_str, 0.5)

    def _get_5m_technical_score(self, df_features: pd.DataFrame) -> tuple:
        """Get technical signal and score from 5m features."""
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
        """Run optimized backtest with XGBoost caching."""

        df_5m = data_dict['5m'].copy()
        df_5m_features = features_dict.get('5m', pd.DataFrame()).copy()
        df_1d_features = features_dict.get('1d', pd.DataFrame()).copy()

        if len(df_5m) == 0 or len(df_5m_features) == 0:
            return {}

        print(f"\n{'='*80}")
        print("OPTIMIZED ADAPTIVE XGBOOST BACKTEST (Pre-computed XGBoost Cache)")
        print(f"{'='*80}")
        print(f"Bars to process: {len(df_5m):,}")

        # Pre-compute XGBoost cache
        print(f"\n[1/2] Pre-computing XGBoost predictions...")
        start = datetime.now()
        self.xgb_cache = self._compute_xgb_cache(df_1d_features)
        elapsed = (datetime.now() - start).total_seconds()
        print(f"✓ XGBoost cache computed in {elapsed:.2f}s")

        # Run backtest with cached predictions
        print(f"\n[2/2] Running backtest...")
        start = datetime.now()

        position_size = 0
        entry_price = 0.0
        entry_bar = 0

        for idx in range(len(df_5m)):
            if (idx + 1) % 30000 == 0:
                print(f"  {idx+1:,}/{len(df_5m):,} bars...")

            current_timestamp = df_5m.index[idx]
            current_price = df_5m.iloc[idx]['Close']

            # Get features
            df_5m_features_slice = df_5m_features.iloc[:min(idx+1, len(df_5m_features))]
            if len(df_5m_features_slice) == 0:
                continue

            # Get signals
            tech_signal, tech_score = self._get_5m_technical_score(df_5m_features_slice)
            xgb_prob = self._get_xgb_probability(current_timestamp)

            # Adaptive confidence
            confidence = 0.50 * xgb_prob + 0.50 * tech_score
            should_execute = confidence >= self.EXECUTE_THRESHOLD and tech_signal != 0

            # Exit logic
            if position_size > 0:
                should_exit = False

                if tech_signal == 1:
                    # Long position
                    if current_price <= entry_price * (1 - self.stop_loss_pct):
                        should_exit = True
                    elif current_price >= entry_price * (1 + self.take_profit_pct):
                        should_exit = True
                elif tech_signal == -1:
                    # Short position
                    if current_price >= entry_price * (1 + self.stop_loss_pct):
                        should_exit = True
                    elif current_price <= entry_price * (1 - self.take_profit_pct):
                        should_exit = True

                if should_exit:
                    profit = position_size * (current_price - entry_price)
                    self.equity += profit
                    self.trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit': profit,
                        'profit_pct': (profit / entry_price * 100) if entry_price > 0 else 0
                    })
                    position_size = 0

            # Entry logic
            if position_size == 0 and should_execute:
                entry_price = current_price
                position_size = max(100, int(self.equity * self.risk_per_trade / (current_price * self.stop_loss_pct)))
                entry_bar = idx

        elapsed = (datetime.now() - start).total_seconds()
        print(f"✓ Backtest complete in {elapsed:.2f}s!")

        if not self.trades:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'final_equity': self.equity
            }

        trades_df = pd.DataFrame(self.trades)
        num_trades = len(trades_df)
        winning = len(trades_df[trades_df['profit'] > 0])

        return {
            'total_return': ((self.equity - self.initial_capital) / self.initial_capital) * 100,
            'num_trades': num_trades,
            'winning_trades': winning,
            'losing_trades': num_trades - winning,
            'win_rate': winning / num_trades * 100 if num_trades > 0 else 0,
            'final_equity': self.equity
        }


def run():
    """Main execution."""
    print("\n" + "="*80)
    print("OPTIMIZED ADAPTIVE XGBOOST BACKTEST")
    print("="*80)

    # Fetch
    print("\n[0/3] Fetching data...")
    start = datetime.now()
    fetcher = MultiTimeframeFetcher()
    data_dict = fetcher.fetch_and_resample(years=2)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"✓ Data fetched in {elapsed:.1f}s")

    # Engineer
    print("\n[1/3] Engineering features...")
    start = datetime.now()
    engineer = MultiTimeframeFeatureEngineer()
    features_dict = engineer.engineer_all_timeframes(data_dict)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"✓ Features engineered in {elapsed:.1f}s")

    # Backtest
    print("\n[2/3] Running optimized backtest...")
    start_total = datetime.now()
    backtester = OptimizedAdaptiveXGBBacktester()
    metrics = backtester.backtest(data_dict, features_dict)
    elapsed_total = (datetime.now() - start_total).total_seconds()

    # Results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Return:         {metrics['total_return']:+.2f}%")
    print(f"Trades:         {metrics['num_trades']}")
    print(f"Win Rate:       {metrics['win_rate']:.1f}%")
    print(f"Final Equity:   ${metrics['final_equity']:,.0f}")
    print(f"Total Runtime:  {elapsed_total:.1f}s")
    print("="*80)

    # Comparison
    print("\nVS PHASE 5-A")
    print("-"*80)
    print(f"{'Metric':<25} {'Phase 5-A':<20} {'Adaptive XGB':<20}")
    print("-"*80)
    print(f"{'Return':<25} {2.00:>8.2f}% {metrics['total_return']:>14.2f}%")
    print(f"{'Trades':<25} {7:>8} {metrics['num_trades']:>14}")
    print(f"{'Win Rate':<25} {57.1:>8.1f}% {metrics['win_rate']:>14.1f}%")
    print("="*80)

    # Save
    output_dir = Path(__file__).parent
    with open(output_dir / 'ADAPTIVE_XGB_OPTIMIZED_RESULTS.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': elapsed_total
        }, f, indent=2)
    print(f"\n✓ Results saved to ADAPTIVE_XGB_OPTIMIZED_RESULTS.json")

    return metrics


if __name__ == "__main__":
    try:
        metrics = run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
