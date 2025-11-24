"""
Step 12: Grid Search - 36パラメータ組み合わせの最適化実行
実行方法: python3 backtest/run_grid_search.py
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.data_fetcher import fetch_usdjpy_data
from features.feature_engineer import engineer_features


def create_hybrid_strategy_with_params(xgb_threshold, seasonality_weights, signal_thresholds):
    """Create HybridTradingStrategy with custom parameters"""

    weekly_weight, monthly_weight = seasonality_weights
    high_threshold, low_threshold = signal_thresholds

    # Dynamic strategy class creation with custom parameters
    import xgboost as xgb

    class CustomHybridStrategy:
        def __init__(self, xgb_model_path=None, use_seasonality=True):
            self.xgb_model = None
            self.use_seasonality = use_seasonality

            if xgb_model_path and Path(xgb_model_path).exists():
                try:
                    self.xgb_model = xgb.Booster()
                    self.xgb_model.load_model(xgb_model_path)
                except Exception as e:
                    print(f"Warning: Could not load XGBoost model: {e}")
                    self.xgb_model = None

            self._init_seasonality_stats()

        def _init_seasonality_stats(self):
            """Initialize seasonality statistics"""
            self.weekly_stats = {
                0: {'mean': 0.0609, 'std': 0.7176, 'signal': +0.1},
                1: {'mean': 0.0514, 'std': 0.5132, 'signal': +0.1},
                2: {'mean': 0.0035, 'std': 0.6183, 'signal': 0.0},
                3: {'mean': -0.0060, 'std': 0.6691, 'signal': -0.05},
                4: {'mean': -0.0237, 'std': 0.7068, 'signal': -0.1},
            }
            self.monthly_stats = {
                1: {'mean': 0.0077, 'std': 0.5925, 'signal': 0.0},
                2: {'mean': 0.0660, 'std': 0.6498, 'signal': +0.1},
                3: {'mean': -0.0300, 'std': 0.5858, 'signal': -0.05},
                4: {'mean': -0.0200, 'std': 0.7115, 'signal': -0.05},
                5: {'mean': 0.0884, 'std': 0.7488, 'signal': +0.1},
                6: {'mean': 0.1023, 'std': 0.4759, 'signal': +0.2},
                7: {'mean': -0.0685, 'std': 0.6404, 'signal': -0.1},
                8: {'mean': -0.0451, 'std': 0.7380, 'signal': -0.1},
                9: {'mean': 0.0319, 'std': 0.5411, 'signal': +0.05},
                10: {'mean': 0.1545, 'std': 0.5384, 'signal': +0.15},
                11: {'mean': -0.0191, 'std': 0.5899, 'signal': 0.0},
                12: {'mean': -0.0640, 'std': 0.8714, 'signal': -0.2},
            }

        def get_seasonality_score(self, date):
            """Calculate seasonality score"""
            dow = date.dayofweek
            month = date.month

            weekly_signal = self.weekly_stats[dow if dow < 5 else 0]['signal']
            weekly_score = 0.5 + weekly_signal

            monthly_signal = self.monthly_stats[month]['signal']
            monthly_score = 0.5 + monthly_signal

            score = weekly_weight * weekly_score + monthly_weight * monthly_score
            return np.clip(score, 0.0, 1.0)

        def generate_predictions(self, df, feature_cols):
            """Generate hybrid predictions"""
            results = pd.DataFrame(index=df.index)

            if self.xgb_model is not None:
                try:
                    available_cols = [col for col in feature_cols if col in df.columns]
                    X = df[available_cols].fillna(0)
                    dmatrix = xgb.DMatrix(X)
                    xgb_probs = self.xgb_model.predict(dmatrix)
                    results['xgb_prob'] = xgb_probs
                except Exception:
                    results['xgb_prob'] = 0.5
            else:
                results['xgb_prob'] = 0.5

            results['seasonality_score'] = df.index.map(self.get_seasonality_score)

            # Generate hybrid signals
            hybrid_signals = []
            for xgb_prob, season_score in zip(results['xgb_prob'], results['seasonality_score']):
                weighted = xgb_prob * (0.5 + 0.5 * season_score)

                if weighted >= high_threshold:
                    hybrid_signals.append(1)
                elif weighted < low_threshold:
                    hybrid_signals.append(0)
                else:
                    hybrid_signals.append(-1)

            results['hybrid_signal'] = hybrid_signals
            return results

        def backtest(self, df_ohlcv, predictions, initial_capital=100000):
            """Run backtest"""
            trades = []
            equity = initial_capital
            position = None
            entry_price = None

            for i in range(1, len(df_ohlcv)):
                if i >= len(predictions):
                    break

                date = df_ohlcv.index[i]
                price = df_ohlcv['Close'].iloc[i]
                signal = predictions['hybrid_signal'].iloc[i]

                # Close position
                if position is not None and signal != 1:
                    return_pct = (price - entry_price) / entry_price * 100
                    equity *= (1 + return_pct / 100)
                    trades.append({
                        'return_pct': return_pct,
                        'win': 1 if return_pct > 0 else 0,
                    })
                    position = None

                # Open position
                if signal == 1 and position is None:
                    position = date
                    entry_price = price

            # Close final position
            if position is not None:
                final_price = df_ohlcv['Close'].iloc[-1]
                return_pct = (final_price - entry_price) / entry_price * 100
                equity *= (1 + return_pct / 100)
                trades.append({
                    'return_pct': return_pct,
                    'win': 1 if return_pct > 0 else 0,
                })

            num_trades = len(trades)
            total_return = (equity - initial_capital) / initial_capital * 100

            if num_trades > 0:
                winning = sum(1 for t in trades if t['win'])
                win_rate = winning / num_trades * 100
                returns = [t['return_pct'] for t in trades]
                avg_return = np.mean(returns)
                max_return = np.max(returns)
                min_return = np.min(returns)

                # Sharpe ratio
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe = 0.0

                # Max drawdown
                eq = initial_capital
                peak = initial_capital
                max_dd = 0
                for trade in trades:
                    eq *= (1 + trade['return_pct'] / 100)
                    if eq > peak:
                        peak = eq
                    dd = (peak - eq) / peak * 100
                    max_dd = max(max_dd, dd)
            else:
                win_rate = 0
                avg_return = 0
                max_return = 0
                min_return = 0
                sharpe = 0
                max_dd = 0

            return {
                'total_return': total_return,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'max_return': max_return,
                'min_return': min_return,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'final_equity': equity,
            }

    return CustomHybridStrategy


def main():
    project_root = Path(__file__).parent.parent

    print("=" * 80)
    print("STEP 12: GRID SEARCH - パラメータ最適化")
    print("=" * 80)

    # Fetch data
    print("\n[1] データ取得中...")
    df_ohlcv = fetch_usdjpy_data()
    print(f"✓ {len(df_ohlcv)} candles取得")

    # Engineer features
    print("[2] 特徴エンジニアリング中...")
    df_features = engineer_features(df_ohlcv)
    print(f"✓ {len(df_features)}行 × {len(df_features.columns)}列の特徴")

    # Load feature columns
    feature_cols_path = project_root / "model" / "feature_columns.json"
    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [col for col in df_features.columns
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

    # Parameter grid
    xgb_thresholds = [0.45, 0.50, 0.55, 0.60]
    seasonality_weights = [(0.25, 0.75), (0.30, 0.70), (0.35, 0.65)]
    signal_thresholds = [(0.55, 0.45), (0.60, 0.40), (0.65, 0.35)]

    total_combinations = len(xgb_thresholds) * len(seasonality_weights) * len(signal_thresholds)

    print(f"\n[3] Grid Search実行 ({total_combinations}パラメータ組み合わせ)...")
    print("=" * 80)

    xgb_model_path = project_root / "model" / "xgb_model.json"
    results = []

    combination = 0
    start_time = time.time()

    for xgb_t, (w_w, w_m), (t_h, t_l) in product(
        xgb_thresholds, seasonality_weights, signal_thresholds
    ):
        combination += 1

        print(f"[{combination:2d}/{total_combinations}] "
              f"XGB={xgb_t:.2f}, W=({w_w:.2f},{w_m:.2f}), "
              f"S=({t_h:.2f},{t_l:.2f})... ", end="", flush=True)

        try:
            # Create strategy with parameters
            StrategyClass = create_hybrid_strategy_with_params(
                xgb_t, (w_w, w_m), (t_h, t_l)
            )

            strategy = StrategyClass(
                xgb_model_path=str(xgb_model_path) if xgb_model_path.exists() else None,
                use_seasonality=True
            )

            # Generate predictions
            predictions = strategy.generate_predictions(df_features, feature_cols)

            # Run backtest
            metrics = strategy.backtest(df_ohlcv, predictions, initial_capital=100000)

            # Store result
            result = {
                'xgb_threshold': xgb_t,
                'seasonality_weekly': w_w,
                'seasonality_monthly': w_m,
                'signal_high': t_h,
                'signal_low': t_l,
                'total_return': metrics['total_return'],
                'num_trades': metrics['num_trades'],
                'win_rate': metrics['win_rate'],
                'avg_return': metrics['avg_return'],
                'max_return': metrics['max_return'],
                'min_return': metrics['min_return'],
                'sharpe': metrics['sharpe'],
                'max_dd': metrics['max_dd'],
                'final_equity': metrics['final_equity'],
            }
            results.append(result)

            print(f"✓ {metrics['num_trades']:2d}取引 | "
                  f"+{metrics['total_return']:6.2f}% | "
                  f"WR {metrics['win_rate']:5.1f}%")

        except Exception as e:
            print(f"✗ エラー: {str(e)[:50]}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"✓ Grid Search完了 ({elapsed:.1f}秒)")
    print("=" * 80)

    # Sort by total return
    results.sort(key=lambda x: x['total_return'], reverse=True)

    # Print top 10
    print(f"\nTOP 10パラメータ組み合わせ:")
    print("-" * 80)

    for idx, result in enumerate(results[:10], 1):
        print(f"\n#{idx}:")
        print(f"  XGBoost: {result['xgb_threshold']:.2f}")
        print(f"  Seasonality: ({result['seasonality_weekly']:.2f}, {result['seasonality_monthly']:.2f})")
        print(f"  Signal: ({result['signal_high']:.2f}, {result['signal_low']:.2f})")
        print(f"  Return: +{result['total_return']:.2f}% | "
              f"Trades: {result['num_trades']} | "
              f"WR: {result['win_rate']:.1f}% | "
              f"Sharpe: {result['sharpe']:.2f}")

    # Save results
    df_results = pd.DataFrame(results)
    output_path = project_root / "backtest" / "grid_search_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ 結果保存: {output_path}")

    # Phase 6判定
    print("\n" + "=" * 80)
    print("PHASE 6: 採用判定")
    print("=" * 80)

    best = results[0]

    print(f"\n最適パラメータ:")
    print(f"  XGBoost: {best['xgb_threshold']:.2f}")
    print(f"  Seasonality: ({best['seasonality_weekly']:.2f}, {best['seasonality_monthly']:.2f})")
    print(f"  Signal: ({best['signal_high']:.2f}, {best['signal_low']:.2f})")

    print(f"\nパフォーマンス:")
    print(f"  総リターン: +{best['total_return']:.2f}%")
    print(f"  取引数: {best['num_trades']}")
    print(f"  勝率: {best['win_rate']:.1f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  最大DD: -{best['max_dd']:.2f}%")

    # Check criteria
    print(f"\n採用判定基準:")
    print(f"  総リターン > +65%: {best['total_return']:+.2f}% "
          f"{'✅' if best['total_return'] > 65 else '❌'}")
    print(f"  最大DD ≤ -1.5%: -{best['max_dd']:.2f}% "
          f"{'✅' if best['max_dd'] <= 1.5 else '❌'}")

    if best['total_return'] > 65 and best['max_dd'] <= 1.5:
        print(f"\n🎉 採用判定: ✅ **ADOPTED**")
        print(f"Step 12ハイブリッド戦略を本番採用します")
        recommendation = "ADOPT"
    elif best['total_return'] > 50:
        print(f"\n⚠️ 採用判定: ⚠️ **CONDITIONAL**")
        print(f"改善傾向が見られます。段階的な採用を検討してください")
        recommendation = "CONDITIONAL"
    else:
        print(f"\n❌ 採用判定: ❌ **REJECT**")
        print(f"さらなる最適化が必要です (Phase 5-B/C の追加実装をご検討ください)")
        recommendation = "REJECT"

    # Save recommendation
    summary = {
        'recommendation': recommendation,
        'best_params': {
            'xgb_threshold': best['xgb_threshold'],
            'seasonality_weekly': best['seasonality_weekly'],
            'seasonality_monthly': best['seasonality_monthly'],
            'signal_high': best['signal_high'],
            'signal_low': best['signal_low'],
        },
        'performance': {
            'total_return': best['total_return'],
            'num_trades': best['num_trades'],
            'win_rate': best['win_rate'],
            'sharpe': best['sharpe'],
            'max_dd': best['max_dd'],
        },
        'execution_time': elapsed,
        'total_combinations': total_combinations,
    }

    summary_path = project_root / "STEP12_GRID_SEARCH_SUMMARY.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ サマリー保存: {summary_path}")

    return recommendation, best


if __name__ == "__main__":
    recommendation, best_result = main()
