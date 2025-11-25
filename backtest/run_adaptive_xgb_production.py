#!/usr/bin/env python3
"""
Step 14: 適応的XGBoost統合 - 本番実装

マルチタイムフレーム信号生成 + XGBoost統合 + 全期間バックテスト
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.cwd()))

from features.multi_timeframe_fetcher import MultiTimeframeFetcher
from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer


class AdaptiveXGBBacktester:
    """適応的XGBoost統合バックテスター"""

    def __init__(self, xgb_model_path: str = "model/xgb_model.json"):
        """初期化"""
        self.xgb_model = None
        self.xgb_feature_cols = None

        # XGBoost モデル読み込み
        if Path(xgb_model_path).exists():
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_model_path)
                print(f"✓ Loaded XGBoost model from {xgb_model_path}")

                # Feature columns from training
                self.xgb_feature_cols = self._get_training_features()
            except Exception as e:
                print(f"⚠ Could not load XGBoost model: {e}")
                self.xgb_model = None

    def _get_training_features(self):
        """XGBoost訓練時の特徴量列を取得"""
        # Step 12/14で使用された40個の特徴量
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_returns', 'volatility_20',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_diff',
            'bb_upper', 'bb_lower', 'bb_middle',
            'atr_14', 'adx_14',
            'hour', 'day_of_week', 'is_london', 'is_newyork', 'is_tokyo',
            'month', 'quarter', 'day_of_month',
            'price_range', 'hl_ratio', 'co_ratio',
            'volume_sma', 'volume_ratio',
            'trend_1h', 'trend_4h', 'trend_1d',
            'momentum_5', 'momentum_10'
        ]
        return features

    def _get_xgb_probability(self, df_1d: pd.DataFrame, idx: int) -> float:
        """1D XGBoostから確率を取得 (バッチ最適化版)"""
        try:
            if self.xgb_model is None or idx < 0 or idx >= len(df_1d):
                return 0.5

            # 該当行のデータを取得
            row = df_1d.iloc[idx]

            # 特徴量を揃える
            X = []
            for col in self.xgb_feature_cols:
                if col in row.index:
                    X.append(float(row[col]) if pd.notna(row[col]) else 0.0)
                else:
                    X.append(0.0)

            # XGBoost予測
            dmatrix = xgb.DMatrix([X])
            prob = float(self.xgb_model.predict(dmatrix)[0])

            return np.clip(prob, 0.0, 1.0)
        except Exception as e:
            return 0.5

    def _get_technical_score_5m(self, df_5m: pd.DataFrame, idx: int) -> Tuple[float, str]:
        """5分足から技術指標スコアを取得"""
        try:
            if idx < 1 or idx >= len(df_5m):
                return 0.5, "insufficient_data"

            row = df_5m.iloc[idx]
            prev_row = df_5m.iloc[idx - 1]

            score = 0.5  # Neutral starting point
            reasons = []

            # MA Crossover (短期 > 長期)
            if 'sma_5' in row.index and 'sma_20' in row.index:
                if pd.notna(row['sma_5']) and pd.notna(row['sma_20']):
                    if row['sma_5'] > row['sma_20']:
                        score += 0.20
                        reasons.append("MA_UP")
                    else:
                        score -= 0.20
                        reasons.append("MA_DOWN")

            # RSI (極値判定)
            if 'rsi_14' in row.index and pd.notna(row['rsi_14']):
                rsi = row['rsi_14']
                if rsi < 30:
                    score += 0.15
                    reasons.append("RSI_OVERSOLD")
                elif rsi > 70:
                    score -= 0.15
                    reasons.append("RSI_OVERBOUGHT")

            # MACD (ゴールデンクロス)
            if 'macd' in row.index and 'macd_signal' in row.index:
                if pd.notna(row['macd']) and pd.notna(row['macd_signal']):
                    if row['macd'] > row['macd_signal']:
                        prev_macd = prev_row.get('macd', 0)
                        prev_signal = prev_row.get('macd_signal', 0)
                        if pd.notna(prev_macd) and pd.notna(prev_signal):
                            if prev_macd <= prev_signal:
                                score += 0.15
                                reasons.append("MACD_GOLDEN_CROSS")
                            else:
                                score += 0.10
                                reasons.append("MACD_POSITIVE")
                    else:
                        score -= 0.10
                        reasons.append("MACD_NEGATIVE")

            # クリップ
            score = np.clip(score, 0.0, 1.0)
            reason = "_".join(reasons) if reasons else "neutral"

            return score, reason
        except Exception as e:
            return 0.5, f"error_{str(e)[:20]}"

    def generate_adaptive_signal(
        self,
        df_5m: pd.DataFrame,
        df_1d: pd.DataFrame,
        idx_5m: int,
        idx_1d: int,
        execute_threshold: float = 0.55
    ) -> Tuple[int, float, str]:
        """
        適応的信号生成

        Returns:
            (signal, confidence, reason)
            signal: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        try:
            # 1D XGBoost確率
            xgb_prob = self._get_xgb_probability(df_1d, idx_1d)

            # 5m技術指標スコア
            tech_score, tech_reason = self._get_technical_score_5m(df_5m, idx_5m)

            # 統合確信度
            confidence = 0.5 * xgb_prob + 0.5 * tech_score

            # シグナル生成
            if confidence >= execute_threshold and tech_score > 0.55:
                signal = 1  # BUY
                reason = f"BUY_conf={confidence:.3f}_xgb={xgb_prob:.3f}_tech={tech_score:.3f}"
            elif confidence <= (1 - execute_threshold) and tech_score < 0.45:
                signal = -1  # SELL
                reason = f"SELL_conf={confidence:.3f}_xgb={xgb_prob:.3f}_tech={tech_score:.3f}"
            else:
                signal = 0  # HOLD
                reason = f"HOLD_conf={confidence:.3f}"

            return signal, confidence, reason
        except Exception as e:
            return 0, 0.5, f"error_{str(e)[:20]}"

    def backtest(self, df_5m: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
        """
        バックテスト実行
        """
        print("\n[Backtest] Starting adaptive XGBoost backtest...")

        trades = []
        equity = 100000
        position = None
        entry_price = None
        entry_time = None
        signals_generated = 0
        signals_executed = 0

        # インデックス対応辞書を作成
        idx_1d_map = {}
        for i, date in enumerate(df_5m.index):
            # 該当する1Dインデックスを検索
            date_1d = pd.Timestamp(date).normalize()
            matches = np.where(df_1d.index.normalize() == date_1d)[0]
            if len(matches) > 0:
                idx_1d_map[i] = matches[-1]  # 最後のマッチを使用
            else:
                idx_1d_map[i] = None

        # バックテスト実行
        for i in range(1, len(df_5m)):
            date = df_5m.index[i]
            price = df_5m['close'].iloc[i]
            idx_1d = idx_1d_map.get(i)

            # シグナル生成
            signal, confidence, reason = self.generate_adaptive_signal(
                df_5m, df_1d, i, idx_1d if idx_1d is not None else max(0, i // 288)
            )

            if signal != 0:
                signals_generated += 1

            # ポジション管理
            if position is None:
                # 新規ポジション
                if signal == 1 and confidence >= 0.55:
                    position = date
                    entry_price = price
                    entry_time = date
                    signals_executed += 1
            else:
                # ポジション決済
                if signal == -1 or confidence < 0.45:
                    return_pct = (price - entry_price) / entry_price * 100
                    equity *= (1 + return_pct / 100)
                    trades.append({
                        'entry_date': entry_time,
                        'entry_price': entry_price,
                        'exit_date': date,
                        'exit_price': price,
                        'return_pct': return_pct,
                        'bars': i - df_5m.index.get_loc(entry_time),
                        'win': 1 if return_pct > 0 else 0
                    })
                    position = None

        # 最終ポジション処理
        if position is not None:
            final_price = df_5m['close'].iloc[-1]
            return_pct = (final_price - entry_price) / entry_price * 100
            equity *= (1 + return_pct / 100)
            trades.append({
                'entry_date': entry_time,
                'entry_price': entry_price,
                'exit_date': df_5m.index[-1],
                'exit_price': final_price,
                'return_pct': return_pct,
                'bars': len(df_5m) - df_5m.index.get_loc(entry_time),
                'win': 1 if return_pct > 0 else 0
            })

        # メトリクス計算
        total_return = (equity - 100000) / 100000 * 100
        num_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['win'])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0

        # Sharpe Ratio
        if num_trades > 1:
            returns = [t['return_pct'] for t in trades]
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Max Drawdown
        eq = 100000
        peak = 100000
        max_dd = 0
        for trade in trades:
            eq *= (1 + trade['return_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)

        metrics = {
            'total_return_pct': total_return,
            'final_equity': equity,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'win_rate_pct': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'signals_generated': signals_generated,
            'signals_executed': signals_executed,
            'trades': trades[:50]  # First 50 trades only
        }

        return metrics

    def run(self):
        """フル実行"""
        print("=" * 80)
        print("STEP 14: ADAPTIVE XGBOOST INTEGRATION - PRODUCTION IMPLEMENTATION")
        print("=" * 80)

        try:
            # データ取得
            print("\n[1/4] Fetching data...")
            fetcher = MultiTimeframeFetcher()
            data_dict = fetcher.fetch_and_resample(years=2)
            df_5m = data_dict['5m'].copy()
            df_1d = data_dict['1d'].copy()
            print(f"✓ Fetched: 5m={len(df_5m)} bars, 1d={len(df_1d)} bars")

            # 特徴量エンジニアリング
            print("\n[2/4] Engineering features...")
            engineer = MultiTimeframeFeatureEngineer()
            features_5m = engineer.engineer_features(df_5m, timeframe='5m')
            features_1d = engineer.engineer_features(df_1d, timeframe='1d')

            # 5mデータの特徴量をマージ
            df_5m = pd.concat([df_5m, features_5m], axis=1)
            df_1d = pd.concat([df_1d, features_1d], axis=1)

            print(f"✓ Features: 5m={len(df_5m.columns)} cols, 1d={len(df_1d.columns)} cols")

            # バックテスト実行
            print("\n[3/4] Running backtest...")
            start_time = time.time()
            backtester = AdaptiveXGBBacktester()
            metrics = backtester.backtest(df_5m, df_1d)
            elapsed = time.time() - start_time
            print(f"✓ Backtest completed in {elapsed:.1f}s")

            # 結果表示
            print("\n[4/4] Results:")
            print("=" * 80)
            print(f"Total Return:           {metrics['total_return_pct']:>8.2f}%")
            print(f"Final Equity:           ${metrics['final_equity']:>12,.2f}")
            print(f"Number of Trades:       {metrics['num_trades']:>8}")
            print(f"Win Rate:               {metrics['win_rate_pct']:>8.2f}%")
            print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:>8.2f}")
            print(f"Max Drawdown:           {metrics['max_drawdown_pct']:>8.2f}%")
            print(f"Signals Generated:      {metrics['signals_generated']:>8}")
            print(f"Signals Executed:       {metrics['signals_executed']:>8}")
            print("=" * 80)

            # 比較分析
            print("\n[Comparison with Phase 5-A Baseline]")
            print("=" * 80)
            phase5a_return = 2.00
            phase5a_trades = 7
            phase5a_winrate = 57.1

            print(f"Phase 5-A:    {phase5a_return:>6.2f}%  |  {phase5a_trades:>3} trades  |  {phase5a_winrate:>5.1f}% WR")
            print(f"Step 14:      {metrics['total_return_pct']:>6.2f}%  |  {metrics['num_trades']:>3} trades  |  {metrics['win_rate_pct']:>5.1f}% WR")

            if metrics['total_return_pct'] > phase5a_return:
                improvement = ((metrics['total_return_pct'] - phase5a_return) / phase5a_return) * 100
                print(f"Improvement:  ✅ +{improvement:.1f}%")
            else:
                print(f"Change:       ⚠️  {metrics['total_return_pct'] - phase5a_return:.2f}pp")

            # 結果保存
            result_file = Path("backtest/STEP14_ADAPTIVE_XGB_RESULTS.json")
            with open(result_file, 'w') as f:
                # trades を JSON シリアライズ可能にする
                metrics['trades'] = [
                    {
                        'entry_date': str(t['entry_date']),
                        'entry_price': float(t['entry_price']),
                        'exit_date': str(t['exit_date']),
                        'exit_price': float(t['exit_price']),
                        'return_pct': float(t['return_pct']),
                        'bars': int(t['bars']),
                        'win': int(t['win'])
                    }
                    for t in metrics['trades']
                ]
                json.dump(metrics, f, indent=2, default=str)

            print(f"\n✓ Results saved to {result_file}")
            print("\n" + "=" * 80)
            print("✓ STEP 14 EXECUTION COMPLETE")
            print("=" * 80)

            return metrics

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    backtester = AdaptiveXGBBacktester()
    metrics = backtester.run()
    sys.exit(0 if metrics else 1)
