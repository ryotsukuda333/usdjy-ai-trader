# Step 11: 予測精度向上（XGBoostモデル最適化） - 計画書

## 概要

Step 10の季節性・周期性活用により +62.46% のハイパフォーマンスを達成しました。
Step 11では、XGBoostモデルの予測精度をさらに向上させるための包括的な改善プログラムを実施します。

---

## 11.1: 現在のモデルパフォーマンス分析 ✅ 完了

### 既存モデルスコア
```
Train Accuracy: 0.7691
Train F1 Score: 0.8200
CV Score: 0.6819
```

### 問題点の識別
1. **オーバーフィット傾向**: Train (0.82) vs CV (0.68) の大きな差
2. **CV スコアが低い**: 0.68 は予測精度改善の余地あり
3. **正則化可能性**: より強い正則化で汎化性能を向上可能

---

## 11.2: ハイパーパラメータチューニング（実装完了）

### 実装ファイル
- `model/advanced_hyperparameter_tuning.py` (Bayesian Optimization対応)
- `model/quick_model_improvement.py` (高速な軽量チューニング)

### チューニング対象パラメータ
```
learning_rate:      [0.001, 0.1] (log scale)
max_depth:          [3, 10]
n_estimators:       [100, 500]
subsample:          [0.5, 1.0]
colsample_bytree:   [0.5, 1.0]
reg_alpha (L1):     [0.001, 10.0] (log scale)
reg_lambda (L2):    [0.001, 10.0] (log scale)
```

### 最適化メソッド選択肢
1. **Bayesian Optimization** (推奨)
   - hyperopt ライブラリ使用
   - 20回イテレーション = より高精度
   - 実行時間: ~5-10分

2. **グリッドサーチ** (フォールバック)
   - hyperopt未インストール時に自動選択
   - 81コンボテスト = 実行時間短い
   - 制限されたパラメータ空間

3. **ランダムサーチ** (軽量)
   - 実装可能 (qはく)
   - 完全なグリッドより高速

### 予想される改善

**保守的な推定**:
```
Original CV F1: 0.6819
Optimized CV F1: 0.71-0.73 (3-7% 改善)
```

**楽観的なシナリオ**:
```
最適なパラメータ組み合わせ: 0.74-0.76
改善率: 8-12%
```

---

## 11.3: 新特徴量の追加と最適化（実装完了）

### 実装ファイル
- `features/advanced_feature_engineer.py` (24個の新特徴量追加)

### 追加される特徴量カテゴリー

#### 1. **季節性特徴量** (4個)
```python
month_sin, month_cos          # 月周期（三角関数エンコード）
day_sin, day_cos              # 日周期（年間パターン）
quarter                        # 四半期
season                        # シーズン (0-3)
seasonal_vol_factor           # 季節的ボラティリティファクタ
```

#### 2. **モメンタム指標** (5個)
```python
rsi_14                        # Relative Strength Index
stoch_k, stoch_d             # Stochastic Oscillator
roc_5, roc_10                # Rate of Change
cci_20                        # Commodity Channel Index
```

#### 3. **ボラティリティシグナル** (5個)
```python
atr_14, atr_pct              # Average True Range
bb_position                   # Bollinger Band ポジション
bb_squeeze                    # Bollinger Band スクイーズ
volatility_regime             # ボラティリティレジーム
```

#### 4. **トレンド信号** (4個)
```python
di_plus, di_minus            # Directional Indicator
adx                          # Average Directional Index
hma_trend                    # Hull Moving Average トレンド
```

#### 5. **サポート/レジスタンス** (3個)
```python
swing_high, swing_low        # スイング高値/安値
distance_to_high/low         # 直近高値/安値との距離
price_position               # 高値/安値間でのポジション
```

#### 6. **取引量指標** (4個)
```python
volume_ma_20, volume_ratio   # 取引量移動平均と比率
obv, obv_ma_20              # オンバランスボリューム
vroc_10                      # 取引量変化率
```

#### 7. **オーダーフロー信号** (4個)
```python
candle_body_size             # ローソク足ボディサイズ
upper_wick, lower_wick       # 上下のヒゲサイズ
close_position               # クローズ位置
consecutive_up/down          # 連続上/下クローズ
```

#### 8. **平均回帰シグナル** (4個)
```python
distance_from_ma20/50/200    # 移動平均からの距離
zscore_20, zscore_50         # Z-スコア
```

### 特徴量エンジニアリングの予想効果

```
元の特徴量: 40個
拡張後: 64-70個

期待される改善:
- CV F1: +0.02-0.05 (2-7%)
- モデルロバストネス向上
- 過学習の軽減
```

### 特徴量追加戦略

**段階的アプローチ**:
```
Phase 1 (軽量): 季節性 + モメンタム (9特徴量) → CV F1テスト
Phase 2 (中程度): + ボラティリティ + トレンド (14特徴量)
Phase 3 (フル): 全特徴量追加 (24特徴量)

各段階でモデルを再訓練して効果測定
```

---

## 11.4: モデル再訓練と検証（実装完了）

### 再訓練パイプライン

```
1. データロード
   ├─ OHLCV データ (3年間)
   ├─ 特徴量エンジニアリング
   └─ (オプション) 高度な特徴量追加

2. ハイパーパラメータ最適化
   ├─ グリッドサーチ / Bayesian最適化
   └─ 最良パラメータ選択

3. モデル訓練
   ├─ 全データで訓練
   ├─ 5-fold CV スコア評価
   └─ 訓練メトリクス計算

4. 検証とテスト
   ├─ CV スコア確認
   ├─ オーバーフィット チェック
   └─ 既存モデルとの比較
```

### 検証メトリクス

```
主要指標:
- CV F1 Score (最重要)
- Train Accuracy vs CV Accuracy (過学習度合い)
- Precision / Recall バランス
- AUC-ROC (ロジスティック回帰の場合)

合格基準:
✅ CV F1 > 0.70
✅ Train vs CV の差 < 0.15
✅ 元モデルより改善
```

### バックテスト統合計画

```
改善モデルをバックテストに統合:

1. 現在のパイプライン (xgb_model.json)
   └─ Seasonality-aware: +62.46%

2. 改善モデルパイプライン (xgb_model_v2.json)
   └─ Expected: +65-68% (推定)

3. 結果比較
   ├─ 取引数
   ├─ 勝率
   ├─ リターン
   └─ Sharpe比

改善検証後に本採用
```

---

## 実装ファイル一覧

### 新規作成ファイル
```
✅ model/advanced_hyperparameter_tuning.py
   ├─ Bayesian Optimization (hyperopt)
   ├─ グリッドサーチフォールバック
   └─ 詳細なトライアルログ出力

✅ model/quick_model_improvement.py
   ├─ 軽量な改善スクリプト
   ├─ 3つの設定をテスト
   └─ 迅速な結果確認

✅ model/improve_model_full_pipeline.py
   ├─ 統合パイプライン
   ├─ ハイパーパラメータ + 特徴量
   └─ モデル比較機能

✅ features/advanced_feature_engineer.py
   ├─ 24個の新特徴量生成
   ├─ 8カテゴリの指標
   └─ モジュラー設計 (段階的追加可能)
```

### 出力ファイル (生成予定)
```
model/xgb_model_v2.json              # 改善モデル
model/xgb_model_improved.json        # Bayesian最適化版
model/step11_improvement.json        # 改善メトリクス
model/improvement_results.json       # 詳細結果
model/feature_columns_improved.json  # 新特徴量リスト
STEP11_RESULTS.md                    # 最終レポート
```

---

## パフォーマンス改善の期待値

### シナリオ別予想

#### **保守的なシナリオ** (現実的)
```
パラメータチューニングのみ:
  CV F1: 0.6819 → 0.71 (3% ↑)

バックテスト影響:
  +62.46% → +64% (2%改善)
```

#### **標準シナリオ** (推奨ケース)
```
パラメータ + 季節性・モメンタム特徴量:
  CV F1: 0.6819 → 0.73 (7% ↑)

バックテスト影響:
  +62.46% → +66% (4-5%改善)
```

#### **楽観的シナリオ** (理想的)
```
フル最適化 (全特徴量 + ハイパーパラメータ):
  CV F1: 0.6819 → 0.75 (10% ↑)

バックテスト影響:
  +62.46% → +68% (6-8%改善)
```

---

## 実装スケジュール

```
Phase 1: データ分析 (11.1) ✅
  ├─ 既存パフォーマンス評価
  ├─ ボトルネック特定
  └─ 改善戦略立案

Phase 2: ハイパーパラメータ最適化 (11.2)
  ├─ Bayesian Optimization実装
  ├─ グリッドサーチ実装
  └─ 最良パラメータ選定

Phase 3: 特徴量拡張 (11.3)
  ├─ 高度な特徴量実装
  ├─ 段階的テスト
  └─ 特徴量選別

Phase 4: 再訓練と検証 (11.4)
  ├─ モデル再訓練
  ├─ パフォーマンス検証
  └─ バックテスト統合
```

---

## リスク管理と注意点

### 潜在的なリスク

1. **過学習リスク**: 特徴量増加 → L1/L2正則化強化で対策
2. **計算時間**: グリッドサーチが長い → Bayesian最適化で短縮
3. **改善の限界**: CV F1が頭打ちの可能性 → 別アプローチ必要
4. **バックテスト乖離**: 訓練データ vs 取引データの乖離

### 対策

```
✓ K-fold CV (5分割) で汎化性能評価
✓ 訓練/テスト分割での検証
✓ 段階的な改善による影響測定
✓ パラメータの物理的妥当性確認
✓ 定期的なモデル再訓練予定
```

---

## 成功基準

```
11.2 ハイパーパラメータチューニング
  ✅ CV F1スコア: > 0.70
  ✅ グリッドサーチ完了: ≥ 18テスト

11.3 特徴量最適化
  ✅ 新特徴量追加: ≥ 15個
  ✅ 特徴量モジュール: 動作確認

11.4 モデル再訓練
  ✅ 改善モデル訓練完了
  ✅ パフォーマンス向上: ≥ 2%
  ✅ ファイル保存: 4個以上

Overall:
  ✅ Step 11完了
  ✅ ドキュメント: STEP11_RESULTS.md
  ✅ Git Commit: Step 11 記録
```

---

## 次のステップ

### Step 11完了後
1. **改善モデルのバックテスト統合**
   - xgb_model_v2.json を main.py で使用
   - パフォーマンス比較 (元モデル vs 改善版)

2. **マルチペア展開** (Step 12候補)
   - EURUSD, GBPUSD, JPYUSD での適用
   - クロスペア相関分析

3. **リアルタイム運用準備**
   - ライブデータフィードの統合
   - 定期的なモデル再訓練
   - リスク管理の自動化

---

## まとめ

Step 11では、XGBoostモデルの予測精度を系統的に向上させます。

**主な活動**:
- ハイパーパラメータの最適化 (11.2)
- 24個の新特徴量の追加 (11.3)
- モデルの再訓練と検証 (11.4)

**期待値**: +2-8% のパフォーマンス向上
**最終目標**: $170,000+ (+70%)のリターン達成

---

**作成日**: 2025-11-24
**ステータス**: 🔵 計画・実装段階
**次フェーズ**: バックテスト統合と結果検証
