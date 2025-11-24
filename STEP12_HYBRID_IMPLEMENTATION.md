# Step 12: 季節性 + XGBoost ハイブリッド統合戦略

**実装日**: 2025-11-24
**ステータス**: ✅ **アーキテクチャ設計・実装完了** → 次フェーズ: バックテスト
**目標**: +62.46% (Step 10) + XGBoostの改善 → **+70-75% 達成目指す**

---

## 📋 戦略概要

### 問題認識

| ステップ | 戦略 | パフォーマンス | 特徴 |
|---------|------|--------|------|
| **Step 10** | 季節性のみ | +62.46% | 確実だが機会を見落とす |
| **Step 11** | XGBoostのみ | +4.85% | 取引多くても精度不足 |
| **Step 12** | 統合戦略 | +70-75% | 両者の長所を統合 |

### 戦略の核心

```
XGBoost: 市場の微細なノイズを学習 (精度が不安定)
季節性: 確実な周期パターンを捉える (機会は限定的)

統合論理:
  XGBoostが「買い」AND 季節性スコアが「高い」
    → ✅ 確信度高い取引

  XGBoostが「買い」BUT 季節性スコアが「低い」
    → ⚠️ 慎重に進める or スキップ

  XGBoostが「売り」OR 季節性が「強く売り」
    → ❌ スキップ (リスク回避)
```

---

## 🏗️ 実装アーキテクチャ

### モジュール構成

```
step12_hybrid_strategy.py
├── HybridTradingStrategy クラス
│   ├── XGBoostモデル統合
│   ├── 季節性スコアリング
│   ├── 信号融合ロジック
│   └── バックテスト機構
│
step12_hybrid_feature_engineering.py
├── SeasonalityFeatureEngineer クラス
│   ├── 週別パターン特徴化
│   ├── 月別パターン特徴化
│   ├── ボラティリティ調整
│   ├── 季節性強度計算
│   └── ラグ特徴の追加
```

---

## 🔧 実装内容

### 1. 季節性スコアリング (HybridTradingStrategy)

#### 週別スコア (30%寄与)

```python
WEEKLY_STATS = {
    0: {'mean': 0.0609%, 'std': 0.7176%, 'signal': +0.1},  # 月 (best)
    1: {'mean': 0.0514%, 'std': 0.5132%, 'signal': +0.1},  # 火 (good)
    2: {'mean': 0.0035%, 'std': 0.6183%, 'signal': 0.0},   # 水 (neutral)
    3: {'mean': -0.0060%, 'std': 0.6691%, 'signal': -0.05}, # 木 (weak)
    4: {'mean': -0.0237%, 'std': 0.7068%, 'signal': -0.1},  # 金 (worst)
}
```

#### 月別スコア (70%寄与)

```python
MONTHLY_STATS = {
    6: {'mean': +0.1023%, 'std': 0.4759%, 'signal': +0.2},  # 6月 (best)
    ...
    12: {'mean': -0.0640%, 'std': 0.8714%, 'signal': -0.2},  # 12月 (worst)
}
```

#### スコア計算式

```
seasonality_score = 0.3 × weekly_score + 0.7 × monthly_score
  where:
    weekly_score = 0.5 + weekly_signal      (range: 0.4-0.6)
    monthly_score = 0.5 + monthly_signal    (range: 0.3-0.7)

Result: 0.0 (避けるべき) ~ 1.0 (理想的)
        中立点: 0.5
```

### 2. ハイブリッド信号融合

#### 融合ロジック

```python
weighted_probability = xgb_probability × (0.5 + 0.5 × seasonality_score)
```

**解釈**:
- 季節性スコアが 1.0 (最高): XGBoost確率を100%使用
- 季節性スコアが 0.5 (中立): XGBoost確率を75%に減衰
- 季節性スコアが 0.0 (最低): XGBoost確率を50%に減衰

#### 信号判定

```
if weighted_probability >= 0.60:
    signal = BUY (買い)
    confidence = weighted_probability

elif weighted_probability < 0.40:
    signal = SELL (売り)
    confidence = 1.0 - weighted_probability

else:
    signal = HOLD (待機)
    confidence = 0.5
```

### 3. 特徴量エンジニアリング

#### 季節性ベース特徴 (25個)

```
1. Weekly Features:
   - weekly_return_bias: 曜日別過去平均リターン
   - weekly_volatility: 曜日別ボラティリティ
   - weekly_return_bias_norm: 正規化版
   - weekly_volatility_norm: 正規化版

2. Monthly Features:
   - monthly_return_bias: 月別過去平均リターン
   - monthly_volatility: 月別ボラティリティ
   - monthly_return_bias_norm: 正規化版
   - monthly_volatility_norm: 正規化版

3. Yearly Features:
   - yearly_return_bias: 年別パターン
   - yearly_volatility: 年別ボラティリティ

4. Volatility Adjustments:
   - volatility_ratio: 実績 / 期待値
   - high_volatility_month: 高ボラティリティフラグ
   - low_volatility_month: 低ボラティリティフラグ

5. Seasonal Signals:
   - seasonal_strength_weekly: 週パターンの強度
   - seasonal_strength_monthly: 月パターンの強度
   - is_best_month: 6月フラグ
   - is_worst_month: 12月フラグ
   - is_best_day: 月曜フラグ
   - is_worst_day: 金曜フラグ
   - seasonality_score: 総合季節性スコア

6. Lagged Features:
   - weekly_pattern_lag5: 1週間前の週パターン
   - weekly_pattern_lag10: 2週間前の週パターン
   - weekly_pattern_lag20: 1ヶ月前の週パターン
   - monthly_pattern_lag20: 1ヶ月前の月パターン
   - monthly_pattern_lag60: 3ヶ月前の月パターン
```

---

## 📊 期待パフォーマンス

### シミュレーション予測

```
BaseLine (Step 10):
  └─ 総リターン: +62.46% (205取引)
  └─ 勝率: 65.85%
  └─ Sharpe: 6.779

期待 (Step 12):
  └─ 総リターン: +70-75% (estimated)
  └─ 勝率: 64-68% (estimated)
  └─ Sharpe: 7.0+ (estimated)

改善メカニズム:
1. 季節性による信号フィルタリング
   - 悪い季節の低精度シグナルを除外 (-3%)
   - 良い季節に確信度を強化 (+3%)

2. XGBoostによる機会拡大
   - Step 10で見落とした取引機会を捉える (+5-7%)

3. リスク調整
   - 季節性フィルタで負けを減らす (-1%)
   - Sharpeを改善する

予測総計: +62.46% → +70-75% (+7-12pp)
```

---

## 🚀 実行ロードマップ

### Phase 1: ✅ 完了 - アーキテクチャ設計

```
✅ Step 12_hybrid_strategy.py 実装 (280行)
✅ Step 12_hybrid_feature_engineering.py 実装 (290行)
✅ SeasonalityFeatureEngineer クラス実装
✅ HybridTradingStrategy クラス実装
```

### Phase 2: ⏳ 進行中 - モデル統合・バックテスト

```
⏳ 季節性特徴のXGBoost統合
⏳ ハイブリッドシグナル生成テスト
⏳ バックテスト実行 (22 → 205取引想定)
⏳ パフォーマンス計測
```

### Phase 3: 🔄 予定 - 最適化・検証

```
🔄 パラメータチューニング
  - XGBoost閾値 (0.55 vs 0.60)
  - 季節性重み (0.3/0.7 vs 他)
  - 信号閾値 (0.60/0.40 vs 他)

🔄 リスク評価
  - 最大ドローダウン測定
  - Sharpe比最適化
  - Win-Loss比改善

🔄 統計検証
  - シミュレーションと実績の乖離確認
  - 信号品質の統計テスト
```

---

## 🔍 技術詳細

### 信号融合の例

#### ケース1: 強気シグナル (買い推奨)

```
日付: 2024-06-10 (月, 6月)
XGBoost確率: 0.75 (強い買い)
Seasonalityスコア: 0.67 (最高クラス)

計算:
  weighted_prob = 0.75 × (0.5 + 0.5 × 0.67) = 0.75 × 0.835 = 0.626

判定: BUY ✅
信頼度: 62.6%
理由: XGBoostの強気 + 最良の季節性
```

#### ケース2: 慎重シグナル (スキップ推奨)

```
日付: 2024-12-20 (金, 12月)
XGBoost確率: 0.72 (強い買い)
Seasonalityスコア: 0.33 (最低クラス)

計算:
  weighted_prob = 0.72 × (0.5 + 0.5 × 0.33) = 0.72 × 0.665 = 0.479

判定: HOLD ⚠️
信頼度: 50%
理由: XGBoostの強気でも最悪の季節性で相殺
```

#### ケース3: 売りシグナル (回避)

```
日付: 2024-01-15 (月, 1月)
XGBoost確率: 0.35 (弱い売り)
Seasonalityスコア: 0.50 (中立)

計算:
  weighted_prob = 0.35 × (0.5 + 0.5 × 0.50) = 0.35 × 0.75 = 0.263

判定: SELL (リスク回避)
信頼度: 73.7%
理由: XGBoostが売りシグナル → 負けのリスク減
```

---

## 📈 検証計画

### バックテスト実行

```bash
# 1. ハイブリッドモデルの準備
python3 model/step12_hybrid_strategy.py

# 2. メインパイプラインでハイブリッド信号を生成
# main.py を修正して以下を実行:
#   - XGBoost予測
#   - 季節性スコアリング
#   - 信号融合
#   - TP/SL設定

# 3. バックテスト実行
python3 main.py

# 4. 結果比較
#   Step 10:   +62.46% (205取引)
#   Step 11:   +4.85% (22取引)
#   Step 12:   +70-75%? (205-220取引)
```

### 成功基準

```
✓ 総リターン > +65% (Step 10比で +2.54pp以上)
✓ 取引数: 200+  (機会をキャプチャ)
✓ 勝率: 64%+ (季節性フィルタで改善)
✓ Sharpe: 6.5+ (リスク調整後)
✓ 最大DD: -1.2%以内 (安全性維持)
```

---

## 🎯 次のステップ

### すぐに実行する

```bash
1. main.py を修正して HybridTradingStrategy を統合
2. バックテスト実行
3. パフォーマンス測定
```

### パラメータチューニング (成功時)

```
- XGBoost閾値を 0.55 → 0.60 に上げて精度重視
- 季節性重み 0.3/0.7 → 0.25/0.75 で季節性重視
- 信号閾値を 0.60/0.40 → 0.65/0.35 で厳選
```

### 追加改善 (2-3日後)

```
- マルチタイムフレーム統合 (4H + 1D)
- 機械学習で最適重みを学習
- アンサンブル (複数季節性指標)
```

---

## 📁 生成ファイル

```
✅ model/step12_hybrid_strategy.py (280行)
   - HybridTradingStrategy クラス
   - 季節性スコアリング
   - 信号融合ロジック

✅ model/step12_hybrid_feature_engineering.py (290行)
   - SeasonalityFeatureEngineer クラス
   - 25個の季節性特徴
   - 特徴ラベル管理

✅ STEP12_HYBRID_IMPLEMENTATION.md (このファイル)
   - 戦略説明
   - 実装詳細
   - パフォーマンス予測
```

---

## 💡 設計の根拠

### なぜこの統合戦略か?

1. **Step 10 (季節性のみ) の問題**
   - 205取引で高勝率だが、見落とす機会も多い
   - 季節性だけでは不十分な市場環境への対応に限界

2. **Step 11 (XGBoostのみ) の問題**
   - 22取引で精度不足（+4.85%）
   - ノイズに過剰反応

3. **統合のメリット**
   - 季節性: **確実性を提供** (基盤)
   - XGBoost: **精密さを提供** (補完)
   - 融合: 両者の長所を活かす

### 数学的根拠

```
P(Success | Hybrid) > P(Success | Seasonality Only)
                    AND > P(Success | XGBoost Only)

直感的には:
  - 季節性が良い + XGBoostが買い → 確実 (95%)
  - 季節性が悪い + XGBoostが買い → 疑わしい (40%)
  - 季節性が悪い + XGBoostが売り → 回避 (5%)
```

---

## 🏁 最終目標

```
Step 10:  +62.46% ✓ (baseline)
Step 11:  +4.85%  (参考: XGBoost単体)
Step 12:  +70-75% 🎯 (target)

達成時:
  ✅ プロジェクト基盤完成
  ✅ 次フェーズ (マルチモデル, リアルトレード) へ進展可能
  ✅ 実用的なAI取引システム確立
```

---

**実装完了日**: 2025-11-24
**ステータス**: ✅ アーキテクチャ完成 → バックテスト準備完了

