# Step 12: クイックリファレンス

**用途**: Phase 5-6の素早い参照用ドキュメント
**最終更新**: 2025-11-24

---

## 📊 現在のステータス

| 項目 | 値 |
|------|-----|
| **フェーズ** | Phase 5完了 → Phase 6準備完了 |
| **進捗** | 85% (5.5/6 フェーズ) |
| **パフォーマンス** | +4.85% (目標: +65%+) |
| **ギャップ** | -60.15pp |

---

## 🎯 パラメータセット

### 現在の設定
```
XGBoost閾値:              0.50
季節性重み (週, 月):      (0.30, 0.70)
信号融合閾値 (買, 売):    (0.60, 0.40)
```

### 推奨調整
```
XGBoost:     0.50 → 0.45 (シグナル増加)
Seasonality: (0.30,0.70) → (0.35,0.65) (週別対応力向上)
Signal:      (0.60,0.40) → (0.55,0.45) (積極化)
```

### 期待改善
```
保守的:   +12-18% (パラメータ調整のみ)
適度:     +35-45% (信号品質向上)
積極的:   +65-80% (完全統合)
```

---

## 🔍 問題点 (3つ)

| # | 問題 | 原因 | 対策 |
|---|------|------|------|
| 1 | シグナル実行率3.7% | ポジション管理制約 | システム改善, 多重時間軸 |
| 2 | 季節性スコア制限 | [0.33,0.67]範囲 | パラメータ調整 |
| 3 | XGBoostノイズ | F1=0.65, 偽陽性94.7% | アンサンブル学習 |

---

## 📋 Phase 6採用判定基準

### MUST条件 (2個とも満たす)
```
✓ 総リターン > +65%
✓ 最大DD ≤ -1.5%
```

### SHOULD条件 (1個以上)
```
✓ 取引数: 150-200
✓ 勝率: ≥62%
```

### 判定フロー
```
MUST両方満たす?
├─ YES → SHOULD確認
│        └─ 1個以上? → ✅ 採用
│        └─ 0個? → ⚠️ 条件付き採用
└─ NO → ❌ 却下 (再最適化)
```

---

## 🚀 実施計画

### Phase 5-A (推奨, 1-2日)
```
1. Grid Search実行 (9組み合わせ)
2. 結果分析
3. 改善版バックテスト
→ 期待: +12-18%
```

### Phase 5-B (オプション, 2-3日)
```
1. 4H時間軸追加
2. マルチTF確認ロジック
3. 信号品質スコア
→ 期待: +35-45%
```

### Phase 5-C (将来, 3-5日)
```
1. アンサンブル統合
2. 複数モデル追加
3. 投票メカニズム
→ 期待: +65-80%
```

---

## 📁 重要ドキュメント

| 文書 | 用途 |
|------|------|
| STEP12_PARAMETER_OPTIMIZATION_ANALYSIS.md | 詳細分析 |
| STEP12_PHASE5_COMPLETION_SUMMARY.md | 完了レポート |
| STEP12_PROJECT_STATUS.md | ステータス概要 |
| STEP12_COMPREHENSIVE_COMPARISON.md | 比較分析 |

---

## 📁 実装ファイル

| ファイル | 説明 |
|---------|------|
| model/step12_hybrid_strategy.py | 戦略実装 |
| model/step12_hybrid_feature_engineering.py | 特徴エンジニアリング |
| backtest/step12_parameter_tuning.py | Grid Search |
| main.py | パイプライン統合 |

---

## 🔧 Grid Search実行方法

### スクリプト実行
```bash
# プロジェクトルートから実行
source venv/bin/activate
python3 backtest/step12_parameter_tuning.py
```

### 出力結果
```
results.csv: 36パラメータ組み合わせの結果
  ├─ xgb_threshold
  ├─ seasonality_weights
  ├─ signal_thresholds
  ├─ total_return (%)
  ├─ num_trades
  ├─ win_rate (%)
  ├─ sharpe_ratio
  └─ max_drawdown (%)
```

---

## 💡 意思決定フロー

```
Grid Search完了
    ↓
TOP 3組み合わせ確認
    ↓
Total Return > 65%?
    ├─ YES → Max DD ≤ -1.5%?
    │         ├─ YES → ✅ 採用 (Phase 6完了)
    │         └─ NO → ⚠️ 条件付き採用
    └─ NO → ❌ 再最適化
           ├─ Phase 5-B実施
           ├─ Phase 5-C実施
           └─ Grid Search再実行
```

---

## 📈 成功メトリクス

```
Phase 6成功: Total Return ≥ +65% AND Max DD ≤ -1.5%

段階別目標:
  Phase 5-A: +12-18%
  Phase 5-B: +35-45%
  Phase 5-C: +65-80%
```

---

## ⏰ スケジュール

```
Week 1:
  Mon (11/25): Grid Search実行開始
  Tue (11/26): 結果分析, パラメータ適用
  Wed (11/27): 改善版バックテスト
  Thu (11/28): Phase 6判定

Week 2: (必要に応じて)
  Mon-Fri: Phase 5-B実施
```

---

## 🎯 目標チェックリスト

- [x] Phase 5分析完了
- [x] Grid Search計画完了
- [ ] Grid Search実行 (Phase 6)
- [ ] パラメータ適用 (Phase 6)
- [ ] 最終判定 (Phase 6)

---

**参照**: STEP12_PARAMETER_OPTIMIZATION_ANALYSIS.md (詳細情報)
**質問**: STEP12_PHASE5_COMPLETION_SUMMARY.md (詳細レポート)
**進捗**: STEP12_PROJECT_STATUS.md (全体ステータス)
