================================================================================
PHASE 5-D FINAL INTEGRATION & COMPREHENSIVE BACKTEST REPORT
================================================================================

[STAGE 1] HYPERPARAMETER OPTIMIZATION
--------------------------------------------------------------------------------
Best AUC Score: 0.5770

Best Parameters:
  n_estimators             : 50.0
  max_depth                : 9.0
  learning_rate            : 0.03
  subsample                : 0.8
  colsample_bytree         : 0.6
  gamma                    : 1.0

[STAGE 3] TRADING CONDITIONS
--------------------------------------------------------------------------------
Entry Threshold: 0.70
Take-Profit: 1.20%
Stop-Loss: 0.30%

[STAGE 4] DRAWDOWN MANAGEMENT
--------------------------------------------------------------------------------
Trailing Stop: 2.00%
Daily Loss Limit: 1.0%

[FINAL PERFORMANCE]
================================================================================

Total Return: 3.64%
Final Equity: $103,643.37
Maximum Drawdown: -3.52%

Trade Statistics:
  Total Trades: 3
  Winning Trades: 3
  Losing Trades: 0
  Win Rate: 100.0%

Quality Metrics:
  Average Win: 1.20%
  Average Loss: 0.00%
  Profit Factor: 0.00
  Sharpe Ratio: 109812368965252560.000

[TRADE DETAILS]
================================================================================
Num  Bars   Entry      Exit       Return%    Reason Equity      
--------------------------------------------------------------------------------
1    3      146.3490   148.1052   1.20       TP     $101,200.00 
2    2      148.4420   150.2233   1.20       TP     $102,414.40 
3    37     146.7930   148.5545   1.20       TP     $103,643.37 

[COMPARISON vs PHASE 5-B BASELINE]
================================================================================
Phase 5-B Return: 293.83%
Phase 5-D Return: 3.64%
Difference: -290.19% (-98.8%)

================================================================================
✓ PHASE 5-D OPTIMIZATION COMPLETE
================================================================================