# Level 3 — 修復有 bug 的回測系統

測 agent coding / debugging：讀 repo → 找根因 → 修 code → 跑測試 → 再修正。

## 任務

`backtester/` 是一個支援多空訊號 `{-1, 0, 1}` 的回測系統，目前**有數個 bug，
一部分測試是 failing 的**。請修復它，讓所有測試通過。

```text
請修復這個回測系統，讓 tests/ 全部通過，且不得修改 tests/。
```

規則：

- **不得修改 `tests/`**，只能改 `backtester/`。
- 只能用 pandas / numpy。
- 修完後測試必須穩定重現（deterministic）。

## 正確規格

`run_backtest(prices, signals, initial_capital=100000, cost_bps=10)` 回傳
`{"equity_curve": pd.Series, "metrics": {...}}`。

```
asset_ret[t] = close[t]/close[t-1] - 1          # 第 0 天視為 0
position[t]  = signal[t-1]                        # 用昨天訊號（避免 look-ahead）；支援 -1/0/1
turnover[t]  = |position[t] - position[t-1]|
strat_ret[t] = position[t]*asset_ret[t] - (cost_bps/10000)*turnover[t]
equity[t]    = initial_capital * cumprod(1+strat_ret)[t]
```

Metrics 定義同 Level 2，但**做多做空皆算持倉**（`win_rate` 的分母為 `position != 0` 的天數）；
`max_drawdown` 是比率 `min((equity - expanding_max)/expanding_max)`，介於 -1~0。

## 執行

```bash
pip install -r ../requirements.txt
python -m pytest tests -q
```

## 交給模型時只給

`README.md`、整個 `backtester/`、`tests/`、`../data/sample_ohlcv_a.csv`、
`../data/signals_short_a.csv`、`../requirements.txt`。
**不要**給 `_grader/`（內含正確參考解答、hidden 測試與 bug 清單）。

## 評分（grader 端）

```bash
python -m pytest tests -q                     # public：修對後應全過
python -m pytest _grader/test_hidden.py -q    # hidden：differential vs 參考解答
```
