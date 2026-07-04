# Level 2 — 建立簡單回測器

測工程整合：訊號→持倉→扣成本→算 metrics，且不得偷看未來。

## 任務

在 `backtester.py` 實作 `run_backtest(prices, signals, initial_capital=100000, cost_bps=10)`，
回傳 `{"equity_curve": pd.Series, "metrics": {...}}`。

**精確定義（reference 與 grader 都照這份，差一步就會被 hidden 測試抓到）**

```
asset_ret[t] = close[t]/close[t-1] - 1            # asset_ret[0] = 0
position[t]  = signal[t-1]                          # 用昨天訊號；position[0]=0
turnover[t]  = |position[t] - position[t-1]|        # position[-1] 視為 0
strat_ret[t] = position[t]*asset_ret[t] - (cost_bps/10000)*turnover[t]
equity[t]    = initial_capital * cumprod(1+strat_ret)[t]
```

Metrics（令 r = strat_ret[1:]，n = len(r)）：

| 指標 | 定義 |
| --- | --- |
| total_return | `equity[-1]/initial_capital - 1` |
| annualized_return | `(equity[-1]/initial_capital)**(252/n) - 1` |
| volatility | `r.std(ddof=1) * sqrt(252)` |
| sharpe_ratio | `sqrt(252) * r.mean()/r.std(ddof=1)`；rf=0；std=0 回傳 0 |
| max_drawdown | `min((equity - expanding_max)/expanding_max)` |
| num_trades | `turnover>0` 的天數（進場、出場各算一次） |
| win_rate | `(strat_ret>0 且 position>0 天數)/(position>0 天數)`；分母 0 回傳 0 |

## 常見坑（hidden 測試會抓）

- 用「當天」訊號配「當天」報酬 → look-ahead bias（應 `shift(1)`）。
- 交易成本沒扣，或扣錯基點。
- Sharpe 沒年化、或年化用錯係數。
- max_drawdown 用報酬率算而非用權益曲線。
- 持倉時點錯一格、num_trades 把「維持持倉」也算成交易。

## 執行 / 評分

```bash
pip install -r ../requirements.txt
python -m pytest tests/test_public.py -q            # 受測者自我檢查
BT_SOLUTION=<模組> python -m pytest tests -q        # grader：public + hidden
```

## 交給模型時只給

`README.md`、`backtester.py`、`tests/test_public.py`、`expected/expected_public.json`、
`../data/sample_ohlcv_a.csv`、`../data/signals_a.csv`、`../requirements.txt`。
**不要**給 `_grader/`、`tests/test_hidden.py`、`../data/*hidden*.csv`。
