# Level 3 埋入的 bug 清單（GRADER ONLY）

受測者不會看到本檔。修對後 public + hidden 應全數通過。

| # | 檔案 | 位置 | bug | 正確做法 |
| - | --- | --- | --- | --- |
| 1 | `backtester/engine.py` | `position = sig.clip(lower=0)` | 沒有 `shift(1)` → 用到當天訊號（look-ahead bias） | `position = sig.shift(1).fillna(0.0)` |
| 2 | `backtester/engine.py` | 同上 `clip(lower=0)` | 把做空部位 (-1) 砍成 0 → 空頭邏輯失效 | 不要 clip，保留 -1 |
| 3 | `backtester/engine.py` | `strat_ret = position * asset_ret` | 沒扣交易成本（`turnover` 算了卻沒用） | `- (cost_bps/10000)*turnover` |
| 4 | `backtester/metrics.py` | `max_dd = (equity - peak).min()` | 少除以 peak → 回撤變成「金額」不是「比率」 | `((equity - peak)/peak).min()` |
| 5 | `backtester/metrics.py` | `win_rate = win_days/inv_days` | `inv_days==0` 時變 NaN（未處理） | 分母為 0 時回傳 `0.0` |

正確版可對照 `reference_engine.py` / `reference_metrics.py`。
