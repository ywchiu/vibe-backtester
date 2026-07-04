# Level 1 — 技術指標計算

測基本執行力：能不能照規格寫純函式、懂不懂金融時間序列、會不會處理 NaN。

## 任務

在 `indicators.py` 中實作以下 5 個函式（檔案已提供簽章與 docstring）：

| 函式 | 說明 |
| --- | --- |
| `sma(close, window)` | 簡單移動平均；前 `window-1` 個值為 NaN |
| `ema(close, window)` | 指數移動平均；`span=window, adjust=False`，以第一個 close 為種子 |
| `daily_return(close)` | `close / close.shift(1) - 1`；第一個值為 NaN |
| `cumulative_return(close)` | `close / close.iloc[0] - 1`；第一個值為 0.0 |
| `max_drawdown(close)` | `min((close - expanding_max) / expanding_max)`，回傳 float |

**精確定義以 `indicators.py` 內每個函式的 docstring 為準。**

## 規則

- 只能使用 pandas / numpy，不得讀網路或外部資料。
- 全部是純函式：相同輸入永遠相同輸出，且**不得修改輸入 Series**。
- 誤差容忍：所有數值以絕對誤差 `1e-6` 比對。
- 不得改動 `tests/`。

## 執行 public 測試（受測者自我檢查）

```bash
pip install -r ../requirements.txt
python -m pytest tests/test_public.py -q
```

期望值在 `expected/expected_public.json`，離線即可執行，不需要參考解答。

## 交給模型時只給這些

- `README.md`（本檔）
- `indicators.py`（stub）
- `tests/test_public.py`
- `expected/expected_public.json`
- `../data/sample_ohlcv_a.csv`
- `../requirements.txt`

**不要**交給模型：`_grader/`（參考解答與產生器）、`tests/test_hidden.py`、`../data/hidden_ohlcv.csv`。

## 評分（grader 端）

```bash
# 1) 若改過參考解答或資料，重新產生 public 期望值
python _grader/gen_expected.py

# 2) 用受測模組跑 public + hidden（differential vs oracle）
BT_SOLUTION=<受測模組名稱> python -m pytest tests -q
```

`conftest.py` 會依 `BT_SOLUTION` 選擇受測模組；未設定時預設 `indicators`，且不會把 `_grader/` 加入 import 路徑，`test_hidden.py` 會自動 skip。
