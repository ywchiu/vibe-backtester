# AI Coding 省錢 Benchmark

配合 `demo/Course_259.md`。核心問題：**同一批 coding 任務，不同模型／方案完成時，誰最省、誰最快、誰最穩？**

題材是一個可自動驗收、不偷看未來、可重現的量化回測系統。資料固定在 `data/`，
每個方案跑同一套任務、同一套測試，才能公平比較。

## 目錄結構

```
benchmark/
├── README.md               # 本檔
├── requirements.txt        # pandas / numpy / pytest
├── data/                   # 固定資料（可重現）
│   ├── sample_ohlcv_a.csv        # 公開 OHLCV
│   ├── signals_a.csv             # 公開多空/多平訊號
│   ├── hidden_ohlcv.csv          # 隱藏 OHLCV（grader-only）
│   └── signals_*_hidden.csv      # 隱藏訊號（grader-only）
├── level1/                 # Level 1：技術指標計算（純函式）
├── level2/                 # Level 2：簡單回測器（訊號→持倉→成本→metrics）
├── level3/                 # Level 3：修復有 bug 的多空回測 repo
└── runs/                   # 各模型/方案的隔離實驗（.gitignore，非基準）

每個 levelN 內：
    README.md            # 任務規格（交給模型）
    <solution>.py        # 受測者要完成的 stub / 待修 repo
    tests/test_public.py # 公開測試（可給受測者）
    _grader/             # grader-only：參考解答、hidden 測試、期望值產生器
```

> Level 4（walk-forward validation）依相同慣例加入 `benchmark/level4/`。

## 安裝

```bash
cd benchmark
pip install -r requirements.txt   # 或 uv pip install -r requirements.txt
```

## 快速開始

```bash
# 受測者自我檢查（public tests）
python -m pytest level1/tests/test_public.py -q
python -m pytest level2/tests/test_public.py -q
python -m pytest level3/tests -q

# grader：完整評分（public + hidden）
BT_SOLUTION=reference_solution python -m pytest level1/tests -q          # Level 1
BT_SOLUTION=reference_solution python -m pytest level2/tests -q          # Level 2
python -m pytest level3/tests -q && \
  python -m pytest level3/_grader/test_hidden.py -q                      # Level 3
```

Level 3 的 hidden 測試以 differential 方式比對 `_grader/` 內的參考解答，
不需 `BT_SOLUTION`（受測者修好 `backtester/` 後直接跑即可）。

## 評分維度（見 Course_259.md 第 2 節）

- 第一層（主評分）：public / hidden tests、no look-ahead、deterministic、edge cases — 全用程式驗證。
- 第二層（輔助）：LLM-as-judge 看可讀性、是否 hard-code 測試答案、可維護性。
- 第三層：成本效率分數 = 任務得分 / (成本 × 人工介入 × 時間)。
