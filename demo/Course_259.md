# [Vibe Coding] AI Coding 省錢 Benchmark：用不同模型打造量化回測系統

這堂課不是「AI coding 做網站省錢」，也不是「用 AI 做會賺錢的策略」。

核心問題只有一個：

> **同一批 coding 任務，不同模型／工具完成時，誰最省、誰最快、誰最穩？而哪一種「模型使用策略」最划算？**

我們用一個可以自動驗收的量化回測系統當題材，把它切成由簡到難的四層任務，量測每個方案的時間、token、成本、成功率與程式品質，最後產出一張可以反覆更新的比較表。

---

# 0. 為什麼用量化回測系統，而不是做網站？

做網站很容易變成主觀評比：UI 好不好看、RWD 細不細、切版漂不漂亮，都不好自動驗證，最後容易變成「這個模型比較會用 Tailwind」而不是「這個模型比較會寫程式」。

量化回測系統剛好相反，適合當 benchmark：

| 優點 | 說明 |
| --- | --- |
| 可以自動驗收 | SMA/RSI/報酬率/最大回撤算對沒有、交易成本有沒有扣、有沒有偷看未來，全部能用 test 驗證 |
| 難度可自然分層 | 從單一函式到修整包 repo，難度階梯很平順，不會被一個小 UI 問題卡死 |
| 容易展示成本差異 | 便宜模型寫得出指標，但回測邏輯、抓 look-ahead bias 就開始拉開差距 |
| 接近真實商業場景 | 比純演算法題更工程化，又比網站更好量測 |

**定位要記住**：我們不是教大家用 AI 做賺錢策略，而是教大家用 AI coding 建立一個**可驗證、可重現、不偷看未來**的金融分析系統。

---

# 1. 任務設計：四層 benchmark

每一層都提供固定資料、public tests 與 hidden tests，並限制互動輪數與時間。誤差容忍一律 `1e-6`。

## Level 1：技術指標計算（測基本執行力）

要求模型實作純函式：

* SMA、EMA
* daily return、cumulative return
* max drawdown

驗收：給固定 CSV → 跑 `pytest` → 比對標準答案。

這一關測的是：能不能照規格寫 function、懂不懂金融時間序列、會不會處理 `NaN`。便宜模型（DeepSeek 這類）通常在這關就表現不錯。

## Level 2：建立簡單回測器（測工程整合）

要求：讀入 OHLCV → 接收買賣訊號 → 模擬持倉 → 扣交易成本 → 輸出 metrics。

Metrics 至少包含：total return、annualized return、volatility、Sharpe ratio、max drawdown、number of trades、win rate。

固定條件：起始資金 `100,000`、交易成本 `10 bps`。

驗收：public test 測基本案例、hidden test 測邊界案例、檢查是否有 look-ahead bias。

這關開始拉開差距，便宜模型常犯：報酬率公式錯、交易成本扣錯、持倉時點錯一格、偷看當天 close 又用當天 close 買、Sharpe 年化錯、max drawdown 算錯。

## Level 3：修復有 bug 的回測系統（測 agent coding／debugging）

給模型一個 starter repo，裡面**故意埋 bug**：transaction cost 沒扣、signal shift 錯誤、max drawdown 寫錯、`NaN` 沒處理、short position 邏輯錯，測試有一半 failing。

任務指令：

```text
請修復這個回測系統，讓所有測試通過，且不得改動測試。
```

這關最適合測 Codex、Claude Code、Fable 這類 agent 型工具：它不是從零生成，而是讀 repo → 找根因 → 改 code → 跑測試 → 再修正，最接近真實工作。

## Level 4：加入 walk-forward validation（高難，可選）

要求 train/test split、rolling window、參數搜尋、防 overfitting、產出 summary report。規格要**寫死**避免自由發揮：

```text
使用 252 天訓練，63 天測試，rolling window。
每次只用訓練區間選參數。
不得使用測試區間資料調參。
```

測的是複雜需求理解、防資料洩漏、架構設計與可維護性 —— 高階模型勝出的地方。

---

# 2. 評估標準：三層

## 第一層：自動測試（主評分，最重要）

能用程式驗證的，全部用程式驗證，**不要交給 LLM judge**。

| 項目 | 權重 |
| --- | --: |
| public tests pass | 20% |
| hidden tests pass | 40% |
| no look-ahead bias | 15% |
| deterministic output | 10% |
| edge cases | 15% |

## 第二層：LLM-as-judge（只做輔助）

LLM judge 評 code 可讀性、架構清晰度、是否過度複雜、是否 hard-code 測試答案、可維護性。

**但不要讓 LLM judge 決定「有沒有完成」** —— 完成與否由 tests 決定。

| 評估項目 | 評估者 |
| --- | --- |
| correctness | pytest / hidden tests |
| performance | benchmark script |
| cost | token logger |
| maintainability | LLM judge + human spot check |
| final ranking | 加權分數 |

## 第三層：成本效率分數（本課最有價值的地方）

不要只比總花費，要比「有效完成成本」：

```text
Cost per accepted task = 總成本 / 通過任務數
```

也可以做一個一看就懂的分數：

```text
Efficiency Score = 任務得分 / (成本 × 人工介入次數 × 時間)
```

---

# 3. 比較對象：不是「模型」，是「方案」

真正要比較的不是哪個模型最強，而是哪一種**使用策略**最划算。

| 方案 | 說明 |
| --- | --- |
| A. 全程 Sonnet | 高品質但可能貴 |
| B. 全程 Fable | 測規劃與複雜任務能力 |
| C. 全程 OpenAI / Codex | 測 agent 修 code 能力 |
| D. 全程 DeepSeek Pro | 測低成本直跑效果 |
| E. 便宜模型 + 高階模型 review | 測省錢分工 |
| F. 高階模型 plan + 便宜模型 implement + Codex review | 主打方案 |

我們預期最有教學價值的是 **F**：

> **高階模型負責想，便宜模型負責寫，Codex／高階模型負責驗。**

對應到分工建議：

| 任務類型 | 適合模型 |
| --- | --- |
| 指標計算、資料清理 | 便宜模型即可 |
| 回測邏輯 | 中階或高階模型 |
| 找 look-ahead bias、審查策略假設 | 高階模型 / Codex |
| 重構架構 | Sonnet / Fable / Codex |

課程結論一句話：

> **AI coding 省錢不是選最便宜的模型，而是把任務拆對，讓不同模型做不同工作。貴模型不要拿來寫所有 code，要拿來規劃、抓錯、審查。**

---

# 4. 實驗流程

每個方案都跑同一套流程，才能公平比較：

```text
1. 建立乾淨 repo
2. 給同一份任務 prompt
3. 限制最多互動輪數（例如 5 輪）
4. 限制最多時間（例如 30 分鐘）
5. 記錄 input / output / cached token
6. 記錄實際花費
7. 跑 public tests
8. 跑 hidden tests
9. 跑 code quality review（LLM judge）
10. 產生比較表
```

---

# 5. 環境準備：怎麼讓每個方案跑同一套任務

下面是把「原生 Claude Code / Codex / OpenRouter 的 DeepSeek」都準備好的實際步驟，讓你可以用同一台機器切換不同方案跑 benchmark。

## 5.1 安裝

```bash
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex
```

```bash
export OPENROUTER_API_KEY="sk-or-xxxxxxxx"
export NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem
```

若寫入 `~/.zshrc`，再執行 `source ~/.zshrc`。

> `OPENROUTER_API_KEY` 是給 Claude Code 的 DeepSeek 實作入口使用，不是拿來改一般 `claude`。
> 公司網路或代理環境如果出現憑證錯誤，再設定 `NODE_EXTRA_CA_CERTS`。
> 這裡不要設定 `ANTHROPIC_BASE_URL` 或 `ANTHROPIC_AUTH_TOKEN`，不然整個 Claude Code 都會改走 OpenRouter。

安裝 OpenRouter MCP 工具到 Claude Code：

```bash
claude mcp add --transport http openrouter https://mcp.openrouter.ai/mcp
claude mcp login openrouter
```

登入時會開 OpenRouter 授權頁面。這把金鑰只給 MCP 連線使用，和你平常的 API 金鑰分開管理。

## 5.2 原生 Claude Code（方案 A/B 的 plan 與高階實作）

開 Claude Code 前，先確認沒有把它改成全域 OpenRouter：

```bash
unset ANTHROPIC_BASE_URL
unset ANTHROPIC_AUTH_TOKEN
claude
```

這樣規劃、審查、與高階模型實作仍然使用原生 Claude Code 的額度。

## 5.3 Codex 外掛（負責挑錯／修 repo，方案 C/E/F 的 review）

在 Claude Code 內：

```text
/plugin marketplace add openai/codex-plugin-cc
/plugin install codex@openai-codex
/reload-plugins
/codex:setup
```

確認命令列工具：

```bash
codex login status
codex exec --skip-git-repo-check "請只回覆 CODEX_OK"
```

## 5.4 OpenRouter 官方 MCP（查模型／查用量，非實作主力）

- https://openrouter.ai/docs/mcp-server

先在 Claude Code 內確認 OpenRouter MCP 工具有出現：

```text
/mcp
```

上面範例安裝後會叫做 `openrouter`。它適合查模型、查用量、問一次性問題，不是把 Claude Code 整個切成 OpenRouter，也不是讓 DeepSeek 自己改檔案。常用工具：

```text
chat-send
models-list
model-get
credits-get
docs-search
generation-get
```

## 5.5 Claude Code 的 DeepSeek 寫碼入口（便宜模型負責寫，方案 D/E/F 的 implement）

- https://openrouter.ai/docs/cookbook/coding-agents/claude-code-integration

這一段是用 Claude Code 呼叫 OpenRouter 的 DeepSeek，不是 Codex，也不是另一套寫碼工具。把下面函式加到 `~/.zshrc`：

```bash
vi ~/.zshrc
```

```bash
claude-deepseek() {
  if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "請先設定 OPENROUTER_API_KEY"
    return 1
  fi

  env \
    ANTHROPIC_BASE_URL="https://openrouter.ai/api" \
    ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY" \
    ANTHROPIC_API_KEY="" \
    ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek/deepseek-v4-flash" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek/deepseek-v4-flash" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek/deepseek-v4-flash" \
    CLAUDE_CODE_SUBAGENT_MODEL="deepseek/deepseek-v4-flash" \
    claude --model sonnet "$@"
}
```

再重新載入：

```bash
source ~/.zshrc
```

`deepseek/deepseek-v4-flash` 請填 OpenRouter 模型頁顯示的完整名稱。如果頁面顯示成 `~deepseek/deepseek-v4-flash`，就把函式裡的四個模型名稱都改成那個。想測方案 D（DeepSeek Pro）時，把模型名稱換成對應的 Pro 版本即可。

先確認這個入口真的有走 OpenRouter：

```bash
cd /Users/david/course/vibe-backtester
claude-deepseek
```

進入 Claude Code 後執行：

```text
/status
```

應該看到：

```text
Auth token: ANTHROPIC_AUTH_TOKEN
Anthropic base URL: https://openrouter.ai/api
```

如果沒有看到上面兩行，代表這次還不是走 OpenRouter，先不要進實作。一般 `claude` 不受影響，仍然可以拿來跑原生 Opus。

---

# 6. 一個方案怎麼跑：以方案 F 為例（plan → review → implement → verify）

下面用 Level 3（修 bug repo）示範方案 F 的完整一輪。其他 Level 只要把任務 prompt 換掉、其餘流程不變。

## 6.1 高階模型：先產生修復計畫（原生 Opus，只規劃）

```text
這一段請用原生 Claude Code 的 Opus，不要切到 OpenRouter。

請只做規劃，不要修改程式碼。

目標：修復 benchmark/level3 這個回測系統，讓所有測試通過，且不得改動測試。
請先讀 repo，找出讓測試 failing 的根因（交易成本、signal shift、max drawdown、
NaN、short position 邏輯等），產出修復計畫到 docs/agent-plans/level3-fix.md。
內容只要包含：目標 / 影響檔案 / 疑似根因 / 修復步驟 / 測試方式 / 風險。

限制：不要改程式碼、不要開始實作、不要擴張範圍、不得改動測試。
```

## 6.2 Codex：快速挑錯（review 計畫）

```text
/codex:adversarial-review --background
請只審查 docs/agent-plans/level3-fix.md，不要實作。
重點看：根因判斷是否正確、是否漏掉某些 failing test 的成因、是否可能引入
look-ahead bias、修復是否會破壞其他已通過的測試。

輸出：目前不能開始實作的問題 / 一定要先修的問題 / 可以之後再改善的地方 / 最後建議。
```

```text
/codex:status
/codex:result
```

如果有「目前不能開始實作的問題」，就回高階模型修計畫；沒有的話就進實作。

## 6.3 便宜模型：實作（DeepSeek V4 Flash，只改計畫內範圍）

```bash
cd /Users/david/course/vibe-backtester
claude-deepseek
```

進入後先 `/status` 確認有顯示 `ANTHROPIC_AUTH_TOKEN` 和 `https://openrouter.ai/api`，再貼：

```text
這一段請用 OpenRouter 的 deepseek/deepseek-v4-flash。

請直接修改檔案，依 docs/agent-plans/level3-fix.md 修復回測系統。
範圍只限計畫內列出的檔案，不得改動測試，不要擴張範圍。

完成後回報：修改了哪些檔案 / 做了什麼 / 跑了哪些測試 / 測試結果 / 還有哪些風險。
完成後請回到原生 Claude Code 做最後確認。
```

## 6.4 驗證：跑 public + hidden tests

```text
python -m pytest benchmark/level3 -q
```

測試失敗時，不要繼續加功能，先請模型修最小錯誤：

```text
測試失敗，請只做最小修正。
先說明失敗原因，再修改程式碼，最後重跑 python -m pytest benchmark/level3 -q。
```

## 6.5 記錄這一輪的量測數據

每跑完一個方案，把下面數字填進比較表：input/output/cached token、實際花費、public/hidden 通過率、是否偵測到 look-ahead bias、人工介入次數、耗時。

---

# 7. 成本比較

分開看花費來源：

- 規劃與審查：Claude Code 原本額度。
- 實作：OpenRouter 後台紀錄。
- Codex 挑錯：Codex / OpenAI 用量。

示範用定價（每一百萬個計費單位；正式示範前請以實際平台價格更新）：

| 模型 | 輸入 | 輸出 |
| --- | --- | --- |
| Opus 4.8 | $5.00 | $25.00 |
| DeepSeek V4 Pro | $0.435 | $0.87 |
| DeepSeek V4 Flash | $0.09 | $0.18 |
| gpt-5.3-codex | $1.75 | $14.00 |

```bash
python3 demo/cost_compare.py
```

```text
情境             總成本    和全用 Opus 相比
-------------------------------------------
全用 Opus    $    2.350    基準
Opus + V4 Pro    $    0.633    -73%
Opus + V4 Flash  $    0.556    -76%
```

```text
全用 Opus        所有階段都用原生 Claude Code 的 Opus 4.8（對應方案 A 的高階版）
Opus + V4 Pro    規劃用 Opus，實作用 OpenRouter 的 DeepSeek V4 Pro，檢查用 Codex（方案 F）
Opus + V4 Flash  規劃用 Opus，實作用 OpenRouter 的 DeepSeek V4 Flash，檢查用 Codex（方案 F）
```

> 金額要用 Claude Code、OpenRouter、Codex 各自的實際用量換算。

---

# 8. 財經／量化題目的三個坑

### 坑一：不要測「策略績效」

不要問「哪個模型做出的策略報酬率最高？」—— 這會變成亂試參數。
要問「哪個模型能正確建立一個不偷看未來、可測試、可重現的回測系統？」

### 坑二：不要接真實交易 API

課程內不接 Binance、IB、券商 API 下單。最多做到 historical data、backtest、paper trading interface、mock broker、report generation。這樣安全，也不會變成投資建議。

### 坑三：資料要固定

不要讓模型自己去抓 Yahoo Finance 或即時資料。提供固定 CSV：

```text
data/
  sample_ohlcv_a.csv
  sample_ohlcv_b.csv
  hidden_ohlcv.csv
```

這樣每次 benchmark 結果才可重現。

---

# 9. 最終產出：一張比較表

課程尾聲把所有方案的量測結果彙整成表（分數為示意，實跑後填入）：

| 方案 | Level 1 | Level 2 | Level 3 | 總成本 | 時間 | 人工介入 | 結論 |
| --- | ------: | ------: | ------: | --: | -: | ---: | --- |
| DeepSeek only | 95 | 75 | 40 | 低 | 中 | 高 | 適合簡單任務 |
| Sonnet only | 98 | 90 | 82 | 高 | 中 | 中 | 穩但成本高 |
| Codex only | 95 | 88 | 90 | 中高 | 快 | 低 | 修 repo 很強 |
| Fable only | 96 | 90 | 88 | 高 | 中 | 低 | 複雜規劃強 |
| 分工模式（F） | 96 | 88 | 85 | 中低 | 中 | 中低 | 最划算 |

這張表就是整堂課的結論：**把任務拆對，讓不同模型做不同工作，比單選一個「最便宜」或「最強」的模型都划算。**

## 9.1 首次實測（2026-07-04，本 repo 的 benchmark）

用本 repo 的 `benchmark/`（Level 1–3）實際跑一輪，每個模型在隔離資料夾 `runs/<model>/`
作答，看不到 `_grader/` 與 hidden 測試，完成後用 `grade.sh` 跑 public + hidden 評分：

| 方案 | L1 (22) | L2 (27) | L3 public (11) | L3 hidden (14) | 總分 /74 | 結論 |
| --- | --: | --: | --: | --: | --: | --- |
| Opus 4.8 | 22 | 27 | 11 | 14 | **74** | 全過（含 hidden） |
| Sonnet 5 | 22 | 27 | 11 | 14 | **74** | 全過（含 hidden） |
| Codex gpt-5.5 | 22 | 27 | 11 | 14 | **74** | 全過；~115k tokens |
| Haiku 4.5 | 22 | 27 | 5 | 0 | **54** | L1/L2 全過，L3 未修 |

實測印證了難度分層：

- **Level 1、2 沒有鑑別度** —— 四個模型全部 100%（含 hidden）。指標計算與規格明確的
  回測器，便宜/快模型也能穩穩寫對 → 這種活交便宜模型即可。
- **Level 3（修 bug repo）才拉開差距** —— 要讀 repo、找 5 個根因、跨檔案修改。
  Opus / Sonnet / Codex 乾淨修好且過 hidden；Haiku 把 L1/L2 寫對，卻沒真正進到
  L3 除錯。這正是「agent 型除錯是高階模型／Codex 勝出處」。
- **無過擬合** —— 三個滿分方案連 hidden（不同資料＋邊界＋look-ahead 不變量）都全過。

> 重跑：`./benchmark/new_run.sh <model>` 建乾淨資料夾 → 讓該模型作答 →
> `./benchmark/grade.sh <model>` 評分。完整紀錄見 `benchmark/RESULTS.md`。
