# [Vibe Coding] 如何透過混用模型將 Fable 5 的效益發揮最大?!



> **同一批 coding 任務，不同模型／工具完成時，誰最省、誰最快、誰最穩？而哪一種「模型使用策略」最划算？**



---
# 1. 環境準備

## 1.1 安裝

```bash
npm install -g @anthropic-ai/claude-code
```

把 OpenRouter 金鑰放進**專案根目錄的 `.env`**（1.3 的函式會自動讀取，不用手動 export）：

```bash
# .env（務必加進 .gitignore，不要 commit）
OPENROUTER_KEY="sk-or-xxxxxxxx"
```

公司網路或代理環境如果出現憑證錯誤，再另外設定：

```bash
export NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem
```

> 這把金鑰只給「1.3 的 OpenRouter 入口」使用，不是拿來改一般 `claude`。
> 也不要在一般 shell 設 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`，不然整個 Claude Code 都會改走 OpenRouter。

## 1.2 原生 Claude Code

開 Claude Code 前，先確認沒有把它改成全域 OpenRouter：

```bash
unset ANTHROPIC_BASE_URL
unset ANTHROPIC_AUTH_TOKEN
claude
```



## 1.3 Claude Code 調用 OpenRouter 模型（DeepSeek 等）

- https://openrouter.ai/docs/cookbook/coding-agents/claude-code-integration

這一段是用 Claude Code 呼叫 OpenRouter 上的模型。把下面這個**通用**函式加到 `~/.zshrc`：它會自動讀當前目錄的 `.env`（金鑰放在 `OPENROUTER_KEY`），第一個參數接模型名稱就能切換——不用為每個模型各寫一份，也不用把金鑰寫死：

```bash
vi ~/.zshrc
```

```bash
claude-openrouter() {
  # 若當前目錄有 .env，載進來（金鑰放這裡，不寫死在函式）
  [ -f .env ] && { set -a; source .env; set +a; }
  # 兩種命名都接受：環境變數 OPENROUTER_API_KEY，或 .env 裡的 OPENROUTER_KEY
  : "${OPENROUTER_API_KEY:=$OPENROUTER_KEY}"
  if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "請先設定 OPENROUTER_API_KEY，或在 .env 放 OPENROUTER_KEY"
    return 1
  fi
  local model="${1:?用法: claude-openrouter <model-slug> [claude 參數...]}"
  shift

  env \
    ANTHROPIC_BASE_URL="https://openrouter.ai/api" \
    ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY" \
    ANTHROPIC_API_KEY="" \
    ANTHROPIC_DEFAULT_OPUS_MODEL="$model" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="$model" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="$model" \
    CLAUDE_CODE_SUBAGENT_MODEL="$model" \
    claude --model sonnet "$@"
}
```

再重新載入：

```bash
source ~/.zshrc
```

用法是第一個參數接 OpenRouter 模型頁上顯示的完整名稱（若頁面顯示成 `~deepseek/...` 開頭就照著填）：

```bash
claude-openrouter deepseek/deepseek-v4-flash    # DeepSeek Flash
claude-openrouter deepseek/deepseek-v4-pro      # 想改測 Pro 就換這個
```

先確認這個入口真的有走 OpenRouter：

```bash
cd /Users/david/course/vibe-backtester
claude-openrouter deepseek/deepseek-v4-flash
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

# 2. 操作步驟：規劃 → 實作 → 驗證

下面示範混用模型**完成三關（Level 1–3）** 的完整一輪：高階模型先規劃、便宜模型照計畫實作、再跑測試驗證。三關分別是——Level 1 實作技術指標、Level 2 實作回測器、Level 3 修好一個故意埋了 bug 的回測程式。

## 2.1 高階模型：先產生實作計畫（原生 Opus，只規劃）

```text
這一段請用原生 Claude Code 的 Opus，不要切到 OpenRouter。

請只做規劃，不要修改任何程式碼。

目標：完成 benchmark 的三關，讓每一關的測試都通過，且不得改動測試。
- Level 1：實作 benchmark/level1/indicators.py 的技術指標（SMA、EMA、每日／累積報酬、最大回撤）。
- Level 2：實作 benchmark/level2/backtester.py 的回測器（讀 OHLCV＋買賣訊號、模擬持倉、
  扣交易成本、算出報酬率／波動率／Sharpe／最大回撤／勝率，且不得偷看未來）。
- Level 3：修好 benchmark/level3/backtester/ 裡故意埋的 bug（交易成本、signal shift、
  max drawdown、NaN、short position 邏輯等），讓測試通過。

請先讀三關的 README 與程式，產出實作計畫到 PLAN.md，每關包含：
目標 / 影響檔案 / 作法或疑似根因 / 實作步驟 / 測試方式 / 風險。

限制：只規劃、不要動程式碼、不要擴張範圍、不得改動任何 tests/ 下的檔案。
```

## 2.2 便宜模型：照計畫完成三關（DeepSeek V4 Flash）

不用進互動模式再貼 prompt，直接一行呼叫（`-p` 就是 headless，任務寫在裡面）：

```bash
cd /Users/david/course/vibe-backtester
claude-openrouter deepseek/deepseek-v4-flash \
  --output-format json --dangerously-skip-permissions \
  -p "$(cat <<'EOF'
依 PLAN.md 完成三關。只改這三處，不得改動任何 tests/ 下的檔案，不要擴張範圍：
  benchmark/level1/indicators.py
  benchmark/level2/backtester.py
  benchmark/level3/backtester/*.py
每關都跑 public tests，直到通過：
  python -m pytest benchmark/level1/tests -q
  python -m pytest benchmark/level2/tests -q
  python -m pytest benchmark/level3/tests -q
完成後回報：改了哪些檔案 / 三關測試結果 / 還有哪些風險。
EOF
)"
```

`claude-openrouter` 會自動讀 `.env`、把端點指向 OpenRouter；`--output-format json` 會把
`num_turns` / `duration_ms` / `total_cost_usd` / `usage`（token）一起吐出來，方便記成本。

> 想改用其他實作者，只要換第一個參數即可：例如原生 Opus 用 `claude -p "..." --model opus`，
> 或 DeepSeek Pro 用 `claude-openrouter deepseek/deepseek-v4-pro -p "..."`。

> 想比照 benchmark 實驗的「乾淨裸模型」跑法，再加上 `--setting-sources ""`（停用 CLAUDE.md /
> skills / hooks，避免 harness 帶偏行為）。這裡不加，模型就會吃到你本地的 CLAUDE.md 與 skills。

## 2.3 驗證：跑三關的 public + hidden tests

三關的 public tests 一次跑：

```bash
benchmark/.venv/bin/python -m pytest benchmark/level1 benchmark/level2 benchmark/level3 -q
```

要連 hidden tests 一起評（受測者看不到 hidden，交給腳本）：

```bash
bash benchmark/grade.sh demo
```

`grade.sh <name>` 會把你在 `benchmark/` 就地改好的解答，覆蓋到一份含 `_grader` 隱藏測試的
乾淨評分樹再跑，輸出每關 public＋hidden 的 pass 數（`<name>` 只是這次評分的標籤，隨便取）。

因為 2.2 是**就地**改 `benchmark/` 底下的檔案，評完想還原成原始題目：

```bash
git checkout benchmark/level1 benchmark/level2 benchmark/level3
```

測試失敗時，不要繼續加功能，回頭叫模型只做最小修正：先說明是哪一關、失敗原因，
再改程式，最後重跑該關的 pytest。

