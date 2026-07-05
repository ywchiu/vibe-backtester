# [Vibe Coding] 如何透過混用模型將 Fable 5 的效益發揮最大?!



> **同一批 coding 任務，不同模型／工具完成時，誰最省、誰最快、誰最穩？而哪一種「模型使用策略」最划算？**



---
# 1. 環境準備

## 1.1 安裝

```bash
npm install -g @anthropic-ai/claude-code
```

```bash
export OPENROUTER_API_KEY="sk-or-xxxxxxxx"
export NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem
```

若寫入 `~/.zshrc`，再執行 `source ~/.zshrc`。

> `OPENROUTER_API_KEY` 是給 Claude Code 的 DeepSeek 實作入口使用，不是拿來改一般 `claude`。
> 公司網路或代理環境如果出現憑證錯誤，再設定 `NODE_EXTRA_CA_CERTS`。
> 這裡不要設定 `ANTHROPIC_BASE_URL` 或 `ANTHROPIC_AUTH_TOKEN`，不然整個 Claude Code 都會改走 OpenRouter。

## 1.2 原生 Claude Code

開 Claude Code 前，先確認沒有把它改成全域 OpenRouter：

```bash
unset ANTHROPIC_BASE_URL
unset ANTHROPIC_AUTH_TOKEN
claude
```



## 1.3 Claude Code 調用 DeepSeek

- https://openrouter.ai/docs/cookbook/coding-agents/claude-code-integration

這一段是用 Claude Code 呼叫 OpenRouter 的 DeepSeek。把下面函式加到 `~/.zshrc`：

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

`deepseek/deepseek-v4-flash` 請填 OpenRouter 模型頁顯示的完整名稱。如果頁面顯示成 `~deepseek/deepseek-v4-flash`，就把函式裡的四個模型名稱都改成那個。想測 DeepSeek Pro 時，把模型名稱換成對應的 Pro 版本即可。

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

## 1.4 headless 自動化跑法（一次跑多個模型、直接量測成本）

上面 1.1–1.3 是「手動、互動式」的操作。若想一次跑多個模型、並直接量到 token／時間／成本，可改用 **headless（非互動）** 模式：把同一組 env 設定包成腳本、用 `claude -p` 直接把結果吐成 JSON。兩種入口：

**Claude 系列（Opus / Sonnet / Haiku / Fable 5）—— 原生 headless**

```bash
claude -p "<任務或計畫>" --model opus \
  --setting-sources "" \
  --output-format json \
  --dangerously-skip-permissions
```

- `--setting-sources ""`：停用 CLAUDE.md / skills / hooks / plugins（但保留登入）。讓每個模型都是「乾淨裸模型」，行為不被 harness 的 skill 帶偏——沒關掉時，便宜模型會被 `writing-plans` skill 帶走、把整段時間拿去寫計畫、一行 code 都沒實作。
- `--output-format json`：回傳裡有 `num_turns` / `duration_ms` / `total_cost_usd` / `usage`，token、時間、成本一次到位（不必另外算）。

**DeepSeek（透過 OpenRouter）—— 同一個 Claude Code，只把 env 改走 OpenRouter**

```bash
env \
  ANTHROPIC_BASE_URL="https://openrouter.ai/api" \
  ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY" \
  ANTHROPIC_API_KEY="" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek/deepseek-v4-flash" \
  claude -p "<任務>" --model sonnet \
    --setting-sources "" --output-format json --dangerously-skip-permissions
```

這就是 1.3 那個 `claude-deepseek` 函式的 headless 版——**同一組 env 變數，只是不進互動模式**。金鑰這次放在專案根目錄的 `.env`（變數名是 `OPENROUTER_KEY`），腳本讀出來再設成 `OPENROUTER_API_KEY`。想測 DeepSeek Pro 就把模型名換成 `deepseek/deepseek-v4-pro`。

---

# 2. 操作步驟：規劃 → 實作 → 驗證

下面用 Level 3（修 bug repo）示範混用模型的完整一輪：高階模型規劃、便宜模型實作、再跑測試驗證。其他 Level 只要把任務 prompt 換掉、其餘流程不變。

## 2.1 高階模型：先產生修復計畫（原生 Opus，只規劃）

```text
這一段請用原生 Claude Code 的 Opus，不要切到 OpenRouter。

請只做規劃，不要修改程式碼。

目標：修復 benchmark/level3 這個回測系統，讓所有測試通過，且不得改動測試。
請先讀 repo，找出讓測試 failing 的根因（交易成本、signal shift、max drawdown、
NaN、short position 邏輯等），產出修復計畫到 docs/agent-plans/level3-fix.md。
內容只要包含：目標 / 影響檔案 / 疑似根因 / 修復步驟 / 測試方式 / 風險。

限制：不要改程式碼、不要開始實作、不要擴張範圍、不得改動測試。
```

## 2.2 便宜模型：實作（DeepSeek V4 Flash，只改計畫內範圍）

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

## 2.3 驗證：跑 public + hidden tests

```text
python -m pytest benchmark/level3 -q
```

測試失敗時，不要繼續加功能，先請模型修最小錯誤：

```text
測試失敗，請只做最小修正。
先說明失敗原因，再修改程式碼，最後重跑 python -m pytest benchmark/level3 -q。
```

