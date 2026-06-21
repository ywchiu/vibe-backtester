# [Vibe Coding] Claude Code 太燒 Token？如何用多模型分工花得更精打細算


---

# 1. 安裝

```bash
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex
npm install -g @musistudio/claude-code-router
```

```bash
export OPENROUTER_API_KEY="sk-or-xxxxxxxx"
export NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem
```

若寫入 `~/.zshrc`，再執行 `source ~/.zshrc`。

> `NODE_EXTRA_CA_CERTS` 要設定，否則 CCR 可能出現 `fetch failed: unable to get local issuer certificate`。

---

# 2. Claude Code Router 設定
- https://github.com/musistudio/claude-code-router

```bash
mkdir -p ~/.claude-code-router
vi ~/.claude-code-router/config.json
```

```json
{
  "LOG": true,
  "API_TIMEOUT_MS": 900000,
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "$OPENROUTER_API_KEY",
      "models": [
        "anthropic/claude-opus-4.8",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash"
      ],
      "transformer": {
        "use": [
          "openrouter",
          ["maxtoken", { "max_tokens": 65536 }],
          "enhancetool",
          "reasoning"
        ]
      }
    }
  ],
  "Router": {
    "think": "openrouter,anthropic/claude-opus-4.8",
    "default": "openrouter,deepseek/deepseek-v4-flash",
    "background": "openrouter,deepseek/deepseek-v4-flash",
    "longContext": "openrouter,deepseek/deepseek-v4-pro",
    "longContextThreshold": 120000
  }
}
```

```bash
ccr restart
ccr model
ccr code
```

---

# 3. Codex plugin

在 Claude Code 內：

```text
/plugin marketplace add openai/codex-plugin-cc
/plugin install codex@openai-codex
/reload-plugins
/codex:setup
```

確認 CLI：

```bash
codex login status
codex exec --skip-git-repo-check "Reply with exactly CODEX_OK"
```

---


# 5. Demo 任務

## 5.1 Opus：先產生 plan

```text
請只做 planning，不要修改 source code。

目標：在本專案新增一條 RSI 策略，行為對照既有的 MA 策略。
請參考 backend/indicators/ma_indicator.py、backend/backtest/ma_backtest.py、
backend/api/ma_routes.py、backend/api/models.py、backend/tests/test_ma_strategy.py
的既有模式。

請先讀 codebase，產出 implementation plan 到 docs/agent-plans/rsi-strategy.md。
內容只要包含：Goal / Affected files / Implementation steps / Test strategy / Risks。

限制：不要改 code、不要開始實作、不要擴張 scope。
```

## 5.2 Codex：快速挑錯

```text
/codex:adversarial-review --background
請只審查 docs/agent-plans/rsi-strategy.md，不要實作。
重點看：檔案入口是否找對、是否漏掉 route/model 註冊、RSI 邊界條件、測試是否足夠、是否可能破壞既有 MA 行為。

輸出：Blockers / Must fix / Nice to have / Final recommendation。
```

```text
/codex:status
/codex:result
```

如果有 blocker，回 Opus 修 plan；沒有 blocker 就進實作。

## 5.3 DeepSeek V4 Flash：實作

```text
請依 docs/agent-plans/rsi-strategy.md 實作 RSI 策略。
範圍只限 plan 內列出的檔案與測試，不要擴張 scope。

完成後回報：
Files changed / Summary / Tests run / Results / Risks。
```

## 5.4 Verify：跑測試

```text
python -m pytest backend/tests -q
```

測試失敗時，不要繼續加功能，先請模型修最小錯誤：

```text
測試失敗，請只做最小修正。
先說明失敗原因，再修改 code，最後重跑 python -m pytest backend/tests -q。
```

---

# 6. 成本比較

review 完之後，用實際 CCR log 的 token 數比較不同模型組合。

示範用定價（每 1M tokens；正式示範前請以實際平台價格更新）：

| 模型 | input | output |
| --- | --- | --- |
| Opus 4.8 | $5.00 | $25.00 |
| DeepSeek V4 Pro | $0.435 | $0.87 |
| DeepSeek V4 Flash | $0.09 | $0.18 |
| gpt-5.3-codex | $1.75 | $14.00 |

```bash
python3 demo/cost_compare.py
```

```text
scenario            total    vs all-opus
------------------------------------------
all-opus       $    2.350     (baseline)
opus+v4-pro    $    0.633           -73%
opus+v4-flash  $    0.556           -76%
```

```text
all-opus     所有階段都用 Opus 4.8
opus+v4-pro    plan 用 Opus，execute/background 用 DeepSeek V4 Pro，review 用 gpt-5.3-codex
opus+v4-flash  plan 用 Opus，execute/background 用 DeepSeek V4 Flash，review 用 gpt-5.3-codex
```

> 金額要用 CCR log（`~/.claude-code-router/logs/` 的 `model_usage` 欄位）或各模型回傳的 usage 換算。
