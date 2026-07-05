# [Vibe Coding] 如何透過混用模型將 Fable 5 的效益發揮最大?!



> **同一批 coding 任務，不同模型／工具完成時，誰最省、誰最快、誰最穩？而哪一種「模型使用策略」最划算？**



---
# 1. 環境準備

## 1.1 安裝

```bash
npm install -g @anthropic-ai/claude-code
```

把 OpenRouter 金鑰放進**專案根目錄的 `.env`**（1.3 的 `claude-openrouter` 腳本會自動讀取，不用手動 export）：

```bash
# .env（務必加進 .gitignore，不要 commit）
OPENROUTER_KEY="sk-or-xxxxxxxx"
```

公司網路或代理環境如果出現憑證錯誤，再另外設定：

```bash
export NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem
```

> 這把金鑰給「1.3 的 `claude-openrouter` 入口」使用，不是拿來改一般 `claude`。
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

這一段是用 Claude Code 呼叫 OpenRouter 上的模型。把它做成一個 **PATH 上的可執行腳本**（不是 `~/.zshrc` 函式）——因為 §2 會在 Claude Code session 內、由 Claude 自己用 Bash 呼叫它，而 shell 函式在非互動 shell 常常載不到、腳本才到處都叫得到。第一個參數接模型名稱，其餘照傳給 `claude`；它會自動讀當前目錄的 `.env`（金鑰放在 `OPENROUTER_KEY`）：

```bash
mkdir -p ~/.local/bin
```
```bash
cat > ~/.local/bin/claude-openrouter <<'SH'
#!/usr/bin/env bash
# 把 Claude Code 指向 OpenRouter 的模型；$1 = model slug，其餘照傳給 claude。
[ -f .env ] && { set -a; source .env; set +a; }   # 當前目錄有 .env 就載入
: "${OPENROUTER_API_KEY:=$OPENROUTER_KEY}"          # 兩種命名都接受
if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "請設定 OPENROUTER_API_KEY，或在當前目錄 .env 放 OPENROUTER_KEY" >&2
  exit 1
fi
model="${1:?用法: claude-openrouter <model-slug> [claude 參數...]}"; shift
exec env \
  ANTHROPIC_BASE_URL="https://openrouter.ai/api" \
  ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY" \
  ANTHROPIC_API_KEY="" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="$model" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="$model" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="$model" \
  CLAUDE_CODE_SUBAGENT_MODEL="$model" \
  claude --model sonnet "$@"
SH
```
```bash
chmod +x ~/.local/bin/claude-openrouter
```

確認 `~/.local/bin` 在 PATH（沒有就加，再 `source`）：

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
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

# 2. 操作步驟：規劃 → 派工 → 驗證

用 **Level 1（技術指標）** 當示範任務，全程待在**同一個原生 Opus 的 Claude Code session**：Opus 先規劃寫成 md、再由你叫 Opus「把任務用 `claude-openrouter -p` 派給 DeepSeek」、最後跑一次測試驗證。你不用跳出 session、也不用自己開新終端機。

先在專案根目錄開 session：

```bash
cd /Users/david/course/vibe-backtester
claude          # 原生 Opus
```

## 2.1 規劃：Opus 產出計畫並寫成 md

在 session 裡輸入：

```text
請只做規劃，不要改任何程式碼。
讀 benchmark/level1（README 與 indicators.py），規劃如何實作這些技術指標
（SMA、EMA、每日／累積報酬、最大回撤），把計畫寫到 plan.md：
目標 / 影響檔案 / 每個函式的作法 / 測試方式 / 風險。
```

## 2.2 派工：同一 session 內，叫 Opus 用 claude-openrouter 把任務派給 DeepSeek

直接叫 Opus 用它的 Bash 工具跑 `claude-openrouter -p`，把 `plan.md` 交給便宜模型實作。在 session 裡輸入：

```text
請在終端機執行下面這行，把 plan.md 的實作任務派給 DeepSeek，完成後把結果摘要給我：

claude-openrouter deepseek/deepseek-v4-flash --dangerously-skip-permissions -p "依 plan.md 實作 benchmark/level1/indicators.py 的技術指標函式；只改這個檔、不要動 tests/；做完自己跑 benchmark/.venv/bin/python -m pytest benchmark/level1/tests -q 確認通過"
```

Opus 會用 Bash 跑這行：`claude-openrouter` 把**一個新的 Claude Code 指向 OpenRouter 的 DeepSeek**、以 headless（`-p`）當實作 agent，直接改檔、跑測試，再把輸出回給你的 Opus session。

> `--dangerously-skip-permissions` 讓 headless 的 DeepSeek 能直接改檔（不會卡在權限詢問）。
> 想記成本，那行再加 `--output-format json`，DeepSeek 這趟的 turns／時間／成本／token 會一起回來。
> 想比照 benchmark 的「乾淨裸模型」，再加 `--setting-sources ""`（停用 CLAUDE.md / skills / hooks）。
> 換實作者只要換 slug：`deepseek/deepseek-v4-pro`；或改回原生 `claude -p "..." --model opus`。

## 2.3 驗證：跑一次測試看是否通過

DeepSeek 那趟通常會自己跑過測試；你要再獨立確認，就在 session 裡叫 Opus 跑（或自己在終端機跑）：

```bash
benchmark/.venv/bin/python -m pytest benchmark/level1/tests -q
```

失敗就回頭叫模型只做最小修正、再跑一次。示範完想還原成原始題目：

```bash
git checkout benchmark/level1
```

## 2.4 把 SOP 包成 skill

整套流程（組 prompt、背景派工、不信自我回報、親自重跑測試、成本記帳）每次都要人肉貼指令。最後一步是把這套 know-how 固化成 **skill**，之後只要口語說「把 X 派給 DeepSeek 跑」，Opus 就會自己照 SOP 走完全程。

skill 已放在 repo 裡，：

```text
.claude/skills/openrouter-dispatch/SKILL.md
```


