---
name: openrouter-dispatch
description: Use when 使用者要把實作任務「派給」OpenRouter 上的模型（DeepSeek、Qwen、GLM 等）當 coding agent 執行——例如「派給 DeepSeek 跑」「叫 deepseek 實作」「用 openrouter 的模型做這個」，或提到 claude-openrouter、deepseek-v4-flash/pro 等 slug。
---

# OpenRouter 派工（openrouter-dispatch）

## Overview

用 `claude-openrouter` wrapper 把一個 headless Claude Code（`-p`）指向 OpenRouter 的模型當實作 agent。核心機制：`ANTHROPIC_BASE_URL` 指到 OpenRouter + model 別名映射到 slug——**不要**自己寫 curl/SDK 呼叫 OpenRouter API，也**不要**重建 benchmark 用的隔離腳本來做日常派工。

## 派工 SOP

**1. 金鑰**：repo 根目錄 `.env` 有 `OPENROUTER_KEY`，wrapper 會自動載入。只需確認 wrapper 存在：`command -v claude-openrouter`（在 `~/.local/bin/`；不存在時照 `demo/Course_259.md` 的安裝段落建立）。

**2. 組 prompt**（三要素缺一不可）：

```
<任務描述>；只改 <目標檔案>、不要動 tests/；
做完自己跑 <repo 絕對路徑>/benchmark/.venv/bin/python -m pytest <測試路徑> -q 確認通過
```

路徑一律用**絕對路徑**——headless agent 的 cwd 和你的 shell cwd 都可能跟預期不同。

**3. 派出**（從 repo 根目錄、背景執行，跑完會收到通知）：

```bash
claude-openrouter deepseek/deepseek-v4-flash <權限模式> \
  --output-format json -p "<上面組好的 prompt>"
```

- **權限模式由使用者決定，不要自行預設跳過權限**：預設建議 `--permission-mode acceptEdits`（允許改檔、其餘照常把關）。若使用者明確給了完整指令或明確要求完全跳過權限（如課程 demo 的做法），照使用者的指令執行即可。
- 模型選擇：`deepseek-v4-flash` 給規格明確的機械實作；`deepseek-v4-pro` 給要推理/除錯的活。
- `--output-format json` 讓 turns／耗時／cost／tokens 隨結果回來，摘要時記帳用。

**4. 驗證（不信自我回報）**——派工 agent 說「全過」不算數，主 session 必須：

```bash
git status --short                    # 抓「只准改 X」之外的意外變更
git diff <目標檔案>                    # 看實作內容
<絕對路徑>/.venv/bin/python -m pytest <測試路徑> -q   # 親自重跑
```

**5. 回報**：測試結果（自己跑的，不是引用它說的）、實作摘要、合規性（有沒有動到不該動的檔）、成本統計（來自 JSON）。

## 何時升級到隔離模式

只有在做**公平比較／benchmark**（模型不能看到 CLAUDE.md、skills、參考解答）時，才用 `benchmark/runs/drive_deepseek.sh` 的做法：repo 外 workspace + `--setting-sources ""` + rsync 排除 `_grader/`。日常派工不需要。

## Common Mistakes

| 錯誤 | 修正 |
| --- | --- |
| 自己寫 curl/requests 呼叫 OpenRouter API | 不必——Claude Code 就是 client，換 base URL 即可 |
| 沿用舊 session 的 scratchpad venv／`.or_key` | 用 repo `.env` + `benchmark/.venv`（都在版本管理的路徑下） |
| prompt 裡用相對路徑 | headless agent cwd 不可控，一律絕對路徑 |
| 引用派工 agent 的「6 passed」當驗收 | 主 session 親自重跑 pytest + `git status` |
| 日常派工套 benchmark 隔離腳本 | 那是公平比較用的；日常一行 `claude-openrouter` 就好 |
| 未經使用者同意就跳過權限模式 | 權限模式是使用者的決定；預設 `acceptEdits` |
