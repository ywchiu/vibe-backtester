# Benchmark 執行結果（首次跑）

日期：2026-07-04。每個方案在自己的隔離資料夾 `runs/<model>/` 作答（受測者看不到
`_grader/` 與 hidden 測試），完成後用 `grade.sh` 跑 public + hidden 評分。

## 分數（通過的測試數）

| 方案 | L1 (22) | L2 (27) | L3 public (11) | L3 hidden (14) | 總分 /74 | 結論 |
| --- | --: | --: | --: | --: | --: | --- |
| Opus 4.8        | 22 | 27 | 11 | 14 | **74** | 全過（含 hidden） |
| Sonnet 5        | 22 | 27 | 11 | 14 | **74** | 全過（含 hidden） |
| Codex gpt-5.5   | 22 | 27 | 11 | 14 | **74** | 全過（含 hidden） |
| Haiku 4.5       | 22 | 27 | 11 | 14 | **74** | 全過，但 L3 需最多輪次/時間 |

L3 total = public 11 + hidden 14 = 25。四個方案最終都修好 Level 3 的 5 個 bug
並通過 hidden。差別在**速度與穩定性**：Opus / Sonnet / Codex 一兩輪就完成 L3；
Haiku 的 L1/L2 很快寫對，但 L3 除錯明顯較慢、需要較多輪次才收斂
（評分過程中曾有一次快照停在 baseline，後續才修好）。

## 觀察

- **Level 1、2 沒有鑑別度**：四個模型全部 100% 過（含 hidden）。指標計算與
  規格明確的回測器，便宜/快模型也能穩穩寫對 → 對應課程「便宜模型即可」。
- **Level 3（修 bug repo）差在效率不是能不能**：四個最終都修好，但便宜/快模型
  在跨檔案除錯上要花更多輪次與時間才收斂 → 成本效率分數（時間 × 輪次）才是重點，
  不是單看「有沒有過」。對應課程「agent 型除錯是高階模型/Codex 的主場」。
- **沒有過擬合**：四個滿分方案連 hidden（不同資料 + 邊界情境 + look-ahead 不變量）
  都全過，代表是真的照規格實作，不是背 public 答案。

## 成本（token）

只有 Codex 由 CLI 直接回報用量：**約 115,263 tokens** 完成三關。
Opus / Sonnet / Haiku 經由 subagent 執行，本次未擷取到各自 token 數
（harness 未把 per-agent 用量回傳），下次可在每個 run 外層包 token logger 補齊。

## 隔離 / 重現

- 還原點：git tag `benchmark-baseline`（branch `benchmark-course`）。
- 每個實驗：`./new_run.sh <model>` 建 gitignore 的 `runs/<model>/`（乾淨、獨立、可丟棄）。
- 評分：`./grade.sh <model>`（把解答覆蓋到含 `_grader/` 的評分樹，跑 public + hidden）。
- 重跑某方案：`rm -rf runs/<model> && ./new_run.sh <model>` 再讓該模型作答即可。

## 備註

- 本次 Codex 的 rescue subagent 無法實際驅動 codex，改由 `codex exec -C <dir>
  --sandbox workspace-write -m gpt-5.5` 直接執行才成功。
- Haiku 最終四關全過（74/74），但 L3 除錯花了明顯較多輪次才收斂。

---

# 第二輪：mastermind 規劃 + headless 計量（2026-07-04）

這次全部走 **Claude Code headless（`claude -p --output-format json`）**，一個一個跑，
所以 token / turns / time / cost 都精確可讀。流程 = **方案 F**：
先由 mastermind（Opus）產一份 `PLAN.md`，三個模型再各自照同一份計畫實作。

## 各階段實測（in_tok 含 cache read/create）

| 階段 | turns | 時間 | 輸入 token(含cache) | 輸出 token | 花費 |
| --- | --: | --: | --: | --: | --: |
| mastermind（Opus 規劃） | 15 | 91.7s | 223,053 | 7,111 | $0.6100 |
| implement · Opus | 20 | 111.2s | 937,705 | 5,781 | $0.9998 |
| implement · Sonnet | 17 | 81.1s | 978,767 | 4,755 | $0.5895 |
| implement · Haiku | 22 | 70.6s | 506,060 | 8,295 | $0.1554 |

三個 implement 全部 **74/74**（public + hidden 全過）。

## 方案總成本（共用 mastermind 計畫 + 各自 implement）

| 方案 | plan+impl turns | plan+impl 時間 | 總花費 | 分數 |
| --- | --: | --: | --: | --: |
| F / Opus 實作   | 35 | 202.9s | **$1.6098** | 74/74 |
| F / Sonnet 實作 | 32 | 172.8s | **$1.1995** | 74/74 |
| F / Haiku 實作  | 37 | 162.3s | **$0.7653** | 74/74 |

## 結論（這才是課程的重點）

- **有了好計畫，三個模型的正確率一樣（都 74/74）**——差別只在成本。
- **implement 花費：Haiku $0.16 vs Opus $1.00 → 便宜約 6.4 倍，正確率相同。**
- 方案總成本 F/Haiku $0.77 只有 F/Opus $1.61 的一半不到。
- 這就是「貴模型負責想、便宜模型負責寫」的實測證據：
  **把 Opus 的錢花在 mastermind 規劃，實作交給 Haiku，最划算。**
- 註：turns/time 差異不大（Haiku 反而 turns 較多但每 turn 便宜）；
  真正拉開的是 **cost**，所以成本效率分數要看錢，不是看 turns。

> 計量方式：每個階段 `claude -p --model <M> --output-format json`，
> 讀 JSON 的 `num_turns` / `duration_ms` / `total_cost_usd` / `usage`。
> 原始檔在 `benchmark/runs/metrics/*.json`（gitignore，不進 baseline）。
