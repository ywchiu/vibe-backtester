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

---

# 第三輪：乾淨隔離 + solo vs Opus-plan 對照（2026-07-04）

依客戶想看的設計：**每個模型「單幹」 vs 「Opus 先規劃再交給它實作」**，
並修正前兩輪的兩個污染源。

## 這次的隔離做法（避免作弊 / 污染）

1. **workspace 建在 repo 外**（scratchpad），只含受測者可見檔案，連 repo 路徑都不
   透露給 agent（python 用 scratch 內的 venv）→ 參考解答在檔案系統上搆不到。
2. **每次全新 process + 全新資料夾**（`rm -rf`）→ 無跨 run、無主 session context 污染。
3. **`claude -p --setting-sources ""`**：停用 CLAUDE.md / skills / hooks / plugins
   （含 superpowers 的 SessionStart 注入），保留 auth。→ 裸模型，行為不被 harness skill 帶偏。
4. 評分一律 hidden 差異測試（受測者看不到）。

> 為什麼第 3 點重要：**未停用 skills 時，Haiku 單幹只有 5/74** —— 它被
> `writing-plans` skill 帶走，把 9 個 turn 全花在寫「計畫文件」，一行 code 都沒實作。
> 停用 skills 後，Haiku 單幹 = **74/74**。這本身是個教訓：harness 的 skill 注入會拖垮便宜模型。

## 2×4 矩陣（全部 74/74；差別只在成本/時間）

| 模型 | 條件 | turns | 時間 | 花費 / token |
| --- | --- | --: | --: | --- |
| Opus 4.8 | solo | 18 | 100.9s | $0.5585 |
| Opus 4.8 | Opus-plan→impl | 14 | 63.2s | $0.4965 |
| Sonnet 5 | solo | 19 | 97.5s | $0.4569 |
| Sonnet 5 | Opus-plan→impl | 18 | 68.7s | $0.3852 |
| Haiku 4.5 | solo | 34 | 134.1s | $0.2181 |
| Haiku 4.5 | Opus-plan→impl | 26 | 108.8s | $0.1936 |
| Codex gpt-5.5 | solo | ~ | 221.0s | 941,959 tok（out 14,740）|
| Codex gpt-5.5 | Opus-plan→impl | ~ | 137.0s | 293,889 tok（out 8,407）|

mastermind 規劃（Opus，一次、共用）：15 turns / 91.7s / **$0.6100**。

## 兩個真正的發現

**A. 計畫讓每個實作者都更快更省（per-run）** —— 有 Opus 計畫後：
- Codex token 941,959 → 293,889（**省 ~69%**），時間 221s → 137s。
- Opus 100.9s→63.2s、Sonnet 97.5s→68.7s、Haiku 134s→109s。
- 花費：Opus $0.56→$0.50、Sonnet $0.46→$0.39、Haiku $0.22→$0.19。
計畫把「探索/試錯」的成本先付掉了，實作階段就短。

**B. 但計畫的 $0.61 固定成本，在「單一交付物」上不划算** ——
all-in 成本（plan + implement）比單幹貴：

| 模型 | 單幹 all-in | Opus-plan+impl all-in | 誰便宜 |
| --- | --: | --: | --- |
| Opus | $0.5585 | $1.1065 | **單幹** |
| Sonnet | $0.4569 | $0.9952 | **單幹** |
| Haiku | $0.2181 | $0.8036 | **單幹** |

## 結論（誠實版，給客戶）

- **規格夠明確時，最省 = 直接用最便宜模型單幹**：Haiku 單幹 **$0.22 拿滿分**，
  比任何「Opus 規劃 + X 實作」方案都便宜。
- **Opus 計畫的價值在「攤提」與「難題」**，不在單一小任務：
  - 攤提：一份 $0.61 計畫餵給多個實作者/多次任務才回本。
  - 難題：當任務規格模糊、便宜模型單幹會失敗時，計畫才是救命的（我們也看到
    污染版 Haiku 單幹只有 5/74 —— 一旦單幹會垮，計畫就從「浪費」變「必要」）。
- 本 benchmark 的 README 規格寫得很死（公式都給了），**偏向單幹有利**；
  真實模糊需求會更偏向「先規劃再實作」。這點要跟客戶講清楚，否則結論會誤導。
- 成本梯度：同樣 74/74，Haiku 約比 Opus 便宜 2.5 倍；Codex 正確但 token/時間最高。
