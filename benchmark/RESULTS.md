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
| Codex gpt-5.5 | solo | ~ | 221.0s | 941,959 tok → **$1.3877** |
| Codex gpt-5.5 | Opus-plan→impl | ~ | 137.0s | 293,889 tok → **$0.7316** |

mastermind 規劃（Opus，一次、共用）：15 turns / 91.7s / **$0.6100**。

> Codex 的 USD 用 gpt-5.5 官方定價換算（輸入 $5/M、輸出 $30/M、cache 讀取 $0.5/M；
> reasoning token 已含在 output 內）。這是「等值 API 成本」；實際走 ChatGPT 訂閱可能不另計費。

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
| Codex gpt-5.5 | $1.3877 | $1.3416 | **F（唯一例外）** |

> **反轉點：Codex 是唯一「先規劃反而更省」的模型** —— 因為它單幹時暴衝探索
> （94萬 token / 221s），計畫把它壓到 29萬 token / 137s，省下的比 $0.61 計畫還多。
> 「探索成本越高的模型，越值得先給計畫」——這是給客戶的一條實用準則。
>
> **⚠ 已被第七輪推翻**：這是單次抽樣的假象。跑 3 次後 Codex solo 平均 $1.28、
> fplan 平均 $1.46（範圍重疊），計畫對 Codex 成本的影響落在雜訊內，無法斷定有幫助。

---

# 第四輪：加入 DeepSeek（完整 6 模型矩陣，2026-07-04）

DeepSeek 走 Course 原本的方案 F 作法：Claude Code 指向 OpenRouter，
`ANTHROPIC_DEFAULT_*_MODEL=deepseek/deepseek-v4-flash|pro`，`claude --model sonnet`。
隔離同前（workspace 在 repo 外、`--setting-sources ""`、fresh process）。
DeepSeek 的 USD 用 OpenRouter 真實定價自算（Claude Code 回報的 total_cost_usd 對
OpenRouter 模型不準，已忽略）。Flash $0.09/$0.18/M、Pro $0.435/$0.87/M。

## 完整矩陣（全部 74/74）—— turns / 時間 / token / 成本 全壓進來

| 實作者 | 條件 | turns | 時間(s) | 總token | 輸出token | 成本 |
| --- | --- | --: | --: | --: | --: | --: |
| **DeepSeek Flash** | solo | 42 | 142.7 | 887,622 | 14,753 | **$0.0800** |
| **DeepSeek Flash** | fplan | 19 | 50.9 | 304,697 | 5,054 | **$0.0277** |
| Haiku 4.5 | solo | 34 | 134.1 | 1,011,054 | 10,933 | $0.2181 |
| Haiku 4.5 | fplan | 26 | 108.8 | 959,509 | 10,756 | $0.1936 |
| DeepSeek Pro | solo | 34 | 165.9 | 529,120 | 9,083 | $0.2216 |
| DeepSeek Pro | fplan | 28 | 106.3 | 422,102 | 6,142 | $0.1744 |
| Sonnet 5 | solo | 19 | 97.5 | 550,299 | 8,409 | $0.4569 |
| Sonnet 5 | fplan | 18 | 68.7 | 660,819 | 5,061 | $0.3852 |
| Opus 4.8 | solo | 18 | 100.9 | 396,350 | 6,023 | $0.5585 |
| Opus 4.8 | fplan | 14 | 63.2 | 339,095 | 4,888 | $0.4965 |
| Codex gpt-5.5 | solo | ~ | 221.0 | 941,959 | 10,677 | $1.3877 |
| Codex gpt-5.5 | fplan | ~ | 137.0 | 293,889 | 6,312 | $0.7316 |
| **mastermind(Opus)** | plan | 15 | 91.7 | 230,164 | 7,111 | $0.6100 |

> 總token = 輸入(含 cache)＋輸出。DeepSeek/Codex 成本用官方 token 定價換算，
> 已忽略 Claude Code 對 OpenRouter 模型不準的 total_cost_usd。
> **DeepSeek 成本用 Anthropic usage 語意計算：`input_tokens` 本身即未快取，
> cache 另計，不再相減**（此處先前算錯、已由 Codex 審查抓出並修正，見文末）。
> 每個 cell 的 public+hidden 評分原始輸出存於 `benchmark/runs/grades.txt`（可稽核，全部 74/74）。

**三個維度各自的贏家（單幹）：**
- **最省錢**：DeepSeek Flash $0.080。
- **最快**：Opus 100.9s / Sonnet 97.5s（DeepSeek Flash 143s、Codex 221s 最慢）。
- **最少 token**：Opus 396k（Haiku 用最多 1.01M，但單價低所以仍便宜）。

**Opus 計畫對三維度的影響（fplan vs solo）：**
- 幾乎都更快：Flash 143→51s、Opus 101→63s、Codex 221→137s。
- token 多數下降：Flash 888k→305k、Codex 942k→294k、Opus 396k→339k；
  但 Sonnet 反而上升(550k→661k)、Haiku 幾乎持平 → 計畫不保證每個模型都省 token。
- turns 多數下降（Flash 42→19、Opus 18→14），Haiku/Sonnet 變動小。

## all-in 成本（單一交付物；fplan 含 $0.61 計畫）

| 模型 | 單幹 | Opus規劃 all-in | 誰便宜 |
| --- | --: | --: | --- |
| **DeepSeek V4 Flash** | **$0.0800** | $0.6377 | **單幹（全場最省）** |
| Haiku 4.5 | $0.2181 | $0.8036 | 單幹 |
| DeepSeek V4 Pro | $0.2216 | $0.7844 | 單幹 |
| Sonnet 5 | $0.4569 | $0.9952 | 單幹 |
| Opus 4.8 | $0.5585 | $1.1065 | 單幹 |
| Codex gpt-5.5 | $1.3877 | $1.3416 | F（唯一例外）|

## 最終結論（含 DeepSeek，數字為修正後）

- **規格明確時，全場最省 = DeepSeek V4 Flash 單幹：$0.080 拿滿分 74/74。**
  比 Haiku($0.218)便宜約 2.7 倍，比 Opus 便宜約 7 倍，比 Codex 便宜約 17 倍。
- **Opus 計畫仍讓每個實作者更快更省（per-run）**：DeepSeek Flash 實作 $0.080→$0.028、
  143s→51s、42→19 turns。但單一交付物加上 $0.61 計畫就不划算（Codex 除外）。
- **方案 F 的真正甜蜜點 = 攤提**：一份 $0.61 的 Opus 計畫，如果餵給 DeepSeek Flash
  連做很多支任務，每支實作只要 ~$0.03 → 量大時平均成本趨近 DeepSeek 的價，
  又拿到 Opus 等級的規劃品質。這才是課程要賣的「貴模型規劃、便宜模型實作」。
- ~~Codex 例外：計畫幫它省最多~~ **（已被第七輪 n=3 推翻——落在雜訊內）**。
- **總排序（單幹，all-in）：Flash $0.080 < Haiku $0.218 < DS Pro $0.222 < Sonnet $0.457
  < Opus $0.559 < Codex $1.388。**（修正 DeepSeek cache 計法後，Haiku 些微低於 DS Pro。）

## 一句話結論（給客戶）

- **規格夠明確時，最省 = 直接用最便宜模型單幹 —— DeepSeek V4 Flash $0.080 拿滿分。**
  （早前未加 DeepSeek 時寫「Haiku 最省」，現以本節為準。）
- **Opus 計畫的價值在「攤提」與「難題」**，不在單一小任務：一份 $0.61 計畫餵給多個
  實作者/多次任務才回本；或任務模糊到便宜模型單幹會失敗時才必要（污染版 Haiku 單幹
  只有 5/74，一旦單幹會垮，計畫就從「浪費」變「必要」）。
- 本 benchmark 的 README 公式都寫死，**偏向單幹有利**；真實模糊需求會更偏向先規劃。

---

# 第五輪：Codex 獨立審查（gpt-5.5 當 reviewer）

把上面的成本分析交給 Codex（gpt-5.5）當獨立審查者，要它讀原始 `metrics/*.json`
**自己重算**、抓錯（正是方案 F 的「Codex 審查」角色）。它花 ~13萬 token，抓到 3 個真問題：

1. **DeepSeek 成本 cache 計法錯（已修）** —— 我原本 `uncached = input − cache`，但 Claude Code
   回報的是 Anthropic 語意，`input_tokens` 本身就已是未快取、cache 另計，不該再減。
   修正後：Flash solo $0.0786→**$0.0800**、fplan →$0.0277；Pro solo $0.2091→**$0.2216**、
   fplan →$0.1744。（Codex 自己語意相反 [input 含 cache]，那邊本來就算對。）
2. **74/74 未存檔、不可稽核（已修）** —— 分數原本只在終端機看到。現已把每個 cell 的
   public+hidden 評分原始輸出存到 `benchmark/runs/grades.txt`，12 個 cell 全部 22+27+11+14=74。
3. **舊結論自相矛盾（已修）** —— 加入 DeepSeek 後仍留著「Haiku 最省」的舊句，已改以
   第四輪為準（最省是 DeepSeek Flash $0.080）。

Codex 也確認：Codex solo $1.3877 與拆解（48/29/23%）**正確**、reasoning token 已含在 output 內、
「gpt-5.5 旗艦定價驅動成本」與「Flash solo 最省」的方向**成立**。

> 這一輪本身就是課程賣點的示範：**便宜模型（DeepSeek）做實作、貴模型/Codex 做審查**，
> Codex 花 ~$0.5 等值就抓出一個我漏掉的計價 bug。審查真的有價值。

---

# 第六輪：獨立性檢查（回應「subagent 污染 / 球員兼裁判」）

**Q：其他 Claude 模型是 subagent 嗎？會不會污染或球員兼裁判？**

1. **不是 subagent。** 矩陣裡的每個 cell 都是獨立的 headless 進程（`claude -p --setting-sources ""`
   / `codex exec` / `claude -p`→OpenRouter），拿不到主 session 的 context。早期用 Agent-tool
   subagent 的那兩輪已作廢、不列入矩陣。
2. **評分不是 LLM 判的**，是確定性 `pytest` + hidden 差異測試（`grades.txt` 可稽核）。
   沒有 Claude 判 Claude。
3. **設計層（規格/參考解答）由 Opus 撰寫 = 潛在利益衝突。** 用兩個方式關掉：
   - **Codex（gpt-5.5）獨立審查成本分析** → 抓到我一個 cache 計價 bug（見第五輪）。
   - **跨 oracle 一致性測試**（`xoracle_run.py`）：在 benchmark **從沒看過的隨機資料**上，
     拿 3 個**非 Claude** 選手的解答當 alternate oracle，和我的參考解答逐點比對。

   結果（Level 1/2/3 全部）：

   | alternate oracle | 與 MY-REF 最大絕對差 | NaN 位置 |
   | --- | --- | --- |
   | Codex gpt-5.5 | **0.00e+00** | 一致 |
   | DeepSeek Pro | **0.00e+00** | 一致 |
   | DeepSeek Flash | **0.00e+00** | 一致 |

   **三個非 Claude 實作在未見資料上與我的參考逐位元相同** → 參考解答不是「Claude 的詮釋」，
   而是任何正確實作都會收斂到的唯一答案；規格無歧義；hidden 測試的公平性等同於一個
   已被交叉驗證的 oracle。「球員兼裁判」在設計層以數據關閉。

殘留、誠實揭露：mastermind 計畫仍由 Opus 寫（fplan 時 Opus-實作拿到同源計畫，理論上略有利，
實測不明顯）；README 公式寫死 → 偏向單幹有利。這兩點屬「設計偏好」，不影響上面的正確性結論。

---

# 第七輪：Codex 變異測試（同條件各跑 3 次）

回應「Codex 要不要再跑一次、會不會每次差很多」。solo / fplan 各跑 3 次：

| cond | 分數 | 時間 min/mean/max | 總token min/mean/max | 成本 min/mean/max |
| --- | --- | --- | --- | --- |
| solo | 3× 74/74 | 205 / 243 / 279 s | 741k / 1.01M / 1.46M | $0.95 / **$1.28** / $1.76 |
| fplan | 3× 74/74 | 205 / 243 / 282 s | 770k / 1.27M / 1.79M | $0.97 / **$1.46** / $2.00 |

## 兩個重點（其中一個推翻先前結論）

1. **分數穩、成本極不穩**：永遠 74/74，但 token/成本 run-to-run 差 **2.0–2.3×**。
   單次數字不能當定值。
2. **⚠ 修正先前結論：「Codex 先規劃反而更省」是單次抽樣的假象。**
   我先前用「單次 solo $1.39 vs 單次 fplan $0.73」得出「計畫幫 Codex 省最多」。
   但跑 3 次後：**solo 平均 $1.28、fplan 平均 $1.46（fplan 反而略高），範圍大幅重疊**
   → 計畫對 Codex 成本的影響**落在雜訊內，無法斷定有幫助**。先前那個 $0.73 只是運氣好的低抽樣
   （甚至比這輪 3 次的最低 $0.97 還低）。

## 推論到整張矩陣（誠實揭露）

- 其他模型（Claude/DeepSeek）**每格也只跑 1 次** → 同樣是點估計，也有 run-to-run 變異，
  我只是還沒對它們做 3× 量測。所以**全表的單次成本都應視為「點估計 ± 不小的誤差」**，
  尤其 agentic 探索型（Codex）誤差最大。
- 穩健的比較應該用「多次中位數 + 範圍」。若要出版級數字，每格至少跑 3–5 次。
- **不變的結論**：分數（正確率）在所有重跑都穩定 74/74；便宜模型（DeepSeek/Haiku）
  的**單價**優勢是數量級差距，不會被 2× 變異吃掉；會被變異吃掉的是 Codex「先規劃更省」這種
  小幅差異。

## 第七輪（續）：便宜模型變異（Haiku / DeepSeek Flash 各 solo 3 次）

| 模型 | 時間 min/mean/max | token max/min | 成本 min/mean/max | 分數 |
| --- | --- | --- | --- | --- |
| DeepSeek Flash | 117/132/158s | 1.63× | $0.032 / **$0.041** / $0.050 | 3× 74/74 |
| Haiku 4.5 | 91/122/158s | 1.66× | $0.214 / **$0.253** / $0.277 | 3× 74/74 |
| （對照）Codex | 205/243/279s | 1.97× | $0.95 / **$1.28** / $1.76 | 3× 74/74 |

**三個層級的成本帶完全不重疊** —— Flash $0.03–0.05 ≪ Haiku $0.21–0.28 ≪ Codex $0.95–1.76，
相鄰層之間即使各取最壞情況仍差 ~3–4×。結論：**跨層（模型單價）排序是變異吃不掉的**；
變異只能吃掉同層內的小差異（如「Codex 先規劃省一點」）。

**更正**：矩陣裡 DeepSeek Flash solo 的 **$0.080 是單次高抽樣**；n=3 平均其實只有 **$0.041**
（range $0.032–0.050，因 DeepSeek 每次 cache 命中差異大）。Flash 比先前報的更便宜。
所有單次成本都應讀作「點估計 ± 誤差」，但跨層結論不受影響。
