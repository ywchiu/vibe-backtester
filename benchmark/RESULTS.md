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
