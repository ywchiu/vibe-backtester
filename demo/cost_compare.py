#!/usr/bin/env python3
"""多模型工作流成本比較（全部用 API 費用估，含 Codex review）。

用法：把每個階段的 input/output token 數填進 STAGES（可從 CCR log 或各模型回應的
usage 欄位抓），然後執行：

    python3 demo/cost_compare.py

會印出三種模型組合的總成本與相對 all-Opus 的節省。
Codex(adversarial review) 以 gpt-5.3-codex 的 API 單價計入。
"""

# 每 1M tokens 美元 (input, output) — 示範用價格；正式示範前請更新
PRICING = {
    "opus-4.8":          (5.00, 25.00),
    "deepseek-v4-pro":   (0.435, 0.87),
    "deepseek-v4-flash": (0.09, 0.18),
    "gpt-5.3-codex":     (1.75, 14.00),
}

# 各階段的 token 用量 (input_tokens, output_tokens) — 換成你實際的數字
# 範例值：以 vibe-backtester 新增 RSI 策略為例的粗估
STAGES = {
    "plan":         (40_000,  8_000),   # Opus 讀 codebase 產 plan
    "review_plan":  (30_000,  6_000),   # Codex adversarial review (plan)
    "execute":      (120_000, 30_000),  # 主力施工（多輪）
    "background":   (30_000,  6_000),   # 小修/測試補齊/lint
}

# 三種組合：每個階段指派一個模型。review 在此流程固定走 Codex(gpt-5.3-codex)。
SCENARIOS = {
    "all-opus": {
        "plan": "opus-4.8", "review_plan": "opus-4.8",
        "execute": "opus-4.8", "background": "opus-4.8",
    },
    "opus+v4-pro": {
        "plan": "opus-4.8", "review_plan": "gpt-5.3-codex",
        "execute": "deepseek-v4-pro", "background": "deepseek-v4-pro",
    },
    "opus+v4-flash": {
        "plan": "opus-4.8", "review_plan": "gpt-5.3-codex",
        "execute": "deepseek-v4-flash", "background": "deepseek-v4-flash",
    },
}


def cost(model, tokens):
    pin, pout = PRICING[model]
    tin, tout = tokens
    return tin / 1_000_000 * pin + tout / 1_000_000 * pout


def main():
    base = None
    print(f"{'scenario':14s} {'total':>10s} {'vs all-opus':>14s}")
    print("-" * 42)
    for name, mapping in SCENARIOS.items():
        total = sum(cost(mapping[stage], STAGES[stage]) for stage in STAGES)
        if base is None:
            base = total
        save = (1 - total / base) * 100
        flag = "(baseline)" if name == "all-opus" else f"-{save:.0f}%"
        print(f"{name:14s} ${total:9.3f} {flag:>14s}")
    print()
    print("註：全部以 API 單價計入（含 Codex review = gpt-5.3-codex）。")
    print("token 數請換成 CCR log / 各模型 usage 的實際值，金額才準。")


if __name__ == "__main__":
    main()
