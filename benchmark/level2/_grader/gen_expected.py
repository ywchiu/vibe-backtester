"""
用參考解答產生 public 期望值（GRADER ONLY）。
輸出：../expected/expected_public.json
用法：python benchmark/level2/_grader/gen_expected.py
"""
import json
from pathlib import Path

import pandas as pd

import reference_solution as ref

LEVEL = Path(__file__).resolve().parents[1]
DATA = LEVEL.parent / "data"
EXPECTED = LEVEL / "expected"


def main():
    prices = pd.read_csv(DATA / "sample_ohlcv_a.csv")
    signals = pd.read_csv(DATA / "signals_a.csv")["Signal"]
    out = ref.run_backtest(prices, signals)

    expected = {
        "initial_capital": 100_000.0,
        "cost_bps": 10.0,
        "equity_curve": [float(x) for x in out["equity_curve"]],
        "metrics": out["metrics"],
    }
    EXPECTED.mkdir(exist_ok=True)
    path = EXPECTED / "expected_public.json"
    path.write_text(json.dumps(expected, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
