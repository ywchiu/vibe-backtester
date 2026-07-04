"""
共用測試設定（同 level1 慣例）。

受測模組預設 backtester，評分時用 BT_SOLUTION 覆寫，例如：
    BT_SOLUTION=reference_solution python -m pytest
只有 BT_SOLUTION 非預設時，才把 _grader/ 加入 import 路徑。
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

LEVEL = Path(__file__).parent
DATA = LEVEL.parent / "data"
EXPECTED = LEVEL / "expected"

sys.path.insert(0, str(LEVEL))

_SOLUTION = os.environ.get("BT_SOLUTION", "backtester")
# 評分模式：BT_GRADE=1，或 BT_SOLUTION 指向非預設模組。
if _SOLUTION != "backtester" or os.environ.get("BT_GRADE"):
    sys.path.insert(0, str(LEVEL / "_grader"))


@pytest.fixture(scope="session")
def impl():
    """回傳受測模組（預設 backtester，可用 BT_SOLUTION 覆寫）。"""
    return importlib.import_module(_SOLUTION)
