"""
共用測試設定。

受測模組預設為 indicators（受測者實作的檔案）。
評分時可用環境變數 BT_SOLUTION 指定其他模組，例如：

    BT_SOLUTION=reference_solution python -m pytest

只有在 BT_SOLUTION 非預設時，才把 _grader/ 加入 import 路徑，
避免受測者不小心 import 到參考解答。
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

LEVEL1 = Path(__file__).parent
DATA = LEVEL1.parent / "data"
EXPECTED = LEVEL1 / "expected"

sys.path.insert(0, str(LEVEL1))

_SOLUTION = os.environ.get("BT_SOLUTION", "indicators")
# 評分模式：BT_GRADE=1（用受測者交回的 indicators.py 跑 hidden），
# 或 BT_SOLUTION 指向非預設模組（例如 reference_solution 自我檢查）。
if _SOLUTION != "indicators" or os.environ.get("BT_GRADE"):
    sys.path.insert(0, str(LEVEL1 / "_grader"))


@pytest.fixture(scope="session")
def impl():
    """回傳受測模組（預設 indicators，可用 BT_SOLUTION 覆寫）。"""
    return importlib.import_module(_SOLUTION)
