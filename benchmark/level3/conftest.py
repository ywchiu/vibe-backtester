"""把 level3 目錄加入 import 路徑，讓 `import backtester` 可用（public 與 hidden 皆需要）。"""
import sys
from pathlib import Path

LEVEL = Path(__file__).parent
DATA = LEVEL.parent / "data"

sys.path.insert(0, str(LEVEL))
