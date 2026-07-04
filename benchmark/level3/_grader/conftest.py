"""把 _grader 目錄加入路徑，讓 hidden 測試能 import reference_engine / reference_metrics。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
