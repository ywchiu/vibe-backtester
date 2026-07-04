#!/usr/bin/env bash
# 評分某個實驗資料夾 runs/<name>/：把受測者的解答覆蓋到一份含 _grader 的
# 乾淨評分樹，跑 public + hidden 測試。受測者本人碰不到 hidden，評分才有意義。
#
# 用法：./grade.sh <name>
set -uo pipefail
cd "$(dirname "$0")"
name="${1:?usage: grade.sh <name>}"
run="runs/$name"
PY="$(pwd)/.venv/bin/python"

G="$(mktemp -d)"
cp -R level1 level2 level3 data requirements.txt "$G/"

# 覆蓋受測者的解答檔
[ -f "$run/level1/indicators.py" ] && cp "$run/level1/indicators.py" "$G/level1/indicators.py"
[ -f "$run/level2/backtester.py" ] && cp "$run/level2/backtester.py" "$G/level2/backtester.py"
[ -d "$run/level3/backtester" ] && cp -R "$run/level3/backtester/." "$G/level3/backtester/"
find "$G" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# 從評分樹內部以相對路徑執行，pytest 才會載入各 level 的 conftest。
cd "$G"
echo "############ 評分：$name ############"
echo "==== Level 1 (public + hidden) ===="
BT_GRADE=1 "$PY" -m pytest level1/tests -q -rN 2>&1 | tail -1
echo "==== Level 2 (public + hidden) ===="
BT_GRADE=1 "$PY" -m pytest level2/tests -q -rN 2>&1 | tail -1
echo "==== Level 3 (public) ===="
"$PY" -m pytest level3/tests -q -rN 2>&1 | tail -1
echo "==== Level 3 (hidden) ===="
"$PY" -m pytest level3/_grader/test_hidden.py -q -rN 2>&1 | tail -1

cd /
rm -rf "$G"
