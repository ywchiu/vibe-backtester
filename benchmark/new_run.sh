#!/usr/bin/env bash
# 建立一個隔離的實驗資料夾 runs/<name>/，內含受測者視角的檔案
# （level1/2/3 + data + requirements，但**不含** _grader 與 hidden 測試）。
# 每個模型/方案在自己的資料夾裡作答，彼此不干擾，也碰不到參考解答。
#
# 用法：./new_run.sh <name>      例如 ./new_run.sh opus
set -euo pipefail
cd "$(dirname "$0")"
name="${1:?usage: new_run.sh <name>}"
dest="runs/$name"

rm -rf "$dest"
mkdir -p "$dest"
rsync -a \
  --exclude '_grader' \
  --exclude '__pycache__' \
  --exclude 'test_hidden.py' \
  --exclude 'BUGS.md' \
  level1 level2 level3 data requirements.txt "$dest/"

echo "建立 ${dest} （受測者視角：無 _grader / 無 hidden 測試）"
echo "受測者可在其中執行： python -m pytest level1/tests level2/tests level3/tests -q"
