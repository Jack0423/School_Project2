#!/bin/bash
# Mac 上雙擊執行：檢查環境、缺套件時問要不要安裝，然後開啟照片管理系統
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
    echo "找不到 Python。請到 https://www.python.org/downloads/ 下載 3.10 以上版本，裝好後再雙擊這個檔案。"
    read -r -p "按 Enter 關閉"
    exit 1
fi
python3 start.py "$@"
status=$?
if [ $status -ne 0 ]; then
    read -r -p "按 Enter 關閉"
fi
exit $status
