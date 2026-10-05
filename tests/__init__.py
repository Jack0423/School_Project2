"""
測試套件。命令列工具在 tools/、訓練與評估在 training/，
這裡把兩個資料夾加進 sys.path，測試才能直接 import 它們（例如 import pick_best）。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("tools", "training"):
    sys.path.append(str(_ROOT / _folder))
