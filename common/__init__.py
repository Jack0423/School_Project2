"""
專案共用模組。

抽出這個套件是為了消除三處重複且已經不一致的定義：
    model.py      NIMABaseline 原本在 3 個檔案各有一份，內容不同
    transforms.py 前處理設定原本散在 7 處，且訓練與推論不一致
    engine.py     run_epoch 原本在 2 個訓練腳本各有一份
"""
from common.model import NIMABaseline, emd_loss
from common.transforms import build_transform
from common.engine import run_epoch, to_mean_score

__all__ = [
    'NIMABaseline',
    'emd_loss',
    'build_transform',
    'run_epoch',
    'to_mean_score',
]
