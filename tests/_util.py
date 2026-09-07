"""
測試共用工具。

設計原則：**資料不存在時「跳過」而非「失敗」**。
data/ 與 *.pth 都在 .gitignore 內，換一台機器 clone 下來不會有這些檔案。
若測試在這種情況直接失敗，整份測試就會變成「反正一定紅的」而被忽略，
失去意義。因此需要真實資料的測試一律標記 skip 並說明缺什麼。
"""
import os
import re
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHOTO_DIR = PROJECT_ROOT / 'data' / 'my_photos'
DATASET_DIR = PROJECT_ROOT / 'data' / 'dataset'


def required_weight_files():
    """
    回傳 ai_inference.py 目前實際使用的權重檔名，例如
    ['nima_aes_dist.pth', 'nima_tech_best.pth']。

    為什麼用「讀原始碼」這種看起來很繞的做法：

      不能寫死清單 —— 已經出過事。美感模型從二元版換成 nima_aes_dist.pth 之後，
      這裡仍在檢查舊的 nima_best.pth。後果不是測試失敗，而是測試**靜默跳過**：
      在只拿到現役兩顆權重的機器上 weights_available() 會回 False，
      所有掛 @requires_weights 的測試全部標記 skip，
      測試跑起來是綠的，但其實一項都沒執行。
      （check_env.py 的註解記載了同一類錯誤在它那邊也發生過一次。）

      不能 import ai_inference —— 那會連帶載入兩顆模型；而這個函式必須在
      「權重根本不存在」的情況下也能正常回答，否則就自相矛盾了。

    因此改成解析 ai_inference.py 的原始碼。解析不到剛好兩個檔名時直接拋錯，
    絕不回傳空清單——空清單會讓 weights_available() 變成恆真，
    那又會退化成另一種靜默失效。
    """
    source = (PROJECT_ROOT / 'ai_inference.py').read_text(encoding='utf-8')
    names = re.findall(
        r"^(?:AES|TECH)_WEIGHTS\s*=\s*BASE_DIR\s*/\s*['\"]([^'\"]+)['\"]",
        source, re.MULTILINE)
    if len(names) != 2:
        raise RuntimeError(
            f'無法從 ai_inference.py 解析出權重檔名（找到 {names}）'
            f'｜AES_WEIGHTS / TECH_WEIGHTS 的寫法可能改了，'
            f'請同步更新 tests/_util.required_weight_files()')
    return names


def weights_available():
    """現役權重是否齊備。清單一律來自 required_weight_files()，不自己維護一份。"""
    return all((PROJECT_ROOT / name).exists() for name in required_weight_files())


def find_photo(*suffixes):
    """在 data/my_photos 找出第一個符合副檔名的檔案，找不到回傳 None。"""
    if not PHOTO_DIR.is_dir():
        return None
    for p in sorted(PHOTO_DIR.iterdir()):
        if p.suffix.lower() in suffixes:
            return p
    return None


def rawpy_available():
    try:
        import rawpy  # noqa: F401
        return True
    except ImportError:
        return False


requires_weights = unittest.skipUnless(
    weights_available(),
    '缺少現役權重：' + ' / '.join(required_weight_files()))

requires_rawpy = unittest.skipUnless(
    rawpy_available(), '未安裝 rawpy（pip install rawpy）')


def make_image(width=640, height=480, kind='detail', seed=0):
    """
    產生合成測試影像，回傳 RGB uint8 numpy array。

    用合成影像而非真實照片，是為了讓這些測試不依賴 .gitignore 掉的資料，
    並且能精確控制要測的性質（銳利／模糊／過曝／死黑／雜訊）。

    kind:
        'detail'  高頻棋盤格，清晰度指標很高
        'blur'    對 detail 做強模糊，清晰度指標很低
        'flat'    純灰平面，對比極低
        'bright'  幾乎全白，死白比例高
        'dark'    幾乎全黑，死黑比例高
        'noise'   中灰底加上強雜訊

    注意 'detail' 與 'blur' 刻意使用中間調（64 / 192）而非純黑白：
    用 0 / 255 的話這張圖同時有 50% 死黑與 50% 死白，
    測「銳利度」時會連帶觸發曝光判斷，把測試的意圖搞混。
    """
    rng = np.random.default_rng(seed)
    _DARK, _LIGHT = 64, 192

    def _checkerboard():
        yy, xx = np.mgrid[0:height, 0:width]
        pattern = ((xx // 2) + (yy // 2)) % 2
        return np.where(pattern == 1, _LIGHT, _DARK).astype(np.uint8)

    if kind == 'detail':
        base = _checkerboard()
    elif kind == 'photo_like':
        # 銳利度接近真實照片的影像。
        #
        # 'detail' 那種未經模糊的棋盤格清晰度指標高達 65,000，
        # 遠高於真實照片（實測 AVA 樣本多在 100~3,000），
        # 銳利到即使被放大 4 倍仍遠高於門檻，反而測不出「放大導致誤判模糊」。
        # 這裡刻意做成原生約 470、放大到 800 寬後約 28，兩邊都離門檻 100 有餘裕。
        import cv2
        yy, xx = np.mgrid[0:height, 0:width]
        pattern = ((xx // 8) + (yy // 8)) % 2
        base = np.where(pattern == 1, _LIGHT, _DARK).astype(np.uint8)
        base = cv2.GaussianBlur(base, (5, 5), 0)
    elif kind == 'blur':
        import cv2
        base = cv2.GaussianBlur(_checkerboard(), (31, 31), 12)
    elif kind == 'flat':
        base = np.full((height, width), 128, dtype=np.uint8)
    elif kind == 'bright':
        base = np.full((height, width), 254, dtype=np.uint8)
    elif kind == 'dark':
        base = np.full((height, width), 1, dtype=np.uint8)
    elif kind == 'noise':
        base = np.clip(128 + rng.normal(0, 40, (height, width)), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f'未知的合成影像類型：{kind}')

    return np.dstack([base, base, base])


def write_image(array, path):
    Image.fromarray(array).save(path)
    return str(path)
