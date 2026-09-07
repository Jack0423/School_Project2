"""
訓練資料集載入的測試。

守的是一個真實踩到的缺陷：訓練跑完 5 個 epoch、印出看似合理的 loss，
但實際上 5,600 張影像**全部都是全黑圖**。

成因有兩層，兩層都要有測試守住：
  1. df.iloc[i] 會把整列轉成共同型別。CSV 一旦同時含整數欄與浮點欄，
     整數的 image 欄會變成 float64，檔名就成了 "53.0.jpg" 而全部找不到。
  2. 找不到檔案時，原本的程式碼靜默塞一張全黑圖代替，
     於是第 1 點完全沒有任何徵兆。
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common import build_transform
from tests._util import make_image, write_image


class _Fixture:
    """建立一個小型的假資料集（3 張可辨識的影像 + CSV）。"""

    def __init__(self, tmpdir, extra_float_columns):
        self.dir = Path(tmpdir)
        self.img_dir = self.dir / 'images'
        self.img_dir.mkdir()
        self.ids = [53, 106, 227]
        for n, i in enumerate(self.ids):
            # 每張亮度不同，方便驗證真的讀到了不同的影像
            arr = np.full((64, 64, 3), 40 + n * 80, dtype=np.uint8)
            write_image(arr, self.img_dir / f'{i}.jpg')

        header = ['image', 'label']
        rows = [[i, n % 2] for n, i in enumerate(self.ids)]
        if extra_float_columns:
            # 這正是 ava_full.csv 的形狀：整數欄 + 浮點欄混在一起
            header += ['mean_score'] + [f'score_{k}' for k in range(1, 11)]
            for r in rows:
                r += [5.5] + [0.1] * 10

        self.csv = self.dir / 'labels.csv'
        self.csv.write_text(
            '\n'.join([','.join(header)] + [','.join(str(v) for v in r) for r in rows]),
            encoding='utf-8')


class TestFilenameResolution(unittest.TestCase):
    """成因 1：檔名不可受其他欄位的型別影響。"""

    def _load_three(self, extra_float_columns):
        from train_nima import AVADataset
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp, extra_float_columns)
            ds = AVADataset(str(fx.csv), str(fx.img_dir),
                            build_transform('eval'), mode='single')
            return torch.stack([ds[i][0] for i in range(3)])

    @staticmethod
    def _pairwise_difference(imgs):
        """
        比較「影像之間」的差異，而不是單一張的全域標準差。

        用 imgs.std() 是不夠的：全黑影像經過 Normalize 之後，
        三個通道的常數值不同（-2.118 / -2.036 / -1.804），
        全域標準差約 0.13，足以讓門檻很低的斷言誤判通過。
        （這個盲點是靠變異測試發現的。）
        """
        n = len(imgs)
        return min((imgs[i] - imgs[j]).abs().mean().item()
                   for i in range(n) for j in range(i + 1, n))

    def test_integer_only_csv_loads_distinct_images(self):
        imgs = self._load_three(extra_float_columns=False)
        self.assertGreater(self._pairwise_difference(imgs), 0.05)

    def test_csv_with_float_columns_still_loads_distinct_images(self):
        """
        這是實際踩到的情境：加上 mean_score 與 score_1~10 之後，
        image 欄被轉成 float64，檔名變成 "53.0.jpg"。
        """
        imgs = self._load_three(extra_float_columns=True)
        self.assertGreater(
            self._pairwise_difference(imgs), 0.05,
            '含浮點欄位的 CSV 讀出來的影像彼此相同，'
            '代表檔名又被浮點化成 "53.0.jpg" 而讀不到檔案')

    def test_both_csv_shapes_give_identical_images(self):
        """兩種 CSV 形狀指向同一批影像，讀出來必須完全一樣。"""
        a = self._load_three(extra_float_columns=False)
        b = self._load_three(extra_float_columns=True)
        self.assertTrue(torch.allclose(a, b))

    def test_filenames_have_no_decimal_point(self):
        from train_nima import AVADataset
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp, extra_float_columns=True)
            ds = AVADataset(str(fx.csv), str(fx.img_dir), None, mode='single')
            for name in ds.filenames:
                self.assertNotIn('.0.jpg', name, f'檔名被浮點化：{name}')


class TestMissingImageFailsLoudly(unittest.TestCase):
    """成因 2：讀不到檔案必須拋錯，不可用全黑圖靜默代替。"""

    def test_avadataset_raises_on_missing_file(self):
        from train_nima import AVADataset
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp, extra_float_columns=False)
            os.remove(fx.img_dir / '106.jpg')
            ds = AVADataset(str(fx.csv), str(fx.img_dir),
                            build_transform('eval'), mode='single')
            with self.assertRaises(RuntimeError) as ctx:
                _ = ds[1]
        self.assertIn('106.jpg', str(ctx.exception))

    def test_koniqdataset_raises_on_missing_file(self):
        from train_tech import KonIQDataset
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp, extra_float_columns=False)
            os.remove(fx.img_dir / '53.jpg')
            ds = KonIQDataset(str(fx.csv), str(fx.img_dir), build_transform('eval'))
            with self.assertRaises(RuntimeError):
                _ = ds[0]

    def test_error_message_is_actionable(self):
        from train_nima import AVADataset
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp, extra_float_columns=False)
            os.remove(fx.img_dir / '53.jpg')
            ds = AVADataset(str(fx.csv), str(fx.img_dir), None, mode='single')
            with self.assertRaises(RuntimeError) as ctx:
                _ = ds[0]
        message = str(ctx.exception)
        self.assertIn('無法讀取訓練影像', message)
        self.assertIn(str(fx.img_dir), message, '訊息應指出是在哪個資料夾找不到')


class TestDistributionMode(unittest.TestCase):
    def test_distribution_labels_are_normalised(self):
        from train_nima import AVADataset
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp, extra_float_columns=True)
            ds = AVADataset(str(fx.csv), str(fx.img_dir),
                            build_transform('eval'), mode='distribution')
            _, label = ds[0]
        self.assertEqual(label.shape, (10,))
        self.assertAlmostEqual(label.sum().item(), 1.0, places=5)

    def test_single_mode_label_is_scalar(self):
        from train_nima import AVADataset
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp, extra_float_columns=True)
            ds = AVADataset(str(fx.csv), str(fx.img_dir),
                            build_transform('eval'), mode='single')
            _, label = ds[0]
        self.assertEqual(label.dim(), 0)


class TestRealDatasetFiles(unittest.TestCase):
    """對實際的標籤檔做冒煙測試（檔案不存在時跳過）。"""

    def _check(self, csv_path, img_dir, mode):
        if not (os.path.exists(csv_path) and os.path.isdir(img_dir)):
            self.skipTest(f'缺少 {csv_path} 或 {img_dir}')
        from train_nima import AVADataset
        ds = AVADataset(csv_path, img_dir, build_transform('eval'), mode=mode)
        imgs = torch.stack([ds[i][0] for i in range(min(8, len(ds)))])
        n = len(imgs)
        pairwise = min((imgs[i] - imgs[j]).abs().mean().item()
                       for i in range(n) for j in range(i + 1, n))
        self.assertGreater(pairwise, 0.05,
                           f'{csv_path} 讀出來的影像彼此相同，疑似全部讀不到檔案')

    def test_legacy_labels_load(self):
        self._check('data/val.csv', 'data/dataset', 'single')

    def test_rebuilt_labels_load_single(self):
        self._check('data/ava_val.csv', 'data/dataset', 'single')

    def test_rebuilt_labels_load_distribution(self):
        self._check('data/ava_val.csv', 'data/dataset', 'distribution')


if __name__ == '__main__':
    unittest.main()
