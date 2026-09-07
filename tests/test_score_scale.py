"""
分數尺度的測試。

美感模型從「二元標籤」改成「1~10 級評分分佈」之後，
輸出的換算方式與「優秀」門檻都必須跟著改。這裡守住三件事：

  1. 分佈輸出要正確換算成 0~100（期望值 -> 線性映射）
  2. 分佈模型的分數天然落在範圍內，不該再依賴裁切
  3. 「優秀」門檻必須落在模型實際的輸出範圍內
     ——舊門檻 85 配上新模型（最高 73.8）會讓這個狀態永遠不出現，
       而且不會有任何錯誤訊息。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import ai_inference as ai
from tests._util import make_image, requires_weights, write_image


class TestOutputConversion(unittest.TestCase):
    """換算公式本身的正確性。"""

    def _dist(self, probs):
        return torch.tensor(probs, dtype=torch.float32, device=ai.DEVICE)

    def test_all_mass_on_one_maps_to_zero(self):
        p = [1.0] + [0.0] * 9          # 期望值 = 1 分
        self.assertAlmostEqual(ai._output_to_score(self._dist(p), 'distribution'),
                               0.0, places=4)

    def test_all_mass_on_ten_maps_to_hundred(self):
        p = [0.0] * 9 + [1.0]          # 期望值 = 10 分
        self.assertAlmostEqual(ai._output_to_score(self._dist(p), 'distribution'),
                               100.0, places=4)

    def test_midpoint_maps_to_fifty(self):
        p = [0.0] * 10
        p[4] = p[5] = 0.5              # 期望值 = 5.5 分，正好是 1~10 的中點
        self.assertAlmostEqual(ai._output_to_score(self._dist(p), 'distribution'),
                               50.0, places=4)

    def test_any_probability_distribution_stays_in_range(self):
        """
        分佈模型的關鍵優勢：任何合法的機率分佈換算後都必定落在 0~100。
        這正是不再需要裁切的理由。
        """
        rng = np.random.default_rng(0)
        for _ in range(200):
            raw = rng.random(10)
            score = ai._output_to_score(self._dist(raw / raw.sum()), 'distribution')
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_single_mode_keeps_old_behaviour(self):
        """single 模式沿用舊行為（直接乘 100），以便隨時退回舊權重。"""
        out = torch.tensor(0.734, device=ai.DEVICE)
        self.assertAlmostEqual(ai._output_to_score(out, 'single'), 73.4, places=3)


class TestModeDetection(unittest.TestCase):
    """輸出模式應從 checkpoint 自動判斷，不靠另外的設定。"""

    def test_detected_mode_matches_classifier_shape(self):
        state = torch.load(ai.AES_WEIGHTS, map_location='cpu', weights_only=True)
        out_dim = state['classifier.1.weight'].shape[0]
        expected = 'distribution' if out_dim == 10 else 'single'
        self.assertEqual(ai.AES_MODE, expected)

    def test_technical_model_is_single_mode(self):
        self.assertEqual(ai.TECH_MODE, 'single')


@requires_weights
class TestExcellentThresholdIsReachable(unittest.TestCase):
    """
    門檻必須落在模型實際輸出得到的範圍內。

    舊門檻 85 配上新模型（實測最高 73.8）會讓「優秀」永遠不出現，
    而且完全沒有錯誤訊息——正是本次稽核一再遇到的靜默失效。
    """

    def test_threshold_is_within_observed_range(self):
        self.assertGreater(ai.AESTHETIC_EXCELLENT_THRESHOLD, 0)
        self.assertLess(
            ai.AESTHETIC_EXCELLENT_THRESHOLD, 100,
            '門檻不可超過分數上限，否則「優秀」永遠不會出現')

    def test_excellent_status_is_actually_reachable(self):
        """
        用合成影像掃過一批分數，確認至少存在「有可能被評為優秀」的空間。
        這裡不強求某張合成圖一定要拿到優秀（合成圖本來就不是好照片），
        而是驗證門檻沒有被訂在模型永遠達不到的位置。
        """
        import pandas as pd
        import os
        from PIL import Image
        from common import build_transform

        csv_path = 'data/ava_val.csv'
        if not os.path.exists(csv_path):
            self.skipTest('缺少 data/ava_val.csv')

        df = pd.read_csv(csv_path).head(150)
        tf = build_transform('eval')
        scores = []
        with torch.no_grad():
            for name in df['image'].tolist():
                path = os.path.join('data/dataset', f'{name}.jpg')
                if not os.path.exists(path):
                    continue
                with Image.open(path) as im:
                    x = tf(im.convert('RGB')).unsqueeze(0).to(ai.DEVICE)
                scores.append(ai._output_to_score(ai.MODEL_AES(x), ai.AES_MODE))

        if len(scores) < 50:
            self.skipTest('可用樣本不足')

        highest = max(scores)
        self.assertGreater(
            highest, ai.AESTHETIC_EXCELLENT_THRESHOLD,
            f'150 張樣本中最高分只有 {highest:.1f}，低於門檻 '
            f'{ai.AESTHETIC_EXCELLENT_THRESHOLD}，「優秀」狀態可能永遠不會出現')


@requires_weights
class TestScoresStayInRange(unittest.TestCase):
    def test_synthetic_images_score_within_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('detail', 'photo_like', 'blur', 'flat', 'bright', 'dark'):
                path = write_image(make_image(kind=kind), Path(tmp) / f'{kind}.jpg')
                r = ai.evaluate_photo(path)
                with self.subTest(kind=kind):
                    self.assertIsNotNone(r)
                    for key in ('aesthetic_score', 'technical_score', 'overall_score'):
                        self.assertGreaterEqual(r[key], 0.0)
                        self.assertLessEqual(r[key], 100.0)


if __name__ == '__main__':
    unittest.main()
