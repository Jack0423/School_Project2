"""
common/transforms.py 的測試。

重點是釘住 B7：推論與驗證必須使用完全相同的前處理。
稽核前推論用 Resize((224,224)) 壓扁長寬比、驗證用 Resize(256)+CenterCrop(224)，
兩者輸入分佈不同，實測讓技術模型的 SRCC 少了 0.0317。
這種偏差不會有任何錯誤訊息，只會讓線上表現默默低於驗證集數字。
"""
import unittest

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from common import build_transform
from common.transforms import INPUT_SIZE, RESIZE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class TestEvalTransform(unittest.TestCase):
    def setUp(self):
        self.tf = build_transform('eval')

    def test_output_is_correct_tensor(self):
        img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        out = self.tf(img)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, INPUT_SIZE, INPUT_SIZE))

    def test_preserves_aspect_ratio_then_crops(self):
        """
        驗證是「等比例縮放後裁切」而非「直接壓扁」。
        做法：同一個圓形圖案分別放進橫幅與直幅畫布，
        壓扁會把圓變成橢圓且兩者變形方向相反；等比例縮放則不會。
        """
        def circle(w, h):
            yy, xx = np.mgrid[0:h, 0:w]
            r = min(w, h) * 0.3
            mask = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < r ** 2
            return Image.fromarray((mask * 255).astype(np.uint8)).convert('RGB')

        wide = self.tf(circle(800, 400))[0]
        tall = self.tf(circle(400, 800))[0]

        def extent(t):
            ys, xs = torch.nonzero(t > t.mean(), as_tuple=True)
            if len(xs) == 0:
                return 0.0
            return (xs.max() - xs.min()).item() / max((ys.max() - ys.min()).item(), 1)

        # 等比例縮放 + 置中裁切下，圓在兩種畫布上的長寬比應該接近
        self.assertAlmostEqual(extent(wide), extent(tall), delta=0.35,
                               msg='橫幅與直幅的圖形比例差異過大，前處理疑似直接壓扁長寬比')

    def test_composition_is_resize_then_centercrop(self):
        """
        直接檢查組成，確保沒有人把它改回 Resize((224,224))。
        """
        kinds = [type(t).__name__ for t in self.tf.transforms]
        self.assertEqual(kinds, ['Resize', 'CenterCrop', 'ToTensor', 'Normalize'])
        self.assertEqual(self.tf.transforms[0].size, RESIZE_SIZE,
                         'Resize 必須傳入單一整數（縮放短邊、保持長寬比），'
                         '傳入 tuple 會直接壓扁')

    def test_normalization_uses_imagenet_stats(self):
        norm = self.tf.transforms[-1]
        self.assertEqual(list(norm.mean), IMAGENET_MEAN)
        self.assertEqual(list(norm.std), IMAGENET_STD)


class TestTrainTransforms(unittest.TestCase):
    def test_all_modes_produce_correct_shape(self):
        img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        for mode in ('eval', 'train_aesthetic', 'train_technical'):
            with self.subTest(mode=mode):
                self.assertEqual(build_transform(mode)(img).shape,
                                 (3, INPUT_SIZE, INPUT_SIZE))

    def test_technical_training_avoids_color_jitter(self):
        """
        技術品質評估對色彩與對比很敏感，色彩抖動會把要偵測的失真洗掉。
        """
        kinds = [type(t).__name__ for t in build_transform('train_technical').transforms]
        self.assertNotIn('ColorJitter', kinds)
        self.assertIn('RandomCrop', kinds)
        self.assertNotIn('RandomResizedCrop', kinds,
                         '技術模型應用 RandomCrop 保留原始像素清晰度')

    def test_aesthetic_training_uses_stronger_augmentation(self):
        kinds = [type(t).__name__ for t in build_transform('train_aesthetic').transforms]
        self.assertIn('RandomResizedCrop', kinds)
        self.assertIn('ColorJitter', kinds)

    def test_unknown_mode_raises_with_helpful_message(self):
        with self.assertRaises(ValueError) as ctx:
            build_transform('nonexistent')
        self.assertIn('nonexistent', str(ctx.exception))


class TestInferenceMatchesEval(unittest.TestCase):
    """B7 的核心保證：推論端用的就是 eval 前處理，不是另一份。"""

    def test_ai_inference_transform_is_eval_transform(self):
        import ai_inference
        expected = [type(t).__name__ for t in build_transform('eval').transforms]
        actual = [type(t).__name__ for t in ai_inference.TRANSFORM.transforms]
        self.assertEqual(actual, expected,
                         '推論前處理與驗證前處理不一致，線上分數會系統性偏離驗證指標')


if __name__ == '__main__':
    unittest.main()
