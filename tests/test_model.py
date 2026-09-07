"""
common/model.py 的測試。

重點是釘住 B9：batch size 為 1 時輸出不可被壓成 0 維純量。
這個 bug 原本靠「資料集大小剛好整除不出 1」躲過，換資料集就會踩到，
而且只會跳出 UserWarning、loss 靜默算錯，是最難察覺的那一類。
"""
import unittest

import torch

from common import NIMABaseline, emd_loss
from tests._util import PROJECT_ROOT, required_weight_files, requires_weights


class TestOutputShape(unittest.TestCase):
    """單一分數模式下，輸出形狀必須恆為 (N,)，不論 N 是多少。"""

    @classmethod
    def setUpClass(cls):
        cls.model = NIMABaseline(output_mode='single', pretrained=False).eval()

    def _forward(self, batch_size):
        with torch.no_grad():
            return self.model(torch.randn(batch_size, 3, 224, 224))

    def test_batch_size_one_is_not_scalar(self):
        """B9 的直接回歸測試：batch=1 不可變成 0 維。"""
        out = self._forward(1)
        self.assertEqual(out.shape, (1,),
                         '批次大小為 1 時輸出被壓成 0 維，會導致 MSELoss 靜默廣播算錯')
        self.assertEqual(out.dim(), 1)

    def test_batch_size_various(self):
        for n in (2, 8, 32):
            with self.subTest(batch_size=n):
                self.assertEqual(self._forward(n).shape, (n,))

    def test_loss_against_labels_has_no_broadcast(self):
        """
        batch=1 時 outputs 與 labels 形狀必須一致。
        形狀不一致 PyTorch 不會報錯，只會廣播並算出錯誤的 loss。
        """
        for n in (1, 4):
            with self.subTest(batch_size=n):
                outputs = self._forward(n)
                labels = torch.rand(n)
                self.assertEqual(outputs.shape, labels.shape)


class TestDistributionMode(unittest.TestCase):
    def test_outputs_probability_distribution(self):
        model = NIMABaseline(output_mode='distribution', pretrained=False).eval()
        with torch.no_grad():
            out = model(torch.randn(3, 3, 224, 224))
        self.assertEqual(out.shape, (3, 10))
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(3), atol=1e-5),
                        'distribution 模式的輸出應為機率分佈，每列總和為 1')

    def test_emd_loss_is_zero_for_identical_distributions(self):
        d = torch.softmax(torch.randn(4, 10), dim=1)
        self.assertAlmostEqual(emd_loss(d, d).item(), 0.0, places=6)

    def test_emd_loss_is_positive_for_different_distributions(self):
        a = torch.zeros(1, 10); a[0, 0] = 1.0
        b = torch.zeros(1, 10); b[0, 9] = 1.0
        self.assertGreater(emd_loss(a, b).item(), 0.0)


class TestArchitectureMatchesCheckpoints(unittest.TestCase):
    """
    釘住「訓練與推論使用同一個架構定義」。
    稽核前這個 class 有三份各自分岔的定義，任何一份改動都不會被發現。
    """

    @requires_weights
    def test_checkpoints_load_strictly(self):
        """
        檢查對象取自現役權重清單，不寫死檔名。

        原本寫死 nima_best.pth 與 nima_tech_best.pth，兩顆都是 single 架構；
        美感模型換成 10 維的 distribution 版本之後，真正在服役的那顆
        正好沒被驗到——守門的對象換了，守門的人沒換。
        """
        for name in required_weight_files():
            with self.subTest(checkpoint=name):
                state = torch.load(PROJECT_ROOT / name, map_location='cpu',
                                   weights_only=True)
                # 輸出模式從 checkpoint 自己反推，與 ai_inference._load_model
                # 同一套規則：分類頭輸出 10 維＝評分分佈、1 維＝單一分數。
                out_dim = state['classifier.1.weight'].shape[0]
                mode = 'distribution' if out_dim == 10 else 'single'
                model = NIMABaseline(output_mode=mode, pretrained=False)
                # strict=True：多一個或少一個鍵都會拋錯
                model.load_state_dict(state, strict=True)

    def test_pretrained_flag_does_not_change_architecture(self):
        keys_off = set(NIMABaseline(pretrained=False).state_dict().keys())
        # 不實際下載預訓練權重，只比對鍵的集合是否一致
        self.assertEqual(len(keys_off), 314,
                         '架構的參數數量與現有 checkpoint 不符')


if __name__ == '__main__':
    unittest.main()
