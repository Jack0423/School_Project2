"""
common/metrics.py 的測試。

指標算錯不會報錯，只會安靜地給出一個看起來合理的錯數字，而且會直接寫進報告。
這裡用手算得出答案的小例子逐項驗證，不需要權重或資料集。
"""
import math
import re
import unittest

import numpy as np

from common import metrics as M
from tests._util import PROJECT_ROOT


class TestDecisionRules(unittest.TestCase):
    """判定方向與嚴格不等式必須與 ai_inference.py 一致，否則報表量的不是系統實際的行為。"""

    def test_threshold_is_strict(self):
        scores = [59.99, 60.0, 60.01]
        self.assertEqual(M.predict_positive(scores, 60, 'below').tolist(), [True, False, False])
        self.assertEqual(M.predict_positive(scores, 60, 'above').tolist(), [False, False, True])

    def test_unknown_direction_raises(self):
        with self.assertRaises(ValueError):
            M.predict_positive([1.0], 0, 'over')

    def test_report_rules_match_ai_inference(self):
        """
        ai_inference.py 若把 < 改成 <=（或反過來），model_report.py 的混淆矩陣
        就會和前台實際的判定差幾張，而且不會有任何錯誤訊息。直接比對兩邊原始碼。
        """
        ai_src = (PROJECT_ROOT / 'ai_inference.py').read_text(encoding='utf-8')
        report_src = (PROJECT_ROOT / 'model_report.py').read_text(encoding='utf-8')
        self.assertRegex(ai_src, r'score_tech\s*<\s*TECH_ISSUE_THRESHOLD',
                         'ai_inference 的警告判定不再是「技術分 < 門檻」，請同步修改 model_report.py')
        self.assertRegex(ai_src, r'score_aes\s*>\s*AESTHETIC_EXCELLENT_THRESHOLD',
                         'ai_inference 的優秀判定不再是「美感分 > 門檻」，請同步修改 model_report.py')
        self.assertRegex(report_src, r"predict_positive\(pred, thr, 'below'\)")
        self.assertRegex(report_src, r"predict_positive\(pred, thr, 'above'\)")


class TestConfusion(unittest.TestCase):

    def test_counts(self):
        pred = [True, True, False, False, True]
        true = [True, False, True, False, True]
        self.assertEqual(M.confusion_counts(pred, true), {'tp': 2, 'fn': 1, 'fp': 1, 'tn': 1})

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            M.confusion_counts([True, False], [True])

    def test_summary_values(self):
        s = M.classification_summary({'tp': 740, 'fn': 161, 'fp': 184, 'tn': 930})
        self.assertAlmostEqual(s['precision'], 740 / 924)
        self.assertAlmostEqual(s['recall'], 740 / 901)
        self.assertAlmostEqual(s['accuracy'], 1670 / 2015)
        self.assertAlmostEqual(s['f1'], 2 * 740 / (2 * 740 + 184 + 161))
        self.assertAlmostEqual(s['positive_rate'], 901 / 2015)

    def test_no_predicted_positive(self):
        """
        完全沒有判定為正時 Precision 無從計算（nan），但 F1 是 0 而不是 nan——
        有真正的正樣本卻一個都沒抓到，這是明確的 0 分。
        若 F1 經由 precision 與 recall 計算，nan 會傳染到 F1。
        """
        s = M.classification_summary({'tp': 0, 'fn': 5, 'fp': 0, 'tn': 10})
        self.assertTrue(math.isnan(s['precision']))
        self.assertEqual(s['recall'], 0.0)
        self.assertEqual(s['f1'], 0.0)

    def test_empty_is_nan_not_zero(self):
        s = M.classification_summary({'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0})
        for key in ('accuracy', 'precision', 'recall', 'f1'):
            self.assertTrue(math.isnan(s[key]), key)

    def test_sweep_matches_single_threshold(self):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 100, 500)
        truth = rng.uniform(0, 100, 500) < 60
        sw = M.threshold_sweep(scores, truth, [40, 60], 'below')
        single = M.classification_summary(
            M.confusion_counts(M.predict_positive(scores, 60, 'below'), truth))
        self.assertAlmostEqual(sw['precision'][1], single['precision'])
        self.assertAlmostEqual(sw['recall'][1], single['recall'])
        self.assertEqual(sw['predicted_positive'][1], int((scores < 60).sum()))


class TestRoc(unittest.TestCase):

    @staticmethod
    def pairwise_auc(scores, labels):
        """AUC 的定義：隨機抽一個正樣本與一個負樣本，正樣本分數較高的機率（同分算一半）。"""
        pos = scores[labels]
        neg = scores[~labels]
        greater = (pos[:, None] > neg[None, :]).sum()
        ties = (pos[:, None] == neg[None, :]).sum()
        return (greater + 0.5 * ties) / (len(pos) * len(neg))

    def test_perfect_and_reversed(self):
        labels = np.array([0, 0, 1, 1], dtype=bool)
        fpr, tpr = M.roc_curve([0.1, 0.2, 0.8, 0.9], labels)
        self.assertAlmostEqual(M.auc_from_roc(fpr, tpr), 1.0)
        fpr, tpr = M.roc_curve([0.9, 0.8, 0.2, 0.1], labels)
        self.assertAlmostEqual(M.auc_from_roc(fpr, tpr), 0.0)

    def test_all_tied_is_half(self):
        """同分的樣本必須一起跨過門檻；逐一跨過的話，AUC 取決於資料排列順序而不是 0.5。"""
        labels = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
        fpr, tpr = M.roc_curve([5.0] * 6, labels)
        self.assertAlmostEqual(M.auc_from_roc(fpr, tpr), 0.5)

    def test_matches_pairwise_definition_with_ties(self):
        rng = np.random.default_rng(1)
        scores = rng.integers(0, 8, 300).astype(float)   # 刻意製造大量同分
        labels = rng.uniform(size=300) < scores / 10
        fpr, tpr = M.roc_curve(scores, labels)
        self.assertAlmostEqual(M.auc_from_roc(fpr, tpr), self.pairwise_auc(scores, labels))

    def test_single_class_raises(self):
        with self.assertRaises(ValueError):
            M.roc_curve([0.1, 0.2], [True, True])


class TestRegressionAndEmd(unittest.TestCase):

    def test_regression_summary(self):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        s = M.regression_summary(true * 2 + 1, true)
        self.assertAlmostEqual(s['plcc'], 1.0)
        self.assertAlmostEqual(s['srcc'], 1.0)
        self.assertAlmostEqual(s['mean_error'], np.mean(true + 1))   # 預測 - 真實，正值 = 高估
        s = M.regression_summary(true, true)
        self.assertEqual(s['rmse'], 0.0)
        self.assertEqual(s['mae'], 0.0)

    def test_emd_hand_computed(self):
        a = np.zeros((1, 10))
        a[0, 0] = 1.0   # 全部投給 1 分
        b = np.zeros((1, 10))
        b[0, 9] = 1.0   # 全部投給 10 分
        # CDF 差：第 1~9 格為 1，第 10 格為 0 -> sqrt(9/10)
        self.assertAlmostEqual(float(M.emd_per_sample(a, b)[0]), math.sqrt(0.9))
        self.assertAlmostEqual(float(M.emd_per_sample(b, b)[0]), 0.0)

    def test_emd_relation_to_training_loss(self):
        """
        文件上說報表的 EMD（論文定義）與訓練用的 emd_loss 差在開根號與 batch 平均。
        驗證這個說法：emd_loss 應等於逐張 EMD 平方的平均。
        """
        import torch
        from common import emd_loss

        rng = np.random.default_rng(2)
        p = rng.dirichlet(np.ones(10), size=8)
        t = rng.dirichlet(np.ones(10), size=8)
        report = M.emd_per_sample(p, t)
        train = emd_loss(torch.tensor(p), torch.tensor(t)).item()
        self.assertAlmostEqual(train, float(np.mean(report ** 2)), places=10)

    def test_binned_error_edges(self):
        true = np.array([0.0, 50.0, 99.0, 100.0])
        pred = true + np.array([1.0, 2.0, 3.0, 5.0])
        rows = M.binned_error(pred, true, [0, 50, 80, 100])
        self.assertEqual([r['count'] for r in rows], [1, 1, 2])   # 100 算進最後一格
        self.assertAlmostEqual(rows[2]['mean_error'], 4.0)
        rows = M.binned_error(pred, true, [0, 50, 60, 80, 100])
        self.assertEqual(rows[2]['count'], 0)
        self.assertTrue(math.isnan(rows[2]['mean_error']))


if __name__ == '__main__':
    unittest.main()
