"""
evaluate_photo() 的行為測試。

這個函式是模型端對外交付的介面（前台唯一會呼叫的東西），
所以除了正確性，也要釘住它的「回傳格式合約」——
前台依賴 dict 的鍵名與型別，任何變動都會直接弄壞介面。
"""
import tempfile
import unittest
from pathlib import Path

import ai_inference as ai
from tests._util import (find_photo, make_image, requires_rawpy,
                         requires_weights, write_image)


@requires_weights
class TestReturnContract(unittest.TestCase):
    """回傳格式合約。前台依賴這些鍵，不可隨意更動。"""

    EXPECTED = {
        'aesthetic_score': float,
        'technical_score': float,
        'overall_score': float,
        'status': str,
        'suggestion': str,
        'technical_issues': list,
        # 本次採用的權重。綜合分寫進資料庫後，沒有這兩個欄位就無法重現，
        # 也分不出兩筆分數是不是在同一組權重下算的。
        'aesthetic_weight': float,
        'technical_weight': float,
    }

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = write_image(make_image(640, 480), Path(cls.tmp.name) / 'p.jpg')
        cls.result = ai.evaluate_photo(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_returns_dict_with_expected_keys(self):
        self.assertIsInstance(self.result, dict)
        self.assertEqual(set(self.result.keys()), set(self.EXPECTED.keys()))

    def test_value_types(self):
        for key, typ in self.EXPECTED.items():
            with self.subTest(key=key):
                self.assertIsInstance(self.result[key], typ)

    def test_technical_issues_are_strings(self):
        for issue in self.result['technical_issues']:
            self.assertIsInstance(issue, str)

    def test_status_is_one_of_known_values(self):
        self.assertIn(self.result['status'], {'正常', '警告', '優秀'})


@requires_weights
class TestScoreRange(unittest.TestCase):
    """
    A5 的回歸測試：分數必須落在 0–100。

    美感模型是拿二元標籤做 MSE 回歸、輸出層沒有 sigmoid，
    原始輸出實測範圍為 -23.6 ~ 145.1，必須經過裁切才對外。
    """

    def test_scores_within_range_for_varied_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('detail', 'blur', 'flat', 'bright', 'dark', 'noise'):
                path = write_image(make_image(kind=kind), Path(tmp) / f'{kind}.jpg')
                result = ai.evaluate_photo(path)
                with self.subTest(kind=kind):
                    self.assertIsNotNone(result)
                    for key in ('aesthetic_score', 'technical_score', 'overall_score'):
                        value = result[key]
                        self.assertGreaterEqual(value, 0.0, f'{kind} 的 {key} 為負值')
                        self.assertLessEqual(value, 100.0, f'{kind} 的 {key} 超過 100')

    def test_clamp_helper_bounds_values(self):
        self.assertEqual(ai._clamp_score(-23.6), 0.0)
        self.assertEqual(ai._clamp_score(145.1), 100.0)
        self.assertEqual(ai._clamp_score(50.0), 50.0)

    def test_overall_is_weighted_sum_of_clamped_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(make_image(kind='detail'), Path(tmp) / 'x.jpg')
            r = ai.evaluate_photo(path)
        expected = (r['aesthetic_score'] * ai.AESTHETIC_WEIGHT
                    + r['technical_score'] * ai.TECHNICAL_WEIGHT)
        self.assertAlmostEqual(r['overall_score'], expected, places=1)


@requires_weights
class TestWeightParameter(unittest.TestCase):
    """
    綜合分權重可逐張指定（期末報告承諾的「不死綁 6:4」）。

    設計成參數而非全域變數，是為了讓下一階段的多執行緒批次分析不會
    在跑到一半時被滑桿改掉權重、產出混著兩種權重的結果。
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = write_image(make_image(640, 480, kind='detail'),
                               Path(cls.tmp.name) / 'w.jpg')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_default_matches_module_constants(self):
        r = ai.evaluate_photo(self.path)
        self.assertAlmostEqual(r['aesthetic_weight'], ai.AESTHETIC_WEIGHT)
        self.assertAlmostEqual(r['technical_weight'], ai.TECHNICAL_WEIGHT)

    def test_single_weight_fills_in_the_other(self):
        """滑桿只會給一個值，另一個必須自動補成 1 - x。"""
        r = ai.evaluate_photo(self.path, aesthetic_weight=0.8)
        self.assertAlmostEqual(r['aesthetic_weight'], 0.8)
        self.assertAlmostEqual(r['technical_weight'], 0.2)

        r = ai.evaluate_photo(self.path, technical_weight=0.25)
        self.assertAlmostEqual(r['aesthetic_weight'], 0.75)
        self.assertAlmostEqual(r['technical_weight'], 0.25)

    def test_overall_follows_the_supplied_weights(self):
        r = ai.evaluate_photo(self.path, aesthetic_weight=0.9)
        expected = r['aesthetic_score'] * 0.9 + r['technical_score'] * 0.1
        self.assertAlmostEqual(r['overall_score'], expected, places=1)

    def test_extreme_weights_reduce_to_a_single_score(self):
        """權重 1/0 時綜合分應等於該項原始分數，不可有額外偏移。"""
        r = ai.evaluate_photo(self.path, aesthetic_weight=1.0)
        self.assertAlmostEqual(r['overall_score'], r['aesthetic_score'], places=1)
        r = ai.evaluate_photo(self.path, aesthetic_weight=0.0)
        self.assertAlmostEqual(r['overall_score'], r['technical_score'], places=1)

    def test_raw_scores_and_status_are_unaffected_by_weights(self):
        """權重只准影響 overall_score，兩個原始分數與狀態都不可被動到。"""
        base = ai.evaluate_photo(self.path)
        tilted = ai.evaluate_photo(self.path, aesthetic_weight=0.1)
        for key in ('aesthetic_score', 'technical_score', 'status'):
            with self.subTest(key=key):
                self.assertEqual(base[key], tilted[key])

    def test_overall_stays_within_range_for_any_valid_weight(self):
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(aesthetic_weight=w):
                r = ai.evaluate_photo(self.path, aesthetic_weight=w)
                self.assertGreaterEqual(r['overall_score'], 0.0)
                self.assertLessEqual(r['overall_score'], 100.0)

    def test_both_weights_must_sum_to_one(self):
        """
        加總不為 1 會讓綜合分跑出 0~100 之外（0.6/0.6 最高算到 120），
        與兩個原始分數放在同一份清單裡比較會產生誤導，因此直接擋下。
        """
        for pair in ((0.6, 0.6), (0.2, 0.2)):
            with self.subTest(pair=pair):
                self.assertIsNone(ai.evaluate_photo(
                    self.path, aesthetic_weight=pair[0], technical_weight=pair[1]))
                self.assertIn('加總必須為 1', ai.LAST_ERROR)

    def test_both_weights_summing_to_one_is_accepted(self):
        r = ai.evaluate_photo(self.path, aesthetic_weight=0.7, technical_weight=0.3)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r['aesthetic_weight'], 0.7)

    def test_invalid_weights_are_rejected(self):
        for bad in (-0.1, 1.5, float('nan'), 'high', None or object()):
            with self.subTest(value=bad):
                self.assertIsNone(ai.evaluate_photo(self.path, aesthetic_weight=bad))
                self.assertIn('權重參數不正確', ai.LAST_ERROR)

    def test_combine_scores_reproduces_the_pipeline_result(self):
        """
        滑桿路徑與完整分析路徑必須算出同一個綜合分。
        兩邊各寫一份公式的話，清單上的分數會與重新分析後的分數對不起來。
        """
        for w in (None, 0.0, 0.35, 1.0):
            with self.subTest(aesthetic_weight=w):
                r = ai.evaluate_photo(self.path, aesthetic_weight=w)
                again = ai.combine_scores(r['aesthetic_score'], r['technical_score'],
                                          aesthetic_weight=w)
                self.assertAlmostEqual(r['overall_score'], round(again, 2), places=2)

    def test_combine_scores_rejects_bad_input(self):
        """
        純算術函式，失敗只可能是呼叫端傳錯——直接拋 ValueError，不吞掉。
        特別要擋住「把模型原始的 1~10 分當成 0~100 分傳進來」這個實際會犯的錯。
        """
        for bad in (-1.0, 100.1, float('nan'), 'x'):
            with self.subTest(score=bad):
                with self.assertRaises(ValueError):
                    ai.combine_scores(bad, 50.0)
        with self.assertRaises(ValueError):
            ai.combine_scores(50.0, 50.0, aesthetic_weight=1.5)

    def test_combine_scores_is_a_plain_weighted_average(self):
        self.assertAlmostEqual(ai.combine_scores(100.0, 0.0, aesthetic_weight=1.0), 100.0)
        self.assertAlmostEqual(ai.combine_scores(100.0, 0.0, aesthetic_weight=0.0), 0.0)
        self.assertAlmostEqual(ai.combine_scores(80.0, 40.0, aesthetic_weight=0.5), 60.0)
        # 邊界值本身要能通過，不可被範圍檢查誤擋
        self.assertAlmostEqual(ai.combine_scores(0.0, 100.0, aesthetic_weight=0.5), 50.0)

    def test_weights_are_validated_before_the_image_is_read(self):
        """
        權重錯是呼叫端的參數問題，不該先花 700~1100 ms 解一張 RAW 檔才發現。
        用一個不存在的路徑呼叫：若失敗訊息講的是權重而不是找不到檔案，
        就證明驗證確實發生在讀檔之前。
        """
        self.assertIsNone(ai.evaluate_photo('絕對不存在的檔案.jpg',
                                            aesthetic_weight=99))
        self.assertIn('權重參數不正確', ai.LAST_ERROR)


@requires_weights
class TestFailureReporting(unittest.TestCase):
    """A4：失敗要能分辨根因，不能一律回報成「未知錯誤」。"""

    def test_missing_file_returns_none_and_records_reason(self):
        result = ai.evaluate_photo('data/my_photos/這個檔案不存在.jpg')
        self.assertIsNone(result)
        self.assertIsNotNone(ai.LAST_ERROR)
        self.assertIn('找不到', ai.LAST_ERROR)

    def test_corrupt_file_is_reported_as_read_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'broken.jpg'
            bad.write_bytes(b'this is definitely not a jpeg')
            result = ai.evaluate_photo(str(bad))
        self.assertIsNone(result)
        self.assertIn('無法讀取影像', ai.LAST_ERROR)

    def test_last_error_is_cleared_on_success(self):
        ai.evaluate_photo('絕對不存在的路徑.jpg')
        self.assertIsNotNone(ai.LAST_ERROR)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(make_image(320, 240), Path(tmp) / 'ok.jpg')
            self.assertIsNotNone(ai.evaluate_photo(path))
        self.assertIsNone(ai.LAST_ERROR, '成功後 LAST_ERROR 應被清空，否則會殘留誤導')


@requires_weights
class TestImageParameter(unittest.TestCase):
    """B3：呼叫端可傳入已解碼影像以避免重複解碼，結果必須完全相同。"""

    def test_matches_loading_from_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(make_image(640, 480, kind='detail'),
                               Path(tmp) / 'x.jpg')
            from_path = ai.evaluate_photo(path)
            from_array = ai.evaluate_photo(path, image=ai._load_image_array(path))
        self.assertEqual(from_path, from_array)

    @requires_rawpy
    def test_matches_for_raw_files(self):
        raw = find_photo('.arw', '.dng')
        if raw is None:
            self.skipTest('data/my_photos 內沒有 RAW 檔可供測試')
        a = ai.evaluate_photo(str(raw))
        b = ai.evaluate_photo(str(raw), image=ai._load_image_array(str(raw)))
        self.assertEqual(a, b)

    def test_invalid_array_is_rejected_not_silently_scored(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(make_image(320, 240), Path(tmp) / 'x.jpg')
            for bad in (np.zeros((64, 64), dtype=np.uint8),
                        np.zeros((64, 64, 3), dtype=np.float32),
                        'not an array'):
                with self.subTest(value=type(bad).__name__):
                    self.assertIsNone(ai.evaluate_photo(path, image=bad))
                    self.assertIn('image 參數格式不正確', ai.LAST_ERROR)

    def test_path_is_not_read_when_image_supplied(self):
        """傳入影像時不應再碰硬碟——路徑不存在也要能算分。"""
        result = ai.evaluate_photo('這個路徑不存在.jpg',
                                   image=make_image(320, 240))
        self.assertIsNotNone(result,
                             '傳入 image 時仍去檢查路徑，代表沒有真正避免重複讀檔')


@requires_weights
class TestStatusDecoupledFromMeasurements(unittest.TestCase):
    """
    釘住「丙案」：狀態只由技術分決定，影像量測結果僅作補充資訊。

    理由是技術分經過驗證（KonIQ 驗證集 SRCC 0.7953），
    而量測門檻沒有；且量測值正確不代表是缺陷——
    刻意以黑色為背景的照片死黑比例本來就高（實測有樣本達 72.8%），
    不該因此把一張美感 100 分的照片降級成警告。
    """

    def test_dark_background_image_is_measured_but_not_downgraded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(make_image(kind='dark'), Path(tmp) / 'dark.jpg')
            r = ai.evaluate_photo(path)

        self.assertTrue(any('曝光不足' in i for i in r['technical_issues']),
                        '死黑比例極高的影像應該要被量測到')
        # 狀態必須與技術分一致，不受量測結果影響
        expected = '警告' if r['technical_score'] < ai.TECH_ISSUE_THRESHOLD else \
                   ('優秀' if r['aesthetic_score'] > ai.AESTHETIC_EXCELLENT_THRESHOLD
                    else '正常')
        self.assertEqual(r['status'], expected,
                         '狀態被影像量測結果影響了，應只由技術分決定')

    def test_analysis_runs_even_for_high_technical_score(self):
        """
        舊版只在技術分 < 60 時才做細項分析，導致技術分高的照片
        即使有明顯問題也完全不會被檢查（實測 DSC04606 技術分 73.3，
        永遠不觸發分析）。

        這裡直接把技術模型換成固定回傳高分，強制製造出
        「技術分很高、但影像確實有可量測問題」的情境。
        用合成影像去碰運氣湊出這個組合並不可靠——
        早期版本就是因為斷言被包在 if 裡，導致條件不成立時整條測試形同虛設
        （這個盲點是靠變異測試發現的）。
        """
        original = ai.MODEL_TECH

        class _AlwaysHigh:
            def __call__(self, _tensor):
                import torch
                return torch.tensor([0.95])       # x100 = 95 分，遠高於門檻

        ai.MODEL_TECH = _AlwaysHigh()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = write_image(make_image(kind='dark'), Path(tmp) / 'd.jpg')
                r = ai.evaluate_photo(path)
        finally:
            ai.MODEL_TECH = original

        self.assertGreaterEqual(r['technical_score'], ai.TECH_ISSUE_THRESHOLD,
                                '測試前提失效：技術分應被固定在門檻之上')
        self.assertTrue(r['technical_issues'],
                        '技術分高於門檻時細項分析就沒有執行，'
                        '代表分析又被技術分閘門擋住了')
        self.assertTrue(any('曝光不足' in i for i in r['technical_issues']))


if __name__ == '__main__':
    unittest.main()
