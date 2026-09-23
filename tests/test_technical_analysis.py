"""
傳統影像分析的測試（對應 B3、B11、B19）。

這裡用合成影像而非真實照片，因為要測的是「指標在已知條件下的行為」，
合成影像能精確控制那些條件，也不依賴 .gitignore 掉的資料。
"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import ai_inference as ai
from tests._util import make_image


class TestDoesNotReadFromDisk(unittest.TestCase):
    """
    B3：分析函式必須接收已解碼的影像，不可自己再讀一次檔。

    重複解碼的代價很高——RAW 全尺寸解碼實測 700~1100 ms，
    比整個模型推論（22.7 ms）高一個數量級。
    """

    def test_analysis_works_without_touching_filesystem(self):
        original_imread = cv2.imread
        original_open = Image.open

        def explode(*args, **kwargs):
            raise AssertionError('傳統影像分析不應該從硬碟讀檔')

        cv2.imread = explode
        Image.open = explode
        try:
            issues = ai._analyze_technical_issues(make_image(kind='detail'))
        finally:
            cv2.imread = original_imread
            Image.open = original_open

        self.assertIsInstance(issues, list)

    def test_accepts_array_not_path(self):
        with self.assertRaises(Exception):
            ai._analyze_technical_issues('some/path.jpg')


class TestNoUpscaling(unittest.TestCase):
    """
    B19：比分析寬度窄的影像不可被放大。

    插值放大本身會讓畫面變糊，清晰度指標因此崩潰。
    實測 AVA 資料集 120 張中有 38 張因此被誤判為對焦不準，
    其中一張原生清晰度 1477.8，放大到 800 後只剩 72.2。
    """

    def test_small_sharp_image_is_not_flagged_as_blurry(self):
        """
        用銳利度接近真實照片的影像（原生約 470，門檻為 100）。
        不能用 'detail' 那種清晰度高達 65,000 的棋盤格——它銳利到
        即使被放大 4 倍仍遠高於門檻，就算 bug 存在也測不出來
        （這個盲點是靠變異測試發現的）。
        """
        small = make_image(width=400, height=300, kind='photo_like')
        issues = ai._analyze_technical_issues(small)
        self.assertFalse(any('模糊' in i for i in issues),
                         '小尺寸但銳利度正常的影像被判為模糊，'
                         '代表分析前又把它放大了（插值放大會讓清晰度指標崩潰）')

    def test_the_upscaling_hazard_is_real(self):
        """
        確認上一項測的是真的危險，而不是一個永遠成立的斷言：
        同一張影像若先放大到 800 寬，清晰度就會掉到門檻以下。
        """
        small = make_image(width=400, height=300, kind='photo_like')
        gray = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        native = cv2.Laplacian(gray, cv2.CV_64F).var()

        upscaled = cv2.resize(small, (800, 600))
        gray_up = cv2.cvtColor(cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR),
                               cv2.COLOR_BGR2GRAY)
        after = cv2.Laplacian(gray_up, cv2.CV_64F).var()

        self.assertGreater(native, ai.BLUR_VAR_THRESHOLD, '測試影像原生就不夠銳利')
        self.assertLess(after, ai.BLUR_VAR_THRESHOLD,
                        '放大後清晰度沒有掉到門檻以下，測試前提已失效')

    def test_genuinely_blurry_image_is_still_flagged(self):
        """確認上一項不是靠「乾脆不偵測模糊」達成的。"""
        blurry = make_image(width=1200, height=900, kind='blur')
        issues = ai._analyze_technical_issues(blurry)
        self.assertTrue(any('模糊' in i for i in issues),
                        '真正模糊的影像應該要被偵測到')

    def test_large_image_is_still_downscaled(self):
        """只縮小、不放大——大圖仍應被縮小以維持門檻值的可比性。"""
        big = make_image(width=2400, height=1800, kind='detail')
        small = make_image(width=800, height=600, kind='detail')
        # 兩者內容性質相同，縮放到同一寬度後判斷結果應一致
        self.assertEqual(
            [i.split('（')[0] for i in ai._analyze_technical_issues(big)],
            [i.split('（')[0] for i in ai._analyze_technical_issues(small)])


class TestExposureDetection(unittest.TestCase):
    def test_dark_image_reports_underexposure(self):
        issues = ai._analyze_technical_issues(make_image(kind='dark'))
        self.assertTrue(any('曝光不足' in i for i in issues))

    def test_bright_image_reports_overexposure(self):
        issues = ai._analyze_technical_issues(make_image(kind='bright'))
        self.assertTrue(any('過曝' in i for i in issues))

    def test_normal_image_reports_neither(self):
        issues = ai._analyze_technical_issues(make_image(kind='detail'))
        self.assertFalse(any('曝光不足' in i or '過曝' in i for i in issues))

    def test_flat_image_reports_low_contrast(self):
        issues = ai._analyze_technical_issues(make_image(kind='flat'))
        self.assertTrue(any('對比度不足' in i for i in issues))


class TestNoiseEstimation(unittest.TestCase):
    """
    B11：雜訊估計必須在原始解析度上計算，並排除已截斷的死黑／死白區域。
    """

    @staticmethod
    def _gray(array):
        return cv2.cvtColor(cv2.cvtColor(array, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    def test_noisy_image_scores_higher_than_clean(self):
        noisy = ai._estimate_noise_sigma(self._gray(make_image(kind='noise')))
        clean = ai._estimate_noise_sigma(self._gray(make_image(kind='flat')))
        self.assertGreater(noisy, clean)

    def test_downscaling_destroys_the_signal(self):
        """
        釘住「為什麼必須用原始解析度」：把影像縮小後再估計，
        雜訊訊號會被低通濾波抹掉。這正是原本判斷方向相反的原因。
        """
        noisy = make_image(width=1600, height=1200, kind='noise')
        full = ai._estimate_noise_sigma(self._gray(noisy))
        shrunk = ai._estimate_noise_sigma(
            self._gray(cv2.resize(noisy, (800, 600))))
        self.assertGreater(full, shrunk,
                           '縮小後雜訊估計值沒有下降，測試前提可能已失效')

    def test_returns_nan_when_everything_is_clipped(self):
        """
        全黑或全白的影像沒有任何未截斷的像素，無法估計雜訊，
        應回傳 nan 讓呼叫端跳過，而不是硬給一個數字。
        """
        allblack = np.zeros((800, 800), dtype=np.uint8)
        self.assertTrue(np.isnan(ai._estimate_noise_sigma(allblack)))

    def test_noise_threshold_matches_the_calibration(self):
        """
        2026-09-23 以 77 張人工盲標樣本校準後啟用（53 張有雜訊、24 張乾淨）。
        門檻取在「乾淨樣本的最大值 1.427」之上，換取零誤報、漏報 5 張。

        這個測試釘住的是那個取捨：技術量測顯示給使用者看，誤報的代價高於漏報。
        要調整請連同 ai_inference 內的校準說明一起更新。
        """
        self.assertIsNotNone(ai.NOISE_SIGMA_THRESHOLD)
        self.assertAlmostEqual(ai.NOISE_SIGMA_THRESHOLD, 1.43, places=2)

    def test_clean_image_is_not_reported_as_noisy(self):
        """誤報是這個功能最不能犯的錯，平坦與模糊的影像都不該被標記。"""
        for kind in ('flat', 'blur'):
            with self.subTest(kind=kind):
                issues = ai._analyze_technical_issues(make_image(kind=kind))
                self.assertFalse(any('雜訊' in i for i in issues))

    def test_noisy_image_is_reported(self):
        issues = ai._analyze_technical_issues(make_image(kind='noise'))
        self.assertTrue(any('雜訊' in i for i in issues),
                        '雜訊明顯的影像沒有被標記')

    def test_noise_note_has_two_levels(self):
        """
        門檻附近（夜景暗部的輕微顆粒）與真正的高 ISO 不該用同一句話。
        實測門檻附近的照片看起來只是輕微，寫成「偏高」會過度警示。
        """
        self.assertIsNone(ai._noise_issue(1.0), '未超過門檻不該有附註')
        self.assertIsNone(ai._noise_issue(float('nan')), '無法估計時不該有附註')
        mild = ai._noise_issue(1.62)
        high = ai._noise_issue(2.50)
        self.assertIn('輕微', mild)
        self.assertIn('偏高', high)
        self.assertIn('1.62', mild)
        self.assertIn('2.50', high)

    def test_noise_level_boundary(self):
        self.assertIn('輕微', ai._noise_issue(ai.NOISE_SIGMA_HIGH))
        self.assertIn('偏高', ai._noise_issue(ai.NOISE_SIGMA_HIGH + 0.01))

    def test_high_frequency_detail_is_a_known_false_positive(self):
        """
        已知限制，刻意用測試記錄下來：這個估計量分不開「細節」與「雜訊」。
        合成的細節影像 sigma 為 53.5，比刻意加雜訊的 39.8 還高。

        實照片上的影響小得多（ISO<=800 的 76 張只有 3 張超過門檻），
        但換相機或拍大量細碎紋理時必須重新校準。
        這個測試若失敗，代表估計方法被改動了，請重新驗證校準是否仍成立。
        """
        gray = cv2.cvtColor(make_image(kind='detail'), cv2.COLOR_RGB2GRAY)
        self.assertGreater(ai._estimate_noise_sigma(gray), ai.NOISE_SIGMA_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
