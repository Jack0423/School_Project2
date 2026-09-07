"""
影像讀取與格式驗證的測試（對應 B1、B2 與 image= 參數驗證）。

最關鍵的是 B1：.DNG 交給 PIL 會「成功」讀到內嵌縮圖而不報錯。
實測某張 DNG 實際影像 6048x4024，PIL 只讀到 256x171，
評分照常回傳、完全沒有警示——是最危險的那種靜默錯誤。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import ai_inference as ai
from tests._util import (find_photo, make_image, requires_rawpy, write_image,
                         PHOTO_DIR)


class TestFormatDispatch(unittest.TestCase):
    def test_raw_extensions_are_registered(self):
        for ext in ('.arw', '.dng', '.cr2', '.nef'):
            with self.subTest(ext=ext):
                self.assertIn(ext, ai.RAW_EXTENSIONS)

    def test_raw_extensions_are_lowercase(self):
        """副檔名比對前會轉小寫，集合內若混入大寫就永遠比不中。"""
        for ext in ai.RAW_EXTENSIONS:
            self.assertEqual(ext, ext.lower())
            self.assertTrue(ext.startswith('.'))

    def test_jpeg_loads_as_rgb_uint8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(make_image(320, 240), Path(tmp) / 'x.jpg')
            arr = ai._load_image_array(path)
        self.assertEqual(arr.ndim, 3)
        self.assertEqual(arr.shape[2], 3)
        self.assertEqual(arr.dtype, np.uint8)

    def test_png_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(make_image(200, 150), Path(tmp) / 'x.png')
            arr = ai._load_image_array(path)
        self.assertEqual(arr.shape[:2], (150, 200))

    def test_non_ascii_filename_loads(self):
        """中文檔名曾是專案裡特別處理過的路徑，確保仍然可用。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(make_image(120, 90), Path(tmp) / '測試照片.jpg')
            arr = ai._load_image_array(path)
        self.assertEqual(arr.shape[:2], (90, 120))


class TestRawIsNotReadByPIL(unittest.TestCase):
    """B1 / B2 的回歸測試。"""

    @requires_rawpy
    def test_dng_is_not_read_as_embedded_thumbnail(self):
        dng = find_photo('.dng')
        if dng is None:
            self.skipTest(f'{PHOTO_DIR} 內沒有 .dng 檔可供測試')

        with Image.open(dng) as im:
            pil_size = im.size                   # PIL 會讀到內嵌縮圖
        arr = ai._load_image_array(str(dng))     # 正確路徑：走 rawpy
        raw_h, raw_w = arr.shape[:2]

        self.assertGreater(raw_w * raw_h, pil_size[0] * pil_size[1] * 10,
                           'DNG 疑似仍被當成一般影像讀取，拿到的是內嵌縮圖而非實際影像')
        self.assertGreater(raw_w, 1000, 'RAW 解碼後的寬度異常小')

    @requires_rawpy
    def test_arw_decodes(self):
        arw = find_photo('.arw')
        if arw is None:
            self.skipTest(f'{PHOTO_DIR} 內沒有 .ARW 檔可供測試')
        arr = ai._load_image_array(str(arw))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertGreater(arr.shape[1], 1000)

    def test_missing_rawpy_raises_actionable_error(self):
        """
        沒安裝 rawpy 時要明確報錯並給出安裝指令，
        而不是讓整個模組因為頂層 import 而無法載入。
        """
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'rawpy':
                raise ImportError('模擬未安裝')
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(RuntimeError) as ctx:
                ai._load_image_array('somewhere/photo.ARW')
        finally:
            builtins.__import__ = real_import

        message = str(ctx.exception)
        self.assertIn('rawpy', message)
        self.assertIn('pip install', message, '錯誤訊息應告訴使用者怎麼解決')


class TestImageArrayValidation(unittest.TestCase):
    """image= 參數的格式檢查，避免呼叫端傳錯而靜默算出錯誤分數。"""

    def test_accepts_valid_rgb_uint8(self):
        self.assertIsNone(ai._validate_image_array(make_image(64, 64)))

    def test_rejects_non_array(self):
        for bad in (None, [1, 2, 3], 'path.jpg',
                    Image.fromarray(make_image(32, 32))):
            with self.subTest(value=type(bad).__name__):
                self.assertIsNotNone(ai._validate_image_array(bad))

    def test_rejects_grayscale(self):
        problem = ai._validate_image_array(np.zeros((64, 64), dtype=np.uint8))
        self.assertIsNotNone(problem)
        self.assertIn('(H, W, 3)', problem)

    def test_rejects_float_dtype(self):
        problem = ai._validate_image_array(np.zeros((64, 64, 3), dtype=np.float32))
        self.assertIsNotNone(problem)
        self.assertIn('uint8', problem)

    def test_bgr_cannot_be_detected(self):
        """
        已知限制，明確寫成測試以免日後有人誤以為驗證涵蓋了通道順序。
        BGR 與 RGB 的形狀和 dtype 完全相同，程式無從分辨。
        """
        import cv2
        rgb = make_image(64, 64)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.assertIsNone(ai._validate_image_array(bgr),
                         'BGR 目前（也只能）通過驗證——'
                         '這是本質限制，防線在 evaluate_photo 的 docstring')


if __name__ == '__main__':
    unittest.main()
