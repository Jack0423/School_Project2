"""
拍攝資訊（shooting_info.py）的測試。

前台在結果面板顯示相機、ISO、快門、光圈、焦距、拍攝時間。守住的是：
  * 有 ExifTool 與沒有 ExifTool（JPG 改用 PIL）兩條路，顯示出來的文字要一樣。
  * Windows 上中文檔名直接放在 ExifTool 命令列會變亂碼，必須用 UTF-8 參數檔傳路徑。
  * 沒有 EXIF 的照片（截圖、編修輸出）與沒有 ExifTool 的 RAW，要顯示原因而不是空白或報錯。
  * 同一張照片只讀一次：ExifTool 每次啟動約 0.2 秒，點照片時不能每次都等。

測試照片用 PIL 產生，不依賴任何被 .gitignore 的資料。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image
from PIL.TiffImagePlugin import IFDRational

import shooting_info as si

requires_exiftool = unittest.skipUnless(shutil.which('exiftool'), '未安裝 ExifTool')

EXPECTED_TEXT = ('TestCam X1\n'
                 'Test 35mm F2.8\n'
                 'ISO 800 · 1/250 秒 · f/2.8 · 35 mm\n'
                 '2026-10-04 10:20:30')


def write_jpg(path, with_exif=True):
    im = Image.new('RGB', (32, 24), (120, 120, 120))
    if not with_exif:
        im.save(path)
        return path
    exif = Image.Exif()
    exif[0x0110] = 'TestCam X1'
    exif[0x8769] = {0x8827: 800, 0x829A: IFDRational(1, 250), 0x829D: IFDRational(28, 10),
                    0x920A: IFDRational(35, 1), 0x9003: '2026:10:04 10:20:30',
                    0xA434: 'Test 35mm F2.8'}
    im.save(path, exif=exif)
    return path


class TestShootingInfo(unittest.TestCase):

    def setUp(self):
        si._cache.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        si._cache.clear()
        self.tmp.cleanup()

    @requires_exiftool
    def test_exiftool_reads_chinese_filename(self):
        path = write_jpg(self.dir / '測試相機.jpg')
        info, reason = si.read_shooting_info(str(path))
        self.assertEqual(reason, '')
        self.assertEqual(si.format_shooting_info(info, reason), EXPECTED_TEXT)

    def test_pil_fallback_matches_exiftool_text(self):
        path = write_jpg(self.dir / 'camera.jpg')
        with mock.patch('shooting_info.shutil.which', return_value=None):
            info, reason = si.read_shooting_info(str(path))
        self.assertEqual(si.format_shooting_info(info, reason), EXPECTED_TEXT)

    def test_raw_without_exiftool_explains_why(self):
        path = self.dir / 'DSC0001.ARW'
        path.write_bytes(b'FAKE')
        with mock.patch('shooting_info.shutil.which', return_value=None):
            info, reason = si.read_shooting_info(str(path))
        self.assertEqual(info, {})
        self.assertIn('ExifTool', si.format_shooting_info(info, reason))

    def test_photo_without_exif_says_so(self):
        path = write_jpg(self.dir / 'screenshot.jpg', with_exif=False)
        with mock.patch('shooting_info.shutil.which', return_value=None):
            text = si.format_shooting_info(*si.read_shooting_info(str(path)))
        self.assertIn('沒有拍攝資訊', text)

    def test_each_photo_is_read_once(self):
        path = write_jpg(self.dir / 'camera.jpg')
        with mock.patch('shooting_info.shutil.which', return_value='exiftool'), \
                mock.patch('shooting_info._read_with_exiftool', return_value={'ISO': 100}) as read:
            si.read_shooting_info(str(path))
            si.read_shooting_info(str(path))
        self.assertEqual(read.call_count, 1)

    def test_partial_info_only_shows_what_exists(self):
        # 實例：手機截圖只有拍攝時間
        text = si.format_shooting_info({'DateTimeOriginal': '2025:10:21 18:00:23'})
        self.assertEqual(text, '2025-10-21 18:00:23')


if __name__ == '__main__':
    unittest.main()
