"""
前台（童小席的 main_v2.py）資料庫層的測試。只測不需要畫面的模組函式。

守的是 2026-10-04 併入時實測到的問題：
  * 星等原本在 recompute_scores 裡自動寫進 .xmp。開資料夾、分析一張、權重滑桿
    每動一格都會重算，於是整個資料夾的 RAW 每次都重寫一次；已有 .xmp 的照片每張要
    呼叫一次 ExifTool，60 張 RAW 拉一格就卡 22.7 秒，Lightroom 的星等也一直被改。
    現在只有「寫入 XMP 星等」按鈕（write_xmp_ratings）會寫。
  * 資料庫原本是相對路徑，從不同位置啟動會各開一個資料庫。

main_v2 會 import ai_inference（連帶載入模型）與 PyQt6，所以整組掛 skip 條件，
並在測試內才 import。資料庫一律換成暫存檔，照片用假的位元組。
"""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._util import PROJECT_ROOT, requires_weights


def _pyqt6_available():
    try:
        import PyQt6.QtWidgets  # noqa: F401
        return True
    except ImportError:
        return False


requires_pyqt6 = unittest.skipUnless(_pyqt6_available(), '未安裝 PyQt6')

FAKE_RAW = b'FAKE-RAW-BYTES\x00\x01' * 32

# 檔名 -> (美感分, 技術分, 是不是目前的模型版本)；None 代表還沒分析
PHOTOS = {
    'a.ARW': (60.0, 80.0, True),      # 0.6 權重 → 68.0，5 星，最高分
    'b.NEF': (40.0, 50.0, True),      # 0.6 權重 → 44.0，3 星
    'c.jpg': (55.0, 70.0, True),      # JPG：Lightroom 不讀 sidecar
    'd.dng': (50.0, 50.0, True),      # DNG：Lightroom 不讀 sidecar
    'e.ARW': (70.0, 90.0, False),     # 舊模型版本的分數：不參加 ★，也不寫星等
    'f.ARW': None,                    # 還沒分析
}


@requires_weights
@requires_pyqt6
class TestFrontendDatabase(unittest.TestCase):

    def setUp(self):
        import main_v2
        self.m = main_v2
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / 'photos'
        self.folder.mkdir()
        self.original_db_path = main_v2.DB_PATH
        patcher = mock.patch.object(main_v2, 'DB_PATH', str(Path(self.tmp.name) / 'photos.db'))
        patcher.start()
        self.addCleanup(patcher.stop)

        main_v2.init_db()
        for name in PHOTOS:
            (self.folder / name).write_bytes(FAKE_RAW)
            main_v2.insert_photo(name, str(self.folder / name))

        # insert_photo 每次自己開連線寫入，全部新增完才開這條連線，否則會互相鎖住
        con = sqlite3.connect(main_v2.DB_PATH)
        for name, scores in PHOTOS.items():
            path = self.folder / name
            if scores is None:
                continue
            aes, tech, current = scores
            con.execute(
                'UPDATE photos_v2 SET aesthetic_score=?, technical_score=?, overall_score=?, '
                'model_version=?, analyzed=1 WHERE file_path=?',
                (aes, tech, aes, main_v2.MODEL_VERSION if current else '舊版', os.path.abspath(path)))
        con.commit()
        con.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _xmp_files(self):
        return sorted(p.name for p in self.folder.glob('*.xmp'))

    def test_db_lives_next_to_the_program(self):
        path = Path(self.original_db_path)
        self.assertTrue(path.is_absolute(), '相對路徑會因啟動位置不同而各開一個資料庫')
        self.assertEqual(path.resolve().parent, PROJECT_ROOT)

    def test_recompute_does_not_write_xmp(self):
        with mock.patch('raw_processor.RawProcessor.safe_update_xmp') as write:
            self.m.recompute_scores(0.6, str(self.folder))

        write.assert_not_called()
        self.assertEqual(self._xmp_files(), [], '重算分數（滑桿、開資料夾）不可以寫 .xmp')

        best = [r[0] for r in self.m.get_photos_in_folder(str(self.folder)) if r[12] == 1]
        self.assertEqual(best, ['a.ARW'], '★ 只能從目前模型版本的分數裡挑')

    def test_xmp_button_writes_current_raw_only(self):
        from raw_processor import RawProcessor
        self.m.recompute_scores(0.6, str(self.folder))

        with mock.patch('builtins.print'):
            written, skipped, failed = self.m.write_xmp_ratings(str(self.folder))

        self.assertEqual((written, skipped, failed), (2, 2, 0))
        self.assertEqual(self._xmp_files(), ['a.xmp', 'b.xmp'],
                         '只寫目前版本、已分析的 RAW；JPG／DNG、舊版本、未分析的都不寫')
        for name, stars in (('a', RawProcessor.map_score_to_rating(68.0)),
                            ('b', RawProcessor.map_score_to_rating(44.0))):
            self.assertIn(f'xmp:Rating="{stars}"',
                          (self.folder / f'{name}.xmp').read_text(encoding='utf-8'))
        for name in PHOTOS:
            self.assertEqual((self.folder / name).read_bytes(), FAKE_RAW, f'{name} 被改到了')

    def test_progress_callback_reports_every_write(self):
        self.m.recompute_scores(0.6, str(self.folder))
        seen = []
        with mock.patch('builtins.print'):
            self.m.write_xmp_ratings(str(self.folder),
                                     lambda done, total, path: seen.append((done, total)))
        self.assertEqual(seen, [(1, 2), (2, 2)])


if __name__ == '__main__':
    unittest.main()
