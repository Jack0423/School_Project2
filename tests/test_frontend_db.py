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
        with mock.patch('raw_processor.RawProcessor.safe_update_xmp') as write, \
                mock.patch('raw_processor.RawProcessor.update_ratings') as batch:
            self.m.recompute_scores(0.6, str(self.folder))

        write.assert_not_called()
        batch.assert_not_called()
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

    def test_auto_write_only_touches_just_analyzed_photos(self):
        """「分析後自動寫入 XMP」只寫剛分析完的那幾張，不可以把整個資料夾重寫一次。"""
        self.m.recompute_scores(0.6, str(self.folder))
        with mock.patch('builtins.print'):
            result = self.m.write_xmp_ratings(
                str(self.folder), only_paths=[str(self.folder / 'a.ARW'), str(self.folder / 'c.jpg')])

        self.assertEqual(result, (1, 1, 0), '寫 a.ARW；c.jpg 是 JPG，略過')
        self.assertEqual(self._xmp_files(), ['a.xmp'], 'b.NEF 沒有剛分析，不可以被寫')

    def test_rating_line_matches_what_xmp_will_write(self):
        """結果面板的「LR 星等」與寫進 .xmp 的星等必須是同一個換算。"""
        from raw_processor import RawProcessor
        cases = {70.0: '★★★★★（69 分以上）',
                 60.0: '★★★☆☆（54～64 分）',
                 47.0: '★★☆☆☆（47～54 分）',
                 40.0: '★☆☆☆☆（未滿 47 分）'}
        for score, text in cases.items():
            with self.subTest(score=score):
                self.assertEqual(self.m.rating_text(score), text)
                self.assertEqual(text.count('★'), RawProcessor.map_score_to_rating(score))

    # ── 結果面板與資料夾統計 ──────────────────────────────
    def _full_row(self, aes=66.0, tech=94.9, overall=77.6, current=True, analyzed=1,
                  suggestion='照片品質良好。', issues='', is_best=0):
        version = self.m.MODEL_VERSION if current else '舊版'
        return ('p.ARW', 'p.ARW', aes, tech, overall, 0.6, 0.4, version, '', suggestion, '', issues,
                is_best, analyzed)

    def test_result_panel_is_compact(self):
        text = self.m.result_html(self._full_row(), 'ILCE-7M3')
        for expected in ('優秀', '正常', '★★★★★', '建議保留', '拍攝資訊', 'ILCE-7M3'):
            self.assertIn(expected, text)
        for removed in ('模型版本', '評分權重', '影像量測附註', '最佳照片'):
            self.assertNotIn(removed, text, f'{removed} 不該出現（沒有附註、不是最佳照片時都不顯示）')

    def test_result_panel_warning_and_escaping(self):
        issues = '疑似對焦不準（清晰度指標 12.0，建議 >= 100）；曝光不足，死黑區域佔比 21.9%'
        text = self.m.result_html(self._full_row(tech=52.0, overall=55.0, issues=issues,
                                                 suggestion='整體技術品質偏低（技術分 52.0，低於門檻 60），可能成因見影像量測附註。',
                                                 is_best=1), '')
        for expected in ('警告', '建議檢視', '影像量測附註', '&gt;= 100', '最佳照片'):
            self.assertIn(expected, text)
        self.assertNotIn('>= 100', text, '量測文字要跳脫，否則 > 會被當成 HTML')
        self.assertEqual(text.count('\u30fb'), 2, '每個附註各一行')   # 條列符號（cp950 編不出來，用跳脫寫）

    def test_result_panel_for_unanalyzed_and_old_version(self):
        self.assertIn('尚未分析', self.m.result_html(self._full_row(analyzed=0), ''))
        self.assertIn('舊版模型', self.m.result_html(self._full_row(current=False), ''))

    def test_folder_summary(self):
        rows = [self._row('a.ARW', 70, 80), self._row('b.ARW', 50, 40), self._row('c.ARW', 40, 90),
                self._row('d.ARW', 60, 70, current=False), self._row('e.ARW', None, None, analyzed=0)]
        self.assertEqual(self.m.folder_summary(rows),
                         '\u30fb'.join(['共 5 張', '已分析 3', '技術警告 1', '美感優秀 1', '需重新分析 1']))

    def test_xmp_failures_are_summarized_for_the_screen(self):
        errors = [('C:/a/1.ARW', '需要 ExifTool'), ('C:/a/2.ARW', '需要 ExifTool'), ('C:/a/3.ARW', '磁碟已滿'),
                  ('C:/a/4.ARW', 'x'), ('C:/a/5.ARW', 'y')]
        self.assertEqual(self.m.summarize_errors(errors),
                         ['需要 ExifTool（1.ARW 等 2 張）', '磁碟已滿（3.ARW）', 'x（4.ARW）', '另有 1 種原因'])
        self.assertEqual(self.m.summarize_errors([('', 'XMP 更新失敗：OSError')]), ['XMP 更新失敗：OSError'])

    def test_xmp_button_reports_reason_without_exiftool(self):
        self.m.recompute_scores(0.6, str(self.folder))
        (self.folder / 'a.xmp').write_text('<x:xmpmeta xmlns:x="adobe:ns:meta/"/>', encoding='utf-8')
        errors = []
        with mock.patch('raw_processor.shutil.which', return_value=None), mock.patch('builtins.print'):
            result = self.m.write_xmp_ratings(str(self.folder), errors=errors)
        self.assertEqual(result, (1, 2, 1))
        self.assertEqual([Path(p).name for p, _ in errors], ['a.ARW'])
        self.assertIn('ExifTool', errors[0][1])

    def test_progress_callback_reports_every_write(self):
        self.m.recompute_scores(0.6, str(self.folder))
        seen = []
        with mock.patch('builtins.print'):
            self.m.write_xmp_ratings(str(self.folder),
                                     lambda done, total, path: seen.append((done, total)))
        self.assertEqual(seen, [(1, 2), (2, 2)])


    # ── 排序 ────────────────────────────────────────────
    def _order(self, label, descending):
        key_index = dict(self.m.SORT_KEYS)[label]
        rows = self.m.get_photos_in_folder(str(self.folder))
        return [r[0] for r in self.m.sort_rows(rows, key_index, descending)]

    def test_sort_by_each_field(self):
        self.m.recompute_scores(0.6, str(self.folder))   # 綜合分：a 68、c 61、d 50、b 44
        # 未分析（f）與舊版本（e）沒有可比的分數，一律排在最後
        self.assertEqual(self._order('綜合分', True), ['a.ARW', 'c.jpg', 'd.dng', 'b.NEF', 'f.ARW', 'e.ARW'])
        self.assertEqual(self._order('綜合分', False), ['b.NEF', 'd.dng', 'c.jpg', 'a.ARW', 'f.ARW', 'e.ARW'])
        self.assertEqual(self._order('美感分', True), ['a.ARW', 'c.jpg', 'd.dng', 'b.NEF', 'f.ARW', 'e.ARW'])
        # b 與 d 的技術分都是 50：同分依檔名
        self.assertEqual(self._order('技術分', False), ['b.NEF', 'd.dng', 'c.jpg', 'a.ARW', 'f.ARW', 'e.ARW'])
        self.assertEqual(self._order('技術分', True), ['a.ARW', 'c.jpg', 'b.NEF', 'd.dng', 'f.ARW', 'e.ARW'])
        self.assertEqual(self._order('檔名', False), ['a.ARW', 'b.NEF', 'c.jpg', 'd.dng', 'e.ARW', 'f.ARW'])
        self.assertEqual(self._order('檔名', True), ['f.ARW', 'e.ARW', 'd.dng', 'c.jpg', 'b.NEF', 'a.ARW'])

    # ── 刪除 ────────────────────────────────────────────
    def _fake_trash(self, fail=()):
        """代替資源回收筒：移到暫存的 trash 資料夾，測試不會動到真的回收筒。"""
        trash_dir = Path(self.tmp.name) / 'trash'
        trash_dir.mkdir(exist_ok=True)

        def trash(path):
            if os.path.basename(path) in fail:
                return False, '測試用的失敗'
            os.replace(path, trash_dir / os.path.basename(path))
            return True, ''
        return trash, trash_dir

    def test_delete_moves_photo_and_raw_sidecar_only(self):
        (self.folder / 'a.xmp').write_text('a 的星等', encoding='utf-8')   # RAW 的 sidecar
        (self.folder / 'c.xmp').write_text('不屬於 c.jpg', encoding='utf-8')
        trash, trash_dir = self._fake_trash()

        moved, failed = self.m.delete_photos(
            [str(self.folder / 'a.ARW'), str(self.folder / 'c.jpg')], trash=trash)

        self.assertEqual((moved, failed), (2, []))
        self.assertEqual(sorted(p.name for p in trash_dir.iterdir()), ['a.ARW', 'a.xmp', 'c.jpg'])
        self.assertTrue((self.folder / 'c.xmp').exists(), '刪 JPG 不可以動到 .xmp')
        remaining = sorted(r[0] for r in self.m.get_photos_in_folder(str(self.folder)))
        self.assertEqual(remaining, ['b.NEF', 'd.dng', 'e.ARW', 'f.ARW'])

        self.m.recompute_scores(0.6, str(self.folder))
        best = [r[0] for r in self.m.get_photos_in_folder(str(self.folder)) if r[12] == 1]
        self.assertEqual(best, ['d.dng'], '刪掉 ★ 之後要重新挑最佳照片')

    def test_failed_delete_keeps_photo_and_record(self):
        trash, _ = self._fake_trash(fail={'b.NEF'})
        moved, failed = self.m.delete_photos([str(self.folder / 'b.NEF')], trash=trash)

        self.assertEqual(moved, 0)
        self.assertEqual([name for name, _ in failed], ['b.NEF'])
        self.assertTrue((self.folder / 'b.NEF').exists())
        self.assertIn('b.NEF', [r[0] for r in self.m.get_photos_in_folder(str(self.folder))])


    # ── 預設權重、篩選與搜尋 ──────────────────────────────
    def test_default_weight_matches_model_side(self):
        import ai_inference
        self.assertEqual(self.m.DEFAULT_AESTHETIC_WEIGHT, ai_inference.AESTHETIC_WEIGHT,
                         '前台、批次工具與 XMP 星等要用同一個預設權重')

    def _row(self, name, aes, tech, analyzed=1, current=True):
        version = self.m.MODEL_VERSION if current else '舊版'
        return (name, name, aes, tech, None, None, None, version, '', '', '', '', 0, analyzed)

    def test_filters_and_search(self):
        rows = [self._row('great.ARW', 70, 80),        # 美感優秀
                self._row('weak.jpg', 30, 90),         # 美感待加強
                self._row('blurry.ARW', 55, 40),       # 技術警告
                self._row('old.ARW', 75, 30, current=False),   # 舊版本：不算優秀也不算警告
                self._row('new.NEF', None, None, analyzed=0)]
        # 依 FILTERS 的順序：全部、美感優秀、美感待加強、技術警告、尚未分析、需重新分析
        names = [[r[0] for r in self.m.filter_rows(rows, i)] for i in range(len(self.m.FILTERS))]
        expected = [[r[0] for r in rows], ['great.ARW'], ['weak.jpg'], ['blurry.ARW'],
                    ['new.NEF'], ['old.ARW']]
        for i, (label, _) in enumerate(self.m.FILTERS):
            with self.subTest(filter=i):
                self.assertEqual(names[i], expected[i], label.encode('cp950', 'replace').decode('cp950'))

        searched = [r[0] for r in self.m.filter_rows(rows, 0, '  arw ')]
        self.assertEqual(searched, ['great.ARW', 'blurry.ARW', 'old.ARW'], '搜尋不分大小寫、忽略前後空白')
        both = [r[0] for r in self.m.filter_rows(rows, 3, 'great')]
        self.assertEqual(both, [], '篩選與搜尋要同時成立')


    # ── 建議文字 ────────────────────────────────────────
    def test_old_suggestions_drop_the_copied_issues(self):
        old_normal = ('照片品質良好。（影像量測附註：曝光不足，死黑區域佔比 55.9%。'
                      '以上為客觀量測值，不一定代表缺陷——例如以黑色為背景的照片死黑比例本來就高）')
        old_low = ('整體技術品質偏低（技術分 52.3，低於門檻 60）。'
                   '可能成因：曝光不足，死黑區域佔比 55.9%；對比度不足，畫面偏灰、層次感弱（對比指標 26.3）。')
        self.assertEqual(self.m.display_suggestion(old_normal), '照片品質良好。')
        self.assertEqual(self.m.display_suggestion(old_low),
                         '整體技術品質偏低（技術分 52.3，低於門檻 60），可能成因見影像量測附註。')
        for current in ('照片品質良好。', '構圖優秀、光影掌握佳，整體技術品質良好。',
                        '整體技術品質偏低（技術分 52.3，低於門檻 60），可能成因見影像量測附註。',
                        '整體技術品質偏低（技術分 41.0，低於門檻 60），但未找出單一明顯成因，可能是壓縮失真或整體畫質不足。'):
            self.assertEqual(self.m.display_suggestion(current), current, '新寫法不可以被改動')


if __name__ == '__main__':
    unittest.main()
