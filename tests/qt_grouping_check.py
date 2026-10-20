"""Real Qt checks isolated from native model initialization; never deletes a file."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
import numpy as np
fake = types.ModuleType('ai_inference')
fake.IMAGE_EXTENSIONS = {'.jpg'}
fake.RAW_EXTENSIONS = {'.arw'}
fake.TECH_ISSUE_THRESHOLD = 40
fake.AESTHETIC_EXCELLENT_THRESHOLD = 60
fake.AESTHETIC_WEIGHT = .6
fake.load_image = lambda path: np.zeros((80, 120, 3), dtype=np.uint8)
fake.evaluate_photo = lambda *args, **kwargs: None
fake.LAST_ERROR = None
fake.aesthetic_grade = lambda score: '普通'
fake.model_version = lambda: 'test-model'
fake.combine_scores = lambda a,b,aesthetic_weight: a*aesthetic_weight+b*(1-aesthetic_weight)
sys.modules['ai_inference'] = fake
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtTest import QTest
import main_v2 as gui
app = QApplication([])


class GroupingUiChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        db = patch.object(gui, 'DB_PATH', str(self.folder/'test.db'))
        db.start()
        self.addCleanup(db.stop)
        self.window = gui.PhotoManagerV2()
        self.addCleanup(self.cleanup_window)
        self.window.current_folder = str(self.folder)
        self.paths = []
        for name, aes, tech, axis in [('甲.jpg',80,40,0),('乙.jpg',60,75,0),('單張.jpg',90,90,1)]:
            path = self.folder/name
            path.write_bytes(b'fixture')
            self.paths.append(str(path))
            vector = [0.0]*1280
            vector[axis] = 1.0
            gui.insert_photo(name,str(path))
            gui.update_analysis(str(path),dict(aesthetic_score=aes,technical_score=tech,feature_vector=vector),.6)
        self.window.photo_paths = {Path(p).name:p for p in self.paths}
        gui.recompute_scores(.6,str(self.folder))
        self.window.refresh_list()
        self.window.show_groups()
        self.view = self.window.group_review

    def cleanup_window(self):
        self.view.exit()
        QThreadPool.globalInstance().waitForDone()
        app.processEvents()
        self.window.close()

    def test_primary_action_follows_folder_analysis_and_review(self):
        candidates = patch.object(gui, 'analysis_candidates', return_value=[])
        candidates.start()
        self.addCleanup(candidates.stop)
        self.view.exit()
        self.window._update_primary_action()
        self.assertEqual(self.window.primary_btn.text(), '整理 1 組相似照片')
        self.assertTrue(self.window.group_btn.isHidden())
        self.assertTrue(self.window.analyze_all_btn.isHidden())
        self.window.primary_btn.click()
        self.assertTrue(self.view.active)
        self.assertEqual(self.window.primary_btn.text(), '完成整理，返回照片清單')
        self.window.primary_btn.click()
        self.assertFalse(self.view.active)
        with patch.object(gui, 'analysis_candidates', return_value=self.paths[:2]):
            self.window._refresh_workflow_state()
            self.assertEqual(self.window.primary_btn.text(), '分析這 2 張照片')
            self.assertFalse(self.window.group_btn.isHidden())
            with patch.object(self.window, 'analyze_all') as analyze:
                self.window.primary_btn.click()
                analyze.assert_called_once()
        self.window._analysis_thread = object()
        self.window._update_primary_action()
        self.assertEqual(self.window.primary_btn.text(), '分析中…')
        self.assertFalse(self.window.primary_btn.isEnabled())
        self.window._analysis_thread = None
        self.window.current_folder = None
        self.window._update_primary_action()
        self.assertEqual(self.window.primary_btn.text(), '選擇照片資料夾')
        self.assertTrue(self.window.folder_btn.isHidden())
        with patch.object(self.window, 'choose_folder') as choose:
            self.window.primary_btn.click()
            choose.assert_called_once()
        self.window.current_folder = str(self.folder)
        empty = dict(groups=[], not_ready=[], no_capture_time=[], exiftool_missing=False)
        with patch.object(gui, 'analysis_candidates', return_value=[]), patch.object(gui, 'group_folder', return_value=empty):
            self.window._refresh_workflow_state()
            self.assertEqual(self.window._primary_action, 'folder')
            self.assertIn('沒有相似照片', self.window.primary_btn.text())

    def test_search_sort_filter_do_not_group_or_scan_analysis_candidates(self):
        self.view.exit()
        with patch.object(gui, 'group_folder', side_effect=AssertionError('unexpected grouping')) as group, patch.object(
                gui, 'analysis_candidates', side_effect=AssertionError('unexpected metadata scan')) as candidates:
            for query in ['a', 'ab', 'abc', '']:
                self.window.search_edit.setText(query)
            self.window.sort_combo.setCurrentIndex(1)
            self.window.toggle_sort_order()
            self.window.filter_combo.setCurrentIndex(1)
            self.window.refresh_list()
            self.window._update_primary_action()
        group.assert_not_called()
        candidates.assert_not_called()

    def test_review_does_not_expand_window_beyond_requested_height(self):
        self.window.show()
        self.view.exit()
        self.window.resize(1420, 820)
        QTest.qWait(40)
        height = self.window.height()
        self.window.show_groups()
        QTest.qWait(80)
        self.assertLessEqual(self.window.height(), height)
        self.assertGreaterEqual(self.window.preview.height(), 220)
        self.assertLessEqual(self.view.footer.geometry().bottom(), self.window.height())

    def test_xmp_write_stays_locked_across_workflow_refresh(self):
        from PyQt6.QtGui import QCloseEvent
        self.view.exit()
        self.window.current_file_path = self.paths[0]
        self.window.analyze_current_btn.setEnabled(False)
        def write(folder, on_progress, errors):
            self.assertTrue(self.window._xmp_busy)
            self.window.refresh_list()
            self.window._refresh_workflow_state()
            self.assertFalse(self.window.primary_btn.isEnabled())
            self.assertEqual(self.window.primary_btn.text(), '寫入 XMP 中…')
            self.assertFalse(self.window.group_btn.isEnabled())
            with patch.object(self.window, '_start_analysis') as analyze:
                self.window.analyze_current()
                self.window.analyze_all()
                self.window._run_primary_action()
            analyze.assert_not_called()
            self.window.show_groups()
            self.assertFalse(self.view.active)
            close = QCloseEvent()
            self.window.closeEvent(close)
            self.assertFalse(close.isAccepted())
            on_progress(1, 1, self.paths[0])
            return 1, 0, 0
        with patch.object(gui, 'write_xmp_ratings', side_effect=write), patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), patch.object(
                gui.QMessageBox, 'information'):
            self.window.write_xmp()
        self.assertFalse(self.window._xmp_busy)
        self.assertTrue(self.window.primary_btn.isEnabled())
        self.assertFalse(self.window.analyze_current_btn.isEnabled())

    def test_unreadable_database_blocks_review_delete(self):
        with patch.object(self.view, 'get_scores', side_effect=OSError('database unavailable')), patch.object(
                gui, 'delete_photos') as delete, patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), patch.object(
                gui.QMessageBox, 'warning') as warning:
            self.view.delete_discarded()
        delete.assert_not_called()
        self.assertIn('未執行刪除', warning.call_args.args[2])

    def test_explicit_refresh_and_review_exit_update_cached_count_and_status(self):
        self.view.exit()
        empty = dict(groups=[], not_ready=[], no_capture_time=[], exiftool_missing=False)
        with patch.object(gui, 'group_folder', return_value=empty) as group:
            self.window._refresh_workflow_state()
            self.window.refresh_list()
            self.window.search_edit.setText('x')
            group.assert_called_once_with(str(self.folder), 'similar')
        self.assertEqual(self.window._group_count, 0)
        self.assertIn('目前沒有', self.window.progress_label.text())
        self.window.search_edit.clear()
        self.window.show_groups()
        self.assertEqual(self.window._group_count, 1)
        with patch.object(gui, 'group_folder', return_value=empty) as group:
            self.view.exit()
            group.assert_called_once_with(str(self.folder), 'similar')
        self.assertEqual(self.window._group_count, 0)
        self.assertNotIn('找到 1 組', self.window.progress_label.text())

    def test_selected_discarded_photo_has_neutral_outline(self):
        self.view.choose_photo(self.paths[0])
        card, image, toggle = self.view.cards[self.paths[0]]
        self.assertIn('#b8c2cc', card.styleSheet())
        self.assertNotIn('background', card.styleSheet())
        self.assertEqual(toggle.text(), '丟掉')
        self.assertEqual(image.graphicsEffect().opacity(), .35)

    def test_inline_state_and_navigation(self):
        self.assertTrue(self.view.active)
        self.assertFalse(self.window.weight_slider.isEnabled())
        self.assertEqual(self.view.review.discard_paths(), [self.paths[0]])
        self.view.move_photo(1)
        self.assertEqual(self.view.selected_path, self.paths[0])
        self.view.toggle_current()
        self.assertEqual(self.view.review.summary(), (2, 0))
        self.assertIn('已調整', self.view.groups.item(0).text())
        self.window.show_groups()
        self.assertFalse(self.view.active)
        self.assertTrue(self.window.weight_slider.isEnabled())
        self.window.show_groups()
        self.assertEqual(self.view.review.summary(), (1, 1))

    def test_cancel_never_calls_delete(self):
        with patch.object(gui, 'delete_photos') as delete, patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.No):
            self.view.delete_discarded()
        delete.assert_not_called()

    def test_all_discarded_warning_cancel(self):
        self.view.toggle(self.paths[1])
        with patch.object(gui, 'delete_photos') as delete, patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.No) as ask:
            self.view.delete_discarded()
        self.assertIn('全部都會丟掉', ask.call_args.args[2])
        delete.assert_not_called()

    def test_delete_exact_decisions_then_regroup(self):
        # Mock the backend: update DB only to emulate success, leave all files intact.
        expected = self.view.review.discard_paths()
        def delete(paths):
            self.assertEqual(paths, expected)
            conn = gui.get_conn()
            conn.executemany('DELETE FROM photos_v2 WHERE file_path = ?', [(p,) for p in paths])
            conn.commit()
            conn.close()
            return len(paths), []
        with patch.object(gui, 'delete_photos', side_effect=delete) as deleted, patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), patch.object(
                gui.QMessageBox, 'information'):
            self.view.delete_discarded()
        deleted.assert_called_once_with(expected)
        self.assertFalse(self.view.active)
        self.assertEqual(self.window._group_count, 0)
        self.assertIn('目前沒有', self.window.progress_label.text())
        self.assertEqual(gui.group_folder(str(self.folder))['groups'], [])
        self.assertTrue(all(Path(p).exists() for p in self.paths))

    def test_delete_failure_keeps_group_and_reason(self):
        with patch.object(gui, 'delete_photos', return_value=(0, [('甲.jpg', 'permission denied')])), patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), patch.object(
                gui.QMessageBox, 'information') as info:
            self.view.delete_discarded()
        self.assertTrue(self.view.active)
        self.assertIn('permission denied', info.call_args.args[2])
        self.assertEqual(len(self.view.review.groups), 1)
        self.assertEqual(self.window._group_count, 1)
        self.assertIn('找到 1 組', self.window.progress_label.text())

    def test_changed_photo_blocks_deletion_and_invalidates_old_group(self):
        Path(self.paths[0]).write_bytes(b'new photo at the same path')
        with patch.object(gui, 'delete_photos') as delete, patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), patch.object(
                gui.QMessageBox, 'warning') as warning:
            self.view.delete_discarded()
        delete.assert_not_called()
        warning.assert_called_once()
        self.assertFalse(gui.get_photo(self.paths[0])[13])
        self.assertEqual(self.view.review.groups, [])
        self.assertTrue(all(Path(p).exists() for p in self.paths))

    def test_kept_photo_changed_while_confirming_blocks_deletion(self):
        kept = self.paths[1]
        def confirm(*args):
            Path(kept).write_bytes(b'replaced reference photo')
            return gui.QMessageBox.StandardButton.Yes
        with patch.object(gui, 'delete_photos') as delete, patch.object(
                gui.QMessageBox, 'question', side_effect=confirm), patch.object(
                gui.QMessageBox, 'warning'):
            self.view.delete_discarded()
        delete.assert_not_called()
        self.assertFalse(gui.get_photo(kept)[13])

    def test_updated_scores_block_old_decisions(self):
        gui.update_analysis(self.paths[0], dict(aesthetic_score=10, technical_score=10,
                            feature_vector=[1.0] + [0.0] * 1279), .6)
        with patch.object(gui, 'delete_photos') as delete, patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), patch.object(
                gui.QMessageBox, 'warning'):
            self.view.delete_discarded()
        delete.assert_not_called()

    def test_xmp_partial_failure_is_presented_as_a_file_failure(self):
        with patch.object(gui, 'delete_photos', return_value=(1, [('甲.xmp', 'locked')])), patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), patch.object(
                gui.QMessageBox, 'information') as info:
            self.view.delete_discarded()
        self.assertIn('已移到資源回收筒：1 張', info.call_args.args[2])
        self.assertIn('未能移走 1 個檔案（含 XMP）', info.call_args.args[2])
        self.assertIn('甲.xmp：locked', info.call_args.args[2])

    def test_failed_delete_preserves_current_photo(self):
        self.window.current_file_path = self.paths[0]
        with patch.object(gui, 'delete_photos', return_value=(0, [('甲.jpg', 'locked')])), patch.object(
                gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), patch.object(
                gui.QMessageBox, 'information'):
            self.view.delete_discarded()
        self.assertEqual(self.window.current_file_path, self.paths[0])

    def test_thumbnail_cache_refreshes_replaced_file_and_ignores_old_result(self):
        from PIL import Image
        from grouping_thumbnails import ThumbnailCache
        self.view.cache.pool.waitForDone()
        app.processEvents()
        cache = ThumbnailCache()
        ready = []
        cache.ready.connect(lambda path, image, error: ready.append(image.pixelColor(0, 0).name()))
        path = self.paths[0]
        with patch('grouping_thumbnails.load_thumbnail', return_value=Image.new('RGB', (20, 30), 'red')) as load:
            cache.request(path)
            cache.pool.waitForDone()
            app.processEvents()
            cache.request(path)
            self.assertEqual(load.call_count, 1)
            # Queue a result, then replace its source before the UI consumes it.
            cache.clear()
            cache.request(path)
            cache.pool.waitForDone()
            Path(path).write_bytes(b'replacement for cached thumbnail')
            load.return_value = Image.new('RGB', (20, 30), 'blue')
            cache.request(path)
            cache.pool.waitForDone()
            ready.clear()
            app.processEvents()
        self.assertEqual(ready, ['#0000ff'])
        self.assertEqual(cache.cache[path][0].pixelColor(0, 0).name(), '#0000ff')

    def test_cache_runs_in_background_deduplicates_and_ignores_old_folder(self):
        import threading
        from PIL import Image
        from grouping_thumbnails import ThumbnailCache
        cache = ThumbnailCache()
        calls = []
        ready = []
        main_thread = threading.get_ident()
        def load(path):
            calls.append((path, threading.get_ident()))
            return Image.new('RGB', (20, 30))
        cache.ready.connect(lambda path, image, error: ready.append(path))
        # Drain host jobs before patching the module-level loader: all caches
        # share the production pool, including the host's queued fixture photos.
        self.view.cache.pool.waitForDone()
        app.processEvents()
        with patch('grouping_thumbnails.load_thumbnail', side_effect=load):
            cache.request('one')
            cache.request('one')
            cache.pool.waitForDone()
            app.processEvents()
            self.assertEqual(ready, ['one'])
            cache.request('one')
            self.assertEqual(ready, ['one', 'one'])
            self.assertEqual(len(calls), 1)
            self.assertNotEqual(calls[0][1], main_thread)
            cache.request('old-folder')
            cache.clear()
            cache.pool.waitForDone()
            app.processEvents()
            self.assertFalse(cache.cache)
            self.assertNotIn('old-folder', ready)

    def test_weight_reenter_refreshes_recommendation(self):
        self.window.show_groups()
        gui.recompute_scores(.9, str(self.folder))
        self.window.show_groups()
        self.assertEqual(self.view.review.groups[0]['best_path'], self.paths[0])

    def test_keyboard_and_folder_reset(self):
        self.window.show()
        self.window.activateWindow()
        QTest.qWait(60)
        QTest.keyClick(self.window, Qt.Key.Key_Space)
        self.assertEqual(self.view.review.summary(), (0, 2))
        QTest.keyClick(self.window, Qt.Key.Key_Escape)
        self.assertFalse(self.view.active)
        self.view.cache.cache['old'] = (None, '')
        self.view.folder_changed()
        self.assertFalse(self.view.cache.cache)

if __name__ == '__main__':
    unittest.main()
