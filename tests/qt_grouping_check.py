"""Real Qt grouping checks using temporary files and deterministic inference metadata."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import types
import unittest
from unittest.mock import patch
fake = types.ModuleType('ai_inference')
fake.IMAGE_EXTENSIONS = {'.jpg'}
fake.RAW_EXTENSIONS = {'.arw'}
fake.TECH_ISSUE_THRESHOLD = 40
fake.AESTHETIC_EXCELLENT_THRESHOLD = 60
fake.AESTHETIC_WEIGHT = .6
fake.load_image = lambda path: object()
fake.evaluate_photo = lambda *args, **kwargs: None
fake.LAST_ERROR = None
fake.aesthetic_grade = lambda score: '優秀' if score >= 60 else '普通'
fake.model_version = lambda: 'test-model'
fake.combine_scores = lambda a,b,aesthetic_weight: a*aesthetic_weight+b*(1-aesthetic_weight)
sys.modules['ai_inference'] = fake
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import main_v2 as gui
from grouping_view import GroupingDialog
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
        self.addCleanup(self.window.close)
        self.window.current_folder = str(self.folder)
        self.paths = []
        for name, aes, tech, axis in [('甲.jpg',80,40,0),('乙.jpg',60,75,0),('單張.jpg',90,90,1)]:
            path = self.folder/name
            path.write_bytes(b'temporary test fixture')
            self.paths.append(str(path))
            vector = [0.0]*1280
            vector[axis] = 1.0
            gui.insert_photo(name,str(path))
            gui.update_analysis(str(path),dict(aesthetic_score=aes,technical_score=tech,feature_vector=vector),.6)
        self.window.photo_paths = {Path(p).name:p for p in self.paths}
        gui.recompute_scores(.6,str(self.folder))
        self.dialog = GroupingDialog(lambda mode:gui.group_folder(str(self.folder),mode),gui.non_best_paths,lambda p:None,self.window._select_group_paths)
        self.addCleanup(self.dialog.close)

    def selected(self):
        return {item.data(Qt.ItemDataRole.UserRole) for item in self.window.photo_list.selectedItems()}

    def test_select_only_non_best(self):
        self.window._select_group_paths([self.paths[2]])
        self.dialog.choose_non_best()
        self.assertEqual(self.selected(),{self.paths[0]})
        self.assertTrue(all(Path(p).exists() for p in self.paths))

    def test_delete_then_regroup(self):
        self.dialog.choose_non_best()
        original = gui.delete_photos
        def delete_temporary(paths):
            self.assertEqual(set(paths),{self.paths[0]})
            return original(paths,trash=lambda p:(Path(p).unlink() is None, ''))
        with patch.object(gui,'delete_photos',side_effect=delete_temporary), patch.object(gui.QMessageBox,'question',return_value=gui.QMessageBox.StandardButton.Yes), patch.object(gui.QMessageBox,'information'):
            self.window.delete_selected()
        self.dialog.refresh()
        self.assertFalse(Path(self.paths[0]).exists())
        self.assertTrue(Path(self.paths[1]).exists())
        self.assertEqual(self.dialog.result['groups'],[])
        self.assertEqual(set(self.dialog.result['singles']),set(self.paths[1:]))
        self.assertFalse(self.dialog.select_button.isEnabled())

    def test_weight_refresh_changes_best_and_selection(self):
        self.assertEqual(self.dialog.result['groups'][0]['best_path'],self.paths[1])
        gui.recompute_scores(.9,str(self.folder))
        self.dialog.choose_non_best()
        group=self.dialog.result['groups'][0]
        self.assertEqual(group['best_path'],self.paths[0])
        self.assertEqual(sum(p['is_best'] for p in group['photos']),1)
        self.assertEqual(self.selected(),{self.paths[1]})

if __name__ == '__main__':
    unittest.main()
