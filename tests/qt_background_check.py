"""Isolated Qt integration check with fake inference, real QThread/signals/SQLite."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import threading
import time
import types
from unittest.mock import patch

fake = types.ModuleType('ai_inference')
fake.IMAGE_EXTENSIONS = {'.jpg'}
fake.RAW_EXTENSIONS = {'.arw'}
fake.TECH_ISSUE_THRESHOLD = 40
fake.AESTHETIC_EXCELLENT_THRESHOLD = 60
fake.AESTHETIC_WEIGHT = .6
fake.LAST_ERROR = None
fake.model_version = lambda: 'test-model'
fake.combine_scores = lambda a,b,aesthetic_weight: a*aesthetic_weight+b*(1-aesthetic_weight)
threads = []
main_id = threading.get_ident()
def evaluate(path, **kwargs):
    threads.append(threading.get_ident())
    time.sleep(.08)
    return dict(aesthetic_score=50, technical_score=60, overall_score=54,
                aesthetic_weight=kwargs['aesthetic_weight'], technical_weight=1-kwargs['aesthetic_weight'],
                feature_vector=[1.0]*1280)
fake.load_image = lambda path: object()
fake.evaluate_photo = evaluate
sys.modules['ai_inference'] = fake
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent
import main_v2 as gui

app = QApplication([])
with tempfile.TemporaryDirectory() as tmp:
    gui.DB_PATH = str(Path(tmp)/'photos.db')
    window = gui.PhotoManagerV2()
    window.current_folder = tmp
    paths = [str(Path(tmp)/f'{i}.jpg') for i in range(3)]
    conn = gui.get_conn()
    for path in paths:
        conn.execute('INSERT INTO photos_v2 (file_path,file_name) VALUES (?,?)',(path,Path(path).name))
    conn.commit()
    conn.close()
    window.photo_paths = {Path(p).name:p for p in paths}
    # Keep reports inside temporary test directory.
    gui.__file__ = str(Path(tmp)/'main_v2.py')
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(10)
    original_update = gui.update_analysis
    def checked_update(*args):
        assert threading.get_ident() == main_id, 'SQLite update off GUI thread'
        return original_update(*args)
    def wait_done():
        deadline = time.monotonic()+8
        while window._analysis_thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.002)
        assert window._analysis_thread is None, 'Thread failed to finish'
    with patch.object(gui, 'update_analysis', checked_update):
        window.analyze_all()
        first = window._analysis_thread
        window.analyze_current()
        assert window._analysis_thread is first
        assert not window.weight_slider.isEnabled()
        close = QCloseEvent()
        with patch.object(gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.No):
            window.closeEvent(close)
        assert not close.isAccepted()
        assert window._analysis_thread is first, 'Answering No must keep the analysis running'
        with patch.object(gui.QMessageBox, 'question', side_effect=AssertionError('delete invoked')):
            window.delete_selected()
        wait_done()
        assert len(ticks) >= 5, 'UI timer blocked'
        assert threads and main_id not in threads
        assert len(set(threads)) == 1, 'Concurrent inference'
        assert window.analyze_all_btn.isEnabled() and window.weight_slider.isEnabled()
        assert not window.group_btn.isEnabled(), 'Prior disabled state lost'
        assert all(gui.get_photo(p)[13] == 1 for p in paths)
        assert Path(window._analysis_output).exists()
        # Single-image analysis uses the same asynchronous owner.
        window.current_file_path = paths[0]
        with patch.object(window, 'show_result'):
            window.analyze_current()
            assert window._analysis_thread is not None
            wait_done()
        # Fatal pipeline errors restore controls and preserve the error.
        with patch('qt_batch_worker.analyze_batch', side_effect=OSError('disk full')):
            window._start_analysis(paths)
            wait_done()
        assert window.analyze_all_btn.isEnabled()
        assert 'disk full' in window._analysis_error
        # DB failures also release thread ownership and restore controls.
        with patch.object(gui, 'update_analysis', side_effect=OSError('DB unavailable')):
            window._start_analysis(paths[:1])
            wait_done()
        assert '資料庫' in window._analysis_error
        assert window.weight_slider.isEnabled()
        many = [str(Path(tmp)/f'many{i}.jpg') for i in range(20)]
        conn = gui.get_conn()
        for path in many:
            conn.execute('INSERT INTO photos_v2 (file_path,file_name) VALUES (?,?)',(path,Path(path).name))
        conn.commit()
        conn.close()
        window._start_analysis(many)
        assert window.cancel_btn.isEnabled()
        window.cancel_analysis()
        wait_done()
        done = sum(gui.get_photo(p)[13] == 1 for p in many)
        assert done < len(many), 'Cancel did not stop the batch'
        assert '已取消' in window.result_label.text()
        assert window.analyze_all_btn.isEnabled() and not window.cancel_btn.isEnabled()
        remaining = [p for p in many if gui.get_photo(p)[13] == 0]
        window._start_analysis(remaining)
        with patch.object(gui.QMessageBox, 'question', return_value=gui.QMessageBox.StandardButton.Yes), \
                patch.object(window, 'close') as close_window:
            close = QCloseEvent()
            window.closeEvent(close)
            assert not close.isAccepted(), 'Must wait for the current photo before closing'
            wait_done()
            for _ in range(5):
                app.processEvents()
            assert close_window.called, 'Window should close after the cancel finishes'
    timer.stop()
    window.close()
print('PASS: real Qt background execution, responsive timer, serial inference, GUI-thread DB, guards, cleanup, failures and cancel')
