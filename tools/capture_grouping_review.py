"""Capture the real Qt review with 40 actual ARWs; source photos stay read-only.

Run from the project: python tools/capture_grouping_review.py PHOTO_FOLDER OUTPUT_DIR
The output includes screenshots, a silent scripted demo and measured event-loop gaps.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import time
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QThreadPool
import main_v2 as gui


def main():
    source, output = map(lambda p: Path(p).resolve(), sys.argv[1:3])
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob('*.ARW'))[:40]
    if len(paths) != 40:
        raise SystemExit('Need at least 40 actual ARW files')
    with tempfile.TemporaryDirectory(prefix='review-40-arw-') as tmp:
        folder = Path(tmp)/'photos'
        folder.mkdir()
        # Links isolate the demonstration database and folder without duplicating RAWs.
        for path in paths:
            (folder/path.name).symlink_to(path)
        gui.DB_PATH = str(Path(tmp)/'photos.db')
        app = QApplication([])
        window = gui.PhotoManagerV2()
        window.resize(1700, 1050)
        window.current_folder = str(folder)
        window.folder_label.setText('40 張 ARW · 真實照片驗證')
        window.photo_paths = {p.name: str(p) for p in folder.iterdir()}
        for name, path in window.photo_paths.items():
            gui.insert_photo(name, path)
        window.refresh_list()
        window.show()
        window.analyze_all()
        metrics = {'photo_count': 40, 'source': str(source), 'scripted_capture': True}
        timer = QTimer()
        state = {'phase': 'analysis', 'tick': time.perf_counter(), 'gaps': [], 'frames': []}
        video = None

        def frame():
            pix = window.grab()
            image = pix.toImage().convertToFormat(gui.QImage.Format.Format_RGB888)
            data = image.bits().asstring(image.sizeInBytes())
            arr = np.frombuffer(data, np.uint8).reshape(image.height(), image.bytesPerLine())
            arr = arr[:, :image.width()*3].reshape(image.height(), image.width(), 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        def tick():
            nonlocal video
            now = time.perf_counter()
            if state['phase'] == 'analysis':
                if window._analysis_thread is not None:
                    return
                metrics['analysis_hint'] = window.progress_label.text()
                metrics['analysis_failures'] = window._analysis_failures
                state['start'] = time.perf_counter()
                window.show_groups()
                state['tick'] = state['start']
                state['phase'] = 'review'
                metrics['groups'] = len(window.group_review.review.groups)
                if not metrics['groups']:
                    raise RuntimeError('Actual 40 ARWs produced no groups')
                return
            state['gaps'].append(now-state['tick'])
            state['tick'] = now
            view = window.group_review
            if 'first_group_thumbnails_seconds' not in metrics and view.cards and all(
                    path in view.cache.cache for path in view.cards):
                metrics['first_group_thumbnails_seconds'] = now-state['start']
            elapsed = now-state['start']
            if elapsed < 2:
                return
            if video is None:
                window.grab().save(str(output/'review-40-arw.png'))
                arr = frame()
                video = cv2.VideoWriter(str(output/'grouping-demo.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'), 10, (arr.shape[1], arr.shape[0]))
                state['demo_start'] = now
                state['step'] = -1
                state['last_frame'] = now - .1
            demo = now-state['demo_start']
            step = int(demo//2)
            if step != state['step']:
                state['step'] = step
                if step == 1: view.move_photo(1)
                elif step == 2: view.toggle_current()
                elif step == 3: view.move_group(1)
                elif step == 4: view.move_group(1)
                elif step == 5:
                    window.grab().save(str(output/'review-adjusted.png'))
                elif step >= 6:
                    timer.stop()
                    video.release()
                    metrics['max_event_loop_gap_seconds'] = max(state['gaps'])
                    metrics['p95_event_loop_gap_seconds'] = float(np.percentile(state['gaps'], 95))
                    (output/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
                    window.group_review.exit()
                    QThreadPool.globalInstance().waitForDone()
                    window.close()
                    app.quit()
                    return
            if now - state['last_frame'] >= .1:
                video.write(frame())
                state['last_frame'] = now
        timer.timeout.connect(tick)
        timer.start(20)
        app.exec()
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
