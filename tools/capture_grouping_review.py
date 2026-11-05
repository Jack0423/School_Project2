"""Capture the real Qt window with 40 actual ARWs; source photos stay read-only.

Run from the project: python tools/capture_grouping_review.py PHOTO_FOLDER OUTPUT_DIR
The output includes screenshots of the main window and the grouping review,
a silent scripted demo and measured event-loop gaps.

截圖方式（2026-11-05 改）：
- 視窗真的開在螢幕最上層，用一般的 Windows 繪圖，字型與平常開程式相同；
  Windows 上直接擷取螢幕上的視窗範圍（含標題列），其他系統改用 Qt 的 grab()。
- 40 張照片以連結放進 output/testphoto_前40張（不複製、不修改原檔），
  畫面上顯示的就是這個資料夾的路徑；結束後刪除這個資料夾。資料庫用暫存檔。
"""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import shutil
import tempfile
import time
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QThreadPool, Qt
import main_v2 as gui

LINK_FOLDER = ROOT / 'output' / 'testphoto_前40張'


def link_photos(paths, folder):
    """硬連結（同一磁碟、不需系統管理員權限）優先，不行再用符號連結。"""
    for path in paths:
        try:
            os.link(path, folder/path.name)
        except OSError:
            (folder/path.name).symlink_to(path)


def square_corners(window):
    """Windows 11 預設圓角，角落會截到後面的桌面；截圖時改成直角。"""
    if sys.platform != 'win32':
        return
    import ctypes
    pref = ctypes.c_int(1)  # DWMWCP_DONOTROUND
    ctypes.windll.dwmapi.DwmSetWindowAttribute(int(window.winId()), 33, ctypes.byref(pref), ctypes.sizeof(pref))


def screen_grab(window, path):
    """截螢幕上這個視窗看得到的範圍（含標題列，不含 Windows 外圍看不見的邊框）。"""
    QApplication.processEvents()
    if sys.platform != 'win32':
        window.grab().save(str(path))
        return
    import ctypes
    from ctypes import wintypes
    from PIL import ImageGrab
    rect = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(  # DWMWA_EXTENDED_FRAME_BOUNDS
        int(window.winId()), 9, ctypes.byref(rect), ctypes.sizeof(rect))
    ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True).save(str(path))


def main():
    source, output = map(lambda p: Path(p).resolve(), sys.argv[1:3])
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob('*.ARW'))[:40]
    if len(paths) != 40:
        raise SystemExit('Need at least 40 actual ARW files')
    folder = LINK_FOLDER
    if folder.exists():
        raise SystemExit(f'{folder} 已存在，請先確認內容後刪除')
    folder.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix='review-40-arw-') as tmp:
            link_photos(paths, folder)
            gui.DB_PATH = str(Path(tmp)/'photos.db')
            run(source, output, folder)
    finally:
        # 只刪連結，原始照片不受影響
        shutil.rmtree(folder)


def run(source, output, folder):
    app = QApplication([])
    window = gui.PhotoManagerV2()
    # 固定在最上層，截圖時不會被其他視窗擋住
    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    window.resize(1700, 1050)
    window.current_folder = str(folder)
    window.photo_paths = {p.name: str(p) for p in folder.iterdir()}
    for name, path in window.photo_paths.items():
        gui.insert_photo(name, path)
    window.refresh_list()
    window.move(30, 30)
    window.show()
    square_corners(window)
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
            # 主視窗：選清單第一張（預設依綜合分由高到低），等預覽與結果面板顯示後截圖
            window.photo_list.setCurrentRow(0)
            window.show_photo(window.photo_list.item(0))
            state['phase'] = 'main'
            state['main_start'] = now
            return
        if state['phase'] == 'main':
            if now - state['main_start'] < 1.5:
                return
            screen_grab(window, output/'main-40-arw.png')
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
            screen_grab(window, output/'review-40-arw.png')
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
                screen_grab(window, output/'review-adjusted.png')
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
