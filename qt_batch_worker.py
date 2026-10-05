"""Qt adapter: decode in the pipeline pool; inference in this one QThread.

Signals are queued to the GUI thread. Never touch widgets or SQLite here.
"""
import threading

from PyQt6.QtCore import QThread, pyqtSignal
from batch_pipeline import analyze_batch
from burst_metadata import read_metadata

# 拍攝時間一次讀幾張。ExifTool 每次啟動約 0.4 秒、每張約 25 毫秒：
# 整個資料夾一次讀完最快（325 張約 8 秒），但取消時得等它讀完；
# 一次 32 張，取消最多多等約 1.2 秒，總共多花的啟動時間也和分析同時進行。
METADATA_CHUNK = 32


class BatchAnalysisThread(QThread):
    progress = pyqtSignal(int, int, object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    # {照片路徑: 連拍欄位}，每讀完一段送一次；都會在 completed／failed 之前送到
    capture_metadata = pyqtSignal(object)

    def __init__(self, paths, output_path, aesthetic_weight, parent=None, metadata_paths=()):
        super().__init__(parent)
        self.paths = tuple(paths)
        self.output_path = str(output_path)
        self.aesthetic_weight = float(aesthetic_weight)
        # 要讀拍攝時間的照片（資料庫裡還沒有的）；由主執行緒查好傳進來，這裡不碰資料庫
        self.metadata_paths = tuple(metadata_paths)
        self._stop = threading.Event()

    def cancel(self):
        """要求停止：手上這張推論完就停（由前台的「取消」按鈕呼叫，可以從主執行緒呼叫）。"""
        self._stop.set()

    @property
    def cancel_requested(self):
        return self._stop.is_set()

    def _read_capture_metadata(self, abort):
        """
        連拍分組要的拍攝時間與相機，和分析同時讀：ExifTool 是另一個行程，只讀檔頭，不搶推論。
        沒有 ExifTool 就不讀（連拍分組會提示安裝）；讀失敗的那段不送，之後重新分析時會再讀。
        """
        paths = self.metadata_paths
        for start in range(0, len(paths), METADATA_CHUNK):
            if self._stop.is_set() or abort.is_set():
                return
            try:
                found = read_metadata(paths[start:start + METADATA_CHUNK])
            except Exception as exc:
                print(f"[拍攝時間] 讀取失敗：{type(exc).__name__}: {exc}")
                continue
            if found is None:
                return
            if found:
                self.capture_metadata.emit(found)

    def run(self):
        abort = threading.Event()
        reader = threading.Thread(target=self._read_capture_metadata, args=(abort,), daemon=True)
        reader.start()
        try:
            summary = analyze_batch(
                self.paths, self.output_path,
                aesthetic_weight=self.aesthetic_weight,
                on_progress=self.progress.emit,
                should_stop=self._stop.is_set,
            )
        except Exception as exc:
            abort.set()
            reader.join()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        reader.join()
        self.completed.emit(summary)
