"""Qt adapter: decode in the pipeline pool; inference in this one QThread.

Signals are queued to the GUI thread. Never touch widgets or SQLite here.
"""
from PyQt6.QtCore import QThread, pyqtSignal
from batch_pipeline import analyze_batch


class BatchAnalysisThread(QThread):
    progress = pyqtSignal(int, int, object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, paths, output_path, aesthetic_weight, parent=None):
        super().__init__(parent)
        self.paths = tuple(paths)
        self.output_path = str(output_path)
        self.aesthetic_weight = float(aesthetic_weight)

    def run(self):
        try:
            summary = analyze_batch(
                self.paths, self.output_path,
                aesthetic_weight=self.aesthetic_weight,
                on_progress=self.progress.emit,
            )
            self.completed.emit(summary)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
