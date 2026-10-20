"""Decode small RGB images off the UI thread; QPixmap is created by the view."""
from io import BytesIO
import os
from pathlib import Path
from PIL import Image, ImageOps
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QImage


def load_thumbnail(path, size=(240, 160)):
    import ai_inference
    image = None
    if Path(path).suffix.lower() in ai_inference.RAW_EXTENSIONS:
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                thumb = raw.extract_thumb()
                flip = raw.sizes.flip
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(BytesIO(thumb.data)) as embedded:
                        orientation = embedded.getexif().get(274, 1)
                        image = ImageOps.exif_transpose(embedded).convert('RGB')
                else:
                    orientation = 1
                    image = Image.fromarray(thumb.data).convert('RGB')
                # Embedded EXIF takes precedence; LibRaw flip is the fallback.
                if orientation == 1:
                    rotation = {3: Image.Transpose.ROTATE_180,
                                5: Image.Transpose.ROTATE_90,
                                6: Image.Transpose.ROTATE_270}.get(flip)
                    if rotation is not None:
                        image = image.transpose(rotation)
        except Exception:
            image = None  # RAW without an embedded thumbnail uses the standard loader.
    if image is None:
        image = Image.fromarray(ai_inference.load_image(path)).convert('RGB')
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


class ThumbnailSignals(QObject):
    done = pyqtSignal(int, str, QImage, str)


class ThumbnailJob(QRunnable):
    def __init__(self, request_id, path, signals):
        super().__init__()
        self.request_id, self.path, self.signals = request_id, path, signals

    def run(self):
        try:
            image = load_thumbnail(self.path)
            data = image.tobytes()
            result = QImage(data, image.width, image.height, image.width * 3,
                            QImage.Format.Format_RGB888).copy()
            error = ''
        except Exception as exc:
            result, error = QImage(), str(exc)
        self.signals.done.emit(self.request_id, self.path, result, error)


class ThumbnailCache(QObject):
    ready = pyqtSignal(str, QImage, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Global pool keeps jobs and their signals safe if a window closes mid-decode.
        self.pool = QThreadPool.globalInstance()
        self.signals = ThumbnailSignals()
        self.signals.done.connect(self._done)
        self.next_request = 0
        self.cache, self.pending, self.signatures = {}, {}, {}

    @staticmethod
    def signature(path):
        try:
            stat = os.stat(path)
            return stat.st_size, stat.st_mtime_ns
        except OSError:
            return None

    def clear(self):
        self.cache.clear()
        self.pending.clear()
        self.signatures.clear()

    def request(self, path):
        signature = self.signature(path)
        if path in self.cache and self.signatures.get(path) == signature:
            image, error = self.cache[path]
            self.ready.emit(path, image, error)
        elif path not in self.pending or self.pending[path][1] != signature:
            self.cache.pop(path, None)
            self.signatures.pop(path, None)
            self.next_request += 1
            self.pending[path] = (self.next_request, signature)
            self.pool.start(ThumbnailJob(self.next_request, path, self.signals))

    def _done(self, request_id, path, image, error):
        pending = self.pending.get(path)
        if pending is None or pending[0] != request_id:
            return
        del self.pending[path]
        if self.signature(path) != pending[1]:
            self.request(path)
            return
        self.cache[path] = image, error
        self.signatures[path] = pending[1]
        self.ready.emit(path, image, error)
