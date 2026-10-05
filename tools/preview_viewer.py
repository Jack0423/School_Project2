import sys
from pathlib import Path

# 這支程式在子資料夾裡；共用模組（ai_inference、common 等）在專案根目錄
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QFileDialog, QGraphicsView, QGraphicsScene, QLabel)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
from raw_processor import RawProcessor

class AdvancedGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.pixmap_item = None
        self.setStyleSheet("background: transparent; border: none;")

    def set_pixmap(self, pixmap):
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if self.pixmap_item is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        cur_scale = self.transform().m11()
        if (cur_scale > 20.0 and factor > 1.0) or (cur_scale < 0.05 and factor < 1.0):
            return
        self.scale(factor, factor)

class MainViewer(QMainWindow):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.setWindowTitle("照片預覽檢視器 (唯讀檢視模式)")
        self.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")
        self.resize(1100, 680)

        self.view = AdvancedGraphicsView()
        self.info_label = QLabel("正在讀取影像...")
        self.info_label.setFixedHeight(45)
        self.info_label.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px; padding: 5px;")

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.info_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        QTimer.singleShot(100, self.run_pipeline)

    def run_pipeline(self):
        try:
            # 僅進行解碼與渲染，不執行任何寫檔或 XMP 修改
            rgb = RawProcessor.decode_image_rgb(self.file_path)

            h, w, ch = rgb.shape
            img_np = np.ascontiguousarray(rgb)
            q_img = QImage(img_np.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.view.set_pixmap(QPixmap.fromImage(q_img))

            ext = os.path.splitext(self.file_path)[1].upper()
            self.info_label.setText(
                f"檔案: {os.path.basename(self.file_path)} ({ext}) | 解析度: {w}x{h} | 狀態: 唯讀預覽 (未更動任何 XMP 檔)"
            )
        except Exception as e:
            self.info_label.setText(f"讀取失敗: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    target = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        target = sys.argv[1]
    else:
        target, _ = QFileDialog.getOpenFileName(None, "選取照片", "", "All Supported (*.arw *.cr3 *.cr2 *.nef *.dng *.jpg *.png)")

    if target:
        viewer = MainViewer(target)
        viewer.show()
        sys.exit(app.exec())