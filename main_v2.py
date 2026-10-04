import os
import shutil
import sys
import sqlite3
import traceback
import numpy as np

import ai_inference
from raw_processor import RawProcessor, SIDECAR_EXTENSIONS
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QListWidget,
    QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QMessageBox,
    QProgressBar, QFrame
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QPoint


class PhotoPreviewLabel(QLabel):
    """照片預覽：滾輪縮放、按住拖曳、雙擊還原。"""

    def __init__(self, text=""):
        super().__init__(text)
        self._original_pixmap = None
        self._zoom = 1.0
        self._dragging = False
        self._last_pos = QPoint()
        self._offset = QPoint()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def setPhotoPixmap(self, pixmap):
        self._original_pixmap = QPixmap(pixmap)
        self.resetView()

    def clearPhoto(self):
        self._original_pixmap = None
        self._zoom = 1.0
        self._offset = QPoint()
        super().clear()

    def resetView(self):
        self._zoom = 1.0
        self._offset = QPoint()
        self._render()

    def _fit_size(self):
        if self._original_pixmap is None:
            return 1, 1

        available_w = max(1, self.width() - 20)
        available_h = max(1, self.height() - 20)

        ow = self._original_pixmap.width()
        oh = self._original_pixmap.height()

        if ow <= 0 or oh <= 0:
            return 1, 1

        ratio = min(available_w / ow, available_h / oh)
        return max(1, int(ow * ratio)), max(1, int(oh * ratio))

    def _render(self):
        if self._original_pixmap is None:
            return

        fit_w, fit_h = self._fit_size()

        target_w = max(1, int(fit_w * self._zoom))
        target_h = max(1, int(fit_h * self._zoom))

        scaled = self._original_pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        canvas = QPixmap(self.size())
        canvas.fill(Qt.GlobalColor.transparent)

        from PyQt6.QtGui import QPainter
        painter = QPainter(canvas)

        x = (self.width() - scaled.width()) // 2 + self._offset.x()
        y = (self.height() - scaled.height()) // 2 + self._offset.y()

        painter.drawPixmap(x, y, scaled)
        painter.end()

        super().setPixmap(canvas)

    def wheelEvent(self, event):
        if self._original_pixmap is None:
            event.ignore()
            return

        delta = event.angleDelta().y()

        if delta == 0:
            event.ignore()
            return

        old_zoom = self._zoom

        if delta > 0:
            self._zoom = min(8.0, self._zoom * 1.20)
        else:
            self._zoom = max(1.0, self._zoom / 1.20)

        if self._zoom <= 1.0001:
            self._zoom = 1.0
            self._offset = QPoint()
        elif old_zoom > 0:
            factor = self._zoom / old_zoom
            self._offset = QPoint(
                int(self._offset.x() * factor),
                int(self._offset.y() * factor)
            )

        self._render()
        event.accept()

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._original_pixmap is not None
        ):
            self._dragging = True
            self._last_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._original_pixmap is not None:
            current = event.position().toPoint()

            if self._zoom > 1.0:
                delta = current - self._last_pos
                self._offset += delta
                self._render()

            self._last_pos = current
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._original_pixmap is not None
        ):
            self.resetView()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._original_pixmap is not None:
            self._render()


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos.db")
MODEL_VERSION = ai_inference.model_version()
TECH_THRESHOLD = ai_inference.TECH_ISSUE_THRESHOLD
AESTHETIC_THRESHOLD = ai_inference.AESTHETIC_EXCELLENT_THRESHOLD
IMAGE_EXTENSIONS = tuple(sorted(ai_inference.IMAGE_EXTENSIONS))


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS photos_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL UNIQUE,
            aesthetic_score REAL,
            technical_score REAL,
            overall_score REAL,
            aesthetic_weight REAL DEFAULT 0.8,
            technical_weight REAL DEFAULT 0.2,
            model_version TEXT DEFAULT '',
            status TEXT DEFAULT '',
            suggestion TEXT DEFAULT '',
            action TEXT DEFAULT '',
            technical_issues TEXT DEFAULT '',
            is_best INTEGER DEFAULT 0,
            analyzed INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def insert_photo(file_name, file_path):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO photos_v2 (file_name, file_path)
        VALUES (?, ?)
    """, (file_name, os.path.abspath(file_path)))
    conn.commit()
    conn.close()


def get_photo(file_path):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT file_name, file_path, aesthetic_score, technical_score,
               overall_score, aesthetic_weight, technical_weight,
               model_version, status, suggestion, action,
               technical_issues, is_best, analyzed
        FROM photos_v2
        WHERE file_path = ?
    """, (os.path.abspath(file_path),))
    row = cur.fetchone()
    conn.close()
    return row


def get_photos_in_folder(folder):
    """目前模型版本優先；舊版本資料保留，但不混入目前排序。"""
    if not folder:
        return []

    folder = os.path.abspath(folder)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT file_name, file_path, aesthetic_score, technical_score,
               overall_score, aesthetic_weight, technical_weight,
               model_version, status, suggestion, action,
               technical_issues, is_best, analyzed
        FROM photos_v2
        ORDER BY
            CASE
                WHEN analyzed = 1 AND model_version = ? THEN 0
                WHEN analyzed = 0 THEN 1
                ELSE 2
            END,
            CASE
                WHEN analyzed = 1 AND model_version = ?
                THEN overall_score
            END DESC,
            file_name ASC
    """, (MODEL_VERSION, MODEL_VERSION))

    rows = [
        row for row in cur.fetchall()
        if os.path.dirname(os.path.abspath(row[1])) == folder
        and os.path.isfile(row[1])
    ]

    conn.close()
    return rows

def update_analysis(file_path, result, aesthetic_weight):
    """status 只表示技術狀態；美感優秀另外由 aesthetic_score 判斷。"""
    aesthetic_score = float(result["aesthetic_score"])
    technical_score = float(result["technical_score"])

    overall_score = round(ai_inference.combine_scores(
        aesthetic_score,
        technical_score,
        aesthetic_weight=aesthetic_weight
    ), 2)

    technical_weight = round(1.0 - aesthetic_weight, 2)

    status = "警告" if technical_score < TECH_THRESHOLD else "正常"

    suggestion = result.get("suggestion", "")
    technical_issues = result.get("technical_issues", "")

    if isinstance(technical_issues, list):
        technical_issues = "；".join(technical_issues)
    else:
        technical_issues = str(technical_issues)

    action = "建議檢視" if status == "警告" else "建議保留"

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE photos_v2
        SET aesthetic_score = ?, technical_score = ?, overall_score = ?,
            aesthetic_weight = ?, technical_weight = ?, model_version = ?,
            status = ?, suggestion = ?, action = ?, technical_issues = ?,
            analyzed = 1
        WHERE file_path = ?
    """, (
        aesthetic_score, technical_score, overall_score,
        aesthetic_weight, technical_weight, MODEL_VERSION,
        status, suggestion, action, technical_issues,
        os.path.abspath(file_path)
    ))

    conn.commit()
    conn.close()

def recompute_scores(aesthetic_weight, folder):
    """
    只用 DB 既有美感分與技術分重算。
    不重新呼叫 evaluate_photo()。
    舊 model_version 不參與重算、排序或最佳照片競爭。
    """
    rows = get_photos_in_folder(folder)
    technical_weight = round(1.0 - aesthetic_weight, 2)

    conn = get_conn()
    cur = conn.cursor()
    analyzed_paths = []

    for row in rows:
        file_path = row[1]
        aesthetic_score = row[2]
        technical_score = row[3]
        model_version = row[7]
        analyzed = row[13]

        if (
            not analyzed
            or aesthetic_score is None
            or technical_score is None
            or model_version != MODEL_VERSION
        ):
            continue

        overall_score = round(ai_inference.combine_scores(
            float(aesthetic_score),
            float(technical_score),
            aesthetic_weight=aesthetic_weight
        ), 2)

        cur.execute("""
            UPDATE photos_v2
            SET overall_score = ?, aesthetic_weight = ?, technical_weight = ?
            WHERE file_path = ?
        """, (
            overall_score,
            aesthetic_weight,
            technical_weight,
            file_path
        ))

        analyzed_paths.append((file_path, overall_score))

    # 目前資料夾全部先取消最佳照片。
    for row in rows:
        cur.execute(
            "UPDATE photos_v2 SET is_best = 0 WHERE file_path = ?",
            (row[1],)
        )

    # ★ 只能從目前模型版本的有效分析結果產生。
    if analyzed_paths:
        best_path = max(analyzed_paths, key=lambda x: x[1])[0]
        cur.execute(
            "UPDATE photos_v2 SET is_best = 1 WHERE file_path = ?",
            (best_path,)
        )

    conn.commit()
    conn.close()


def write_xmp_ratings(folder, on_progress=None):
    """
    把目前資料夾 RAW 的星等寫進旁邊的 .xmp（李奇翰的 RawProcessor.safe_update_xmp）。

    只由「寫入 XMP 星等」按鈕呼叫，不在 recompute_scores 裡自動寫：
    已有 .xmp 的照片每張都要呼叫一次 ExifTool（實測約 0.4 秒），
    放在重算裡時，開資料夾、分析一張、滑桿每動一格都會把整個資料夾重寫一次
    （60 張 RAW 拉一格就卡 22.7 秒），而且會一直覆蓋使用者在 Lightroom 打的星等。

    只寫目前模型版本、已分析的 RAW；星等依資料庫裡目前權重下的綜合分。
    回傳 (寫入張數, 略過的非 RAW 張數, 失敗張數)。
    """
    targets = []
    skipped = 0
    for row in get_photos_in_folder(folder):
        file_path = row[1]
        overall_score = row[4]
        model_version = row[7]
        analyzed = row[13]

        if not analyzed or overall_score is None or model_version != MODEL_VERSION:
            continue
        # JPG、PNG、DNG：Lightroom 不讀它們旁邊的 .xmp
        if os.path.splitext(file_path)[1].lower() not in SIDECAR_EXTENSIONS:
            skipped += 1
            continue
        targets.append((file_path, overall_score))

    written = 0
    failed = 0
    for index, (file_path, overall_score) in enumerate(targets, start=1):
        try:
            ok, _ = RawProcessor.safe_update_xmp(file_path, overall_score)
        except Exception as e:
            print(f"XMP 更新失敗：{file_path}：{e}")
            ok = False

        if ok:
            written += 1
        else:
            failed += 1

        if on_progress:
            on_progress(index, len(targets), file_path)

    return written, skipped, failed


def is_raw_file(file_path):
    return os.path.splitext(file_path)[1].lower() in ai_inference.RAW_EXTENSIONS


def load_preview_image(file_path):
    """統一使用 ai_inference 的影像解碼流程。"""
    return ai_inference.load_image(file_path)

def numpy_to_pixmap(img):
    img = np.ascontiguousarray(img)
    h, w, ch = img.shape
    qimg = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class PhotoManagerV2(QWidget):
    def __init__(self):
        super().__init__()
        init_db()

        self.photo_paths = {}
        self.current_folder = None
        self.current_file_path = None
        self.current_image_buffer = None
        self.aesthetic_weight = 0.8

        self.setWindowTitle("智慧照片篩選與管理系統")
        self.resize(1420, 820)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #181818;
                color: #F2F2F2;
                font-size: 14px;
            }
            QPushButton {
                background-color: #303030;
                color: #F5F5F5;
                border: 1px solid #4A4A4A;
                border-radius: 6px;
                padding: 9px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #444444; }
            QPushButton:pressed { background-color: #555555; }
            QPushButton:disabled { color: #777777; background-color: #242424; }
            QListWidget {
                background-color: #242424;
                color: #EEEEEE;
                border: 1px solid #3A3A3A;
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item { padding: 7px; border-radius: 4px; }
            QListWidget::item:selected { background-color: #4A4A4A; color: white; }
            QLabel { color: #EAEAEA; }
            QSlider::groove:horizontal {
                height: 6px; background: #3A3A3A; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #F2F2F2; width: 16px; margin: -5px 0; border-radius: 8px;
            }
            QProgressBar {
                background-color: #2A2A2A; color: #F2F2F2;
                border: 1px solid #444444; border-radius: 5px;
                text-align: center; min-height: 20px;
            }
            QProgressBar::chunk { background-color: #A8A8A8; border-radius: 4px; }
            QFrame#panel { background-color: #242424; border: 1px solid #383838; border-radius: 8px; }
        """)

        title = QLabel("智慧照片篩選與管理系統")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #FFFFFF;")

        self.folder_btn = QPushButton("選擇資料夾")
        self.analyze_current_btn = QPushButton("分析目前照片")
        self.analyze_all_btn = QPushButton("開始批次分析")
        self.xmp_btn = QPushButton("寫入 XMP 星等")
        self.xmp_btn.setToolTip("把目前資料夾 RAW 的星等寫進旁邊的 .xmp 給 Lightroom 讀取，不會修改原始照片")
        self.group_btn = QPushButton("相似照片分組")
        self.group_btn.setEnabled(False)
        self.group_btn.setToolTip("預留宋宇宸的相似照片分組 JSON 接口")

        top_buttons = QHBoxLayout()
        top_buttons.addWidget(self.folder_btn)
        top_buttons.addWidget(self.analyze_current_btn)
        top_buttons.addWidget(self.analyze_all_btn)
        top_buttons.addWidget(self.xmp_btn)
        top_buttons.addWidget(self.group_btn)
        top_buttons.addStretch()

        top = QVBoxLayout()
        top.addWidget(title)
        top.addLayout(top_buttons)

        # 左：目前資料夾照片清單
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left = QVBoxLayout(left_panel)
        left_title = QLabel("照片清單")
        left_title.setStyleSheet("font-size: 17px; font-weight: 700;")
        self.folder_label = QLabel("尚未選擇資料夾")
        self.folder_label.setStyleSheet("color: #AAAAAA;")
        self.folder_label.setWordWrap(True)
        self.count_label = QLabel("照片數量：0 張")
        self.photo_list = QListWidget()
        left.addWidget(left_title)
        left.addWidget(self.folder_label)
        left.addWidget(self.count_label)
        left.addWidget(self.photo_list)

        # 中：照片預覽
        center_panel = QFrame()
        center_panel.setObjectName("panel")
        center = QVBoxLayout(center_panel)
        center_title = QLabel("照片預覽")
        center_title.setStyleSheet("font-size: 17px; font-weight: 700;")
        self.preview = PhotoPreviewLabel("請選擇照片")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(650, 560)
        self.preview.setStyleSheet("""
            background-color: #101010;
            color: #888888;
            border: 1px solid #333333;
            border-radius: 6px;
            font-size: 17px;
        """)
        self.file_label = QLabel("目前尚未選取照片")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("color: #BDBDBD;")
        center.addWidget(center_title)
        center.addWidget(self.preview, 1)
        center.addWidget(self.file_label)

        # 右：評分與權重
        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right = QVBoxLayout(right_panel)
        right_title = QLabel("AI 分析結果")
        right_title.setStyleSheet("font-size: 17px; font-weight: 700;")

        self.result_label = QLabel("尚未分析")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.result_label.setStyleSheet("background-color: #1D1D1D; border-radius: 6px; padding: 12px;")

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #3A3A3A;")

        weight_title = QLabel("照片排序偏好")
        weight_title.setStyleSheet("font-weight: 700;")
        self.weight_label = QLabel("美感 80%   /   技術 20%")
        self.weight_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(80)
        self.weight_hint = QLabel("調整權重僅重新計算排序，不會重新執行 AI 分析。")
        self.weight_hint.setWordWrap(True)
        self.weight_hint.setStyleSheet("color: #9A9A9A; font-size: 12px;")

        self.model_label = QLabel(f"模型版本\n{MODEL_VERSION}")
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet("color: #888888; font-size: 11px;")

        right.addWidget(right_title)
        right.addWidget(self.result_label, 1)
        right.addWidget(divider)
        right.addWidget(weight_title)
        right.addWidget(self.weight_label)
        right.addWidget(self.weight_slider)
        right.addWidget(self.weight_hint)
        right.addWidget(self.model_label)

        body = QHBoxLayout()
        body.addWidget(left_panel, 24)
        body.addWidget(center_panel, 52)
        body.addWidget(right_panel, 24)

        # 下：進度列，之後可直接接背景批次分析 on_progress callback
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("尚未開始分析")
        self.progress_label = QLabel("目前處理：—")
        self.progress_label.setStyleSheet("color: #AFAFAF;")

        bottom = QVBoxLayout()
        bottom.addWidget(self.progress_bar)
        bottom.addWidget(self.progress_label)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(body, 1)
        root.addLayout(bottom)

        self.folder_btn.clicked.connect(self.choose_folder)
        self.photo_list.itemClicked.connect(self.show_photo)
        self.analyze_current_btn.clicked.connect(self.analyze_current)
        self.analyze_all_btn.clicked.connect(self.analyze_all)
        self.xmp_btn.clicked.connect(self.write_xmp)
        self.weight_slider.valueChanged.connect(self.weight_changed)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇照片資料夾")
        if not folder:
            return

        self.current_folder = os.path.abspath(folder)
        self.current_file_path = None
        self.current_image_buffer = None

        # 以目前硬碟內容同步這個資料夾；其他資料夾的 DB 紀錄不受影響。
        existing_paths = set()

        for file_name in os.listdir(self.current_folder):
            file_path = os.path.abspath(
                os.path.join(self.current_folder, file_name)
            )
            if os.path.isfile(file_path) and file_name.lower().endswith(IMAGE_EXTENSIONS):
                existing_paths.add(file_path)
                insert_photo(file_name, file_path)

        # 清除「這個資料夾內已從硬碟刪除」的舊 DB 紀錄。
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT file_path FROM photos_v2")
        for (db_path,) in cur.fetchall():
            abs_db_path = os.path.abspath(db_path)
            if (
                os.path.dirname(abs_db_path) == self.current_folder
                and abs_db_path not in existing_paths
            ):
                cur.execute(
                    "DELETE FROM photos_v2 WHERE file_path = ?",
                    (db_path,)
                )
        conn.commit()
        conn.close()

        # 刪除舊照片後重新決定目前資料夾的 ★。
        recompute_scores(self.aesthetic_weight, self.current_folder)

        self.folder_label.setText(self.current_folder)
        self.refresh_list()
        self.result_label.setText("尚未選取照片")
        self.preview.clearPhoto()
        self.preview.setText("請選擇照片")
        self.file_label.setText("目前尚未選取照片")

    @staticmethod
    def clean_name(text):
        for prefix in ("★ ", "△ ", "✓ ", "↻ "):
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    def show_photo(self, item):
        file_name = self.clean_name(item.text())
        file_path = self.photo_paths.get(file_name)
        if not file_path:
            return

        self.current_file_path = file_path
        self.current_image_buffer = None
        self.file_label.setText(f"目前選取：{file_name}")

        try:
            img = load_preview_image(file_path)
            self.current_image_buffer = img
            pixmap = numpy_to_pixmap(img)
            self.preview.setPhotoPixmap(pixmap)
        except Exception as e:
            self.preview.setText(f"照片預覽失敗：{e}")

        self.show_result(file_path)

    def analyze_current(self):
        if not self.current_file_path:
            self.result_label.setText("請先選擇一張照片。")
            return

        self.result_label.setText("AI 分析中，請稍候…")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("分析目前照片…")
        self.progress_label.setText(f"目前處理：{os.path.basename(self.current_file_path)}")
        QApplication.processEvents()

        try:
            if self.current_image_buffer is None:
                self.current_image_buffer = load_preview_image(self.current_file_path)
            result = ai_inference.evaluate_photo(
                self.current_file_path,
                image=self.current_image_buffer,
                aesthetic_weight=self.aesthetic_weight
            )
        except Exception as e:
            traceback.print_exc()
            self.result_label.setText(f"AI 分析失敗：{e}")
            self.progress_bar.setFormat("分析失敗")
            return

        if not result:
            error_msg = getattr(ai_inference, "LAST_ERROR", None) or "未知錯誤"
            self.result_label.setText(f"AI 分析失敗：{error_msg}")
            self.progress_bar.setFormat("分析失敗")
            self.progress_label.setText("目前處理：—")
            return

        update_analysis(self.current_file_path, result, self.aesthetic_weight)
        recompute_scores(self.aesthetic_weight, self.current_folder)
        self.refresh_list()
        self.show_result(self.current_file_path)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("完成")
        self.progress_label.setText("目前處理：—")

    def analyze_all(self):
        if not self.current_folder or not self.photo_paths:
            self.result_label.setText("請先選擇照片資料夾。")
            return

        self.analyze_all_btn.setEnabled(False)
        self.analyze_current_btn.setEnabled(False)
        # 跑到一半換資料夾或寫 XMP，最後的重算與星等會用到另一個資料夾或還沒算完的分數
        self.folder_btn.setEnabled(False)
        self.xmp_btn.setEnabled(False)

        try:
            # 目前仍為同步版本；背景執行緒之後由宋宇宸的批次管線接口接入。
            candidates = []
            for file_name, file_path in self.photo_paths.items():
                row = get_photo(file_path)
                if not row or row[13] == 0 or row[7] != MODEL_VERSION:
                    candidates.append((file_name, file_path))

            if not candidates:
                self.progress_bar.setValue(100)
                self.progress_bar.setFormat("目前資料夾已是最新模型版本")
                QMessageBox.information(self, "完成", "目前資料夾沒有需要重新分析的照片。")
                return

            total = len(candidates)
            failed = 0
            self.progress_bar.setRange(0, total)

            for index, (file_name, file_path) in enumerate(candidates, start=1):
                self.progress_bar.setValue(index - 1)
                self.progress_bar.setFormat(f"{index - 1} / {total}")
                self.progress_label.setText(f"目前處理：{file_name}")
                QApplication.processEvents()

                try:
                    img = load_preview_image(file_path)
                    result = ai_inference.evaluate_photo(
                        file_path,
                        image=img,
                        aesthetic_weight=self.aesthetic_weight
                    )
                except Exception as e:
                    failed += 1
                    print(f"分析失敗：{file_name}，原因：{e}")
                    traceback.print_exc()
                    continue

                if not result:
                    failed += 1
                    error_msg = getattr(ai_inference, "LAST_ERROR", None) or "未知錯誤"
                    print(f"分析失敗：{file_name}，原因：{error_msg}")
                    continue
                update_analysis(file_path, result, self.aesthetic_weight)

            recompute_scores(self.aesthetic_weight, self.current_folder)
            self.refresh_list()
            self.progress_bar.setValue(total)
            self.progress_bar.setFormat(f"完成 {total - failed} / {total}")
            self.progress_label.setText("目前處理：—")

            self.result_label.setText(
                f"批次分析完成\n\n成功：{total - failed} 張\n失敗：{failed} 張\n\n"
                "★ 目前權重下最佳照片\n△ 技術品質需檢視\n✓ 已分析\n\n"
                "權重滑桿只重算既有分數，不會重新執行模型。"
            )
            QMessageBox.information(self, "完成", "目前資料夾的 AI 分析已完成。")

        finally:
            self.analyze_all_btn.setEnabled(True)
            self.analyze_current_btn.setEnabled(True)
            self.folder_btn.setEnabled(True)
            self.xmp_btn.setEnabled(True)

    def write_xmp(self):
        if not self.current_folder:
            self.result_label.setText("請先選擇照片資料夾。")
            return

        answer = QMessageBox.question(
            self, "寫入 XMP 星等",
            "依目前權重下的綜合分，把這個資料夾 RAW 的星等寫進旁邊的 .xmp。\n"
            "原始照片不會被修改，但會覆蓋在 Lightroom 裡打過的星等。\n\n"
            "確定要寫入嗎？"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        controls = (self.folder_btn, self.analyze_current_btn, self.analyze_all_btn,
                    self.xmp_btn, self.weight_slider)
        for control in controls:
            control.setEnabled(False)

        def on_progress(done, total, file_path):
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
            self.progress_bar.setFormat(f"寫入 XMP {done} / {total}")
            self.progress_label.setText(f"目前處理：{os.path.basename(file_path)}")
            QApplication.processEvents()

        try:
            written, skipped, failed = write_xmp_ratings(self.current_folder, on_progress)
        finally:
            for control in controls:
                control.setEnabled(True)

        self.progress_bar.setFormat("XMP 寫入完成")
        self.progress_label.setText("目前處理：—")

        if not (written or skipped or failed):
            message = "這個資料夾還沒有分析過的照片可以寫入。"
        else:
            message = f"已寫入 {written} 張 RAW 的星等。"
            if skipped:
                message += f"\n略過 {skipped} 張 JPG／PNG／DNG（Lightroom 不讀它們旁邊的 .xmp）。"
            if failed:
                message += f"\n失敗 {failed} 張，原因請看主控台。"
                if shutil.which("exiftool") is None:
                    message += "\n已經有 .xmp 的照片，要安裝 ExifTool 才能更新星等。"
        QMessageBox.information(self, "寫入 XMP 星等", message)

    def weight_changed(self, value):
        aesthetic_percent = int(value)
        technical_percent = 100 - aesthetic_percent
        self.aesthetic_weight = aesthetic_percent / 100.0

        self.weight_label.setText(
            f"美感 {aesthetic_percent}%   /   技術 {technical_percent}%"
        )

        if self.current_folder:
            recompute_scores(self.aesthetic_weight, self.current_folder)
            self.refresh_list()
            if self.current_file_path:
                self.show_result(self.current_file_path)

    def refresh_list(self):
        self.photo_list.clear()
        self.photo_paths.clear()

        if not self.current_folder:
            self.count_label.setText("照片數量：0 張")
            return

        rows = get_photos_in_folder(self.current_folder)

        for row in rows:
            file_name = row[0]
            file_path = row[1]
            technical_score = row[3]
            model_version = row[7]
            is_best = row[12]
            analyzed = row[13]

            self.photo_paths[file_name] = file_path

            if analyzed == 0:
                display = file_name
            elif model_version != MODEL_VERSION:
                display = f"↻ {file_name}"
            elif is_best == 1:
                display = f"★ {file_name}"
            elif (
                technical_score is not None
                and float(technical_score) < TECH_THRESHOLD
            ):
                display = f"△ {file_name}"
            else:
                display = f"✓ {file_name}"

            self.photo_list.addItem(display)

        self.count_label.setText(f"照片數量：{len(rows)} 張")

    def show_result(self, file_path):
        row = get_photo(file_path)

        if not row:
            self.result_label.setText("尚未分析")
            return

        (
            file_name,
            file_path,
            aesthetic_score,
            technical_score,
            overall_score,
            aesthetic_weight,
            technical_weight,
            model_version,
            _stored_status,
            suggestion,
            _stored_action,
            technical_issues,
            is_best,
            analyzed,
        ) = row

        if analyzed == 0:
            self.result_label.setText(
                "尚未分析\n\n"
                "請按「分析目前照片」或「開始批次分析」。"
            )
            return

        if model_version != MODEL_VERSION:
            old_version = model_version or "未記錄"
            self.result_label.setText(
                "此照片為舊模型分析結果，不納入目前版本排序。\n\n"
                f"舊模型版本\n{old_version}\n\n"
                f"目前模型版本\n{MODEL_VERSION}\n\n"
                "請重新分析此照片，或執行批次分析。"
            )
            return

        technical_status = (
            "警告"
            if float(technical_score) < TECH_THRESHOLD
            else "正常"
        )

        aesthetic_mark = (
            "優秀"
            if float(aesthetic_score) >= AESTHETIC_THRESHOLD
            else "—"
        )

        best_text = "是" if is_best == 1 else "否"
        action = "建議檢視" if technical_status == "警告" else "建議保留"
        issues_text = technical_issues if technical_issues else "無"

        self.result_label.setText(
            f"美感分數　{aesthetic_score:.2f}\n"
            f"技術分數　{technical_score:.2f}\n"
            f"綜合分數　{overall_score:.2f}\n\n"
            f"美感標記　{aesthetic_mark}\n"
            f"技術狀態　{technical_status}\n"
            f"最佳照片　{best_text}\n"
            f"系統建議　{action}\n\n"
            f"評分權重\n"
            f"美感 {int(round(aesthetic_weight * 100))}% / "
            f"技術 {int(round(technical_weight * 100))}%\n\n"
            f"建議\n{suggestion}\n\n"
            f"影像量測附註\n{issues_text}\n"
            "以上為客觀量測值，不一定代表照片缺陷。\n\n"
            f"模型版本\n{model_version}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoManagerV2()
    window.show()
    sys.exit(app.exec())
