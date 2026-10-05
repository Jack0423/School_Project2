import html
import os
import shutil
import sys
import sqlite3
import traceback
import numpy as np

import ai_inference
from raw_processor import RawProcessor, SIDECAR_EXTENSIONS, RATING_THRESHOLDS
from shooting_info import read_shooting_info, format_shooting_info
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QListWidget,
    QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QMessageBox, QCheckBox,
    QProgressBar, QFrame, QComboBox, QAbstractItemView, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QPoint, QFile


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

# 預設權重與模型端相同（美感 0.6／技術 0.4），批次工具與 XMP 星等也用這個值。
# 原本前台是 0.8：325 張實拍的美感分與技術分標準差差不多（9.4、9.9），
# 0.8 時美感對排名的影響是技術的 3.8 倍，技術分幾乎不起作用；0.6 時是 1.4 倍。
DEFAULT_AESTHETIC_WEIGHT = ai_inference.AESTHETIC_WEIGHT

# 照片清單的排序選項：(顯示文字, 資料列中的欄位位置)；None 代表依檔名
SORT_KEYS = (
    ("綜合分", 4),
    ("美感分", 2),
    ("技術分", 3),
    ("檔名", None),
)


def sort_rows(rows, key_index, descending):
    """
    依使用者選的欄位排序資料列（欄位順序同 get_photos_in_folder）。

    依分數排序時，只有「目前模型版本、已分析」的照片有可比的分數，排在前面；
    未分析與舊版本（↻）的照片排在後面，依檔名排列。分數相同時依檔名。
    """
    if key_index is None:
        return sorted(rows, key=lambda r: r[0].lower(), reverse=descending)

    def comparable(row):
        return row[13] and row[7] == MODEL_VERSION and row[key_index] is not None

    scored = sorted((r for r in rows if comparable(r)), key=lambda r: r[0].lower())
    scored.sort(key=lambda r: r[key_index], reverse=descending)   # 穩定排序：同分仍依檔名
    others = sorted((r for r in rows if not comparable(r)),
                    key=lambda r: (1 if r[13] else 0, r[0].lower()))
    return scored + others


def rating_text(overall_score):
    """
    寫進 XMP 時會是幾星，例如「★★★☆☆（54～64 分）」。結果面板很窄，括號裡只寫綜合分的範圍。

    綜合分跟著權重滑桿變，星等也會跟著變；要再寫一次 XMP，Lightroom 裡的星等才會更新。
    """
    stars = RawProcessor.map_score_to_rating(overall_score)
    lowers = dict(RATING_THRESHOLDS)
    if stars == 5:
        span = f"{lowers[5]} 分以上"
    elif stars == 1:
        span = f"未滿 {lowers[2]} 分"
    else:
        span = f"{lowers[stars]}～{lowers[stars + 1]} 分"
    return f"{'★' * stars}{'☆' * (5 - stars)}（{span}）"


def display_suggestion(text):
    """
    建議的顯示文字。

    2026-10-04 以前分析的照片，模型端在建議裡抄了一份影像量測附註，
    畫面上會和下面的「影像量測附註」重複（死黑比例那句幾乎一模一樣）。
    資料庫裡的舊文字在這裡整理成新寫法，不必為了文字把照片重新分析一次。
    """
    text = text or ""
    # 技術分正常時的舊寫法：「照片品質良好。（影像量測附註：……）」
    text = text.split("（影像量測附註：", 1)[0]
    # 技術分偏低時的舊寫法：「……低於門檻 60）。可能成因：A；B。」
    head, sep, _ = text.partition("）。可能成因：")
    if sep:
        text = head + "），可能成因見影像量測附註。"
    return text


# 結果面板的顏色：綠＝好、橙＝要注意、金＝星等與最佳照片、灰＝附註說明
_GOOD = "#8FD19E"
_WARN = "#F2B04A"
_GOLD = "#F5C451"
_DIM = "#8A8A8A"


def _rating_span(stars):
    lowers = dict(RATING_THRESHOLDS)
    if stars == 5:
        return f"{lowers[5]} 分以上"
    if stars == 1:
        return f"未滿 {lowers[2]} 分"
    return f"{lowers[stars]}～{lowers[stars + 1]} 分"


RATING_HELP = "Lightroom 星等依綜合分換算：" + "、".join(
    f"{_rating_span(s)} {s} 星" for s in (5, 4, 3, 2, 1))


def result_html(row, shooting_text):
    """
    結果面板的內容（HTML）。

    原本 15 行左右的「欄位　值」清單，同一件事分好幾行講（美感分數／美感標記、技術分數／技術狀態、
    系統建議／建議）。改成：分數與等級同一行，用顏色分好壞；建議只留一句；
    量測附註條列，沒有就不顯示；最佳照片只在「是」的時候顯示。
    """
    (_, _, aes, tech, overall, _, _, version, _, suggestion, _, issues, is_best, analyzed) = row
    blocks = []

    if not analyzed:
        blocks.append("<b>尚未分析</b><br>請按「分析目前照片」或「開始批次分析」。")
    elif version != MODEL_VERSION:
        blocks.append("<b>舊版模型的分數</b><br>不納入目前的排序，請重新分析。")
    else:
        aes, tech, overall = float(aes), float(tech), float(overall)
        warn = tech < TECH_THRESHOLD
        grade = ai_inference.aesthetic_grade(aes)
        grade_color = _GOOD if grade in ("優秀", "良好") else (_WARN if grade == "待加強" else "")
        stars = RawProcessor.map_score_to_rating(overall)

        table_rows = (
            ("美感", f"{aes:.1f}", grade, grade_color),
            ("技術", f"{tech:.1f}", "警告" if warn else "正常", _WARN if warn else _GOOD),
            ("綜合", f"{overall:.1f}", "★" * stars + "☆" * (5 - stars), _GOLD),
        )
        cells = "".join(
            f'<tr><td>{name}</td><td align="right">&nbsp;&nbsp;{value}&nbsp;&nbsp;</td>'
            f'<td style="color:{color or "#EAEAEA"};">{html.escape(mark)}</td></tr>'
            for name, value, mark, color in table_rows)
        head = f'<span style="color:{_GOLD};">★ 這個資料夾目前的最佳照片</span><br>' if is_best else ""
        blocks.append(
            head + f'<table cellspacing="0" cellpadding="1">{cells}</table>'
            f'<span style="color:{_DIM}; font-size:12px;">星等＝寫入 Lightroom 的評等</span>')

        verdict = "建議檢視" if warn else "建議保留"
        blocks.append(f'<b style="color:{_WARN if warn else _GOOD};">{verdict}</b><br>'
                      f"{html.escape(display_suggestion(suggestion))}")

        items = [i for i in (issues or "").split("；") if i.strip()]
        if items:
            blocks.append(
                f'<b>影像量測附註</b> <span style="color:{_DIM}; font-size:12px;">僅供參考，不一定是缺陷</span><br>'
                + "<br>".join(f"・{html.escape(i)}" for i in items))

    shooting = "<br>".join(html.escape(line) for line in shooting_text.splitlines())
    blocks.append(f"<b>拍攝資訊</b><br>{shooting}")
    return "<br><br>".join(blocks)


def folder_summary(rows):
    """標題列右邊的資料夾統計，例如「共 325 張・已分析 325・技術警告 154・美感優秀 40」。"""
    current = [r for r in rows if _is_current(r)]
    parts = [f"共 {len(rows)} 張", f"已分析 {len(current)}"]
    warn = sum(r[3] < TECH_THRESHOLD for r in current)
    excellent = sum(ai_inference.aesthetic_grade(r[2]) == "優秀" for r in current)
    old = sum(1 for r in rows if r[13] and r[7] != MODEL_VERSION)
    if warn:
        parts.append(f"技術警告 {warn}")
    if excellent:
        parts.append(f"美感優秀 {excellent}")
    if old:
        parts.append(f"需重新分析 {old}")
    return "・".join(parts)


def _is_current(row):
    """目前模型版本、已分析過，分數可以拿來比較與篩選。"""
    return bool(row[13]) and row[7] == MODEL_VERSION and row[2] is not None and row[3] is not None


# 照片清單的篩選選項：(顯示文字, 條件)
FILTERS = (
    ("全部照片", lambda r: True),
    ("美感優秀", lambda r: _is_current(r) and ai_inference.aesthetic_grade(r[2]) == "優秀"),
    ("美感待加強", lambda r: _is_current(r) and ai_inference.aesthetic_grade(r[2]) == "待加強"),
    ("技術警告（△）", lambda r: _is_current(r) and r[3] < TECH_THRESHOLD),
    ("尚未分析", lambda r: not r[13]),
    ("需重新分析（↻）", lambda r: bool(r[13]) and r[7] != MODEL_VERSION),
)


def filter_rows(rows, filter_index, search_text=""):
    """只留下符合篩選條件、而且檔名含搜尋字的資料列（不分大小寫，順序不變）。"""
    keep = FILTERS[filter_index][1]
    needle = search_text.strip().lower()
    return [row for row in rows if keep(row) and needle in row[0].lower()]


def _move_to_trash(path):
    """移到資源回收筒（Windows）或垃圾桶（macOS），可以還原。回傳 (是否成功, 失敗原因)。"""
    ok, _ = QFile.moveToTrash(path)
    return ok, "" if ok else "無法移到資源回收筒（檔案可能不存在或正在使用中）"


def delete_photos(paths, trash=_move_to_trash):
    """
    把照片移到資源回收筒，並刪掉它們在資料庫裡的紀錄。不會永久刪除。

    相機原生 RAW 旁邊的 .xmp（「寫入 XMP 星等」建立的）一起移走；
    刪 JPG 時不動 .xmp——RAW+JPG 同檔名時，那個 .xmp 屬於 RAW。
    回傳 (移走的張數, [(檔名, 失敗原因), ...])。
    """
    moved = 0
    failed = []
    conn = get_conn()
    cur = conn.cursor()

    for path in paths:
        path = os.path.abspath(path)
        ok, reason = trash(path)
        if not ok:
            failed.append((os.path.basename(path), reason))
            continue

        moved += 1
        cur.execute("DELETE FROM photos_v2 WHERE file_path = ?", (path,))

        base, ext = os.path.splitext(path)
        sidecar = base + ".xmp"
        if ext.lower() in SIDECAR_EXTENSIONS and os.path.exists(sidecar):
            trash(sidecar)

    conn.commit()
    conn.close()
    return moved, failed


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
            aesthetic_weight REAL DEFAULT 0.6,
            technical_weight REAL DEFAULT 0.4,
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


def write_xmp_ratings(folder, on_progress=None, only_paths=None):
    """
    把目前資料夾 RAW 的星等寫進旁邊的 .xmp（李奇翰的 RawProcessor.safe_update_xmp）。

    兩個地方會呼叫：「寫入 XMP 星等」按鈕（整個資料夾），以及勾選「分析後自動寫入 XMP」時
    分析完的那幾張（only_paths）。不在 recompute_scores 裡寫：
    已有 .xmp 的照片每張都要呼叫一次 ExifTool（實測約 0.4 秒），
    放在重算裡時，開資料夾、分析一張、滑桿每動一格都會把整個資料夾重寫一次
    （60 張 RAW 拉一格就卡 22.7 秒），而且會一直覆蓋使用者在 Lightroom 打的星等。

    只寫目前模型版本、已分析的 RAW；星等依資料庫裡目前權重下的綜合分。
    回傳 (寫入張數, 略過的非 RAW 張數, 失敗張數)。
    """
    if only_paths is not None:
        only_paths = {os.path.abspath(p) for p in only_paths}

    targets = []
    skipped = 0
    for row in get_photos_in_folder(folder):
        file_path = row[1]
        overall_score = row[4]
        model_version = row[7]
        analyzed = row[13]

        if only_paths is not None and os.path.abspath(file_path) not in only_paths:
            continue
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
        self.aesthetic_weight = DEFAULT_AESTHETIC_WEIGHT

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
            QComboBox {
                background-color: #303030; color: #F5F5F5;
                border: 1px solid #4A4A4A; border-radius: 6px; padding: 6px 10px;
            }
            QCheckBox { color: #EAEAEA; spacing: 6px; }
            QLineEdit {
                background-color: #303030; color: #F5F5F5;
                border: 1px solid #4A4A4A; border-radius: 6px; padding: 6px 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #242424; color: #EEEEEE;
                selection-background-color: #4A4A4A;
            }
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
        # 期末報告寫的「分析後自動寫入 XMP」。預設不勾：平常不在照片資料夾產生檔案
        self.auto_xmp_check = QCheckBox("分析後自動寫入 XMP")
        self.auto_xmp_check.setToolTip("勾選後，每次分析完會把剛分析的 RAW 星等寫進旁邊的 .xmp")
        top_buttons.addWidget(self.auto_xmp_check)
        top_buttons.addWidget(self.group_btn)
        top_buttons.addStretch()

        # 右上：排序偏好（權重滑桿）。原本放在右下，移上來讓分析結果有更多空間
        default_percent = round(DEFAULT_AESTHETIC_WEIGHT * 100)
        weight_title = QLabel("排序偏好")
        weight_title.setStyleSheet("font-weight: 700;")
        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(default_percent)
        self.weight_slider.setFixedWidth(170)
        self.weight_slider.setToolTip("調整美感與技術分數的比重。只重新計算綜合分、排序與星等，不會重新執行 AI 分析。")
        self.weight_label = QLabel(f"美感 {default_percent}%  /  技術 {100 - default_percent}%")
        self.weight_label.setMinimumWidth(150)
        # 完整版本字串太長，只顯示最後的 rev，完整字串放在滑鼠提示
        self.model_label = QLabel(f"模型 {MODEL_VERSION.rsplit('|', 1)[-1]}")
        self.model_label.setToolTip(MODEL_VERSION)
        self.model_label.setStyleSheet("color: #777777; font-size: 11px;")
        top_buttons.addWidget(weight_title)
        top_buttons.addWidget(self.weight_slider)
        top_buttons.addWidget(self.weight_label)
        top_buttons.addWidget(self.model_label)

        # 標題列右邊：目前資料夾與統計（原本在照片清單上方，移上來讓清單變窄）
        self.folder_label = QLabel("尚未選擇資料夾")
        self.folder_label.setStyleSheet("color: #AAAAAA;")
        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.folder_label)

        top = QVBoxLayout()
        top.addLayout(title_row)
        top.addLayout(top_buttons)

        # 左：目前資料夾照片清單
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left = QVBoxLayout(left_panel)
        left_title = QLabel("照片清單")
        left_title.setStyleSheet("font-size: 17px; font-weight: 700;")
        self.count_label = QLabel("照片數量：0 張")

        # 排序：欄位＋升降冪
        self.sort_combo = QComboBox()
        for label, _ in SORT_KEYS:
            self.sort_combo.addItem(f"依{label}")
        self.sort_descending = True
        self.sort_order_btn = QPushButton()
        self._update_sort_order_text()
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("排序"))
        sort_row.addWidget(self.sort_combo, 1)
        sort_row.addWidget(self.sort_order_btn)

        # 篩選與搜尋只影響清單顯示哪些照片；批次分析與刪除照樣以整個資料夾為準
        self.filter_combo = QComboBox()
        for label, _ in FILTERS:
            self.filter_combo.addItem(label)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("顯示"))
        filter_row.addWidget(self.filter_combo, 1)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜尋檔名，例如 DSC021")
        self.search_edit.setClearButtonEnabled(True)

        # 可以用 Ctrl／Shift 多選，按 Delete 鍵或下面的按鈕刪除
        self.photo_list = QListWidget()
        self.photo_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.select_warn_btn = QPushButton("選取技術警告")
        self.select_warn_btn.setToolTip("一次選取所有 △（技術分低於警告門檻）的照片，刪除前可以再取消幾張")
        self.delete_btn = QPushButton("刪除選取")
        self.delete_btn.setToolTip("移到資源回收筒，可以還原；RAW 旁邊的 .xmp 也會一起移走")
        delete_row = QHBoxLayout()
        delete_row.addWidget(self.select_warn_btn)
        delete_row.addWidget(self.delete_btn)

        left.addWidget(left_title)
        left.addWidget(self.count_label)
        left.addLayout(sort_row)
        left.addLayout(filter_row)
        left.addWidget(self.search_edit)
        left.addWidget(self.photo_list)
        left.addLayout(delete_row)

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

        self.result_label = QLabel("尚未選取照片")
        self.result_label.setTextFormat(Qt.TextFormat.RichText)
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.result_label.setStyleSheet("background-color: #1D1D1D; border-radius: 6px; padding: 12px;")

        right.addWidget(right_title)
        right.addWidget(self.result_label, 1)

        # 照片清單縮窄，讓預覽大一點
        body = QHBoxLayout()
        body.addWidget(left_panel, 17)
        body.addWidget(center_panel, 58)
        body.addWidget(right_panel, 25)

        # 下：進度列，之後可直接接背景批次分析 on_progress callback
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("color: #AFAFAF;")
        self._set_idle("尚未開始分析")

        bottom = QVBoxLayout()
        bottom.addWidget(self.progress_bar)
        bottom.addWidget(self.progress_label)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(body, 1)
        root.addLayout(bottom)

        self.folder_btn.clicked.connect(self.choose_folder)
        self.photo_list.itemClicked.connect(self.show_photo)
        self.photo_list.itemSelectionChanged.connect(self._selection_changed)
        self.analyze_current_btn.clicked.connect(self.analyze_current)
        self.analyze_all_btn.clicked.connect(self.analyze_all)
        self.xmp_btn.clicked.connect(self.write_xmp)
        self.auto_xmp_check.toggled.connect(self.auto_xmp_toggled)
        self.weight_slider.valueChanged.connect(self.weight_changed)
        self.sort_combo.currentIndexChanged.connect(self.sort_field_changed)
        self.sort_order_btn.clicked.connect(self.toggle_sort_order)
        self.filter_combo.currentIndexChanged.connect(self.refresh_list)
        self.search_edit.textChanged.connect(self.refresh_list)
        self.select_warn_btn.clicked.connect(self.select_warnings)
        self.delete_btn.clicked.connect(self.delete_selected)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.photo_list,
                  activated=self.delete_selected)

    # ── 下方進度列與提示 ──────────────────────────────────
    # 所有地方都經過這三個方法。原本各處自己設定，留下幾個不會更新的情況：
    # 分析單張時進度範圍還停在上一次批次（顯示「完成」但進度列是空的）、
    # 分析失敗後一直顯示「目前處理：壞檔名」、換資料夾後還顯示上一個資料夾的狀態、
    # 選取改變後還顯示「已選取 N 張技術警告的照片」。
    def _set_progress(self, done, total, text):
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(text)

    def _set_idle(self, bar_text="待命"):
        self._set_progress(0, 1, bar_text)
        self.progress_label.setText("目前處理：—")

    def _selection_changed(self):
        count = len(self.photo_list.selectedItems())
        if count > 1:
            self.progress_label.setText(f"已選取 {count} 張照片，按「刪除選取」或 Delete 鍵可移到資源回收筒")
        else:
            self.progress_label.setText("目前處理：—")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇照片資料夾")
        if not folder:
            return

        self.current_folder = os.path.abspath(folder)
        self.current_file_path = None
        self.current_image_buffer = None
        self._set_idle()

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
        self._set_progress(0, 1, "分析目前照片…")
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
            self.result_label.setText(f"AI 分析失敗：{html.escape(str(e))}")
            self._set_idle("分析失敗")
            return

        if not result:
            error_msg = getattr(ai_inference, "LAST_ERROR", None) or "未知錯誤"
            self.result_label.setText(f"AI 分析失敗：{html.escape(str(error_msg))}")
            self._set_idle("分析失敗")
            return

        update_analysis(self.current_file_path, result, self.aesthetic_weight)
        recompute_scores(self.aesthetic_weight, self.current_folder)
        self.refresh_list()
        self.show_result(self.current_file_path)

        xmp = self._auto_write_xmp([self.current_file_path])
        self._set_progress(1, 1, "完成")
        if xmp and (xmp[0] or xmp[2]):
            self.progress_label.setText(
                "已寫入 XMP 星等" if xmp[0] else "XMP 星等寫入失敗，原因請看主控台")
        else:
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
        self.delete_btn.setEnabled(False)
        self.select_warn_btn.setEnabled(False)
        self.auto_xmp_check.setEnabled(False)

        try:
            # 目前仍為同步版本；背景執行緒之後由宋宇宸的批次管線接口接入。
            candidates = []
            for file_name, file_path in self.photo_paths.items():
                row = get_photo(file_path)
                if not row or row[13] == 0 or row[7] != MODEL_VERSION:
                    candidates.append((file_name, file_path))

            if not candidates:
                self._set_progress(1, 1, "目前資料夾已是最新模型版本")
                self.progress_label.setText("目前處理：—")
                QMessageBox.information(self, "完成", "目前資料夾沒有需要重新分析的照片。")
                return

            total = len(candidates)
            failed = 0
            analyzed_paths = []

            for index, (file_name, file_path) in enumerate(candidates, start=1):
                self._set_progress(index - 1, total, f"{index - 1} / {total}")
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
                analyzed_paths.append(file_path)

            recompute_scores(self.aesthetic_weight, self.current_folder)
            self.refresh_list()

            # 星等要等整批分析完、重算綜合分之後才寫，才會用到目前權重下的分數
            xmp = self._auto_write_xmp(analyzed_paths)
            xmp_text = ""
            if xmp:
                xmp_text = f"XMP 星等：寫入 {xmp[0]} 張 RAW"
                if xmp[2]:
                    xmp_text += f"，失敗 {xmp[2]} 張（原因請看主控台）"
                xmp_text += "<br><br>"

            self._set_progress(total, total, f"完成 {total - failed} / {total}")
            self.progress_label.setText("目前處理：—")

            self.result_label.setText(
                f"<b>批次分析完成</b><br>成功 {total - failed} 張，失敗 {failed} 張<br><br>"
                + xmp_text +
                "★ 目前權重下的最佳照片<br>△ 技術分偏低，建議檢視<br>✓ 已分析<br>↻ 舊版模型的分數，需重新分析"
            )
            QMessageBox.information(self, "完成", "目前資料夾的 AI 分析已完成。")

        finally:
            self.analyze_all_btn.setEnabled(True)
            self.analyze_current_btn.setEnabled(True)
            self.folder_btn.setEnabled(True)
            self.xmp_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            self.select_warn_btn.setEnabled(True)
            self.auto_xmp_check.setEnabled(True)

    def auto_xmp_toggled(self, checked):
        if not checked:
            return
        answer = QMessageBox.question(
            self, "分析後自動寫入 XMP",
            "勾選後，每次分析完會把剛分析的 RAW 星等寫進旁邊的 .xmp，給 Lightroom 讀取。\n\n"
            "沒有 .xmp 的照片會新建一個小檔案（約 250 bytes）；\n"
            "已經有的只改星等，但會覆蓋在 Lightroom 裡打過的星等。\n"
            "原始照片不會被修改。\n\n"
            "確定要開啟嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.auto_xmp_check.setChecked(False)

    def _auto_write_xmp(self, paths):
        """勾選「分析後自動寫入 XMP」時，寫剛分析完的照片。沒勾或沒有照片時回傳 None。"""
        if not self.auto_xmp_check.isChecked() or not paths:
            return None

        def on_progress(done, total, file_path):
            self._set_progress(done, total, f"寫入 XMP {done} / {total}")
            self.progress_label.setText(f"目前處理：{os.path.basename(file_path)}")
            QApplication.processEvents()

        return write_xmp_ratings(self.current_folder, on_progress, only_paths=paths)

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
                    self.xmp_btn, self.weight_slider, self.delete_btn, self.select_warn_btn,
                    self.auto_xmp_check)
        for control in controls:
            control.setEnabled(False)

        def on_progress(done, total, file_path):
            self._set_progress(done, total, f"寫入 XMP {done} / {total}")
            self.progress_label.setText(f"目前處理：{os.path.basename(file_path)}")
            QApplication.processEvents()

        try:
            written, skipped, failed = write_xmp_ratings(self.current_folder, on_progress)
        finally:
            for control in controls:
                control.setEnabled(True)

        self._set_progress(1, 1, "XMP 寫入完成")
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

        _, key_index = SORT_KEYS[self.sort_combo.currentIndex()]
        all_rows = get_photos_in_folder(self.current_folder)
        self.rows_by_name = {row[0]: row for row in all_rows}
        self.folder_label.setText(f"{self.current_folder}　　{folder_summary(all_rows)}")
        # 批次分析用 photo_paths 決定要分析哪些照片，所以放整個資料夾，不受篩選影響
        self.photo_paths.update({row[0]: row[1] for row in all_rows})
        rows = filter_rows(sort_rows(all_rows, key_index, self.sort_descending),
                           self.filter_combo.currentIndex(), self.search_edit.text())

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

        if len(rows) == len(all_rows):
            self.count_label.setText(f"照片數量：{len(rows)} 張")
        else:
            self.count_label.setText(f"照片數量：顯示 {len(rows)} 張／共 {len(all_rows)} 張")

    def _update_sort_order_text(self):
        by_name = SORT_KEYS[self.sort_combo.currentIndex()][1] is None
        if by_name:
            self.sort_order_btn.setText("Z → A" if self.sort_descending else "A → Z")
        else:
            self.sort_order_btn.setText("高 → 低" if self.sort_descending else "低 → 高")

    def sort_field_changed(self):
        # 換欄位時用最常用的方向：分數由高到低，檔名由 A 到 Z
        self.sort_descending = SORT_KEYS[self.sort_combo.currentIndex()][1] is not None
        self._update_sort_order_text()
        self.refresh_list()

    def toggle_sort_order(self):
        self.sort_descending = not self.sort_descending
        self._update_sort_order_text()
        self.refresh_list()

    def select_warnings(self):
        """一次選取所有技術警告（△）的照片。只選取不刪除，使用者可以再取消幾張。"""
        self.photo_list.clearSelection()
        count = 0
        for i in range(self.photo_list.count()):
            item = self.photo_list.item(i)
            row = self.rows_by_name.get(self.clean_name(item.text()))
            if (row and row[13] and row[7] == MODEL_VERSION
                    and row[3] is not None and float(row[3]) < TECH_THRESHOLD):
                item.setSelected(True)
                count += 1

        if count:
            self.progress_label.setText(
                f"已選取 {count} 張技術警告的照片，看過後按「刪除選取的照片」；按住 Ctrl 點一下可以取消")
        else:
            self.progress_label.setText("目前資料夾沒有技術警告的照片")

    def delete_selected(self):
        names = [self.clean_name(item.text()) for item in self.photo_list.selectedItems()]
        paths = [self.photo_paths[name] for name in names if name in self.photo_paths]
        if not paths:
            self.progress_label.setText("請先選取要刪除的照片（可以按住 Ctrl 或 Shift 多選）")
            return

        preview = "\n".join(names[:8]) + (f"\n……共 {len(names)} 張" if len(names) > 8 else "")
        answer = QMessageBox.question(
            self, "刪除照片",
            f"要把這 {len(paths)} 張照片移到資源回收筒嗎？\n"
            "RAW 旁邊的 .xmp 也會一起移走。之後可以從資源回收筒還原。\n\n"
            f"{preview}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        moved, failed = delete_photos(paths)

        if self.current_file_path and os.path.abspath(self.current_file_path) in {
                os.path.abspath(p) for p in paths} and not os.path.exists(self.current_file_path):
            self.current_file_path = None
            self.current_image_buffer = None
            self.preview.clearPhoto()
            self.preview.setText("請選擇照片")
            self.file_label.setText("目前尚未選取照片")
            self.result_label.setText("尚未選取照片")

        # 刪掉的可能是 ★，重新決定目前資料夾的最佳照片
        recompute_scores(self.aesthetic_weight, self.current_folder)
        self.refresh_list()

        message = f"已移到資源回收筒：{moved} 張。"
        if failed:
            message += f"\n失敗 {len(failed)} 張：\n" + "\n".join(
                f"{name}：{reason}" for name, reason in failed[:8])
        self.progress_label.setText("目前處理：—")
        QMessageBox.information(self, "刪除照片", message)

    def show_result(self, file_path):
        row = get_photo(file_path)

        if not row:
            self.result_label.setText("尚未分析")
            self.result_label.setToolTip("")
            return

        # 拍攝資訊與分析無關，還沒分析的照片也顯示；同一張照片只讀一次（約 0.2 秒）
        shooting = format_shooting_info(*read_shooting_info(file_path))
        self.result_label.setText(result_html(row, shooting))

        overall, model_version, analyzed = row[4], row[7], row[13]
        if analyzed and model_version == MODEL_VERSION and overall is not None:
            self.result_label.setToolTip(
                f"這張寫入 Lightroom 會是 {rating_text(float(overall))}\n{RATING_HELP}")
        else:
            self.result_label.setToolTip(RATING_HELP)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoManagerV2()
    window.show()
    sys.exit(app.exec())
