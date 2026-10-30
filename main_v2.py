import html
import json
import os
import shutil
import sys
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import numpy as np

import ai_inference
from raw_processor import RawProcessor, SIDECAR_EXTENSIONS, RATING_THRESHOLDS
from shooting_info import read_shooting_info, format_shooting_info
from qt_batch_worker import BatchAnalysisThread
from grouping_review_view import GroupReviewView
from photo_grouping import DEFAULT_THRESHOLD, group_records
from burst_metadata import DEFAULT_MAX_SECONDS, build_burst_check, camera_identity, parse_capture_time
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QMessageBox, QCheckBox,
    QProgressBar, QFrame, QComboBox, QAbstractItemView, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QPoint, QFile, QTimer


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
    (_, file_path, aes, tech, overall, _, _, version, _, suggestion, _, issues, is_best, analyzed) = row
    blocks = []
    # 只有相機 RAW 會寫 .xmp（write_xmp_ratings）；JPG、HEIC、DNG 等顯示星等會讓人以為也會寫進 Lightroom
    writes_rating = os.path.splitext(file_path)[1].lower() in SIDECAR_EXTENSIONS

    if not analyzed:
        reason = row[8] if row[8] in ("檔案已變更", "檔案版本待確認") else "尚未分析"
        blocks.append(f"<b>{reason}</b><br>請按「分析目前照片」或「開始批次分析」。")
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
            ("綜合", f"{overall:.1f}", "★" * stars + "☆" * (5 - stars) if writes_rating else "", _GOLD),
        )
        rating_note = ("星等＝寫入 Lightroom 的評等" if writes_rating
                       else "此格式不產生 Lightroom 星等（只有相機 RAW 檔會寫入）")
        cells = "".join(
            f'<tr><td>{name}</td><td align="right">&nbsp;&nbsp;{value}&nbsp;&nbsp;</td>'
            f'<td style="color:{color or "#EAEAEA"};">{html.escape(mark)}</td></tr>'
            for name, value, mark, color in table_rows)
        head = f'<span style="color:{_GOLD};">★ 這個資料夾目前的最佳照片</span><br>' if is_best else ""
        blocks.append(
            head + f'<table cellspacing="0" cellpadding="1">{cells}</table>'
            f'<span style="color:{_DIM}; font-size:12px;">{rating_note}</span>')

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
    old = sum(1 for r in rows if (r[13] and r[7] != MODEL_VERSION)
              or r[8] in ("檔案已變更", "檔案版本待確認"))
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
    ("需重新分析（↻）", lambda r: (bool(r[13]) and r[7] != MODEL_VERSION)
     or r[8] in ("檔案已變更", "檔案版本待確認")),
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


def _delete_permanently(path):
    """永久刪除，無法還原。只用在沒有資源回收筒的磁碟，而且使用者已經確認過。"""
    try:
        os.remove(path)
        return True, ""
    except OSError as exc:
        return False, f"無法刪除（{exc.strerror or exc}）"


def no_trash_drive(path):
    """
    照片所在的磁碟沒有資源回收筒時，回傳磁碟代號（例如 "K:"），否則回傳 None。

    Windows 的記憶卡、隨身碟（DRIVE_REMOVABLE）與網路磁碟（DRIVE_REMOTE）沒有資源回收筒，
    移到資源回收筒會一張一張失敗。macOS 的外接磁碟有自己的垃圾桶，一律回傳 None。
    """
    if sys.platform != "win32":
        return None
    import ctypes
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    if not drive:
        return None
    kind = ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")
    return drive if kind in (2, 4) else None


def confirm_permanent_delete(parent, drive, count):
    """沒有資源回收筒的磁碟：預設返回，使用者明確按下「永久刪除」才回傳 True。"""
    box = QMessageBox(QMessageBox.Icon.Warning, "刪除照片",
                      f"這個磁碟（{drive}）沒有資源回收筒，刪除後無法復原，共 {count} 張。\n"
                      "RAW 旁邊的 .xmp 也會一起刪除。\n\n"
                      "想保留還原的機會，可以先用檔案總管把照片複製到電腦，"
                      "再用本程式開啟那個資料夾整理。", parent=parent)
    back = box.addButton("返回", QMessageBox.ButtonRole.RejectRole)
    delete = box.addButton(f"永久刪除 {count} 張", QMessageBox.ButtonRole.DestructiveRole)
    box.setDefaultButton(back)
    box.setEscapeButton(back)
    box.exec()
    return box.clickedButton() is delete


def delete_result_message(moved, failed, permanent=False):
    message = f"{'已永久刪除' if permanent else '已移到資源回收筒'}：{moved} 張。"
    if failed:
        message += f"\n未能移走 {len(failed)} 個檔案（含 XMP）：\n" + "\n".join(
            f"{name}：{reason}" for name, reason in failed[:8])
    return message


def delete_photos(paths, trash=None, permanent=False):
    """
    把照片移到資源回收筒，並刪掉它們在資料庫裡的紀錄。
    permanent=True 時改為永久刪除，只用在沒有資源回收筒的磁碟（見 no_trash_drive），
    而且必須先經過 confirm_permanent_delete。

    相機原生 RAW 旁邊的 .xmp（「寫入 XMP 星等」建立的）一起移走；
    刪 JPG 時不動 .xmp——RAW+JPG 同檔名時，那個 .xmp 屬於 RAW。
    回傳 (移走的照片張數, [(照片或 XMP 檔名, 失敗原因), ...])。
    """
    if trash is None:
        trash = _delete_permanently if permanent else _move_to_trash
    done = "已刪除" if permanent else "已移到資源回收筒"
    moved = 0
    failed = []
    with closing(get_conn()) as conn, conn:
        for path in paths:
            path = os.path.abspath(path)
            try:
                ok, reason = trash(path)
            except Exception as exc:
                ok, reason = False, str(exc)
            if not ok:
                failed.append((os.path.basename(path), reason))
                continue

            moved += 1
            conn.execute("DELETE FROM photos_v2 WHERE file_path = ?", (path,))

            base, ext = os.path.splitext(path)
            sidecar = base + ".xmp"
            if ext.lower() in SIDECAR_EXTENSIONS and os.path.exists(sidecar):
                try:
                    ok, reason = trash(sidecar)
                except Exception as exc:
                    ok, reason = False, str(exc)
                if not ok:
                    failed.append((os.path.basename(sidecar),
                                   f"照片{done}，但 XMP 未能移走：" + reason))
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
            analyzed INTEGER DEFAULT 0,
            feature BLOB,
            capture_meta TEXT
        )
    """)
    # 分組用的兩欄（2026-10-05 加入）。CREATE TABLE IF NOT EXISTS 不會幫已經存在的 photos.db 加欄位，
    # 舊的資料庫要自己補，否則一寫入就是 no such column。
    #   feature       1280 維特徵（float32，5 KB／張），和同一列的 model_version 綁在一起
    #   capture_meta  連拍判定用的 EXIF 欄位（JSON）；NULL＝還沒讀過，{}＝讀過但沒有
    columns = {row[1] for row in cur.execute("PRAGMA table_info(photos_v2)")}
    for name, kind in (("file_size", "INTEGER"), ("file_mtime_ns", "INTEGER"),
                       ("feature", "BLOB"), ("capture_meta", "TEXT")):
        if name not in columns:
            cur.execute(f"ALTER TABLE photos_v2 ADD COLUMN {name} {kind}")
    conn.commit()
    conn.close()


def insert_photo(file_name, file_path):
    file_path = os.path.abspath(file_path)
    signature = file_signature(file_path)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO photos_v2 (file_name, file_path, file_size, file_mtime_ns)
        VALUES (?, ?, ?, ?)
    """, (file_name, file_path, *(signature or (None, None))))
    conn.commit()
    conn.close()
    sync_photo_files([file_path])


def file_signature(path):
    """快速辨識原路徑的照片是否被覆寫；不讀取整張 RAW。"""
    try:
        stat = os.stat(path)
        return stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


def sync_photo_files(paths=None, folder=None):
    """檔案變更或舊紀錄沒有指紋時，讓分數、特徵與拍攝資訊失效。"""
    targets = None if paths is None else {os.path.abspath(p) for p in paths}
    folder = None if folder is None else os.path.abspath(folder)
    with closing(get_conn()) as conn, conn:
        query = "SELECT file_path, file_size, file_mtime_ns FROM photos_v2"
        if targets is None and folder is not None:
            # 只讀這個資料夾的紀錄，不掃整個資料庫；子資料夾在下面用 dirname 排除
            prefix = os.path.join(folder, "")
            rows = conn.execute(query + " WHERE substr(file_path, 1, ?) = ?",
                                (len(prefix), prefix)).fetchall()
        elif targets is None:
            rows = conn.execute(query).fetchall()
        else:
            rows = [row for path in targets
                    for row in conn.execute(query + " WHERE file_path=?", (path,))]
        for path, size, mtime in rows:
            if targets is not None and path not in targets:
                continue
            if folder is not None and os.path.dirname(path) != folder:
                continue
            signature = file_signature(path)
            if signature is not None and signature == (size, mtime):
                continue
            status = "檔案版本待確認" if size is None or mtime is None else "檔案已變更"
            conn.execute("""
                UPDATE photos_v2 SET aesthetic_score=NULL, technical_score=NULL,
                    overall_score=NULL, model_version='', status=?, suggestion='',
                    action='', technical_issues='', is_best=0, analyzed=0,
                    feature=NULL, capture_meta=NULL, file_size=?, file_mtime_ns=?
                WHERE file_path=?
            """, (status, *(signature or (None, None)), path))


def get_photo(file_path):
    sync_photo_files([file_path])
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

    sync_photo_files(folder=folder)

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

def update_analysis(file_path, result, aesthetic_weight, expected_signature=None):
    """status 只表示技術狀態；美感優秀另外由 aesthetic_score 判斷。"""
    signature = file_signature(file_path)
    if expected_signature is not None and signature != expected_signature:
        sync_photo_files([file_path])
        raise ValueError("分析期間照片已變更，請重新分析")
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

    # 沒有特徵時存 NULL，不能留著上一次的：那是別的模型版本算的
    feature = result.get("feature_vector")
    feature = None if feature is None else feature_blob(feature)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE photos_v2
        SET aesthetic_score = ?, technical_score = ?, overall_score = ?,
            aesthetic_weight = ?, technical_weight = ?, model_version = ?,
            status = ?, suggestion = ?, action = ?, technical_issues = ?,
            feature = ?, analyzed = 1, file_size = ?, file_mtime_ns = ?
        WHERE file_path = ?
    """, (
        aesthetic_score, technical_score, overall_score,
        aesthetic_weight, technical_weight, MODEL_VERSION,
        status, suggestion, action, technical_issues, feature,
        *(signature or (None, None)),
        os.path.abspath(file_path)
    ))

    conn.commit()
    conn.close()

FEATURE_DIM = 1280


def feature_blob(vector):
    """
    1280 維特徵存成 little-endian float32（5 KB／張，325 張約 1.6 MB）。
    模型輸出本來就是 float32，存回去不失真；格式寫死 little-endian，Windows 與 Mac 存的資料庫可以互通。
    形狀或數值不對時回傳 None（當成沒有特徵），不讓壞資料進到分組。
    """
    array = np.asarray(vector, dtype="<f4")
    if array.shape != (FEATURE_DIM,) or not np.isfinite(array).all():
        return None
    return array.tobytes()


def store_capture_metadata(metadata):
    """把 {照片路徑: 連拍欄位} 存進資料庫（分析時背景讀到的拍攝時間，見 qt_batch_worker）。"""
    conn = get_conn()
    conn.executemany(
        "UPDATE photos_v2 SET capture_meta = ? WHERE file_path = ?",
        [(json.dumps(item, ensure_ascii=False), os.path.abspath(path))
         for path, item in metadata.items()])
    conn.commit()
    conn.close()


def paths_missing_capture_metadata(paths):
    """paths 之中還沒讀過拍攝時間的（照原順序）。"""
    conn = get_conn()
    missing = {row[0] for row in conn.execute(
        "SELECT file_path FROM photos_v2 WHERE capture_meta IS NULL")}
    conn.close()
    return [p for p in paths if os.path.abspath(p) in missing]


def analysis_candidates(paths):
    """
    「開始批次分析」要跑的照片（照 paths 的順序）：
      * 還沒分析，或是舊模型版本的分數
      * 分數是目前版本、但沒有分組用的特徵：加入分組功能前分析的，第一次會整個資料夾重跑一次
      * 有 ExifTool、但還沒讀過拍攝時間：分析時還沒裝 ExifTool 的話，不重跑就永遠缺這幾張的連拍資料
    """
    paths = list(paths)
    sync_photo_files(paths)
    exiftool = shutil.which("exiftool") is not None
    conn = get_conn()
    rows = {
        row[0]: row[1:]
        for row in conn.execute(
            "SELECT file_path, analyzed, model_version, feature IS NOT NULL, capture_meta IS NOT NULL "
            "FROM photos_v2")
    }
    conn.close()

    candidates = []
    for path in paths:
        row = rows.get(os.path.abspath(path))
        if row is None:
            candidates.append(path)
            continue
        analyzed, version, has_feature, has_meta = row
        if not analyzed or version != MODEL_VERSION or not has_feature or (exiftool and not has_meta):
            candidates.append(path)
    return candidates


GROUP_MODES = ("similar", "burst")


def group_folder(folder, mode="similar", threshold=DEFAULT_THRESHOLD,
                 max_seconds=DEFAULT_MAX_SECONDS, camera_match="model"):
    """
    相似照片分組（F06），給前台呼叫。先核對檔案狀態、讓過期分析失效，再讀取分組資料。
    不跑模型、不呼叫 ExifTool，可以直接在主執行緒呼叫。

    mode：
      "similar"  內容相似度達門檻（預設 0.85）就同一組，而且必須和組內每一張都達門檻
      "burst"    連拍：另外要同一台相機、整組拍攝時間在 max_seconds（預設 5 秒）內
    兩個門檻是 2026-09-23 用 325 張人工標註校準的（見 photo_grouping、burst_metadata 的常數說明）。
    分組本體和命令列工具（photo_grouping.py、run_bursts.py）是同一份程式。

    回傳 dict：
      groups           2 張以上的組：[{"group_id", "count", "best_path",
                         "photos": [{"path", "file_name", "overall_score", "is_best", "similarity_to_best"}, ...]}, ...]
                       group_id 從 1 連號；組依組內最高綜合分由高到低排，組內照片也是高到低。
                       is_best 是組內綜合分（目前權重）最高的那張，也就是建議保留的
      singles          可以分組、但沒有相似照片的
      not_ready        還不能分組的：沒分析、舊模型版本、或沒有特徵（加入分組功能前分析的）
                       → 按「開始批次分析」就會補上
      no_capture_time  只有連拍模式：讀不到拍攝時間或相機的照片（截圖、編修輸出的圖沒有 EXIF，
                       或分析時沒裝 ExifTool），這些不會被分進連拍組
      exiftool_missing 只有連拍模式：有照片還沒讀過拍攝時間，而且這台電腦沒有 ExifTool
    三個清單都是照片路徑，依檔名排序。
    """
    if mode not in GROUP_MODES:
        raise ValueError(f"mode 必須是 {GROUP_MODES} 其中之一")

    folder = os.path.abspath(folder)
    sync_photo_files(folder=folder)
    conn = get_conn()
    rows = conn.execute(
        "SELECT file_path, file_name, overall_score, analyzed, model_version, feature, capture_meta "
        "FROM photos_v2 ORDER BY file_name").fetchall()
    conn.close()

    ready, not_ready, capture_meta = [], [], {}
    for path, name, score, analyzed, version, feature, meta in rows:
        if os.path.dirname(os.path.abspath(path)) != folder or not os.path.isfile(path):
            continue
        if (not analyzed or version != MODEL_VERSION or score is None
                or feature is None or len(feature) != FEATURE_DIM * 4):
            not_ready.append(path)
            continue
        ready.append({"path": path, "file_name": name, "overall_score": score,
                      "feature_vector": np.frombuffer(feature, dtype="<f4")})
        if meta is not None:
            capture_meta[path] = json.loads(meta)

    can_pair = None
    no_capture_time = []
    exiftool_missing = False
    if mode == "burst":
        can_pair, _ = build_burst_check(capture_meta, max_seconds=max_seconds, camera_match=camera_match)
        for record in ready:
            item = capture_meta.get(record["path"])
            if (item is None or parse_capture_time(item) is None
                    or camera_identity(item, camera_match) is None):
                no_capture_time.append(record["path"])
        exiftool_missing = (len(capture_meta) < len(ready)) and shutil.which("exiftool") is None

    groups = group_records(ready, threshold, can_pair)

    names = {record["path"]: record["file_name"] for record in ready}
    multi = []
    for group in groups:
        if group["count"] < 2:
            continue
        group = dict(group, group_id=len(multi) + 1)
        for photo in group["photos"]:
            photo["file_name"] = names[photo["path"]]
        multi.append(group)

    by_name = lambda paths: sorted(paths, key=lambda p: os.path.basename(p).lower())  # noqa: E731
    return {
        "mode": mode,
        "groups": multi,
        "singles": by_name(g["best_path"] for g in groups if g["count"] == 1),
        "not_ready": by_name(not_ready),
        "no_capture_time": by_name(no_capture_time),
        "exiftool_missing": exiftool_missing,
    }


def non_best_paths(grouping):
    """group_folder 結果中，每組除了建議保留那張以外的照片（「選取每組非最佳」用）。"""
    return [photo["path"]
            for group in grouping["groups"]
            for photo in group["photos"]
            if not photo["is_best"]]


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


def summarize_errors(errors, limit=3):
    """[(照片路徑, 原因)] → 給畫面顯示的幾行文字：相同原因合併成一行並註明張數，最多 limit 種。"""
    grouped = {}
    for path, reason in errors:
        grouped.setdefault(reason, []).append(os.path.basename(path))
    lines = []
    for reason, names in list(grouped.items())[:limit]:
        where = names[0] if len(names) == 1 else f"{names[0]} 等 {len(names)} 張"
        lines.append(f"{reason}（{where}）" if names[0] else reason)
    if len(grouped) > limit:
        lines.append(f"另有 {len(grouped) - limit} 種原因")
    return lines


def write_xmp_ratings(folder, on_progress=None, only_paths=None, errors=None):
    """
    把目前資料夾 RAW 的星等寫進旁邊的 .xmp（李奇翰的 RawProcessor.update_ratings）。

    兩個地方會呼叫：「寫入 XMP 星等」按鈕（整個資料夾），以及勾選「分析後自動寫入 XMP」時
    分析完的那幾張（only_paths）。不在 recompute_scores 裡寫：
    修改已有的 .xmp 要呼叫 ExifTool（每次啟動約 0.4 秒；批次寫入時同星等共用一次），
    放在重算裡時，開資料夾、分析一張、滑桿每動一格都會把整個資料夾重寫一次
    （60 張 RAW 拉一格就卡 22.7 秒），而且會一直覆蓋使用者在 Lightroom 打的星等。

    只寫目前模型版本、已分析的 RAW；星等依資料庫裡目前權重下的綜合分。
    回傳 (寫入張數, 略過的非 RAW 張數, 失敗張數)；errors 給 list 的話，失敗原因以 (照片路徑, 原因) 加進去。
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

    # 批次寫入：已有 .xmp 的照片同星等一組交給一次 ExifTool（原本每張啟動一次，325 張要將近 2 分鐘）
    try:
        results = RawProcessor.update_ratings(targets, on_progress, errors)
    except Exception as e:
        if errors is not None:
            errors.append(("", f"XMP 更新失敗：{type(e).__name__}: {e}"))
        return 0, skipped, len(targets)
    written = sum(1 for ok, _ in results if ok)
    failed = len(targets) - written

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

        self._analysis_thread = None
        self._xmp_busy = False
        self._analysis_summary = None
        self._analysis_error = None
        self.photo_paths = {}
        self.current_folder = None
        self._folder_drive = None
        self.current_file_path = None
        self.current_image_buffer = None
        self._group_count = 0
        self._pending_analysis_count = 0
        self._workflow_refresh_timer = QTimer(self)
        self._workflow_refresh_timer.setSingleShot(True)
        self._workflow_refresh_timer.timeout.connect(self._refresh_workflow_state)
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
        self.group_btn.clicked.connect(self.show_groups)
        self.group_btn.setToolTip("在主視窗逐組比較照片，選擇保留或丟掉")

        self.primary_btn = QPushButton("選擇照片資料夾")
        self.primary_btn.setStyleSheet(
            "QPushButton { background: #23754b; color: white; padding: 11px 22px; }"
            "QPushButton:hover { background: #2b8a59; }"
            "QPushButton:disabled { background: #303a34; color: #aaaaaa; }")
        self.primary_btn.clicked.connect(self._run_primary_action)
        self.analyze_all_btn.hide()
        self.group_btn.hide()
        top_buttons = QHBoxLayout()
        top_buttons.addWidget(self.primary_btn)
        top_buttons.addWidget(self.folder_btn)
        top_buttons.addWidget(self.analyze_current_btn)
        top_buttons.addWidget(self.group_btn)
        top_buttons.addWidget(self.xmp_btn)
        # 期末報告寫的「分析後自動寫入 XMP」。預設不勾：平常不在照片資料夾產生檔案
        self.auto_xmp_check = QCheckBox("分析後自動寫入 XMP")
        self.auto_xmp_check.setToolTip("勾選後，每次分析完會把剛分析的 RAW 星等寫進旁邊的 .xmp")
        top_buttons.addWidget(self.auto_xmp_check)
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

        # 分析中才能按：手上這張做完就停，已分析的保存，其餘維持未分析
        self.cancel_btn = QPushButton("取消分析")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.cancel_btn)

        bottom = QVBoxLayout()
        bottom.addLayout(progress_row)
        bottom.addWidget(self.progress_label)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(body, 1)
        root.addLayout(bottom)

        self.group_review = GroupReviewView(
            self, lambda mode: group_folder(self.current_folder, mode=mode),
            lambda path: tuple(get_photo(path)[2:5]), self._preview_group_photo,
            lambda paths, permanent=False: (delete_photos(paths, permanent=True) if permanent
                                            else delete_photos(paths)),
            lambda: recompute_scores(self.aesthetic_weight, self.current_folder),
            body, left_panel, center, bottom, file_signature,
            trash_drive=no_trash_drive, confirm_permanent=confirm_permanent_delete,
            result_message=delete_result_message)

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

        self._update_primary_action()

    def _refresh_workflow_state(self, grouping=None, update_status=True):
        """Refresh expensive folder facts only when photo data changes.

        List search, sorting and filtering consume these cached facts without
        grouping photos or checking analysis metadata again.
        """
        self._workflow_refresh_timer.stop()
        if self.current_folder:
            paths = [row[1] for row in get_photos_in_folder(self.current_folder)]
            self._pending_analysis_count = len(analysis_candidates(paths))
            if grouping is None:
                grouping = group_folder(self.current_folder, "similar")
            self._group_count = len(grouping['groups'])
        else:
            self._group_count = self._pending_analysis_count = 0
        self._update_primary_action()
        if update_status and self._analysis_thread is None and not self._xmp_busy:
            self._update_group_status()

    def _update_group_status(self, grouping=None):
        count = len(grouping['groups']) if grouping is not None else self._group_count
        burst = grouping is not None and grouping.get('mode') == 'burst'
        kind = "連拍" if burst else "相似"
        if count:
            self.progress_label.setText(
                f"找到 {count} 組{kind}照片" +
                ("，可逐組比較與選擇保留" if self.group_review.active else
                 ("，可先整理已分析的照片" if self._pending_analysis_count else
                  f"，按『整理 {count} 組相似照片』開始")))
        else:
            self.progress_label.setText(f"目前沒有可整理的{kind}照片組")

    def _update_primary_action(self):
        """One visible next step, derived from actual folder and review state."""
        self.group_btn.hide()
        self.folder_btn.setText("更換資料夾")
        self.folder_btn.setVisible(bool(self.current_folder))
        self.primary_btn.setEnabled(self._analysis_thread is None and not self._xmp_busy)
        if self._xmp_busy:
            self._primary_action = None
            self.primary_btn.setText("寫入 XMP 中…")
        elif self._analysis_thread is not None:
            self._primary_action = None
            self.primary_btn.setText("分析中…")
        elif getattr(self, 'group_review', None) and self.group_review.active:
            self._primary_action = 'review'
            self.primary_btn.setText("完成整理，返回照片清單")
        elif not self.current_folder:
            self._primary_action = 'folder'
            self.primary_btn.setText("選擇照片資料夾")
        else:
            pending = self._pending_analysis_count
            if pending:
                self._primary_action = 'analyze'
                self.primary_btn.setText(f"分析這 {pending} 張照片")
                # Partial failures must not prevent reviewing already analyzed groups.
                ready_groups = self._group_count
                self.group_btn.setText(f"先整理已分析的 {ready_groups} 組")
                self.group_btn.setVisible(ready_groups > 0)
            else:
                count = self._group_count
                self._primary_action = 'review' if count else 'folder'
                self.primary_btn.setText(f"整理 {count} 組相似照片" if count else
                                         "沒有相似照片，選擇其他資料夾")

    def _run_primary_action(self):
        self._update_primary_action()
        if self._primary_action == 'folder':
            self.choose_folder()
        elif self._primary_action == 'analyze':
            self.analyze_all()
        elif self._primary_action == 'review':
            self.show_groups()

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

        self.group_review.folder_changed()
        self.current_folder = os.path.abspath(folder)
        self._folder_drive = no_trash_drive(self.current_folder)
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
        self._refresh_workflow_state()
        self.refresh_list()
        self.result_label.setText("尚未選取照片")
        self.preview.clearPhoto()
        self.preview.setText("請選擇照片")
        self.file_label.setText("目前尚未選取照片")

    def show_groups(self):
        if self._analysis_thread is not None or self._xmp_busy:
            return
        if not self.current_folder:
            QMessageBox.information(self, "照片分組", "請先選擇資料夾並完成批次分析。")
            return
        if self.group_review.active:
            self.group_review.exit()
        else:
            self.group_review.enter()

    def _preview_group_photo(self, path):
        if path not in self.photo_paths.values():
            return
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.show_photo(item)

    @staticmethod
    def clean_name(text):
        for prefix in ("★ ", "△ ", "✓ ", "↻ "):
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    def show_photo(self, item):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path:
            return
        file_name = os.path.basename(file_path)

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

    # ── 分析（宋宇宸的背景執行緒，2026-10-05 併入）────────────────────
    # 單張與批次都交給 qt_batch_worker.BatchAnalysisThread：4 條執行緒解碼、
    # 模型只在一條背景執行緒循序推論；進度用 Qt 訊號回到主執行緒，資料庫也只在主執行緒寫。
    # 實測 325 張 ARW：84 秒 → 39.7 秒，分析期間畫面最長只停 0.26 秒。
    def analyze_current(self):
        if self._analysis_thread is not None or self._xmp_busy or self.group_review.active:
            return
        if not self.current_file_path:
            self.result_label.setText("請先選擇一張照片。")
            return
        self._start_analysis([self.current_file_path])

    def analyze_all(self):
        if self._analysis_thread is not None or self._xmp_busy or self.group_review.active:
            return
        if not self.current_folder or not self.photo_paths:
            self.result_label.setText("請先選擇照片資料夾。")
            return

        candidates = analysis_candidates(self.photo_paths.values())

        if not candidates:
            self._set_progress(1, 1, "目前資料夾已是最新模型版本")
            self.progress_label.setText("目前處理：—")
            QMessageBox.information(self, "完成", "目前資料夾沒有需要重新分析的照片。")
            return
        self._start_analysis(candidates)

    def _start_analysis(self, paths):
        if self._analysis_thread is not None:
            return
        paths = list(paths)
        sync_photo_files(paths)
        # 權重與資料夾在開始時固定；同一時間只有一個分析，
        # 模型不會被同時呼叫，ai_inference.LAST_ERROR 也不會互相覆蓋。
        self._analysis_paths = {str(Path(path).resolve()): os.path.abspath(path)
                                for path in paths}
        self._analysis_weight = self.aesthetic_weight
        self._analysis_signatures = {path: file_signature(path) for path in self._analysis_paths.values()}
        self._analysis_folder = self.current_folder
        self._analysis_summary = None
        self._analysis_error = None
        self._analysis_ok_paths = []
        self._close_after_cancel = False
        self._analysis_failures = []

        # 每次分析的完整結果（含相似照片分組要用的 1280 維特徵）存在 reports/batches/
        output_dir = Path(__file__).resolve().parent / "reports" / "batches"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"batch-{uuid4().hex}.jsonl"
            worker = BatchAnalysisThread(paths, output, self._analysis_weight, self,
                                         metadata_paths=paths_missing_capture_metadata(paths))
        except Exception as exc:
            self.result_label.setText(f"無法啟動分析：{html.escape(str(exc))}")
            return

        self._analysis_thread = worker
        self._update_primary_action()
        self._analysis_output = str(output)
        # 記下原本的啟用狀態，結束後照原樣恢復
        self._analysis_controls = [
            (control, control.isEnabled()) for control in (
                self.primary_btn, self.folder_btn, self.analyze_current_btn, self.analyze_all_btn,
                self.xmp_btn, self.weight_slider, self.delete_btn,
                self.select_warn_btn, self.group_btn, self.auto_xmp_check,
            )
        ]
        for control, _ in self._analysis_controls:
            control.setEnabled(False)

        self._set_progress(0, len(paths), f"0 / {len(paths)}")
        if len(paths) == 1:
            self.progress_label.setText(f"目前處理：{os.path.basename(paths[0])}")
        self.result_label.setText("AI 背景分析中，可以繼續瀏覽照片。")

        worker.progress.connect(self._analysis_progress)
        worker.capture_metadata.connect(self._analysis_capture_metadata)
        worker.completed.connect(self._analysis_completed)
        worker.failed.connect(self._analysis_failed)
        worker.finished.connect(self._analysis_finished)
        self.cancel_btn.setEnabled(True)
        worker.start()

    def _analysis_progress(self, done, total, record):
        # 這個方法在主執行緒執行；每個資料庫操作自己開關連線
        name = os.path.basename(record["path"])
        if record.get("ok"):
            path = self._analysis_paths.get(record["path"], record["path"])
            try:
                update_analysis(path, record, self._analysis_weight,
                                expected_signature=self._analysis_signatures.get(path))
                self._analysis_ok_paths.append(path)
            except ValueError as exc:
                self._analysis_failures.append((name, str(exc)))
            except Exception as exc:
                self._analysis_error = f"資料庫寫入失敗：{exc}"
        else:
            self._analysis_failures.append((name, record.get("error") or "未知錯誤"))
        self._set_progress(done, total, f"{done} / {total}")
        self.progress_label.setText(f"已處理：{name}")

    def _analysis_capture_metadata(self, metadata):
        # 主執行緒：連拍分組要的拍攝時間，分析時在背景讀到的
        try:
            store_capture_metadata({path: item for path, item in metadata.items()
                                    if file_signature(path) == self._analysis_signatures.get(path)})
        except Exception as exc:
            print(f"[拍攝時間] 寫入資料庫失敗：{exc}")

    def _analysis_completed(self, summary):
        self._analysis_summary = summary

    def _analysis_failed(self, error):
        self._analysis_error = error

    def _analysis_finished(self):
        worker = self._analysis_thread
        total = len(self._analysis_paths)
        ok = len(self._analysis_ok_paths)
        processed = ok + len(self._analysis_failures)
        cancelled = worker.cancel_requested and processed < total
        xmp = None
        try:
            recompute_scores(self._analysis_weight, self._analysis_folder)
            self.refresh_list()
            # 星等要等整批分析完、重算綜合分之後才寫，才會用到目前權重下的分數
            xmp = self._auto_write_xmp(self._analysis_ok_paths)
            xmp_note = ""
            if xmp:
                xmp_note = f"<br><br>XMP 星等：寫入 {xmp[0]} 張 RAW"
                if xmp[2]:
                    xmp_note += f"，失敗 {xmp[2]} 張：" + "<br>".join(
                        html.escape(line) for line in summarize_errors(self._xmp_errors))

            if self._analysis_error:
                self._set_progress(ok, total, "分析未完整完成")
                self.result_label.setText(
                    f"<b>分析未完整完成</b><br>{html.escape(self._analysis_error)}<br><br>"
                    f"已完成 {ok} 張，結果都已存進資料庫。")
            elif cancelled:
                self._set_progress(processed, total, f"已取消：完成 {processed} / {total}")
                text = (f"<b>已取消分析</b><br>已分析 {ok} 張，結果都已存進資料庫；"
                        f"其餘 {total - processed} 張維持「尚未分析」，下次按「開始批次分析」會接著跑。")
                if self._analysis_failures:
                    text += f"<br>另有 {len(self._analysis_failures)} 張分析失敗，下次也會重試。"
                self.result_label.setText(text + xmp_note)
            elif total == 1 and self._analysis_failures:
                self._set_progress(0, 1, "分析失敗")
                self.result_label.setText(
                    f"AI 分析失敗：{html.escape(self._analysis_failures[0][1])}")
            elif total == 1 and self.current_file_path and \
                    os.path.abspath(self.current_file_path) in self._analysis_ok_paths:
                self._set_progress(1, 1, "完成")
                self.show_result(self.current_file_path)
            else:
                failed = total - ok
                self._set_progress(total, total, f"完成：成功 {ok} / 失敗 {failed}")
                text = f"<b>批次分析完成</b><br>成功 {ok} 張，失敗 {failed} 張"
                if self._analysis_failures:
                    text += "<br>" + "<br>".join(
                        f"・{html.escape(name)}：{html.escape(reason)}"
                        for name, reason in self._analysis_failures[:5])
                text += xmp_note
                text += ("<br><br>★ 目前權重下的最佳照片<br>△ 技術分偏低，建議檢視"
                         "<br>✓ 已分析<br>↻ 舊版模型的分數，需重新分析")
                self.result_label.setText(text)
        except Exception as exc:
            self.result_label.setText(f"結果更新失敗：{html.escape(str(exc))}")
        finally:
            if xmp and (xmp[0] or xmp[2]):
                self.progress_label.setText(
                    "已寫入 XMP 星等" if xmp[0] else "XMP 星等寫入失敗，原因見分析結果")
            else:
                self.progress_label.setText("目前處理：—")
            for control, enabled in self._analysis_controls:
                control.setEnabled(enabled)
            self.cancel_btn.setEnabled(False)
            self._analysis_thread = None
            worker.deleteLater()
            try:
                self._refresh_workflow_state(
                    update_status=not cancelled and not self._analysis_error)
            except Exception as exc:
                self._update_primary_action()
                self.progress_label.setText(f"分組統計更新失敗：{exc}")

            if self._close_after_cancel:
                QTimer.singleShot(0, self.close)

    def cancel_analysis(self):
        """手上這張推論完就停；不強制中斷正在跑的 GPU／RAW 解碼。"""
        if self._analysis_thread is None:
            return
        self._analysis_thread.cancel()
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("正在取消，手上這張做完就停…")

    def closeEvent(self, event):
        if self._xmp_busy:
            event.ignore()
            self.progress_label.setText("正在寫入 XMP，完成後即可關閉視窗。")
            return
        if self._analysis_thread is None:
            event.accept()
            return
        # 分析中關視窗：問要不要取消；取消的話等手上這張做完再自動關閉
        event.ignore()
        answer = QMessageBox.question(
            self, "分析還在進行",
            "背景分析還在進行，要取消分析並關閉視窗嗎？\n已經分析完的照片會保留。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if answer == QMessageBox.StandardButton.Yes and self._analysis_thread is not None:
            self._close_after_cancel = True
            self.cancel_analysis()
            self.progress_label.setText("正在取消，完成後會自動關閉視窗…")

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

        self._xmp_errors = []
        return write_xmp_ratings(self.current_folder, on_progress, only_paths=paths,
                                 errors=self._xmp_errors)

    def write_xmp(self):
        if self._analysis_thread is not None or self._xmp_busy or self.group_review.active:
            return
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

        controls = (self.primary_btn, self.group_btn, self.folder_btn, self.analyze_current_btn, self.analyze_all_btn,
                    self.xmp_btn, self.weight_slider, self.delete_btn, self.select_warn_btn,
                    self.auto_xmp_check)
        controls = [(control, control.isEnabled()) for control in controls]
        self._xmp_busy = True
        self._update_primary_action()
        for control, _ in controls:
            control.setEnabled(False)

        def on_progress(done, total, file_path):
            self._set_progress(done, total, f"寫入 XMP {done} / {total}")
            self.progress_label.setText(f"目前處理：{os.path.basename(file_path)}")
            QApplication.processEvents()

        try:
            errors = []
            written, skipped, failed = write_xmp_ratings(self.current_folder, on_progress, errors=errors)
        finally:
            self._xmp_busy = False
            for control, enabled in controls:
                control.setEnabled(enabled)
            self._update_primary_action()

        self._set_progress(1, 1, "XMP 寫入完成")
        self.progress_label.setText("目前處理：—")

        if not (written or skipped or failed):
            message = "這個資料夾還沒有分析過的照片可以寫入。"
        else:
            message = f"已寫入 {written} 張 RAW 的星等。"
            if skipped:
                message += f"\n略過 {skipped} 張非相機 RAW 的照片（JPG、PNG、HEIC、DNG 等；Lightroom 不讀它們旁邊的 .xmp）。"
            if failed:
                message += f"\n失敗 {failed} 張：\n" + "\n".join(summarize_errors(errors))
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
            # Score ordering can affect grouping. Refresh once after adjustment,
            # never from list refreshes or for each intermediate slider value.
            self._workflow_refresh_timer.start(250)
            self.refresh_list()
            if self.current_file_path:
                self.show_result(self.current_file_path)

    def refresh_list(self):
        self.photo_list.clear()
        self.photo_paths.clear()

        if not self.current_folder:
            self.count_label.setText("照片數量：0 張")
            self._update_primary_action()
            return

        _, key_index = SORT_KEYS[self.sort_combo.currentIndex()]
        all_rows = get_photos_in_folder(self.current_folder)
        self.rows_by_name = {row[0]: row for row in all_rows}
        hint = (f"　　{self._folder_drive} 是記憶卡、隨身碟或網路磁碟：讀取較慢、刪除無法復原，建議先複製到電腦"
                if self._folder_drive else "")
        self.folder_label.setText(f"{self.current_folder}　　{folder_summary(all_rows)}{hint}")
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
                display = (f"↻ {file_name}" if row[8] in ("檔案已變更", "檔案版本待確認")
                           else file_name)
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

            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            self.photo_list.addItem(item)

        if len(rows) == len(all_rows):
            self.count_label.setText(f"照片數量：{len(rows)} 張")
        else:
            self.count_label.setText(f"照片數量：顯示 {len(rows)} 張／共 {len(all_rows)} 張")

        self._update_primary_action()

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
            path = item.data(Qt.ItemDataRole.UserRole)
            row = self.rows_by_name.get(os.path.basename(path)) if path else None
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
        if self._analysis_thread is not None or self._xmp_busy or self.group_review.active:
            return
        paths = [item.data(Qt.ItemDataRole.UserRole)
                 for item in self.photo_list.selectedItems()
                 if item.data(Qt.ItemDataRole.UserRole)]
        names = [os.path.basename(path) for path in paths]
        if not paths:
            self.progress_label.setText("請先選取要刪除的照片（可以按住 Ctrl 或 Shift 多選）")
            return

        # 記憶卡、隨身碟沒有資源回收筒：先講清楚，不要一張一張失敗
        drive = no_trash_drive(paths[0])
        if drive:
            if not confirm_permanent_delete(self, drive, len(paths)):
                return
        else:
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

        moved, failed = delete_photos(paths, permanent=bool(drive))

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
        self._refresh_workflow_state()
        self.refresh_list()

        self._update_group_status()
        QMessageBox.information(self, "刪除照片", delete_result_message(moved, failed, bool(drive)))

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
