import sys
from pathlib import Path

# 這支程式在子資料夾裡；共用模組（ai_inference、common 等）在專案根目錄
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

import numpy as np

from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.QtGui import QImage, QPixmap

import ai_inference
from arw_viewer_gui import ARWViewerPro


class ARWViewerProV2(ARWViewerPro):

    def process_gpu_pipeline(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext in ai_inference.RAW_EXTENSIONS:
                self.label.setText("正在解碼 RAW 影像...")
            else:
                self.label.setText("正在載入標準影像格式...")

            QApplication.processEvents()

            # 使用 ai_inference 與模型完全相同的解碼流程：
            # RAW_EXTENSIONS、RGB、JPG EXIF 方向都一致。
            img_np = ai_inference.load_image(file_path)

            if img_np is None:
                raise RuntimeError("影像解碼失敗")

            img_np = np.ascontiguousarray(img_np)

            if img_np.ndim != 3 or img_np.shape[2] != 3:
                raise RuntimeError(f"影像格式不正確：{img_np.shape}")

            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # 同一份 RGB buffer 同時給預覽與 evaluate_photo()
            self.image_buffer = img_np

            h, w, ch = img_np.shape

            q_img = QImage(
                img_np.data,
                w,
                h,
                ch * w,
                QImage.Format.Format_RGB888,
            )

            self.pixmap = QPixmap.fromImage(q_img)
            self.update_display()

            self.run_ai_assessment()

        except Exception as e:
            self.label.setText(f"[FAIL] 影像載入失敗：{e}")

    def run_ai_assessment(self):
        if self.image_buffer is None:
            self.score_label.setText("[FAIL] 尚未取得 RGB 影像")
            return

        # 關鍵：不再解碼第二次
        res = ai_inference.evaluate_photo(
            self.file_path,
            image=self.image_buffer,
        )

        if not res:
            self.score_label.setText("[FAIL] AI 讀取該照片格式失敗")
            return

        aesthetic_score = float(res["aesthetic_score"])
        technical_score = float(res["technical_score"])
        overall_score = float(res["overall_score"])

        technical_status = (
            "警告"
            if technical_score < ai_inference.TECH_ISSUE_THRESHOLD
            else "正常"
        )

        # 優秀／良好／普通／待加強，與主程式相同
        aesthetic_mark = ai_inference.aesthetic_grade(aesthetic_score)

        info = (
            f"🎨 美感分數：{aesthetic_score:.2f}\n"
            f"🛠️ 技術分數：{technical_score:.2f}\n"
            f"⭐ 綜合分數：{overall_score:.2f}\n\n"
            f"美感標記：{aesthetic_mark}\n"
            f"技術狀態：{technical_status}"
        )

        issues = res.get("technical_issues", [])

        if issues:
            info += (
                "\n\n🔍 影像量測附註：\n"
                + "\n".join(f"  • {issue}" for issue in issues)
                + "\n以上為客觀量測值，不一定代表照片缺陷。"
            )

        self.score_label.setText(info)

        # 紅色只表示技術分低於門檻，
        # technical_issues 本身不會觸發警告。
        if technical_status == "警告":
            self.score_label.setStyleSheet("""
                background-color: #5c1d1d;
                color: #f87171;
                padding: 10px;
                border-radius: 5px;
            """)
        else:
            self.score_label.setStyleSheet("""
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 10px;
                border-radius: 5px;
            """)


def choose_file():
    # 可以：
    # python tools/arw_viewer_gui_v2.py /完整路徑/photo.DNG
    if len(sys.argv) >= 2:
        return os.path.abspath(os.path.expanduser(sys.argv[1]))

    # 沒傳路徑就開選檔視窗，不再寫死測試照片
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "選擇照片",
        "",
        "照片 (*.jpg *.jpeg *.png *.tif *.tiff *.arw *.dng *.nef *.cr2 *.cr3 *.raf);;"
        "所有檔案 (*)",
    )

    return file_path


if __name__ == "__main__":
    app = QApplication(sys.argv)

    file_path = choose_file()

    if not file_path:
        sys.exit(0)

    if not os.path.isfile(file_path):
        print(f"[FAIL] 找不到照片：{file_path}")
        sys.exit(1)

    viewer = ARWViewerProV2(file_path)
    viewer.show()

    sys.exit(app.exec())
