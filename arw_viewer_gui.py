import sys
import os
import rawpy
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer
from PIL import Image  # 引入 PIL 用來處理 JPG 檔

# ── 1. 引入你封裝好的 AI 大腦 ────────────────────────────────
try:
    from ai_inference import evaluate_photo
except (ImportError, FileNotFoundError) as e:
    # ImportError：ai_inference.py 不存在，或它的依賴套件缺失
    # FileNotFoundError：ai_inference.py 找不到 .pth 權重檔
    #   —— 寧可關閉評分功能，也不要顯示隨機權重產生的假分數
    evaluate_photo = None
    AI_LOAD_ERROR = str(e)
    print(f"⚠️ AI 評分功能已關閉：{e}")
else:
    AI_LOAD_ERROR = None

class ARWViewerPro(QMainWindow):
    def __init__(self, file_path):
        super().__init__()
        self.setWindowTitle("AI 雙核心照片品質評估系統")
        self.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")
        self.file_path = file_path
        
        self.resize(1000, 650)
        self.setMinimumSize(400, 300)

        # 影像顯示區域
        self.label = QLabel("正在初始化影像載入管線...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(1, 1)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # ── 2. AI 分數顯示文字標籤 ────────────────────────────
        self.score_label = QLabel("🤖 AI 正在健檢照片中，請稍候...")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Weight.Bold))
        self.score_label.setStyleSheet("""
            background-color: #2a2a2a; 
            border-radius: 5px; 
            padding: 10px; 
            margin-top: 5px;
        """)
        
        # 佈局設定
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15) 
        layout.addWidget(self.label)
        layout.addWidget(self.score_label) 
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        QTimer.singleShot(100, lambda: self.process_gpu_pipeline(file_path))

    def process_gpu_pipeline(self, file_path):
        # 3. 自動偵測環境
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # ── 🌟 核心修正：根據格式分流處理 🌟 ──
            if ext in ['.arw']:
                if device == 'cpu':
                    self.label.setText("正在使用 CPU 管線解碼 RAW 影像...")
                    QApplication.processEvents()
                # 執行你原本寫的 RAW 檔處理
                with rawpy.imread(file_path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=1.0, user_flip=None)
                gpu_tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
                img_np = (gpu_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                # JPG/PNG 走這條路，直接用 PIL 讀取，相容性 100%
                self.label.setText("正在載入標準影像格式...")
                QApplication.processEvents()
                pil_img = Image.open(file_path).convert('RGB')
                img_np = np.array(pil_img)

            # 轉成 PyQt 畫面渲染
            img_np = np.ascontiguousarray(img_np)
            h, w, ch = img_np.shape
            q_img = QImage(img_np.data, w, h, ch * w, QImage.Format.Format_RGB888)
            
            self.image_buffer = img_np
            self.pixmap = QPixmap.fromImage(q_img)
            self.update_display()
            
            # 4. 畫面出來了，叫 AI 算分數
            self.run_ai_assessment()

        except Exception as e:
            self.label.setText(f"❌ 影像載入失敗: {str(e)}")

    def run_ai_assessment(self):
        if evaluate_photo is None:
            # 把真正的原因顯示出來（例如「找不到 nima_best.pth」），
            # 而不是只丟一句看不出所以然的「未就緒」
            reason = AI_LOAD_ERROR or "原因不明"
            self.score_label.setText(f"❌ AI 評分模組未就緒\n{reason}")
            return
            
        res = evaluate_photo(self.file_path)

        if res:
            info = (
                f"🎨 美感構圖：{res['aesthetic_score']} 分  |  "
                f"🛠️ 技術畫質：{res['technical_score']} 分  |  "
                f"⭐ 綜合分數：{res['overall_score']} 分\n"
                f"💡 AI 評語：[{res['status']}] {res['suggestion']}"
            )
            if res['technical_issues']:
                info += "\n🔍 技術問題細項：\n" + "\n".join(f"  • {issue}" for issue in res['technical_issues'])
            self.score_label.setText(info)
            
            if res['status'] == "優秀":
                self.score_label.setStyleSheet("background-color: #1b4d3e; color: #a3e635; padding: 10px; border-radius: 5px;")
            elif res['status'] == "警告":
                self.score_label.setStyleSheet("background-color: #5c1d1d; color: #f87171; padding: 10px; border-radius: 5px;")
            else:
                self.score_label.setStyleSheet("background-color: #2a2a2a; color: #ffffff; padding: 10px; border-radius: 5px;")
        else:
            self.score_label.setText("❌ AI 讀取該照片格式失敗")

    def update_display(self):
        if hasattr(self, 'pixmap'):
            label_size = self.label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = self.pixmap.scaled(
                    label_size, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 現在隨便你放 JPG 還是 ARW，通通都能測了！
    test_file = "data/my_photos/DSC05860-2.jpg" 
    
    if not os.path.exists(test_file):
        print(f"❌ 找不到實測照片：{test_file}，請修正路徑")
        sys.exit(1)
        
    viewer = ARWViewerPro(test_file)
    viewer.show()
    sys.exit(app.exec())