"""
環境自檢腳本 —— 一次列出這個專案跑起來需要的東西，缺什麼、為什麼缺。

執行方式：
    python check_env.py

設計原則：**每一項獨立檢查，互不影響**。
原本的寫法把所有 import 放在檔案最上方，只要有一個套件沒裝，
腳本第一行就 ModuleNotFoundError 崩潰，一項結果都印不出來
（實測：因為 rawpy 沒裝，整支腳本在第 3 行就死了）。
現在改成每項各自 try，就算全部套件都沒裝也能產出一份完整報告。

檢查項目分成兩級：
    [必要] 缺了推論就完全跑不動  -> 記為 FAIL，腳本以離開代碼 1 結束
    [選用] 缺了只影響部分功能    -> 記為 WARN，不影響離開代碼

換一台電腦時先跑這支：最後的總結會列出還缺什麼、怎麼補（權重、套件、ExifTool）。
"""
import hashlib
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_ICON = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}

results = []


def record(status, title, detail="", hint=""):
    results.append((status, title, detail, hint))
    print(f"{_ICON[status]} {title}")
    if detail:
        print(f"        {detail}")
    if hint:
        print(f"        -> {hint}")


def check_module(module_name, title, required, version_attr="__version__", hint=""):
    """
    檢查單一套件能不能匯入，並回報版本。
    required=True 時缺套件記為 FAIL，否則記為 WARN。
    回傳已匯入的模組物件，失敗時回傳 None（讓後續的深入檢查可以跳過）。
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        record(FAIL if required else WARN, title,
               f"無法匯入 {module_name}：{type(e).__name__}: {e}", hint)
        return None
    version = getattr(mod, version_attr, "版本不明")
    record(OK, title, f"{module_name} {version}")
    return mod


def check_python():
    record(OK, "Python 直譯器",
           f"{platform.python_version()}  ({sys.executable})")


def check_gpu(torch_mod):
    """
    GPU 檢查刻意分成兩層：

      第一層 torch.cuda.is_available() 只回答「機器上有沒有 CUDA 裝置」。
      第二層才是重點 —— 實際送一次運算到 GPU。
      因為 is_available() 不檢查「這版 PyTorch 編譯出來的 kernel
      支不支援這張卡的架構」。RTX 50 系列是 sm_120，若 PyTorch 只編到 sm_90，
      is_available() 會回 True，但任何一次真正的運算都會炸
      "no kernel image is available for execution on the device"。
      只做第一層檢查，會給出「CUDA 可用」這種完全錯誤的結論。
    """
    if torch_mod is None:
        record(WARN, "GPU 加速", "PyTorch 未安裝，略過檢查")
        return

    if not torch_mod.cuda.is_available():
        record(WARN, "GPU 加速", "未偵測到可用的 CUDA 裝置",
               "推論會走 CPU。本專案 CPU 單張約 50 ms，一般使用足夠")
        return

    try:
        name = torch_mod.cuda.get_device_name(0)
        cap = torch_mod.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
    except Exception as e:
        record(WARN, "GPU 加速", f"無法讀取 GPU 資訊：{type(e).__name__}: {e}")
        return

    try:
        torch_mod.zeros(1, device="cuda").add_(1)
        torch_mod.cuda.synchronize()
        record(OK, "GPU 加速",
               f"{name}（{sm}）可實際執行運算，CUDA {torch_mod.version.cuda}")
    except Exception as e:
        record(WARN, "GPU 加速",
               f"偵測到 {name}（{sm}），但 PyTorch {torch_mod.__version__} "
               f"無法在它上面執行運算：{type(e).__name__}: {e}",
               f"這是 PyTorch 版本與顯示卡架構不相容，不是程式問題。"
               f"推論會自動降級 CPU；要啟用 GPU 需改裝支援 {sm} 的 PyTorch")


def check_rawpy():
    """
    rawpy 的真實檢查：不只是 import，而是真的去解一張 RAW 檔。

    原本的寫法是 try 區塊裡只有一行 print("RawPy 狀態: 已就緒")，
    那個 try/except 永遠不會失敗，等於什麼都沒檢查。
    """
    try:
        import rawpy
    except Exception as e:
        record(WARN, "RAW 檔支援 (rawpy)",
               f"無法匯入 rawpy：{type(e).__name__}: {e}",
               "缺少它只影響 .ARW/.DNG 等 RAW 檔，JPG/PNG 不受影響。"
               "安裝：pip install rawpy")
        return

    version = getattr(rawpy, "__version__", "版本不明")

    photo_dir = BASE_DIR / "data" / "my_photos"
    samples = []
    if photo_dir.is_dir():
        samples = [p for p in sorted(photo_dir.iterdir())
                   if p.suffix.lower() in (".arw", ".dng", ".cr2", ".nef", ".raf")]

    if not samples:
        # 新電腦上通常沒有這個資料夾，屬於正常情況，不記為警告
        record(OK, "RAW 檔支援 (rawpy)",
               f"rawpy {version} 可匯入（{photo_dir} 內沒有 RAW 檔，未實測解碼；"
               f"放一張 .ARW 或 .DNG 進去就會實際解碼一次）")
        return

    sample = samples[0]
    try:
        with rawpy.imread(str(sample)) as raw:
            sizes = raw.sizes
        record(OK, "RAW 檔支援 (rawpy)",
               f"rawpy {version}，實測解碼 {sample.name} 成功 "
               f"({sizes.raw_width}x{sizes.raw_height})")
    except Exception as e:
        record(WARN, "RAW 檔支援 (rawpy)",
               f"rawpy {version} 可匯入，但解碼 {sample.name} 失敗："
               f"{type(e).__name__}: {e}")


def check_pyqt():
    """
    GUI 框架檢查。照片管理主程式 main_v2.py 需要它；
    缺少它時模型推論與命令列工具（run_batch.py 等）照樣能用，因此只記為 WARN。
    """
    try:
        from PyQt6.QtCore import QT_VERSION_STR
        from PyQt6.QtWidgets import QApplication
    except Exception as e:
        record(WARN, "GUI 框架 (PyQt6)",
               f"無法匯入 PyQt6：{type(e).__name__}: {e}",
               "沒有它就開不了照片管理主程式 main_v2.py（命令列工具不受影響）。安裝：pip install PyQt6")
        return

    try:
        app = QApplication.instance() or QApplication([])
        app.quit()
        record(OK, "GUI 框架 (PyQt6)", f"Qt {QT_VERSION_STR}，可建立 QApplication")
    except Exception as e:
        record(WARN, "GUI 框架 (PyQt6)",
               f"PyQt6 可匯入但無法建立 QApplication：{type(e).__name__}: {e}",
               "常見於沒有顯示裝置的環境（遠端連線、CI）")


# 這裡的檔名必須與 ai_inference.py 的 AES_WEIGHTS / TECH_WEIGHTS 保持一致。
#
# 曾經不一致過：美感模型換成分佈版本（nima_aes_dist.pth）之後，本檔案仍在檢查
# 舊的 nima_best.pth，於是「自檢全過、實際 import 卻拋 FileNotFoundError」——
# 一個為了預防缺檔而存在的檢查，剛好對缺檔視而不見。
# 不直接 import ai_inference 取值，是因為那會連帶載入模型；
# 本腳本必須在套件或權重缺失時仍能跑完並產出完整報告。
#
# 第三欄是組上目前使用的權重的 SHA-256 前 8 碼（與 ai_inference.model_version() 記的相同）。
# 下載不完整或拿到舊版時檔名一樣、也可能載得起來，但分數會和大家不同，
# 而且前台會把資料庫裡的分數全部當成舊版本。換了權重請一併更新這裡。
REQUIRED_WEIGHTS = (("nima_aes_dist.pth", "美感", "71dbd9e1"),
                    ("nima_tech_best.pth", "技術", "e39a98f4"))

# 兩個權重放在 repo 裡（README「模型權重」一節），下載整個專案就有。
WEIGHTS_HINT = "權重包含在 GitHub 的專案裡，請重新下載整個專案"

# 退回用的舊權重。缺了不影響評分，只是無法切回舊模型做對照。
OPTIONAL_WEIGHTS = (("nima_best.pth", "美感（舊二元版，退回用）"),)


def check_weights():
    """
    權重檔檢查。沒有權重，ai_inference 會直接拋 FileNotFoundError，
    整個評分功能等於不存在，所以列為必要項。
    """
    for filename, label, expected in REQUIRED_WEIGHTS:
        path = BASE_DIR / filename
        if not path.exists():
            record(FAIL, f"{label}模型權重", f"找不到 {path}",
                   f"沒有權重就無法評分。{WEIGHTS_HINT}（{filename} 應該在 {BASE_DIR}）")
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        if digest == expected:
            record(OK, f"{label}模型權重", f"{filename} ({size_mb:.1f} MB，指紋 {digest})")
        else:
            record(WARN, f"{label}模型權重",
                   f"{filename} ({size_mb:.1f} MB) 的指紋是 {digest}，組上目前用的是 {expected}",
                   "可能是下載不完整或拿到舊版：評分照樣能跑，但分數會和大家的不一樣。請重新下載")

    for filename, label in OPTIONAL_WEIGHTS:
        path = BASE_DIR / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            record(OK, f"{label}模型權重", f"{filename} ({size_mb:.1f} MB)")
        else:
            record(WARN, f"{label}模型權重", f"找不到 {filename}",
                   "評分不受影響，但無法切回舊模型做對照實驗。")


def check_exiftool():
    """
    ExifTool 不是 Python 套件，pip 裝不到，要另外安裝並加進 PATH。
    沒有它時：修改已存在的 .xmp（保留 Lightroom 調色）會失敗、RAW 的拍攝資訊讀不到；
    新建 .xmp 與評分本身不受影響，因此記為 WARN。
    """
    cmd = shutil.which("exiftool")
    if cmd is None:
        if sys.platform == "darwin":
            hint = "Mac：brew install exiftool（或到 https://exiftool.org 下載 macOS 安裝檔）"
        else:
            hint = ("Windows：到 https://exiftool.org 下載 Windows 版，"
                    "把 exiftool(-k).exe 改名為 exiftool.exe，連同 exiftool_files 資料夾放進 PATH 裡的資料夾；"
                    "或用下載頁的 Windows 安裝程式（會自動加入 PATH）。裝完要重開命令列視窗")
        record(WARN, "ExifTool（XMP 星等、拍攝資訊）", "找不到 exiftool 指令（不在 PATH 上）",
               "缺少它時已存在的 .xmp 無法更新、RAW 讀不到拍攝資訊；評分不受影響。" + hint)
        return
    try:
        result = subprocess.run([cmd, "-ver"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        version = result.stdout.decode("utf-8", "replace").strip()
        if result.returncode != 0 or not version:
            raise RuntimeError(result.stderr.decode("utf-8", "replace").strip() or f"離開代碼 {result.returncode}")
    except Exception as e:
        record(WARN, "ExifTool（XMP 星等、拍攝資訊）", f"找到 {cmd}，但執行失敗：{type(e).__name__}: {e}",
               "Windows 版要連同 exiftool_files 資料夾一起放，只複製 exe 會無法執行")
        return
    record(OK, "ExifTool（XMP 星等、拍攝資訊）", f"版本 {version}（{cmd}）")


def verify():
    print("=" * 68)
    print(" 專案環境自檢")
    print(f" 專案位置：{BASE_DIR}")
    print("=" * 68)

    print("\n--- 基礎環境 ---")
    check_python()

    print("\n--- 推論核心（缺少任一項就無法評分）---")
    torch_mod = check_module("torch", "PyTorch", required=True)
    check_module("torchvision", "TorchVision", required=True)
    check_module("numpy", "NumPy", required=True)
    check_module("PIL", "Pillow", required=True)
    check_module("cv2", "OpenCV", required=True)
    check_weights()

    print("\n--- 硬體加速 ---")
    check_gpu(torch_mod)

    print("\n--- 訓練用（只跑推論的話可以沒有）---")
    check_module("pandas", "pandas", required=False)
    check_module("scipy", "SciPy", required=False)
    check_module("matplotlib", "matplotlib（報表圖表）", required=False)

    print("\n--- 照片管理主程式 main_v2.py ---")
    check_pyqt()
    check_rawpy()
    check_exiftool()

    # ── 總結 ──────────────────────────────────────────
    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]

    print("\n" + "=" * 68)
    print(f" 總計 {len(results)} 項： "
          f"通過 {len(results) - len(fails) - len(warns)}．"
          f"警告 {len(warns)}．失敗 {len(fails)}")
    print("=" * 68)

    # 總結區只留一行摘要：有些底層錯誤訊息（例如 CUDA 的）本身帶換行，
    # 直接印出來會把總結撐爛，詳情上面各項已經完整列過了。
    def _oneline(text, limit=100):
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[:limit] + "..."

    # 有補救方法的就印補救方法：換電腦的人要知道的是下一步做什麼
    # 補救方法是自己寫的、一行內，整句印出來（截斷的話剛好截掉路徑或安裝步驟）
    def _summary(detail, hint):
        return " ".join(hint.split()) if hint else _oneline(detail)

    if fails:
        print("\n必須修正（否則無法評分）：")
        for _, title, detail, hint in fails:
            print(f"  - {title}：{_summary(detail, hint)}")
    if warns:
        print("\n部分功能受限：")
        for _, title, detail, hint in warns:
            print(f"  - {title}：{_summary(detail, hint)}")
    if not fails and not warns:
        print("\n所有項目正常。")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(verify())
