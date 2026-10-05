"""
一鍵啟動照片管理系統：檢查環境 → 缺套件時問要不要自動安裝 → 開啟主程式（main_v2.py）。

    python start.py           # Windows 也可以雙擊 start_windows.bat，Mac 雙擊 start_mac.command
    python start.py --check   # 只檢查、不開主程式

前提是已經裝好 Python 3.10 以上，並下載了整個專案資料夾（權重也在裡面）。
這裡只檢查開主程式需要的東西；GPU、訓練用套件等完整的逐項檢查請用 check_env.py。
"""
import argparse
import hashlib
import importlib
import importlib.util
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

# 開主程式需要的套件：pip 名稱 → import 名稱。版本一律讀 requirements.txt，不在這裡另寫一份。
# pandas、SciPy、matplotlib 只有訓練與報表用到，不在這裡裝。
RUNTIME_PACKAGES = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
                    "Pillow": "PIL", "opencv-python": "cv2", "rawpy": "rawpy", "pillow-heif": "pillow_heif",
                    "PyQt6": "PyQt6"}


def requirement_specs(path=ROOT / "requirements.txt"):
    """requirements.txt 裡的 {pip 名稱: 安裝寫法}，例如 {'torch': 'torch==2.11.0+cu128'}。"""
    specs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            specs[re.split(r"[=<>!~ ]", line, maxsplit=1)[0]] = line
    return specs


def has_nvidia_gpu():
    """有 NVIDIA 顯卡（驅動附的 nvidia-smi 在 PATH 上）。Mac 沒有 CUDA 版的 PyTorch。"""
    return sys.platform != "darwin" and shutil.which("nvidia-smi") is not None


def install_commands(missing, specs, nvidia):
    """
    把缺的套件換成 pip 指令（一或兩條）。
    PyTorch 在 requirements.txt 裡是 CUDA 版（+cu128），一般 PyPI 沒有，要從 PyTorch 的索引裝；
    沒有 NVIDIA 顯卡（包括所有 Mac）改裝同版本的 CPU 版，PyPI 上 Windows 與 Apple 晶片都有。
    """
    pip = [sys.executable, "-m", "pip", "install"]
    torch_like = [specs.get(name, name) for name in missing if name in ("torch", "torchvision")]
    others = [specs.get(name, name) for name in missing if name not in ("torch", "torchvision")]
    commands = []
    if torch_like:
        if nvidia:
            commands.append(pip + torch_like + ["--index-url", CUDA_INDEX])
        else:
            commands.append(pip + [spec.split("+", 1)[0] for spec in torch_like])
    if others:
        commands.append(pip + others)
    return commands


def missing_packages():
    return [name for name, module in RUNTIME_PACKAGES.items()
            if importlib.util.find_spec(module) is None]


def ask(question):
    """輸入 y 才算同意；沒有輸入（例如被別的程式呼叫）一律當成不同意。"""
    try:
        return input(f"{question} (y/N) ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def install_missing():
    """缺套件時列出要裝什麼、問要不要裝。回傳 True 表示套件都齊了。"""
    missing = missing_packages()
    if not missing:
        return True
    nvidia = has_nvidia_gpu()
    commands = install_commands(missing, requirement_specs(), nvidia)
    print("缺少這些套件：" + "、".join(missing))
    if "torch" in missing:
        print("PyTorch 會裝" + ("NVIDIA 顯卡版（約 3 GB，要等一段時間）" if nvidia else "CPU 版"))
    print("會執行：")
    for command in commands:
        print("  python " + " ".join(command[1:]))
    if not ask("要自動安裝嗎？"):
        print("沒有安裝。也可以自己執行上面的指令，或照 README 的「安裝」一節。")
        return False
    for command in commands:
        if subprocess.run(command).returncode != 0:
            print("[FAIL] 安裝失敗，原因請看上面 pip 的訊息")
            return False
    importlib.invalidate_caches()
    still = missing_packages()
    if still:
        print("[FAIL] 裝完還是找不到：" + "、".join(still))
        return False
    print("[ OK ] 套件安裝完成")
    return True


def check_weights():
    """缺權重回傳 False（不能評分）；版本不同只提醒。"""
    from check_env import REQUIRED_WEIGHTS   # 只用它的常數；check_env 的檢查都在 verify() 裡
    ok = True
    for filename, label, expected in REQUIRED_WEIGHTS:
        path = ROOT / filename
        if not path.exists():
            print(f"[FAIL] 找不到{label}模型權重 {filename}。權重放在 GitHub 的專案裡，請重新下載整個專案。")
            ok = False
        elif hashlib.sha256(path.read_bytes()).hexdigest()[:8] != expected:
            print(f"[WARN] {filename} 和專案目前用的版本不同（可能下載不完整），分數會和大家的不一樣")
    return ok


def check_exiftool():
    """沒有 ExifTool 照樣能用，只提醒；Mac 有 Homebrew 時可以直接裝。"""
    if shutil.which("exiftool"):
        return
    print("[WARN] 沒有 ExifTool：照片照樣能評分，但已有 .xmp 的 RAW 無法更新星等，"
          "RAW 也讀不到拍攝資訊與連拍時間。")
    if sys.platform == "darwin" and shutil.which("brew") and ask("要用 Homebrew 安裝 ExifTool 嗎？"):
        subprocess.run(["brew", "install", "exiftool"])
        return
    print("       安裝方式見 README 的「ExifTool」一節（https://exiftool.org）。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="一鍵啟動照片管理系統")
    parser.add_argument("--check", action="store_true", help="只檢查環境，不開主程式")
    args = parser.parse_args(argv)

    if sys.version_info < (3, 10):
        print(f"[FAIL] 需要 Python 3.10 以上，目前是 {platform.python_version()}（{sys.executable}）。"
              "請到 https://www.python.org/downloads/ 下載安裝。")
        return 1
    if not install_missing() or not check_weights():
        return 1
    check_exiftool()
    if args.check:
        print("[ OK ] 環境檢查完成")
        return 0
    print("開啟照片管理系統")
    return subprocess.call([sys.executable, str(ROOT / "main_v2.py")], cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())
