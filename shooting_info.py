"""
讀照片的拍攝資訊（相機、鏡頭、ISO、快門、光圈、焦距、拍攝時間），給前台顯示。

讀法：
  * 有 ExifTool 時一律用它。RAW（ARW、CR3、NEF、DNG……）與 JPG 都讀得到，
    相機拍的 JPG 和 RAW 一樣帶有這些資訊。
  * 沒有 ExifTool 時，JPG／TIFF 改用 PIL 讀 EXIF；RAW 讀不到，回報需要 ExifTool。
  * 截圖、經過編修輸出的 JPG／PNG 常常完全沒有 EXIF，這時回傳空的結果，由畫面說明。

ExifTool 每次啟動約 0.2 秒，所以結果依路徑快取，同一張照片只讀一次。
路徑以 UTF-8 參數檔傳給 ExifTool：直接放在命令列時，Windows 上的中文檔名會變成亂碼。
"""
import json
import os
import shutil
import subprocess

from PIL import Image

TAGS = ("Model", "LensModel", "ISO", "ExposureTime", "FNumber", "FocalLength", "DateTimeOriginal")

# PIL 讀 EXIF 用的標記編號（沒有 ExifTool 時的備援）
_PIL_IFD0 = {0x0110: "Model"}
_PIL_EXIF_IFD = 0x8769
_PIL_EXIF = {0x8827: "ISO", 0x829A: "ExposureTime", 0x829D: "FNumber",
             0x920A: "FocalLength", 0x9003: "DateTimeOriginal", 0xA434: "LensModel"}

_cache = {}


def _read_with_exiftool(exe, path):
    args = ["-j", "-fast", *[f"-{tag}" for tag in TAGS], path]
    result = subprocess.run(
        [exe, "-charset", "filename=utf8", "-@", "-"],
        input=("\n".join(args) + "\n").encode("utf-8"),
        capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    data = json.loads(result.stdout.decode("utf-8"))
    return {k: v for k, v in data[0].items() if k in TAGS} if data else {}


def _read_with_pil(path):
    with Image.open(path) as im:
        exif = im.getexif()
        raw = {name: exif.get(tag) for tag, name in _PIL_IFD0.items()}
        sub = exif.get_ifd(_PIL_EXIF_IFD)
        raw.update({name: sub.get(tag) for tag, name in _PIL_EXIF.items()})

    values = {k: v for k, v in raw.items() if v not in (None, "")}
    # PIL 給的是有理數，換成和 ExifTool 一樣的寫法
    if "ExposureTime" in values:
        t = float(values["ExposureTime"])
        values["ExposureTime"] = f"1/{round(1 / t)}" if 0 < t < 1 else f"{t:g}"
    if "FNumber" in values:
        values["FNumber"] = float(values["FNumber"])
    if "FocalLength" in values:
        values["FocalLength"] = f"{float(values['FocalLength']):.1f} mm"
    for key in ("Model", "LensModel", "DateTimeOriginal"):
        if key in values:
            values[key] = str(values[key]).strip("\x00 ")
    return values


def read_shooting_info(path):
    """
    回傳 (資訊, 讀不到的原因)。資訊是 {ExifTool 標記名: 值}，沒有任何拍攝資訊時是空 dict。
    原因只在完全讀不到時才有，例如 RAW 但沒有安裝 ExifTool。
    """
    path = os.path.abspath(path)
    if path in _cache:
        return _cache[path]

    exe = shutil.which("exiftool")
    ext = os.path.splitext(path)[1].lower()
    info, reason = {}, ""
    try:
        if exe:
            info = _read_with_exiftool(exe, path)
            if info is None:
                info, reason = {}, "ExifTool 讀取失敗"
        elif ext in (".jpg", ".jpeg", ".tif", ".tiff"):
            info = _read_with_pil(path)
        else:
            reason = "讀這種格式的拍攝資訊需要安裝 ExifTool"
    except Exception as e:
        info, reason = {}, f"讀取失敗：{e}"

    _cache[path] = (info, reason)
    return info, reason


def format_shooting_info(info, reason=""):
    """把 read_shooting_info 的結果排成給人看的幾行文字。"""
    if not info:
        return reason or "這張照片沒有拍攝資訊（常見於截圖或經過編修輸出的照片）"

    # 攝影常見的寫法，不加欄位名稱：相機、鏡頭、一行曝光參數、拍攝時間。
    # 曝光參數之間用「 · 」：「｜」加全形空白太長，在結果面板裡被切斷（例如「127」與「mm」分到兩行）；
    # 只用全形空白的話，畫面上會被縮成一般空白而擠在一起。
    lines = []
    if info.get("Model"):
        lines.append(str(info["Model"]))
    if info.get("LensModel"):
        lines.append(str(info["LensModel"]))

    exposure = []
    if "ISO" in info:
        exposure.append(f"ISO {info['ISO']}")
    if "ExposureTime" in info:
        exposure.append(f"{info['ExposureTime']} 秒")
    if "FNumber" in info:
        exposure.append(f"f/{float(info['FNumber']):g}")
    if "FocalLength" in info:
        exposure.append(str(info["FocalLength"]).replace(".0 mm", " mm"))
    if exposure:
        lines.append(" · ".join(exposure))

    if "DateTimeOriginal" in info:
        date, _, time = str(info["DateTimeOriginal"]).partition(" ")
        lines.append(f"{date.replace(':', '-')} {time}".rstrip())

    return "\n".join(lines) if lines else "這張照片沒有拍攝資訊（常見於截圖或經過編修輸出的照片）"
