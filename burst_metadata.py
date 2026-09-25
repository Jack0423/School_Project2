import json
import math
import re
from datetime import datetime
from pathlib import Path


def normalized_path(path):
    return str(Path(path).expanduser().resolve())


_FRACTION = re.compile(r'\.(\d+)')


def _normalize_fraction(text):
    """
    把小數秒補到 6 位。

    ExifTool 寫出來的小數秒位數取決於相機，Sony 實測是兩位（例如 21:55:35.32）。
    Python 3.11 之前的 datetime.fromisoformat 只接受 3 位或 6 位，兩位會直接拋
    ValueError——在 Python 3.10 上（本專案的開發環境）連拍時間會整個解析失敗，
    退回只精確到秒的 DateTimeOriginal，同一秒內的連拍因此分不出先後；
    若相機只寫了 SubSecDateTimeOriginal，那張照片會變成「沒有拍攝時間」而被排除。

    這種差異不會有任何錯誤訊息，只會讓分組結果在不同 Python 版本上不一樣
    （開發機 3.10 / 批次管線開發者的 macOS 3.12）。
    """
    return _FRACTION.sub(lambda m: '.' + (m.group(1) + '000000')[:6], text, count=1)


def parse_capture_time(item):
    # 優先使用含小數秒及時區的時間。
    for field in ("SubSecDateTimeOriginal", "DateTimeOriginal"):
        value = item.get(field)

        if not isinstance(value, str):
            continue

        # EXIF：2026:05:13 21:55:35.32+08:00
        # 轉成：2026-05-13T21:55:35.32+08:00
        value = value.strip()
        text = _normalize_fraction(value[:10].replace(":", "-") + "T" + value[11:])

        try:
            return datetime.fromisoformat(text)
        except ValueError:
            continue

    return None


CAMERA_MATCH_CHOICES = ("serial", "model", "ignore")


def camera_identity(item, camera_match="model"):
    """
    判斷這張照片出自哪一台相機，回傳可比較的識別值，無法判斷時回傳 None。

    依序嘗試三種欄位，前一種讀不到才往下退：

        SerialNumber          機身序號。最準，同型號的兩台相機也分得出來。
        InternalSerialNumber  Sony 等廠牌把序號放在 MakerNotes 的這個欄位，
                              準確度與上一項相同，只是欄位名不同。
        Make + Model          廠牌與型號。同型號的兩台相機會被視為同一台。

    camera_match 決定最低可接受的層級：
        'serial'  只接受序號（上面前兩項），讀不到就不配對——最嚴格。
        'model'   序號讀不到時退到廠牌＋型號（預設）。
        'ignore'  完全不比對相機，只看拍攝時間。

    為什麼預設不是最嚴格的 'serial'：
        實測 Sony ILCE-7M4 的 JPG 完全沒有序號欄位、ARW 只有 InternalSerialNumber。
        只認 SerialNumber 的話，整個資料夾一組連拍都找不出來，而且不會有任何錯誤訊息。
        退到型號的風險是「兩台同型號相機」被當成同一台，但那兩張照片還必須同時通過
        拍攝時間與內容相似度兩道門檻（預設 5 秒內、0.85），實際誤判的機會很低。

    回傳值的第一個元素是判斷層級，比較時一併納入：
    以序號認定的照片，不會和只能以型號認定的照片配成一組。
    """
    if camera_match not in CAMERA_MATCH_CHOICES:
        raise ValueError(f"camera_match 必須是 {CAMERA_MATCH_CHOICES} 其中之一")

    if camera_match == "ignore":
        return "ignored"

    for field in ("SerialNumber", "InternalSerialNumber"):
        serial = str(item.get(field) or "").strip()
        if serial:
            return ("serial", serial)

    if camera_match == "serial":
        return None

    model = str(item.get("Model") or "").strip()
    if not model:
        return None

    return ("model", str(item.get("Make") or "").strip(), model)


DEFAULT_MAX_SECONDS = 5.0
"""
整組連拍的時間跨度上限，2026-09-23 由 2 秒放寬。

人工標註顯示，抽查的 15 張未分組照片中有 3 張是「相似度夠高（0.92~0.93）、
但拍攝時間差了 3~4 秒」而被排除——實際按快門的節奏沒有那麼緊湊。
放寬到 5 秒後重新標註有變動的 35 組，沒有出現錯誤的合併。
"""


def build_burst_check(metadata_path, max_seconds=DEFAULT_MAX_SECONDS, camera_match="model"):
    """
    回傳 can_pair(路徑A, 路徑B)，以及一份識別層級的統計（給使用者確認判斷依據）。
    """
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("時間門檻必須是非負的有限數字")

    with open(metadata_path, encoding="utf-8") as file:
        items = json.load(file)

    metadata = {
        normalized_path(item["SourceFile"]): item
        for item in items
        if item.get("SourceFile")
    }

    info = {
        path: (parse_capture_time(item), camera_identity(item, camera_match))
        for path, item in metadata.items()
    }

    stats = {"serial": 0, "model": 0, "ignored": 0, "無法識別": 0, "無拍攝時間": 0}
    for capture_time, camera in info.values():
        if camera is None:
            stats["無法識別"] += 1
        elif camera == "ignored":
            stats["ignored"] += 1
        else:
            stats[camera[0]] += 1
        if capture_time is None:
            stats["無拍攝時間"] += 1

    def can_pair(path_a, path_b):
        time_a, camera_a = info.get(
            normalized_path(path_a), (None, None)
        )
        time_b, camera_b = info.get(
            normalized_path(path_b), (None, None)
        )

        if time_a is None or time_b is None:
            return False

        if camera_a is None or camera_a != camera_b:
            return False

        # 有時區和沒有時區的時間，不直接混著比較。
        if (time_a.tzinfo is None) != (time_b.tzinfo is None):
            return False

        difference = abs((time_a - time_b).total_seconds())
        return difference <= max_seconds

    return can_pair, stats
