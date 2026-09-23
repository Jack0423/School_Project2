import json
import math
from datetime import datetime
from pathlib import Path


def normalized_path(path):
    return str(Path(path).expanduser().resolve())


def parse_capture_time(item):
    # 優先使用含小數秒及時區的時間。
    for field in ("SubSecDateTimeOriginal", "DateTimeOriginal"):
        value = item.get(field)

        if not isinstance(value, str):
            continue

        # EXIF：2026:05:13 21:55:35.32+08:00
        # 轉成：2026-05-13T21:55:35.32+08:00
        value = value.strip()
        text = value[:10].replace(":", "-") + "T" + value[11:]

        try:
            return datetime.fromisoformat(text)
        except ValueError:
            continue

    return None


def build_burst_check(metadata_path, max_seconds=2.0):
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("時間門檻必須是非負的有限數字")

    with open(metadata_path, encoding="utf-8") as file:
        items = json.load(file)

    metadata = {
        normalized_path(item["SourceFile"]): item
        for item in items
        if item.get("SourceFile")
    }

    def camera_identity(item):
        # 僅靠相機型號，無法確認是同一台相機。
        serial = str(item.get("SerialNumber") or "").strip()

        if not serial:
            return None

        return (
            str(item.get("Make") or "").strip(),
            str(item.get("Model") or "").strip(),
            serial,
        )

    info = {
        path: (parse_capture_time(item), camera_identity(item))
        for path, item in metadata.items()
    }

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

    return can_pair
