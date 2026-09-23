import json
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ai_inference


SUPPORTED_EXTENSIONS = ai_inference.RAW_EXTENSIONS | {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
}


def collect_photos(folder):
    """遞迴收集資料夾中的照片。"""
    folder = Path(folder).expanduser().resolve()

    if not folder.is_dir():
        raise ValueError(f"不是有效資料夾：{folder}")

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def analyze_batch(paths, output_path, aesthetic_weight=0.6,
                  on_progress=None):
    """
    解碼使用 4 個執行緒。
    推論只在呼叫本函式的執行緒依序執行。

    同一時間只能啟動一個批次；
    前端單張分析也不能同時呼叫模型。
    """
    weight = float(aesthetic_weight)

    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("美感權重必須介於 0 與 1")

    # 建立本批次的固定路徑清單與權重。
    paths = [str(Path(p).expanduser().resolve()) for p in paths]
    output_path = Path(output_path).expanduser().resolve()

    if output_path in {Path(p) for p in paths}:
        raise ValueError("結果檔案不能和輸入照片相同")

    total = len(paths)
    success_count = 0
    failed_count = 0
    path_iterator = iter(paths)

    # 使用 x 模式，避免不小心覆寫之前的結果。
    with output_path.open("x", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = deque()

            def submit_next():
                path = next(path_iterator, None)
                if path is None:
                    return

                future = pool.submit(
                    ai_inference._load_image_array,
                    path,
                )
                pending.append((path, future))

            # 限制預讀數量，避免整個資料夾的 RAW 同時占用記憶體。
            for _ in range(min(4, total)):
                submit_next()

            completed = 0

            while pending:
                path, future = pending.popleft()
                image = None

                try:
                    image = future.result()
                    decode_error = None
                except Exception as exc:
                    decode_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

                # Future 會保留解碼結果，取出後釋放它的參照。
                del future

                if decode_error is not None:
                    record = {
                        "path": path,
                        "ok": False,
                        "stage": "decode",
                        "error": decode_error,
                        "aesthetic_weight": weight,
                        "technical_weight": 1 - weight,
                    }
                    failed_count += 1
                else:
                    try:
                        # 不可放進 pool.submit()：
                        # 這裡就是唯一的模型推論消費者。
                        result = ai_inference.evaluate_photo(
                            path,
                            image=image,
                            aesthetic_weight=weight,
                            # 分組需要特徵；evaluate_photo 預設不回傳。
                            return_features=True,
                        )

                        # 在下一次推論前立即讀取錯誤原因。
                        error = (
                            ai_inference.LAST_ERROR
                            if result is None else None
                        )
                    finally:
                        del image

                    if result is None:
                        record = {
                            "path": path,
                            "ok": False,
                            "stage": "inference",
                            "error": error or "推論失敗",
                            "aesthetic_weight": weight,
                            "technical_weight": 1 - weight,
                        }
                        failed_count += 1
                    else:
                        record = {
                            "path": path,
                            "ok": True,
                            **result,
                        }
                        success_count += 1

                # 每完成一張就保存一行，不必把全部特徵留在記憶體。
                output.write(
                    json.dumps(record, ensure_ascii=False,
                               allow_nan=False) + "\n"
                )
                output.flush()

                completed += 1
                submit_next()

                if on_progress is not None:
                    on_progress(completed, total, record)

    return {
        "total": total,
        "success": success_count,
        "failed": failed_count,
        "output": str(output_path),
    }
