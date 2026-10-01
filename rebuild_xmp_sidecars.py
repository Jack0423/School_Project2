import os
import sys


def batch_refresh_xmp(target_dir):
    # 載入不到模型直接退出，絕不採用假分數寫入硬碟。
    # raw_processor 也會載入 ai_inference，所以一起放在這裡，失敗時才會印出下面的說明而不是 traceback。
    try:
        import ai_inference
        from ai_inference import evaluate_photo
        from raw_processor import RawProcessor, SIDECAR_EXTENSIONS
    except (ImportError, FileNotFoundError) as e:
        print("[致命錯誤] 未能載入 ai_inference 模組！")
        print(f"原因：{e}")
        print("原則：載入不到模型即明確失敗，絕不使用假分數產生錯誤 XMP 星等。")
        sys.exit(1)

    if not os.path.exists(target_dir):
        print(f"[錯誤] 目錄不存在: {target_dir}")
        return

    count = 0
    fail_count = 0
    skipped = 0

    for file_name in sorted(os.listdir(target_dir)):
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in SIDECAR_EXTENSIONS:
            # JPG、PNG、DNG 等：Lightroom 不讀它們的 sidecar，同名 RAW 的星等也不能被蓋掉
            if ext in ai_inference.IMAGE_EXTENSIONS:
                skipped += 1
            continue

        full_path = os.path.join(target_dir, file_name)
        try:
            # 1. 取得一致的解碼陣列（RGB）
            rgb = RawProcessor.decode_image_rgb(full_path)

            # 2. 傳入推論端進行真實評估
            result = evaluate_photo(full_path, image=rgb)

            # 3. 處理失敗防禦（回傳為 None 時印出 LAST_ERROR）
            if result is None:
                err_msg = getattr(ai_inference, 'LAST_ERROR', '未知錯誤')
                print(f"[FAIL] {file_name}: {err_msg}")
                fail_count += 1
                continue

            # 4. 正確從 dict 提取 overall_score
            score = result['overall_score']

            # 5. 使用 ExifTool 安全修改星等，不損害原有修圖設定
            success, rating = RawProcessor.safe_update_xmp(full_path, score)
            if success:
                print(f"[OK] {file_name} -> 綜合分: {score:.1f} | 寫入星等: {rating}★")
                count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"[FAIL] {file_name}: {e}")
            fail_count += 1

    print(f"\n執行完畢：成功更新 {count} 張，失敗 {fail_count} 張。")
    if skipped:
        print(f"另有 {skipped} 張非 RAW 照片（JPG、PNG、DNG 等）沒有寫 .xmp：Lightroom 不讀它們的 sidecar。")


if __name__ == "__main__":
    target = input("請輸入照片目錄路徑 (直接按 Enter 使用當前目錄): ").strip()
    target_dir = target if target else os.getcwd()
    batch_refresh_xmp(target_dir)
