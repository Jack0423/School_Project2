import os
import shutil
import subprocess

# 解碼與 RAW 格式清單一律以推論端為準，不另外維護一份。
# 自己解碼的話，參數或 JPG 轉正只要有一點不同，分數就會和評分時不一樣，而且不會報錯。
from ai_inference import RAW_EXTENSIONS, load_image

# Lightroom 只讀「相機原生 RAW」旁邊的 .xmp sidecar。
# JPG、PNG、TIFF 與 DNG 的中繼資料寫在檔案本身，旁邊的 .xmp 會被忽略；
# 而且 RAW+JPG 同時拍攝時兩張檔名相同，會寫到同一個 .xmp，後寫的蓋掉先寫的
# （2026-10-01 實測：DSC04606.ARW 的 4 星被同名 JPG 蓋成 2 星）。
# 因此只幫這些格式寫 sidecar。
SIDECAR_EXTENSIONS = RAW_EXTENSIONS - {'.dng'}

# 綜合分 → 星等：(星等, 綜合分下限)，都不到就是 1 星。
#
# 2026-10-04 依實際照片重訂。原本的 65／54／43／32 是在前台預設權重 0.8、只有 28 張照片時訂的；
# 權重改為 0.6 後綜合分整體變高，935 張實拍中 1 星 0%、5 星 28%，星等幾乎分不開。
# 改以預設權重 0.6 下的 935 張實拍（F:\testing_photo 的 0810 共 610 張、testphoto 共 325 張）為準，
# 目標 1～5 星約 10%／20%／40%／20%／10%，實測 9%／19%／41%／22%／10%。
# 兩個資料夾各自的分布不同（0810 偏高；testphoto 多為高 ISO 夜景、偏低），
# 星等反映的是照片本身，不是在資料夾內排名。
RATING_THRESHOLDS = ((5, 69), (4, 64), (3, 54), (2, 47))


class RawProcessor:
    @staticmethod
    def is_raw(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext in RAW_EXTENSIONS

    @staticmethod
    def map_score_to_rating(score):
        """依據綜合分映射星等（門檻見 RATING_THRESHOLDS）"""
        for rating, lower in RATING_THRESHOLDS:
            if score >= lower:
                return rating
        return 1

    @staticmethod
    def decode_image_rgb(file_path):
        """統一使用推論端的 load_image()：RAW 解碼參數、半尺寸與 JPG 轉正都與評分時相同"""
        return load_image(file_path)

    @staticmethod
    def safe_update_xmp(image_path, score):
        """
        安全更新或建立 XMP（只處理 SIDECAR_EXTENSIONS 的照片，其餘不寫、回傳失敗）：
        1. 若已有 .xmp sidecar：用 ExifTool 原地只改 Rating，保留所有 Lightroom 調色設定。
           沒有 ExifTool 或執行失敗時不做任何修改、回傳失敗——
           自己用文字替換去改，會把 ExifTool 寫過的 .xmp 改成無法解析的 XML，卻仍回報成功
           （ExifTool 把 Rating 寫成 <xmp:Rating> 元素，替換時插進一個沒有宣告命名空間的屬性）。
        2. 若無 .xmp sidecar：新建獨立的 .xmp 檔案，絕對不碰原始照片檔！

        一次寫很多張時請用 update_ratings()：已有 .xmp 的照片在這裡每張都要啟動一次 ExifTool。
        """
        rating = RawProcessor.map_score_to_rating(score)
        base_name, ext = os.path.splitext(image_path)
        xmp_path = f"{base_name}.xmp"

        if ext.lower() not in SIDECAR_EXTENSIONS:
            print(f"[略過] {os.path.basename(image_path)}：Lightroom 不讀 {ext} 旁邊的 .xmp")
            return False, rating

        # 情況一：已有 sidecar 檔，只用 ExifTool 原地更新 Rating，保留其他修圖標籤
        if os.path.exists(xmp_path):
            exiftool_cmd = shutil.which("exiftool")
            if not exiftool_cmd:
                print(f"[XMP 未更新] {os.path.basename(xmp_path)} 已存在，需要 ExifTool 才能安全修改"
                      f"｜請安裝 ExifTool：https://exiftool.org")
                return False, rating
            ok, error = _exiftool_set_rating(exiftool_cmd, rating, [xmp_path])
            if not ok:
                print(f"[ExifTool 錯誤] {error}")
            return ok, rating

        # 情況二：尚無 sidecar 檔，新建 .xmp 檔，完全不碰、不寫入原始檔本身
        return _create_sidecar(xmp_path, rating), rating

    @staticmethod
    def update_ratings(items, on_progress=None):
        """
        一次寫很多張照片的星等。items 是 [(照片路徑, 綜合分), ...]，回傳同順序的 [(是否成功, 星等), ...]。

        結果與逐張呼叫 safe_update_xmp 相同，差在已有 .xmp 的照片：
        ExifTool 每次啟動約 0.4 秒（啟動本身慢，不是寫檔慢），逐張改 325 張要將近 2 分鐘。
        這裡把同一個星等的照片交給同一次 ExifTool，最多啟動 5 次。
        某一組失敗時退回逐張處理，才知道是哪幾張失敗。

        on_progress(完成張數, 總張數, 照片路徑)：沒有 .xmp 的每張回報一次，已有 .xmp 的每組回報一次。
        """
        results = [None] * len(items)
        groups = {}          # 星等 -> [(索引, 照片路徑, .xmp 路徑)]
        done = 0
        total = len(items)

        for index, (image_path, score) in enumerate(items):
            rating = RawProcessor.map_score_to_rating(score)
            base_name, ext = os.path.splitext(image_path)
            xmp_path = f"{base_name}.xmp"
            if ext.lower() not in SIDECAR_EXTENSIONS:
                results[index] = (False, rating)
            elif os.path.exists(xmp_path):
                groups.setdefault(rating, []).append((index, image_path, xmp_path))
                continue
            else:
                results[index] = (_create_sidecar(xmp_path, rating), rating)
            done += 1
            if on_progress:
                on_progress(done, total, image_path)

        exiftool_cmd = shutil.which("exiftool") if groups else None
        if groups and not exiftool_cmd:
            print("[XMP 未更新] 已存在的 .xmp 需要 ExifTool 才能安全修改｜請安裝 ExifTool：https://exiftool.org")

        for rating, members in groups.items():
            if exiftool_cmd:
                ok, error = _exiftool_set_rating(exiftool_cmd, rating, [xmp for _, _, xmp in members])
            else:
                ok, error = False, ""
            if ok:
                for index, _, _ in members:
                    results[index] = (True, rating)
            elif exiftool_cmd:
                # 整組失敗時逐張重試，找出是哪幾張壞掉
                print(f"[ExifTool 錯誤] {rating} 星這組有檔案沒有更新，改逐張處理：{error}")
                for index, image_path, _ in members:
                    results[index] = RawProcessor.safe_update_xmp(image_path, items[index][1])
            else:
                for index, _, _ in members:
                    results[index] = (False, rating)
            done += len(members)
            if on_progress:
                on_progress(done, total, members[-1][1])

        return results


def _exiftool_set_rating(exiftool_cmd, rating, xmp_paths):
    """
    用一次 ExifTool 把多個 .xmp 的 Rating 改成同一個值，回傳 (是否全部成功, 錯誤訊息)。

    路徑放在 UTF-8 參數檔（從標準輸入傳）而不是命令列：Windows 上中文檔名直接放命令列會變亂碼，
    也不會因為檔案太多而超過命令列長度上限。
    """
    args = [f"-XMP:Rating={rating}", "-overwrite_original", *xmp_paths]
    try:
        result = subprocess.run(
            [exiftool_cmd, "-charset", "filename=utf8", "-@", "-"],
            input=("\n".join(args) + "\n").encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    error = result.stderr
    if isinstance(error, bytes):
        error = error.decode("utf-8", "replace")
    return result.returncode == 0, (error or "").strip()


def _create_sidecar(xmp_path, rating):
    """新建只含星等的 .xmp。用 "x" 模式開檔：檢查之後若有別的程式剛好建立了同名 .xmp，寧可失敗也不覆蓋它。"""
    try:
        xmp_content = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="{rating}"/>
 </rdf:RDF>
</x:xmpmeta>"""
        with open(xmp_path, "x", encoding="utf-8") as f:
            f.write(xmp_content.strip())
        return True
    except Exception as e:
        print(f"[新建 XMP 失敗] {e}")
        return False
