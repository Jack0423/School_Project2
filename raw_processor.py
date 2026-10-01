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


class RawProcessor:
    @staticmethod
    def is_raw(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext in RAW_EXTENSIONS

    @staticmethod
    def map_score_to_rating(score):
        """依據新模型綜合分尺度映射星等"""
        if score >= 65:
            return 5
        elif score >= 54:
            return 4
        elif score >= 43:
            return 3
        elif score >= 32:
            return 2
        else:
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

            cmd = [
                exiftool_cmd,
                f"-XMP:Rating={rating}",
                "-overwrite_original",
                xmp_path
            ]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except Exception as e:
                print(f"[ExifTool 例外] {e}")
                return False, rating

            if result.returncode == 0:
                return True, rating
            print(f"[ExifTool 錯誤] {result.stderr.strip()}")
            return False, rating

        # 情況二：尚無 sidecar 檔，新建 .xmp 檔，完全不碰、不寫入原始檔本身
        # 用 "x" 模式開檔：檢查之後若有別的程式剛好建立了同名 .xmp，寧可失敗也不覆蓋它
        try:
            xmp_content = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="{rating}"/>
 </rdf:RDF>
</x:xmpmeta>"""
            with open(xmp_path, "x", encoding="utf-8") as f:
                f.write(xmp_content.strip())
            return True, rating
        except Exception as e:
            print(f"[新建 XMP 失敗] {e}")
            return False, rating
