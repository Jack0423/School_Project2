import hashlib
import os
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

# ==========================================
# 1. 模型架構與前處理（改由 common/ 提供，確保與訓練時完全一致）
# ==========================================
# 這兩者原本在本檔案內另外寫了一份，與 train_nima.py / train_tech.py 各自分岔。
# 前處理的分岔尤其有實質影響：本檔案原本用 Resize((224,224)) 壓扁長寬比，
# 訓練驗證用的卻是 Resize(256)+CenterCrop(224)。實測同一組權重只換前處理，
# 技術模型在 KonIQ 驗證集上 PLCC 0.8056→0.8315、SRCC 0.7636→0.7953。
from common import NIMABaseline, build_transform

# ==========================================
# 2. 全域初始化設定 (只在模組載入時執行一次)
# ==========================================
# 本檔案的 console 輸出一律用 [ OK ] / [WARN] / [FAIL] / [INFO] 純文字標籤，
# 不使用 emoji——與 check_env.py 的既有慣例一致。
#
# 這不是風格偏好，是這個模組會被前台 import：
# Windows 繁中版主控台預設編碼是 cp950，print 一個 emoji 會拋 UnicodeEncodeError，
# 而這幾行 print 位在模組層級，等於「一句純裝飾的訊息把整個 import 弄掛」，
# 前台會直接失去 AI 功能。訊息本身的中文在 cp950 沒問題，只有 emoji 不行。
# 前台 arw_viewer_gui.py 的 Qt 標籤要放 emoji 沒關係——那不經過主控台編碼。
def _select_device():
    """
    選出一個「實際上真的能跑」的推論裝置。

    torch.cuda.is_available() 只回答「機器上有沒有 CUDA 裝置」，
    它不檢查「目前這版 PyTorch 編譯出來的 kernel 支不支援這張卡的架構」。
    例如 RTX 50 系列是 sm_120，若安裝的 PyTorch 只編到 sm_90，
    is_available() 仍會回 True，但任何一次實際運算都會炸：
        CUDA error: no kernel image is available for execution on the device
    而且錯誤是在推論當下才爆，很容易被誤判成「這張照片有問題」。

    因此這裡在啟動時就真的跑一次極小的 GPU 運算來探測，
    失敗就明確降級到 CPU 並告知原因，讓程式仍然可用。
    """
    if not torch.cuda.is_available():
        print("[INFO] [AI 模組] 未偵測到可用的 CUDA 裝置，使用 CPU 推論。")
        return torch.device("cpu")

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "未知裝置"

    try:
        # 真正送一次運算到 GPU；synchronize 確保非同步錯誤在這裡就被抓到
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
        print(f"[ OK ] [AI 模組] GPU 可用：{gpu_name}")
        return torch.device("cuda")
    except Exception as e:
        cap = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        print(
            f"[WARN] [AI 模組] 偵測到 GPU（{gpu_name}，運算能力 sm_{cap.replace('.', '')}），"
            f"但目前安裝的 PyTorch {torch.__version__} 無法在它上面執行運算，已自動改用 CPU。\n"
            f"    實際錯誤：{type(e).__name__}: {e}\n"
            f"    這不是照片或程式的問題，是 PyTorch 版本與顯示卡架構不相容；"
            f"要啟用 GPU 需改裝對應 CUDA 版本的 PyTorch。"
        )
        return torch.device("cpu")


# 偵測硬體：能用 GPU 就用，不能用就明確降級 CPU（不會讓程式整個掛掉）
DEVICE = _select_device()

# 影像前處理：與訓練時的驗證集使用完全相同的設定（見 common/transforms.py）
TRANSFORM = build_transform('eval')

# 綜合分權重的「預設值」：美感為主、技術為輔。
#
# 這兩個常數只是預設，不再是唯一的權重來源——evaluate_photo() 可以逐張傳入
# 不同的權重（見該函式的 aesthetic_weight / technical_weight 參數）。
# 期末報告承諾要讓使用者自行拉動美感與技術的佔比，而不是死綁在 6:4，
# 前台的滑桿就是把值傳進那兩個參數，不需要（也不應該）去改動這裡。
AESTHETIC_WEIGHT = 0.6
TECHNICAL_WEIGHT = 0.4

# 對外顯示的分數範圍
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# 技術分低於此門檻視為整體品質不足。
# 注意：這個門檻「不再」用來決定要不要跑細項分析——細項分析現在無條件執行。
# 舊版把細項分析掛在這個門檻之下，導致技術分高的照片即使有明顯曝光或對比
# 問題也完全不會被檢查（實測 DSC04606 標註為欠曝，技術分 73.3，永遠不觸發）。
TECH_ISSUE_THRESHOLD = 60

# 美感分高於此門檻時給「優秀」評價。
#
# 這個門檻必須跟著模型的輸出尺度走。舊的二元模型輸出沒有界限（實測 -32.8 ~ 154.9），
# 門檻 85 是針對那個會爆表的尺度訂的；換成分佈模型後輸出範圍是 21.1 ~ 73.8，
# 沿用 85 會導致**沒有任何一張照片會被評為優秀**，等於功能靜默失效。
#
# 新門檻 62 的訂法：以舊模型的選取比例為基準做校準，
# 讓「優秀」的嚴格程度維持在使用者已經習慣的水準。
#   舊模型 >85    選出 494 張（35.3%），其中真實 AVA 分數 >6 者佔 88.9%
#   新模型 >62    選出 497 張（35.5%），其中真實 AVA 分數 >6 者佔 92.4%
# 也就是選取比例幾乎相同，但準確度更高。
#
# 若想更嚴格，可參考實測：>68 選出 16.9%、準確度 96.2%。
AESTHETIC_EXCELLENT_THRESHOLD = 62

# RAW 檔副檔名。這些格式必須交給 rawpy，不能讓 PIL 處理：
#   .ARW  —— PIL 直接失敗（ValueError: Invalid dimensions），至少會報錯。
#   .DNG  —— 更危險。PIL「開得起來」，但讀到的是檔案內嵌的預覽縮圖。
#            實測 DSC01044-Enhanced-NR.dng 實際影像 6048x4024，
#            PIL 只讀到 256x171，而且完全不會報錯——等於安靜地拿縮圖去評分。
# 因此這裡以副檔名強制分流，不依賴 PIL 是否「成功」開啟。
RAW_EXTENSIONS = {
    '.arw', '.dng', '.cr2', '.cr3', '.nef', '.nrw',
    '.raf', '.orf', '.rw2', '.pef', '.srw', '.dcr',
}

# RAW 預設以「半尺寸」解碼（rawpy 的 half_size：2x2 像素合成一個，跳過去馬賽克）。
#
# 為什麼：解碼佔整個流程 83.5% 的時間，是真正的瓶頸。
# 以 325 張 Sony ARW（15 GB）實測整條批次管線：
#     全尺寸  每張 366~373 ms   325 張約 2 分鐘
#     半尺寸  每張  80~ 85 ms   325 張約 26~30 秒   （4.3 倍，調換測試順序驗證過不是快取造成）
# 模型只吃 224x224、細項分析只用 800px 寬，半尺寸的 3024x2012 綽綽有餘。
#
# 代價（120 張實測）：綜合分平均差 0.34~0.56、最大 2.1，
# 而 60 張中有 3 張的狀態判定改變——都是分數本來就壓在門檻上的照片
# （例如技術分 59.63 -> 60.84，剛好越過 60 這條線）。
# 該批有 14/60 的技術分落在門檻 ±3 分內，這種邊緣照片被一點點差異推過去是必然的。
#
# 需要「和之前完全一致」、或要最準的雜訊判斷時，改用全尺寸：
#     evaluate_photo(path, half_size=False)
#
# 雜訊偵測在兩種尺寸下各有一組門檻（見 NOISE_SIGMA_THRESHOLD_HALF）。
# 半尺寸量出來的數值比全尺寸高 1.4~2.5 倍，而且乾淨與有雜訊兩群重疊得更多，
# 只抓得到明顯的高 ISO 雜訊——這是換取 4.3 倍速度的代價之一。
RAW_HALF_SIZE = True

# 其餘解碼參數刻意與前台顯示用的設定保持一致，
# 確保「前台傳入已解碼影像」與「本模組自行解碼」得到完全相同的結果。
_RAW_POSTPROCESS = dict(
    use_camera_wb=True,
    no_auto_bright=True,
    user_flip=None,
)


def raw_postprocess_params(half_size=None):
    """
    回傳 rawpy.postprocess() 要用的參數。

    呼叫端若自己解碼 RAW（前台為了顯示、RAW 模組為了寫 XMP），
    請用這個函式取得參數，不要自己抄一份：
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(**ai_inference.raw_postprocess_params())

    參數不一致會讓傳進 evaluate_photo(image=...) 的影像與本模組自行解碼的不同，
    算出來的分數就跟著不同，而且不會有任何錯誤訊息。
    """
    return dict(_RAW_POSTPROCESS,
                half_size=RAW_HALF_SIZE if half_size is None else bool(half_size))

# 傳統影像分析門檻值（可依實測樣本再微調）
BLUR_VAR_THRESHOLD = 100.0       # Laplacian 變異數，低於此值視為模糊/失焦/手震
OVEREXPOSED_RATIO_THRESHOLD = 0.05   # 死白像素比例
UNDEREXPOSED_RATIO_THRESHOLD = 0.05  # 死黑像素比例
# 雜訊判斷門檻（全尺寸解碼、JPG/PNG 用）。設為 None 代表停用：不計算、也不回報。
#
# 2026-09-23 以人工標註校準後啟用（先前因樣本不足而設為 None）。
# 下面所有 sigma 數值都是在「全尺寸」解碼上量的；RAW 預設的半尺寸另有一組，
# 見 NOISE_SIGMA_THRESHOLD_HALF。
#
# 校準資料：同一台 Sony ILCE-7M4 的 90 張 ARW，分兩輪盲標
#   （ISO 與程式算出的數值都不顯示、順序打亂，避免標註者被影響）：
#     第一輪  ISO 100~125 共 30 張、ISO 10000 共 30 張
#     第二輪  ISO 2000~6400 共 30 張（補中間地帶，真正的界線在這裡）
#   標註者判斷「看不看得出顆粒或色斑」，扣掉 13 張「不確定」後得到
#   53 張有雜訊、24 張乾淨。第一輪的判斷與 ISO 完全一致（53/53），
#   可視為標註品質的佐證。
#
# 量測結果：
#     有雜訊   sigma 1.319 ~ 2.79
#     乾淨     sigma 0.430 ~ 1.427
#   兩群在 1.32~1.43 之間重疊，所以沒有任何門檻能同時零誤報與零漏報。
#
# 為什麼選 1.43（剛好在乾淨樣本的最大值之上）：
#     門檻 0.95   漏報 0 張、誤報 3 張
#     門檻 1.43   漏報 5 張、誤報 0 張   <- 採用
#     門檻 1.90   漏報 19 張、誤報 0 張
#   技術量測是顯示給使用者看的「附註」，誤報的代價高於漏報：
#   把一張乾淨照片說成有雜訊，會直接損害整個評分的可信度；
#   漏掉幾張輕微雜訊的（都是 ISO 2000~2500、sigma 1.32~1.41），使用者通常不會察覺。
#
# 已知限制：這個估計量對「高頻細節」與「雜訊」無法完全區分。
#   合成測試影像中，細節豐富那張的 sigma 是 53.5，比刻意加雜訊那張的 39.8 還高。
#   實照片上的影響較小：把同一批 ISO<=800 的 76 張全部算過，只有 3 張超過門檻
#   （sigma 1.60~1.65，皆為 ISO 400~800，本身可能就有輕微雜訊），約 4%。
#   換相機或換題材（大量樹葉、砂石等細碎紋理）時應重新校準。
NOISE_SIGMA_THRESHOLD = 1.43

# 分級門檻：超過這個值才說「偏高」，1.43~2.0 之間只說「輕微」。
#
# 為什麼要分級：門檻附近（1.6 左右）的照片多半是夜景暗部的輕微顆粒，
# 一律寫成「雜訊偏高」會讓使用者覺得系統太敏感——實際看圖確實只是輕微。
# 標註樣本中量測值超過 2.0 的 32 張全部是 ISO 5000~10000，
# 1.43~2.0 之間的 16 張則是 ISO 2000~6400，因此以 2.0 分界。
#
# 附註不影響任何分數與狀態判定，純粹是顯示給使用者的說明文字。
NOISE_SIGMA_HIGH = 2.0

# RAW 以半尺寸解碼時用的門檻（2026-09-25 以同一批 77 張盲標樣本重新校準）。
#
# 為什麼要另一組：上面的 1.43 是在全尺寸上校準的，RAW 改成預設半尺寸之後，
# 同一張照片量出來的 sigma 高 1.4~2.5 倍（半尺寸跳過去馬賽克，雜訊沒有被插值抹平，
# 細節在每個像素裡也更密）。沿用 1.43 的話，ISO 100~125 的 30 張裡有 7 張被標記、
# 4 張被寫成「偏高」，完全違背當初「零誤報」的取捨。
#
# 半尺寸量測結果：
#     有雜訊   sigma 2.054 ~ 4.369
#     乾淨     sigma 0.717 ~ 3.016
#   重疊區 2.05~3.02 比全尺寸（1.32~1.43）寬得多：兩張 ISO 100、細節很密的乾淨照片
#   （DSC02032、DSC02033）在半尺寸落到 2.88、3.02，和 ISO 5000 的照片混在一起。
#
# 沿用同一個取捨（誤報的代價高於漏報），門檻取在乾淨樣本最大值 3.016 之上：
#     門檻 2.05   誤報 2 張、漏報  0 張
#     門檻 3.02   誤報 0 張、漏報 21 張   <- 採用
#   漏報的 21 張是 ISO 2000~5000。也就是說半尺寸只抓得到明顯的高 ISO 雜訊；
#   以 325 張實測，ISO 4000~6400 的 65 張只標記 14 張（全尺寸是 65 張全部）。
#   要完整的雜訊判斷請用 evaluate_photo(path, half_size=False)。
#
# 設為 None 則半尺寸時完全不做雜訊判斷。
NOISE_SIGMA_THRESHOLD_HALF = 3.02

# 半尺寸的分級門檻。有效標註樣本中超過 3.02 的只有兩群：
# ISO 5000 兩張（3.07、3.08）與 ISO 10000 全部 30 張（3.50 以上），
# 取兩群中間的 3.3，讓「偏高」對應 ISO 10000 那一群。
# 已知出入：ISO 400~800 的三張暗部夜景（DSC02141~02143）半尺寸是 3.37~3.97，
# 會被寫成「偏高」；全尺寸時它們只是「輕微」，人工複查也認為其中兩張「略嚴」。
NOISE_SIGMA_HIGH_HALF = 3.3

# 雜訊估計取樣設定：在原始解析度上取 GRID x GRID 個 TILE_SIZE 見方的區塊，
# 把所有未截斷的像素合併成單一樣本池。這與「整張圖計算」是同一個估計量，
# 只是改用抽樣像素，實測誤差多在 0.05 以內，速度快約 6 倍（125 ms → 20 ms）。
NOISE_TILE_SIZE = 512
NOISE_TILE_GRID = 4
# 亮度落在此區間外的像素視為已截斷（死黑／死白），排除於雜訊估計之外
NOISE_VALID_MIN = 16
NOISE_VALID_MAX = 240
# 有效像素太少時無法可靠估計，直接跳過雜訊判斷而非硬給一個數字
NOISE_MIN_VALID_PIXELS = 10000
CONTRAST_STD_THRESHOLD = 30.0    # 亮度標準差，過低代表畫面偏灰、對比不足

# 權重檔一律以「本檔案所在目錄」為基準解析。
# 用相對路徑會依賴當前工作目錄，GUI 由捷徑開啟、或被前台程式匯入時就會找不到權重。
BASE_DIR = Path(__file__).resolve().parent

# 美感模型改用「1~10 級評分分佈」訓練的版本（搭配 emd_loss）。
# 舊的二元標籤版本 nima_best.pth 仍保留在專案內，把下面這行改回去即可退回。
#
# 為什麼換：在同一組 1,400 張驗證照片、以 AVA 群眾平均評分為對照答案的比較中
#     舊版 nima_best.pth（二元、3,920 張）      SRCC 0.7512   輸出 -0.328 ~ 1.549
#     二元 + 新資料（二元、5,600 張）            SRCC 0.7542   輸出 -0.322 ~ 1.482
#     分佈 + emd_loss（分佈、5,600 張）          SRCC 0.7925   輸出  2.899 ~ 7.642
# 值得注意的是「資料增加 43%」幾乎沒有幫助（+0.003），
# 真正有效的是換掉標籤方式（+0.041）。
# 而且分佈模型的輸出天然落在 1~10，換算成 0~100 後必定在範圍內，
# 不再需要靠裁切——舊版有 35.4% 的照片會被壓在 0 或 100 兩端，
# 那些照片彼此之間的排序資訊等於消失。
AES_WEIGHTS = BASE_DIR / 'nima_aes_dist.pth'
TECH_WEIGHTS = BASE_DIR / 'nima_tech_best.pth'

# 評分分佈的級距（1~10 分），用來把機率分佈換算成期望值
_SCORE_LEVELS = torch.arange(1, 11, dtype=torch.float32, device=DEVICE)

_MODEL_VERSION_CACHE = None

# 評分流程的版本號，會出現在 model_version() 的最後面（|rev=N）。
#
# 權重指紋與解碼尺寸以外，任何會讓「同一張照片算出不同結果」的程式修改，
# 都要把這個數字加 1。權重沒換、程式卻改了算法時，只有它能讓資料庫分出新舊——
# 否則修改前存進去的分數會和修改後的混在同一份清單裡排序，不會有任何錯誤訊息。
#   1  2026-09-23 以前（commit 252261c 為止）
#   2  2026-09-25  JPG 依 EXIF 方向轉正（直幅照片的分數會改變）；
#                  RAW 半尺寸改用另一組雜訊門檻（只改附註文字，分數不變）
SCORING_REVISION = 2


def model_version(half_size=None):
    """
    回傳目前這組設定的版本字串，例如：

        nima_aes_dist.pth@71dbd9e1+nima_tech_best.pth@e39a98f4|raw=half|rev=2

    給資料庫記錄「這筆分數是哪一版模型、哪一種解碼設定算出來的」用。
    三段各記一件事：權重內容（檔名＋指紋）、RAW 解碼尺寸、評分流程版本（SCORING_REVISION）。

    參數:
        half_size: 與 evaluate_photo 的同名參數相同。評分時傳了 half_size=False，
            這裡也要傳同一個值，版本字串才會記成 raw=full。不傳就是模組預設。

    為什麼連解碼尺寸也要記：RAW 改用半尺寸解碼後，同一張照片的分數會有
    0.3~0.6 分的差異（邊緣照片甚至會換狀態）。只記權重的話，
    資料庫裡半尺寸與全尺寸算出來的分數會被當成同一版而混在一起排序。

    為什麼需要:
        這學期美感模型換代後輸出尺度整個改變（舊的會爆到 100 以上、新的約 21~74），
        但資料庫裡沒有任何欄位分得出哪幾筆是舊模型算的，最後只能整批清空重跑
        （reset_db_analysis.py 就是為此而寫）。每筆分數附上版本之後，
        下次換模型只要重跑版本對不上的那些，不必全部重來。

    為什麼不只用檔名:
        訓練腳本加 --force 就會把新權重存成同一個檔名，檔名因此不足以識別。
        這裡在檔名後面接上檔案內容 SHA-256 的前 8 碼，內容一變版本就跟著變。

    為什麼是函式而不是模組層級的常數:
        算指紋要把兩個約 9 MB 的權重檔讀過一遍。做成常數會讓每一次
        import ai_inference 都付這個成本，即使呼叫端根本用不到版本字串。
        第一次呼叫後結果會快取，之後都是直接回傳。

    [注意] 特徵向量（見 evaluate_photo 的 return_features）也綁在同一組權重上。
      版本字串變了，舊的特徵就不能再跟新的算相似度。
    """
    global _MODEL_VERSION_CACHE
    if _MODEL_VERSION_CACHE is None:
        parts = []
        for path in (AES_WEIGHTS, TECH_WEIGHTS):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
            parts.append(f'{path.name}@{digest}')
        _MODEL_VERSION_CACHE = '+'.join(parts)
    # 解碼尺寸不快取：它是模組層級的預設值，呼叫端可能在執行期間改掉。
    raw_half = RAW_HALF_SIZE if half_size is None else bool(half_size)
    return f'{_MODEL_VERSION_CACHE}|raw={"half" if raw_half else "full"}|rev={SCORING_REVISION}'


def _load_model(weight_path, label):
    """
    載入單顆模型的權重，並自動判斷它是哪一種輸出模式。

    輸出模式直接從 checkpoint 反推（分類頭輸出 10 維＝評分分佈、1 維＝單一分數），
    而不是寫死在設定裡。這樣同一份程式碼可以載入任一種權重，
    要退回舊模型只需改權重檔路徑，不必同時記得改模式設定——
    兩者不同步會安靜地算出錯誤分數。

    權重檔不存在時直接拋出 FileNotFoundError，而不是印個警告就放行——
    沒有權重的模型是隨機初始化的，照樣能輸出數字，但那些數字是純噪音。
    寧可在啟動時大聲失敗，也不要讓使用者拿到一個看起來正常的假分數。
    """
    if not weight_path.exists():
        raise FileNotFoundError(
            f"[AI 模組] 找不到{label}權重檔：{weight_path}"
            f"｜沒有權重就無法評分（隨機初始化的模型只會產生無意義的分數）"
            f"｜請確認 {weight_path.name} 位於：{BASE_DIR}"
        )

    state = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    out_dim = state['classifier.1.weight'].shape[0]
    mode = 'distribution' if out_dim == 10 else 'single'

    # pretrained=False：稍後 load_state_dict 會完整覆蓋所有權重，
    # 先下載 ImageNet 預訓練權重只是浪費時間與流量
    model = NIMABaseline(output_mode=mode, pretrained=False).to(DEVICE)
    model.load_state_dict(state)
    model.eval()   # 關閉 Dropout，確保推論結果可重現
    print(f"[ OK ] [AI 模組] 成功載入{label}權重：{weight_path.name}（{mode} 模式）")
    return model, mode


# 實例化並載入兩顆大腦的權重（任一顆失敗都會直接中止模組載入，不會留下半殘的模型）
MODEL_AES, AES_MODE = _load_model(AES_WEIGHTS, '美感')
MODEL_TECH, TECH_MODE = _load_model(TECH_WEIGHTS, '技術')


def _output_to_score(output, mode):
    """
    把模型輸出換算成 0~100 的分數。

    distribution 模式：先取評分期望值 sum(i * p_i)，得到 1~10 分，
                       再線性映射到 0~100。因為機率分佈的期望值必定落在
                       1~10 之間，換算結果天然就在 0~100，不需要裁切。
    single 模式：      沿用舊行為，直接乘以 100。這條路徑沒有任何界限保護，
                       仍然依賴後續的 _clamp_score（保留是為了能退回舊權重）。
    """
    if mode == 'distribution':
        mean_score = (output * _SCORE_LEVELS).sum().item()
        return (mean_score - 1.0) / 9.0 * 100.0
    return output.item() * 100.0


# ==========================================
# 3. 影像讀取（依格式分流）
# ==========================================
_EXIF_ORIENTATION = 0x0112   # EXIF 的 Orientation 標記編號；1 代表不用轉


def _load_image_array(img_path, half_size=None):
    """
    讀取影像檔，一律回傳 RGB numpy array，形狀 (H, W, 3)、dtype uint8。

    half_size 只影響 RAW：None 用模組預設（RAW_HALF_SIZE，目前為半尺寸），
    False 強制全尺寸。JPG/PNG 不受影響。

    依副檔名分流：RAW 交給 rawpy，其餘交給 PIL。
    不用「先試 PIL、失敗再試 rawpy」的寫法，因為 .DNG 會讓 PIL
    「成功」讀到內嵌縮圖而不報錯，這種 fallback 永遠不會被觸發。

    兩條路都會把照片轉正：rawpy 依 RAW 內的方向資訊（user_flip=None），
    JPG 等依 EXIF 的 Orientation 標記（見下方說明）。
    """
    ext = os.path.splitext(img_path)[1].lower()

    if ext in RAW_EXTENSIONS:
        # rawpy 採延遲匯入：沒裝它的環境仍然可以正常評分 JPG/PNG。
        # 若寫成檔案頂層的 import，缺少 rawpy 會讓整個模組無法載入。
        try:
            import rawpy
        except ImportError as e:
            raise RuntimeError(
                f"讀取 RAW 檔（{ext}）需要 rawpy 套件，但它並未安裝"
                f"｜請執行：pip install rawpy"
                f"｜或改用 JPG/PNG 格式（不需要 rawpy）"
            ) from e

        with rawpy.imread(img_path) as raw:
            return raw.postprocess(**raw_postprocess_params(half_size))

    with Image.open(img_path) as im:
        # 依 EXIF 方向轉正（2026-09-25 起，SCORING_REVISION 2）。
        #
        # 相機與手機直拿拍照時，JPG 的像素仍是橫的，只在 EXIF 記一個
        # 「顯示時要轉 90°」的 Orientation 標記；PIL 不會自動套用它。
        # 原本模型看到的是躺著的照片，同一張照片的 ARW 卻是正的（rawpy 會轉）。
        # 實測把照片轉 90° 再評分：美感分平均差 4.2、技術分平均差 3.9（最大 10.5，
        # 可以跨過警告門檻），特徵的餘弦相似度只剩 0.54~0.85——
        # 同一個鏡頭的 JPG 與 ARW 會分不到同一組。
        #
        # 訓練資料不受影響：AVA 7,000 張只有 8 張帶旋轉標記，KonIQ 沒有。
        # 只在標記不是 1 時才轉，沒有標記或本來就正的照片不會多複製一次整張影像。
        if im.getexif().get(_EXIF_ORIENTATION, 1) != 1:
            im = ImageOps.exif_transpose(im)
        return np.array(im.convert('RGB'))


def load_image(img_path, half_size=None):
    """
    讀取照片，回傳與 evaluate_photo() 內部完全相同的 RGB 影像
    （形狀 (H, W, 3)、dtype uint8，RAW 依 half_size 解碼，JPG 依 EXIF 轉正）。

    給自己也要解碼照片的呼叫端（例如前台要顯示照片）：顯示與評分共用同一份，
    解碼只做一次，而且保證模型看到的就是畫面上那一張：

        rgb = ai_inference.load_image(path)
        # ……拿 rgb 去顯示……
        result = ai_inference.evaluate_photo(path, image=rgb)

    自己用 PIL 或 rawpy 解碼的話，參數或轉正方式只要有一點不同，
    分數就會和本模組自行解碼時不一樣，而且不會有任何錯誤訊息。

    參數:
        half_size: 只影響 RAW，意義與 evaluate_photo 的同名參數相同。
            傳了 False，之後呼叫 evaluate_photo 也要傳 half_size=False。

    讀不到檔案或格式不支援時直接拋出例外（不像 evaluate_photo 回傳 None）。
    """
    return _load_image_array(img_path, half_size)


def _validate_image_array(image):
    """
    檢查外部傳入的影像陣列格式。格式不對就明確報錯，不要默默算出錯的分數
    ——傳入 BGR、float 或灰階影像都會讓評分結果錯誤但不會拋例外。
    回傳錯誤說明字串；格式正確則回傳 None。
    """
    if not isinstance(image, np.ndarray):
        return f"需要 numpy array，實際收到 {type(image).__name__}"
    if image.ndim != 3 or image.shape[2] != 3:
        return f"需要形狀 (H, W, 3) 的 RGB 影像，實際收到 {image.shape}"
    if image.dtype != np.uint8:
        return f"需要 dtype uint8，實際收到 {image.dtype}"
    return None


# ==========================================
# 4. 技術問題細項分析（傳統影像分析，非深度學習）
# ==========================================
# 技術分數模型是用單一整體分數（KonIQ MOS）訓練出來的，資料本身沒有
# 「模糊/曝光/雜訊/對比」這種細項標籤，深度模型無法直接拆解原因。
# 因此這裡另外用傳統影像分析針對這四個面向做量化判斷，作為補充資訊。
#
# 這些分析現在「無條件執行」，不再由技術分決定要不要跑，
# 而且結果只作為客觀量測資訊，不參與狀態判定（理由見 evaluate_photo 內的說明）。
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_NOISE_SCALE = float(np.sqrt(np.pi / 2) / 6)


def _estimate_noise_sigma(gray_full):
    """
    Immerkaer 快速雜訊估計，但修正了原本兩個讓判斷完全失效的問題。

    參數:
        gray_full (np.ndarray): **原始解析度**的灰階影像。

    回傳:
        float: 雜訊估計值；有效像素不足時回傳 nan（代表無法判斷）。

    修正一 —— 必須在原始解析度上計算
      原本是先把影像縮到 800 寬再估計。高 ISO 產生的是「細粒」雜訊，
      降取樣等於做了一次低通濾波，雜訊直接被平滑掉；反而是細節豐富的
      清晰照片保留較多高頻成分，被誤判成雜訊。
      實測（8 張標註樣本，2 張為高 ISO）：
        800 寬 ：高 ISO 兩張的雜訊排名為第 4、第 6  ← 判斷方向相反
        1600 寬：第 3、第 4                          ← 仍失敗
        原始解析度：第 1、第 2                        ← 正確

    修正二 —— 必須排除已截斷的死黑／死白區域
      訊號被削平的地方本來就量不到雜訊。實測 DSC06685（標註高 ISO）
      有 50.5% 的像素是死黑，只有 14.3% 的像素落在有效亮度區間，
      整張平均會被大量的零稀釋。排除截斷區後，該張從 1.42 上升到 2.13，
      與非高 ISO 樣本的最高值（1.80）才拉開差距。
    """
    h, w = gray_full.shape
    tile = min(NOISE_TILE_SIZE, h, w)
    ys = np.linspace(0, h - tile, NOISE_TILE_GRID).astype(int)
    xs = np.linspace(0, w - tile, NOISE_TILE_GRID).astype(int)

    total = 0.0
    count = 0
    for y in ys:
        for x in xs:
            sub = gray_full[y:y + tile, x:x + tile]
            conv = np.abs(cv2.filter2D(sub.astype(np.float32), -1, _NOISE_KERNEL))
            valid = (sub >= NOISE_VALID_MIN) & (sub <= NOISE_VALID_MAX)
            # filter2D 在邊界會補值，邊緣一圈的結果不可信，一律排除
            valid[:1, :] = valid[-1:, :] = valid[:, :1] = valid[:, -1:] = False
            total += float(conv[valid].sum())
            count += int(valid.sum())

    if count < NOISE_MIN_VALID_PIXELS:
        return float('nan')
    return _NOISE_SCALE * total / count


def _noise_thresholds(raw_half_size):
    """回傳 (門檻, 分級門檻)。半尺寸解碼的 RAW 用另一組，理由見 NOISE_SIGMA_THRESHOLD_HALF。"""
    if raw_half_size:
        return NOISE_SIGMA_THRESHOLD_HALF, NOISE_SIGMA_HIGH_HALF
    return NOISE_SIGMA_THRESHOLD, NOISE_SIGMA_HIGH


def _noise_issue(noise_sigma, raw_half_size=False):
    """
    依雜訊估計值回傳附註文字；未超過門檻、無法估計、或功能停用時回傳 None。

    raw_half_size=True 代表影像是半尺寸解碼的 RAW，改用半尺寸校準的門檻。

    分成兩級的理由見 NOISE_SIGMA_HIGH 的說明：
    門檻附近的照片多半只是夜景暗部的輕微顆粒，寫成「偏高」會過度警示。
    """
    threshold, high = _noise_thresholds(raw_half_size)
    if threshold is None or np.isnan(noise_sigma):
        return None
    if noise_sigma <= threshold:
        return None
    if noise_sigma > high:
        return f"雜訊偏高（噪點指標 {noise_sigma:.2f}），可能是高 ISO 或弱光環境拍攝"
    return f"雜訊輕微（噪點指標 {noise_sigma:.2f}），多半來自暗部或弱光"


def _analyze_technical_issues(image, raw_half_size=False):
    """
    參數:
        image (np.ndarray): 已解碼的 RGB 影像，形狀 (H, W, 3)、dtype uint8。
        raw_half_size (bool): 影像是不是半尺寸解碼的 RAW。只影響雜訊門檻——
            同一張照片在半尺寸上量到的雜訊值高 1.4~2.5 倍，用全尺寸的門檻會大量誤報。
            其餘四項都先縮到 800 寬才計算，兩種尺寸結果相近。

    改為接收「已解碼的影像」而非檔案路徑，原因有二：
      1. 原本用 cv2.imread(img_path) 會把同一張圖再解碼一次
         （模型推論已經解過一次了）。RAW 檔全尺寸解碼實測要 700~1100 ms，
         重複解碼的代價比整個模型推論（22.7 ms）還高一個數量級。
      2. cv2.imread 會自動套用 EXIF 旋轉，PIL 的 Image.open 則不會。
         原本模型看的是未旋轉的影像、傳統分析看的卻是旋轉後的影像，
         兩者對不上。統一由呼叫端解碼一次再往下傳，這個不一致自然消失。
         （2026-09-25 起 _load_image_array 會依 EXIF 轉正，兩者看的都是正的照片。）
    """
    issues = []

    # 內部運算沿用 OpenCV 的 BGR 慣例
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 雜訊估計必須在原始解析度上做（原因見 _estimate_noise_sigma 的說明），
    # 所以先留一份未縮放的灰階影像。其餘四項仍在縮放後的影像上計算。
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 統一縮放到固定寬度上限，避免解析度不同造成門檻值失準。
    #
    # 只縮小、不放大：原本寫成 `if w != target_w` 會把比 800 窄的影像「放大」到 800，
    # 而插值放大本身就會讓畫面變糊，清晰度指標因此崩潰、被誤判為對焦不準。
    # 實測 AVA 資料集（寬度多為 187~640）120 張中有 38 張因此被誤判，
    # 例如 100397.jpg 原生清晰度 1477.8（相當銳利），放大到 800 後只剩 72.2（0.05 倍）。
    h, w = img_bgr.shape[:2]
    target_w = 800
    if w > target_w:
        scale = target_w / w
        img_bgr = cv2.resize(img_bgr, (target_w, int(h * scale)))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. 模糊/失焦/手震：Laplacian 變異數越低代表邊緣銳利度越差
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_var < BLUR_VAR_THRESHOLD:
        issues.append(f"疑似對焦不準或手震導致畫面模糊（清晰度指標 {blur_var:.1f}，建議 >= {BLUR_VAR_THRESHOLD:.0f}）")

    # 2. 曝光：統計死白／死黑像素比例
    total_pixels = gray.size
    overexposed_ratio = float(np.sum(gray >= 250)) / total_pixels
    underexposed_ratio = float(np.sum(gray <= 5)) / total_pixels
    if overexposed_ratio > OVEREXPOSED_RATIO_THRESHOLD:
        issues.append(f"畫面過曝，死白區域佔比 {overexposed_ratio * 100:.1f}%")
    if underexposed_ratio > UNDEREXPOSED_RATIO_THRESHOLD:
        issues.append(f"曝光不足，死黑區域佔比 {underexposed_ratio * 100:.1f}%")

    # 3. 雜訊估計。門檻設為 None 時整項跳過（估計本身要約 20 ms，停用就不必算）。
    if _noise_thresholds(raw_half_size)[0] is not None:
        issue = _noise_issue(_estimate_noise_sigma(gray_full), raw_half_size)
        if issue is not None:
            issues.append(issue)

    # 4. 對比度：亮度標準差過低代表畫面偏灰、層次不足
    contrast_std = float(np.std(gray))
    if contrast_std < CONTRAST_STD_THRESHOLD:
        issues.append(f"對比度不足，畫面偏灰、層次感弱（對比指標 {contrast_std:.1f}）")

    return issues


# ==========================================
# 4. 提供給前端 GUI 呼叫的 API 接口
# ==========================================
# 最近一次 evaluate_photo 失敗的原因字串（成功時為 None）。
# evaluate_photo 的回傳格式維持不變（成功回 dict、失敗回 None），
# 前端若想把「為什麼失敗」顯示給使用者，可選擇性讀取這個變數，
# 不必再把所有失敗都籠統顯示成「讀取照片格式失敗」。
LAST_ERROR = None


def _clamp_score(score):
    """
    把模型輸出的分數限制在 0~100。

    現役的兩個模型各自需不需要這一步：
      美感（分佈模式）：期望值必定在 1~10，換算後天然落在 0~100，裁切不會生效。
      技術（single 模式）：輸出層只有 Linear、沒有接 sigmoid，理論上可能超出 0~100，
                         這一步是它的保險。

    這個函式最早是為舊的二元美感模型（nima_best.pth）加的：它用 0/1 標籤配 MSE 訓練，
    輸出乘以 100 後會出現負分或超過 100 分（實測 40 張樣本範圍 -23.6 ~ 145.1）。
    換成分佈模型後這個問題已經根治；保留裁切是為了技術模型，以及能退回舊權重做對照。
    """
    return max(SCORE_MIN, min(SCORE_MAX, score))


def _resolve_weights(aesthetic_weight, technical_weight):
    """
    決定這一次評分要用的綜合分權重，回傳 (美感權重, 技術權重)。

    三種傳法:
        兩個都不傳    用模組預設值（AESTHETIC_WEIGHT / TECHNICAL_WEIGHT）
        只傳其中一個  另一個自動補成 1 - x。滑桿本來就只會給一個值，
                      這是前台最常走的路徑，也避免呼叫端自己算 1-x 時算錯。
        兩個都傳      必須加總為 1，否則視為呼叫端弄錯而直接失敗。

    為什麼設計成「每次呼叫傳入」而不是「改模組層的全域變數」:
        下一階段前台要改寫成多執行緒的批次分析（期末報告 3.2）。
        若滑桿變動時是去改 AESTHETIC_WEIGHT 這個全域變數，
        正在背景執行緒跑到一半的照片會突然換成新權重，
        同一批結果會混著兩種權重、而且完全不會報錯，事後也無從分辨。
        用參數就沒有這個問題——每張照片用哪組權重在呼叫的當下就固定了。

    為什麼不允許加總不為 1 的權重（例如 0.6 / 0.6）:
        綜合分是加權「平均」，不是加權「總和」。權重加起來不是 1 的話，
        算出來的分數就不再落在 0~100 的尺度上（0.6/0.6 會算出最高 120），
        與美感分、技術分放在同一個清單裡比較會產生誤導。
        要調整佔比請維持總和為 1，例如 0.5/0.5 或 0.8/0.2。

    參數不合法時拋出 ValueError，由 evaluate_photo() 轉成統一的失敗出口。
    """
    def check(value, name):
        # NaN 會讓所有比較都回 False，因此這個範圍檢查同時擋掉了 NaN。
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} 必須是數字，收到 {type(value).__name__}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必須介於 0 與 1 之間，收到 {value}")
        return float(value)

    if aesthetic_weight is None and technical_weight is None:
        return AESTHETIC_WEIGHT, TECHNICAL_WEIGHT

    if technical_weight is None:
        w_aes = check(aesthetic_weight, 'aesthetic_weight')
        return w_aes, 1.0 - w_aes

    if aesthetic_weight is None:
        w_tech = check(technical_weight, 'technical_weight')
        return 1.0 - w_tech, w_tech

    w_aes = check(aesthetic_weight, 'aesthetic_weight')
    w_tech = check(technical_weight, 'technical_weight')
    # 浮點數相加幾乎不可能剛好等於 1（例如 0.7 + 0.3 = 0.9999999999999999），
    # 所以用容差比較而不是 ==。
    if abs(w_aes + w_tech - 1.0) > 1e-6:
        raise ValueError(
            f"兩個權重加總必須為 1，收到 {w_aes} + {w_tech} = {w_aes + w_tech}"
            f"｜只想調整佔比的話，傳其中一個就好，另一個會自動補成 1 - x"
        )
    return w_aes, w_tech


def combine_scores(aesthetic_score, technical_score,
                   aesthetic_weight=None, technical_weight=None):
    """
    由「已知的兩個分數」算出綜合分。不載入影像、不呼叫模型，純算術。

    這是給前台權重滑桿用的。滑桿拉動時**不應該**重新呼叫 evaluate_photo()：
    美感分與技術分不會因為權重而改變，重跑一次每張要 500 ms 以上
    （實測解碼佔 83.5%），1,000 張照片每拉一次滑桿就要凍結數分鐘。
    正確做法是從資料庫已存的兩個分數直接重算綜合分、重新排序、重標最佳照片。

    evaluate_photo() 內部也是呼叫本函式來算綜合分，因此兩條路徑
    保證使用同一條公式——否則遲早出現「清單上的分數與重新分析後的分數對不起來」。

    參數:
        aesthetic_score (float): 美感分，0~100。
        technical_score (float): 技術分，0~100。
        aesthetic_weight / technical_weight: 見 _resolve_weights()。

    回傳:
        float: 綜合分，0~100。

    與 evaluate_photo() 的錯誤處理不同——參數不合法時本函式直接拋 ValueError，
    而不是回傳 None。因為這裡唯一的失敗原因就是呼叫端傳錯東西（程式 bug），
    不像評分會因為檔案損毀之類的資料問題而失敗，沒有理由安靜地吞掉。
    """
    w_aes, w_tech = _resolve_weights(aesthetic_weight, technical_weight)

    for value, name in ((aesthetic_score, 'aesthetic_score'),
                        (technical_score, 'technical_score')):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} 必須是數字，收到 {type(value).__name__}")
        # 範圍檢查同時擋掉 NaN（NaN 的所有比較都是 False）。
        # 這裡也擋得掉一個實際會發生的錯誤：把模型原始的 1~10 分
        # 當成 0~100 分傳進來——那會安靜地算出一個小到離譜的綜合分。
        if not SCORE_MIN <= value <= SCORE_MAX:
            raise ValueError(
                f"{name} 必須介於 {SCORE_MIN:.0f} 與 {SCORE_MAX:.0f} 之間，收到 {value}"
                f"｜注意本函式吃的是換算後的 0~100 分，不是模型原始的 1~10 分"
            )

    return aesthetic_score * w_aes + technical_score * w_tech


def _is_half_size_raw(img_path, half_size):
    """
    這張照片是不是「以半尺寸解碼的 RAW」。決定雜訊用哪一組門檻。

    img_path 在傳入 image= 時可能是 None 或其他非路徑的值，一律視為非 RAW，
    也就是沿用全尺寸門檻——與這個參數加入之前的行為相同。
    """
    try:
        ext = os.path.splitext(os.fspath(img_path))[1].lower()
    except TypeError:
        return False
    if ext not in RAW_EXTENSIONS:
        return False
    return RAW_HALF_SIZE if half_size is None else bool(half_size)


def _fail(message, exc=None):
    """
    統一的失敗出口：記錄原因、印到 console，回傳 None。

    傳入 exc 時會一併印出完整 traceback。原本的寫法只印 str(e)，
    像 CUDA kernel 不相容這種錯誤會變成一句看不出根因的「未知錯誤」，
    追查時完全沒有線索。
    """
    global LAST_ERROR
    LAST_ERROR = message
    print(f"[FAIL] [AI 模組] {message}")
    if exc is not None:
        traceback.print_exc()
    return None


def evaluate_photo(img_path, image=None, aesthetic_weight=None, technical_weight=None,
                   return_features=False, half_size=None):
    """
    對指定路徑的照片進行美感與技術品質評估，並計算綜合分數。

    參數:
        img_path (str): 照片的磁碟絕對或相對路徑。
        image (np.ndarray, optional): 已解碼的影像，形狀 (H, W, 3)、dtype uint8。
            呼叫端若為了顯示等目的已經解碼過同一張照片，傳進來即可省下一次解碼。
            實測解碼成本：RAW 全尺寸約 700~1100 ms、15 MB JPG 約 220 ms，
            都遠高於模型推論本身的 22.7 ms。
            不傳（預設 None）時本函式會自行依副檔名讀檔，行為與過去相同。

            [注意] 通道順序必須是 RGB，不是 OpenCV 慣用的 BGR。
              形狀與 dtype 會被檢查，但 RGB 與 BGR 的形狀、dtype 完全相同，
              程式無法分辨——傳錯只會安靜地算出錯誤分數，不會報錯。
              用 cv2.imread() 取得的影像必須先經
              cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 轉換再傳入。

              最保險的做法是用 load_image(img_path) 取得影像：它就是本函式內部的
              解碼流程（RGB、RAW 參數一致、JPG 依 EXIF 轉正）。自己用 PIL 讀的話
              要記得轉正，否則直幅 JPG 會以躺著的樣子被評分。

        aesthetic_weight (float, optional): 綜合分中美感分的佔比，介於 0 與 1。
        technical_weight (float, optional): 綜合分中技術分的佔比，介於 0 與 1。
            兩者都不傳時使用模組預設的 0.6 / 0.4。
            只傳其中一個時，另一個自動補成 1 - x——前台滑桿只會產生一個值，
            走這條路徑即可，例如 evaluate_photo(p, aesthetic_weight=0.8)。
            兩個都傳時必須加總為 1，否則本函式回傳 None 並記錄原因。

            [注意] 權重只影響 overall_score。aesthetic_score 與 technical_score 是
              模型的原始輸出，不受權重影響；status（優秀／正常／警告）也不受影響
              ——狀態只由技術分決定、「優秀」只由美感分決定。
              也就是說調整滑桿會改變照片的「排序」與「最佳照片」是哪一張，
              但不會讓一張原本被標記為警告的照片變成正常。

        return_features (bool, optional): True 時結果多一個 'feature_vector'，
            是美感模型分類頭之前的 1280 維特徵（list[float]），給相似照片／連拍分組用
            （見 batch_pipeline.py、photo_grouping.py）。與評分是同一次前向運算，不多花推論時間。
            預設 False：回傳格式與過去完全相同。刻意不預設開啟——前台與資料庫用不到它，
            而 1280 個數字會讓 score.py --json 之類「整包輸出結果」的地方每張多出上千行。

            [注意] 特徵只在同一組權重之間可比較。換了美感權重檔（例如重新訓練）之後，
              舊的特徵與新的特徵不能混在一起算相似度，必須整批重跑。

        half_size (bool, optional): RAW 的解碼尺寸。None 用模組預設（目前為半尺寸，
            見 RAW_HALF_SIZE），False 強制全尺寸。批次掃描用預設就好；
            要與舊資料一致、或要最準的雜訊判斷時傳 False。JPG/PNG 不受影響。

            [注意] 傳了 image= 時，影像是呼叫端自己解碼的，這個參數改為「告訴本函式
              那張 RAW 是用哪種尺寸解碼的」——雜訊門檻依此選擇（半尺寸另有一組）。
              請用 raw_postprocess_params(x) 解碼，並在這裡傳同一個 x：
                  rgb = raw.postprocess(**raw_postprocess_params(half_size=False))
                  evaluate_photo(path, image=rgb, half_size=False)
              兩邊都不傳就是兩邊都用模組預設，自然一致。
              判斷是不是 RAW 看的是 img_path 的副檔名，所以傳 image= 時仍要給真實路徑。

    回傳:
        dict: 包含美感分數、技術分數、綜合分數、系統建議與技術問題細項的字典，
              另含本次實際採用的兩個權重（aesthetic_weight / technical_weight）。
              權重一併回傳的原因：綜合分寫進資料庫後，若不知道它是用哪組權重算的，
              這個數字就無法重現，也無法與其他權重下算出的分數放在一起比較。
              若讀取失敗則回傳 None（失敗原因可從模組層級的 LAST_ERROR 取得）。
    """
    global LAST_ERROR
    LAST_ERROR = None

    # 0. 權重先驗證。刻意放在讀檔與推論「之前」——權重錯是呼叫端的參數問題，
    #    沒有必要先花 700~1100 ms 解一張 RAW 檔才發現這件事。
    try:
        w_aes, w_tech = _resolve_weights(aesthetic_weight, technical_weight)
    except ValueError as e:
        return _fail(f"綜合分權重參數不正確：{e}")

    # 1. 取得影像
    # 讀檔獨立成一段來攔錯，是因為「照片開不開得起來」和「模型算不算得動」
    # 是兩種完全不同的問題，混在同一個 except 裡會分不出來。
    # 而且不能只靠例外型別判斷：PIL 對不支援的檔案不一定丟 UnidentifiedImageError，
    # 例如 .ARW 實測丟的是 ValueError: Invalid dimensions。
    if image is None:
        if not os.path.exists(img_path):
            return _fail(f"找不到指定的照片檔案：{img_path}")
        try:
            image = _load_image_array(img_path, half_size)
        except Exception as e:
            return _fail(
                f"無法讀取影像：{img_path}"
                f"｜常見原因：格式不支援、檔案損毀、缺少 RAW 解碼套件、或沒有讀取權限"
                f"｜{type(e).__name__}: {e}",
                e,
            )
    else:
        problem = _validate_image_array(image)
        if problem is not None:
            return _fail(f"傳入的 image 參數格式不正確：{problem}")

    try:
        img_tensor = TRANSFORM(Image.fromarray(image)).unsqueeze(0).to(DEVICE)

        # 2. 雙核心大腦進行不帶梯度的推論
        with torch.no_grad():
            if return_features:
                aes_output, features = MODEL_AES(img_tensor, return_features=True)
                feature_vector = features[0].cpu().tolist()
            else:
                aes_output = MODEL_AES(img_tensor)
            score_aes = _output_to_score(aes_output, AES_MODE)
            score_tech = _output_to_score(MODEL_TECH(img_tensor), TECH_MODE)

        # 3. 分數裁切到 0~100
        # 必須在算綜合分「之前」裁切，否則超出範圍的分數會把綜合分一起拉出範圍
        # （舊二元美感模型時期實測有一張 美感=145.09 導致 綜合=116.67）。
        # 後面的狀態判定（TECH_ISSUE_THRESHOLD、AESTHETIC_EXCELLENT_THRESHOLD）
        # 也一併吃裁切後的值，確保整份結果的每個數字都在同一個一致的區間內。
        score_aes = _clamp_score(score_aes)
        score_tech = _clamp_score(score_tech)

        # 4. 綜合分數。刻意呼叫對外公開的 combine_scores()，而不是在這裡再寫一次
        # 加權公式——前台滑桿走的是同一個函式，共用實作才不會兩邊算出不同的分數。
        # 兩個權重加總必為 1（由 _resolve_weights 保證），而兩個分數都已裁切到
        # 0~100，因此綜合分不需要再裁切一次就必定落在 0~100。
        overall_score = combine_scores(score_aes, score_tech, w_aes, w_tech)

        # 5. 傳統影像分析：無條件執行，不再由技術分決定要不要跑
        # 技術分（KonIQ MOS）與細項分析量的是不同東西：
        #   技術分  —— 學習自 KonIQ-10k 人工評分的「整體感知品質」（訓練集 8,058 張），
        #              擅長模糊、雜訊、壓縮失真這類「失真」問題。
        #   細項分析 —— 曝光、對比這類技術分本來就不擅長判斷的面向
        #              （KonIQ 的標註裡沒有曝光這個維度）。
        # 舊版把後者掛在前者之下，等於用一個訊號去決定要不要看另一個訊號，
        # 結果是技術分高的照片即使明顯欠曝也完全不會被檢查。
        technical_issues = _analyze_technical_issues(
            image, raw_half_size=_is_half_size_raw(img_path, half_size))

        # 6. 根據兩個獨立訊號給予綜合評價
        # 狀態只由「技術分」決定，影像量測結果僅作為補充說明，不影響狀態。
        #
        # 為什麼量測結果不參與狀態判定：
        #   技術分是驗證過的訊號——在 KonIQ 驗證集 2,015 張（未參與訓練）上
        #   PLCC 0.8315、SRCC 0.7953，與人類評分高度一致。
        #   而傳統量測的門檻沒有經過同等驗證，更關鍵的是：
        #   量測值本身正確，不代表它就是「缺陷」。
        #   例如刻意以黑色為背景的照片，死黑比例本來就高（實測 AVA 樣本中
        #   有照片達 72.5%），那是創作選擇而非曝光失誤；若讓它把一張
        #   美感分很高的照片降級成「警告」，只會製造誤導。
        #   因此讓已驗證的訊號決定狀態，未驗證的量測值只提供客觀數據。
        low_tech = score_tech < TECH_ISSUE_THRESHOLD

        if low_tech:
            status = "警告"
            suggestion = f"整體技術品質偏低（技術分 {score_tech:.1f}，低於門檻 {TECH_ISSUE_THRESHOLD}）"
            if technical_issues:
                suggestion += "。可能成因：" + "；".join(technical_issues) + "。"
            else:
                suggestion += "，但未找出單一明顯成因，可能是壓縮失真或整體畫質不足。"
        else:
            if score_aes > AESTHETIC_EXCELLENT_THRESHOLD:
                status = "優秀"
                suggestion = "構圖優秀、光影掌握佳，整體技術品質良好。"
            else:
                status = "正常"
                suggestion = "照片品質良好。"
            if technical_issues:
                suggestion += (
                    "（影像量測附註：" + "；".join(technical_issues)
                    + "。以上為客觀量測值，不一定代表缺陷——例如以黑色為背景的照片死黑比例本來就高）"
                )

        # 7. 包裝成前端最好拿取、最好讀的字典格式
        result = {
            "aesthetic_score": round(score_aes, 2),
            "technical_score": round(score_tech, 2),
            "overall_score": round(overall_score, 2),
            "status": status,
            "suggestion": suggestion,
            "technical_issues": technical_issues,
            # 本次實際採用的權重。前台可直接寫進資料庫，
            # 日後才分辨得出一筆綜合分是在哪組權重下算出來的。
            "aesthetic_weight": w_aes,
            "technical_weight": w_tech,
        }
        if return_features:
            result["feature_vector"] = feature_vector
        return result

    # 以下把「推論階段」的失敗再依根因分類。原本一律歸為「未知錯誤」，
    # 會讓 GPU 驅動問題被誤讀成照片格式問題，害人查錯方向。
    except RuntimeError as e:
        return _fail(
            f"推論失敗（硬體或驅動層錯誤，與這張照片的內容無關）：{type(e).__name__}: {e}"
            f"｜若訊息含 'no kernel image is available'，代表 PyTorch 版本與顯示卡架構不相容",
            e,
        )
    except Exception as e:
        return _fail(f"處理照片時發生未預期的錯誤：{type(e).__name__}: {e}", e)

# ==========================================
# 測試專用區塊 (只有單獨執行此檔案時才會跑)
# ==========================================
if __name__ == '__main__':
    print("\n--- 正在進行 AI 模組獨立效能測試 ---")
    test_target = "data/my_photos/DSC05860-2.jpg"  # 換成你硬碟裡確定有的照片

    res = evaluate_photo(test_target)
    if res:
        print(f"測試成功！回傳結果：\n{res}")
