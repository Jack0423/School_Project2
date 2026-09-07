import os
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

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

# rawpy 解碼參數，刻意與前台 arw_viewer_gui.py 顯示用的設定保持一致，
# 確保「前台傳入已解碼影像」與「本模組自行解碼」得到完全相同的結果。
_RAW_POSTPROCESS = dict(
    use_camera_wb=True,
    half_size=False,
    no_auto_bright=True,
    user_flip=None,
)

# 傳統影像分析門檻值（可依實測樣本再微調）
BLUR_VAR_THRESHOLD = 100.0       # Laplacian 變異數，低於此值視為模糊/失焦/手震
OVEREXPOSED_RATIO_THRESHOLD = 0.05   # 死白像素比例
UNDEREXPOSED_RATIO_THRESHOLD = 0.05  # 死黑像素比例
# 雜訊判斷門檻。設為 None 代表「計算但不對外回報」——目前刻意停用，原因見下。
#
# 為什麼停用（而不是隨便填一個數字）：
#   估計方法本身已修好，17 張標註樣本上的排序明顯改善（詳見 _estimate_noise_sigma），
#   但「該用哪個數值當門檻」目前無法負責任地決定：
#
#   正樣本（確定有雜訊）      DSC07029 2.77 / DSC06685 2.16 / IMG_0989 9.71
#   負樣本最高（確定沒雜訊）  保時捷.jpg 4.20
#
#   保時捷那張是細節極多的車輛照片，而且經過 Lightroom 降噪處理，
#   指標卻比兩張真正的高 ISO 還高——一張降過噪的照片讀數最高，
#   等於直接證明這個估計法量到的不是雜訊，而是細節與銳化產生的高頻。
#   正負樣本重疊，任何單一門檻都必然同時誤報與漏報。
#
#   曾嘗試「只在平坦區域估計」並掃過 32 組參數，確實找得到能分開的組合，
#   但只有 3 個正樣本，且結果對參數極度敏感（grid=8 分得開、grid=12 就不行），
#   屬於過擬合，不予採用。
#
#   對照片篩選工具而言誤報的代價高於漏報：把好照片標成有雜訊會直接損害可信度。
#   因此在取得足夠標註樣本（建議各 20 張以上正負樣本）之前，維持停用。
NOISE_SIGMA_THRESHOLD = None

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
def _load_image_array(img_path):
    """
    讀取影像檔，一律回傳 RGB numpy array，形狀 (H, W, 3)、dtype uint8。

    依副檔名分流：RAW 交給 rawpy，其餘交給 PIL。
    不用「先試 PIL、失敗再試 rawpy」的寫法，因為 .DNG 會讓 PIL
    「成功」讀到內嵌縮圖而不報錯，這種 fallback 永遠不會被觸發。
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
            return raw.postprocess(**_RAW_POSTPROCESS)

    return np.array(Image.open(img_path).convert('RGB'))


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


def _analyze_technical_issues(image):
    """
    參數:
        image (np.ndarray): 已解碼的 RGB 影像，形狀 (H, W, 3)、dtype uint8。

    改為接收「已解碼的影像」而非檔案路徑，原因有二：
      1. 原本用 cv2.imread(img_path) 會把同一張圖再解碼一次
         （模型推論已經解過一次了）。RAW 檔全尺寸解碼實測要 700~1100 ms，
         重複解碼的代價比整個模型推論（22.7 ms）還高一個數量級。
      2. cv2.imread 會自動套用 EXIF 旋轉，PIL 的 Image.open 則不會。
         原本模型看的是未旋轉的影像、傳統分析看的卻是旋轉後的影像，
         兩者對不上。統一由呼叫端解碼一次再往下傳，這個不一致自然消失。
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
        issues.append(f"疑似對焦不準或手震導致畫面模糊（清晰度指標 {blur_var:.1f}，建議 ≥ {BLUR_VAR_THRESHOLD:.0f}）")

    # 2. 曝光：統計死白／死黑像素比例
    total_pixels = gray.size
    overexposed_ratio = float(np.sum(gray >= 250)) / total_pixels
    underexposed_ratio = float(np.sum(gray <= 5)) / total_pixels
    if overexposed_ratio > OVEREXPOSED_RATIO_THRESHOLD:
        issues.append(f"畫面過曝，死白區域佔比 {overexposed_ratio * 100:.1f}%")
    if underexposed_ratio > UNDEREXPOSED_RATIO_THRESHOLD:
        issues.append(f"曝光不足，死黑區域佔比 {underexposed_ratio * 100:.1f}%")

    # 3. 雜訊估計 —— 目前停用（NOISE_SIGMA_THRESHOLD is None），原因見該常數的說明。
    # 門檻設回數值即可重新啟用，估計方法本身已修正並驗證過排序正確性。
    if NOISE_SIGMA_THRESHOLD is not None:
        noise_sigma = _estimate_noise_sigma(gray_full)
        if not np.isnan(noise_sigma) and noise_sigma > NOISE_SIGMA_THRESHOLD:
            issues.append(f"雜訊偏高（噪點指標 {noise_sigma:.2f}），可能是高 ISO 或弱光環境拍攝")

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

    為什麼需要這一步：
      美感模型是拿「二元標籤（0/1）」用 MSE 回歸訓練出來的，
      而且輸出層只有 Linear、沒有接 sigmoid，
      所以預測值本來就可能落在 [0, 1] 之外，乘以 100 後就會出現
      負分或超過 100 分（實測 40 張樣本範圍為 -23.6 ~ 145.1）。

    這只是止血，不是根治：
      裁切之後分數看起來合理了，但美感分本質上仍是
      「二元分類機率 x 100」，與技術分（KonIQ MOS）不是同一種尺度，
      兩者用 0.6:0.4 加權相加在統計意義上並不嚴謹。
      根治需要重新訓練美感模型（輸出層加 sigmoid，或改用
      AVA 原始的 10 級分佈標籤搭配 emd_loss）。
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


def evaluate_photo(img_path, image=None, aesthetic_weight=None, technical_weight=None):
    """
    對指定路徑的照片進行美感與技術品質評估，並計算綜合分數。

    參數:
        img_path (str): 照片的磁碟絕對或相對路徑。
        image (np.ndarray, optional): 已解碼的影像，形狀 (H, W, 3)、dtype uint8。
            呼叫端若為了顯示等目的已經解碼過同一張照片，傳進來即可省下一次解碼。
            實測解碼成本：RAW 全尺寸約 700~1100 ms、15 MB JPG 約 220 ms，
            都遠高於模型推論本身的 22.7 ms。
            不傳（預設 None）時本函式會自行依副檔名讀檔，行為與過去相同。

            ⚠ 通道順序必須是 RGB，不是 OpenCV 慣用的 BGR。
              形狀與 dtype 會被檢查，但 RGB 與 BGR 的形狀、dtype 完全相同，
              程式無法分辨——傳錯只會安靜地算出錯誤分數，不會報錯。
              用 cv2.imread() 取得的影像必須先經
              cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 轉換再傳入。
              PIL 的 np.array(Image.open(p).convert('RGB')) 與
              rawpy 的 raw.postprocess() 都已經是 RGB，可直接使用。

        aesthetic_weight (float, optional): 綜合分中美感分的佔比，介於 0 與 1。
        technical_weight (float, optional): 綜合分中技術分的佔比，介於 0 與 1。
            兩者都不傳時使用模組預設的 0.6 / 0.4。
            只傳其中一個時，另一個自動補成 1 - x——前台滑桿只會產生一個值，
            走這條路徑即可，例如 evaluate_photo(p, aesthetic_weight=0.8)。
            兩個都傳時必須加總為 1，否則本函式回傳 None 並記錄原因。

            ⚠ 權重只影響 overall_score。aesthetic_score 與 technical_score 是
              模型的原始輸出，不受權重影響；status（優秀／正常／警告）也不受影響
              ——狀態只由技術分決定、「優秀」只由美感分決定。
              也就是說調整滑桿會改變照片的「排序」與「最佳照片」是哪一張，
              但不會讓一張原本被標記為警告的照片變成正常。

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
            image = _load_image_array(img_path)
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
            score_aes = _output_to_score(MODEL_AES(img_tensor), AES_MODE)
            score_tech = _output_to_score(MODEL_TECH(img_tensor), TECH_MODE)

        # 3. 分數裁切到 0~100
        # 必須在算綜合分「之前」裁切，否則超出範圍的美感分會把綜合分一起拉出範圍
        # （實測有一張 美感=145.09 導致 綜合=116.67）。
        # 後面的狀態判定門檻（score_tech < 60、score_aes > 85）也一併吃裁切後的值，
        # 確保整份結果的每個數字都在同一個一致的區間內。
        score_aes = _clamp_score(score_aes)
        score_tech = _clamp_score(score_tech)

        # 4. 綜合分數。刻意呼叫對外公開的 combine_scores()，而不是在這裡再寫一次
        # 加權公式——前台滑桿走的是同一個函式，共用實作才不會兩邊算出不同的分數。
        # 兩個權重加總必為 1（由 _resolve_weights 保證），而兩個分數都已裁切到
        # 0~100，因此綜合分不需要再裁切一次就必定落在 0~100。
        overall_score = combine_scores(score_aes, score_tech, w_aes, w_tech)

        # 5. 傳統影像分析：無條件執行，不再由技術分決定要不要跑
        # 技術分（KonIQ MOS）與細項分析量的是不同東西：
        #   技術分  —— 學習自 10,073 張人工評分的「整體感知品質」，
        #              擅長模糊、雜訊、壓縮失真這類「失真」問題。
        #   細項分析 —— 曝光、對比這類技術分本來就不擅長判斷的面向
        #              （KonIQ 的標註裡沒有曝光這個維度）。
        # 舊版把後者掛在前者之下，等於用一個訊號去決定要不要看另一個訊號，
        # 結果是技術分高的照片即使明顯欠曝也完全不會被檢查。
        technical_issues = _analyze_technical_issues(image)

        # 6. 根據兩個獨立訊號給予綜合評價
        # 狀態只由「技術分」決定，影像量測結果僅作為補充說明，不影響狀態。
        #
        # 為什麼量測結果不參與狀態判定：
        #   技術分是驗證過的訊號——在 2,015 張未參與訓練的 KonIQ 照片上
        #   PLCC 0.8056、SRCC 0.7636，與人類評分高度一致。
        #   而傳統量測的門檻沒有經過同等驗證，更關鍵的是：
        #   量測值本身正確，不代表它就是「缺陷」。
        #   例如刻意以黑色為背景的照片，死黑比例本來就高（實測 AVA 樣本中
        #   有照片達 72.5%），那是創作選擇而非曝光失誤；若讓它把一張
        #   美感 100 分的照片降級成「警告」，只會製造誤導。
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
