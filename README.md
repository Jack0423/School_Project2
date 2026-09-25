# 雙核心照片品質評估系統

以兩個獨立的 NIMA（Neural Image Assessment）模型分別評估照片的**美感**與**技術品質**，
再加權合成單一綜合分數，並輸出可讀的問題診斷。

| 核心 | 評什麼 | 訓練資料 | 輸出 |
|---|---|---|---|
| 美感模型 | 構圖、色彩、主體等主觀吸引力 | AVA（群眾 1–10 分評分分佈） | 10 級分佈取期望值 → 換算 0–100 |
| 技術模型 | 銳利度、曝光、雜訊等客觀品質 | KonIQ-10k（MOS 分數） | 連續值 → 換算 0–100 |

兩者皆以 MobileNetV2 為骨幹（`common/model.py`），加上 OpenCV 的傳統影像量測
（Laplacian 清晰度、過曝／死黑像素比、對比標準差）作為**附註性**的問題細項。

---

## 實測指標

於 2026-09-19 在對應驗證集上量測（完整紀錄見 [docs/verification_log.md](docs/verification_log.md)）：

| 模型 | PLCC | SRCC | 驗證集 |
|---|---|---|---|
| 美感（`nima_aes_dist.pth`，評分分佈版） | 0.8872 | 0.7925 | AVA 驗證集 1,400 張，對照答案為群眾平均分 |
| 技術（`nima_tech_best.pth`） | 0.8314 | 0.7951 | KonIQ 驗證集 2,015 張 |

隨時可用 `python eval_models.py` 重現。新舊美感模型（二元標籤 vs 評分分佈）的比較
請用 `compare_aesthetic_models.py`，它統一以 AVA 群眾平均分為對照答案。

兩點限制：挑選最佳 epoch 與量測上面的數字用的是同一份驗證集，沒有另外保留測試集；
KonIQ 的驗證集是自己隨機切的 20%，**不是**官方的 test 切分（剛好也是 2,015 張），
不能直接和論文數字比較。

---

## 安裝

需要 Python 3.10+（開發環境為 3.10.11 / Windows 11）。

```bash
pip install -r requirements.txt
```

**CUDA 版本不可隨意更動。** `requirements.txt` 鎖定 `torch==2.11.0+cu128`，
一般 PyPI 索引沒有這些版本，必須指定 PyTorch 官方索引：

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

原因：開發用的 RTX 5070 Ti 是 sm_120 架構，較舊的 cu121 只編譯到 sm_90，
`torch.cuda.is_available()` 會回 `True` 但實際運算拋
`CUDA error: no kernel image is available`。詳見 `requirements.txt` 內的完整說明。

裝完先自檢：

```bash
python check_env.py
```

它會逐項檢查每個依賴（每項獨立 try，缺套件也能印出完整報告），
`[必要]` 項目缺失會以離開代碼 1 結束。

---

## 需要另外準備的檔案

以下兩類**不在版本庫內**（見 `.gitignore`），需自行取得：

### 1. 模型權重（4 個 `.pth`，約 36 MB）

放在專案根目錄。`ai_inference.py` 實際載入的是：

- `nima_aes_dist.pth` — 美感（評分分佈版，目前採用）
- `nima_tech_best.pth` — 技術

另有兩個歷史權重供對照：`nima_aes_binary.pth`（舊二元美感）、`nima_best.pth`。

**權重缺席時系統會明確失敗，絕不產生任何分數**——寧可關閉評分功能，
也不要顯示隨機權重算出的假數字（`tests/test_startup_guards.py` 守住這條）。

### 2. 資料集（僅訓練與評估需要，約 2.1 GB）

標註 CSV 已包含在版本庫內（`data/*.csv`，僅相對檔名，無個資），影像檔需自備：

```
data/
├── dataset/                 AVA 影像，約 7,000 張
├── koniq/
│   ├── 512x384/             KonIQ-10k 影像，約 10,373 張
│   └── koniq10k_distributions_sets.csv
├── ava/
│   └── ground_truth_dataset.csv   AVA 原始評分分佈
├── ava_full.csv                     AVA 完整標籤（build_ava_labels.py 產生，含 1–10 級分佈）
├── ava_train.csv / ava_val.csv      美感，現役模型 nima_aes_dist.pth 用的切分
├── train_full.csv                   舊二元標籤的完整檔（split_data.py 的來源）
├── train.csv / val.csv              美感（二元標籤，舊模型 nima_best.pth 用）
├── train_tech.csv / val_tech.csv    技術（KonIQ MOS）
└── my_photos/               自己的測試照片（含 RAW）
```

---

## 使用

### 評估單張照片（程式介面）

```python
from ai_inference import evaluate_photo

result = evaluate_photo("path/to/photo.jpg")
# {
#   'aesthetic_score': 58.3, 'technical_score': 71.2, 'overall_score': 63.5,
#   'status': '正常', 'suggestion': '照片品質良好。',
#   'technical_issues': [...],
#   'aesthetic_weight': 0.6, 'technical_weight': 0.4
# }
```

支援 JPG / PNG，以及 `.ARW` / `.DNG` 等 RAW 檔（需 `rawpy`，依副檔名分流，
不會讓 PIL 誤讀內嵌縮圖）。失敗時回傳 `None`，原因可從模組層級的
`LAST_ERROR` 取得，且會依根因分類（硬體／驅動錯誤不會被誤報成照片格式問題）。

幾個實用細節：

- **已解碼的影像可直接傳入**：`evaluate_photo(p, image=arr)` 省下一次解碼
  （RAW 全尺寸解碼約 700–1100 ms，遠高於推論本身：CPU 約 21 ms、GPU 約 12 ms）。
  ⚠ 通道順序必須是 **RGB**，傳入 BGR 不會報錯，只會安靜地算出錯誤分數。
  **前台要顯示照片時請用 `ai_inference.load_image(p)` 解碼**，再把同一份傳進 `image=`：
  它就是本模組內部的解碼流程（RGB、RAW 參數一致、JPG 依 EXIF 轉正），顯示與評分保證是同一張。
- **直幅 JPG 會自動轉正**：相機與手機直拿拍照時，JPG 的像素仍是橫的，只在 EXIF 記方向。
  2026-09-25 前模型看到的是躺著的照片（同一張的 ARW 卻是正的），轉 90° 實測技術分最多差 10.5 分。
  現在依 EXIF 轉正；沒有方向標記的照片分數完全不變。
- **權重可調**：`evaluate_photo(p, aesthetic_weight=0.8)`，另一個自動補成 0.2。
- **權重只影響 `overall_score`**：`status` 只由技術分決定、「優秀」只由美感分決定，
  調整權重會改變排序，但不會把「警告」變成「正常」。
- **RAW 預設半尺寸解碼**：整條批次管線實測快 4.3 倍（325 張 15 GB 的 ARW：2 分鐘 → 26～30 秒）。
  模型只吃 224×224、細項分析只用 800px 寬，半尺寸的 3024×2012 綽綽有餘。
  代價是綜合分平均差 0.34～0.56（最大 2.1），分數壓在門檻上的照片可能換狀態
  （實測 60 張中 3 張）。要與舊資料一致時：`evaluate_photo(p, half_size=False)`。
  **雜訊附註在半尺寸下比較不靈敏**：兩種尺寸各有一組校準過的門檻（全尺寸 1.43、半尺寸 3.02，
  都是零誤報），但半尺寸只抓得到明顯的高 ISO 雜訊（77 張標註樣本漏報 21 張，全尺寸只漏 5 張）。
  要完整的雜訊判斷請用 `half_size=False`。
  **自己解碼 RAW 的呼叫端**（前台顯示、XMP 模組）請用 `ai_inference.raw_postprocess_params()`
  取得參數，不要自己抄一份，否則傳進來的影像與本模組自行解碼的不同，分數會安靜地不一樣。
  解碼時若傳了 `half_size=False`，呼叫 `evaluate_photo(p, image=rgb, half_size=False)` 也要傳同一個值，
  雜訊才會用對門檻。
- **版本標籤**：`ai_inference.model_version()` 回傳版本字串，例如
  `nima_aes_dist.pth@71dbd9e1+nima_tech_best.pth@e39a98f4|raw=half|rev=2`，
  三段分別記「權重內容（檔名＋SHA-256 前 8 碼）」「RAW 解碼尺寸」「評分流程版本」。
  資料庫每筆分數存一份，之後換模型**或改了評分算法**，只要重跑版本對不上的那些，不必整批清空重來。
  評分時傳了 `half_size=False`，這裡也要傳 `model_version(half_size=False)`。
  最後的 `rev` 在權重沒換、但程式改變了評分結果時遞增（`SCORING_REVISION`，目前為 2：
  JPG 轉正與半尺寸雜訊門檻）。
- **需要特徵向量時**：`evaluate_photo(p, return_features=True)` 會多回傳 `feature_vector`
  （美感模型分類頭之前的 1280 維特徵，給相似照片分組用）。預設不回傳，回傳格式與上面相同。

### 批次分析與重複照片挑選（宋宇宸）

先批次評分，再選一種方式挑出「同一組裡要留哪張」：

```bash
python run_batch.py <照片資料夾> --weight 0.8 --output batch_01.jsonl
```

`run_batch.py` 以 4 條執行緒解碼、單一執行緒推論（模型不可多執行緒呼叫），
每張結果即時寫一行 JSONL（分數、權重、1280 維特徵或失敗原因）。整批固定一組權重。
接著三種挑選模式，依「誰決定哪些照片算同一組」而不同：

| 模式 | 由誰分組 | 需要 EXIF | 指令 |
|---|---|---|---|
| 相似照片檢索 | 程式，看畫面內容 | 否 | `photo_grouping.py` |
| 連拍候選 | 程式，看內容＋拍攝時間＋相機 | 是 | `run_bursts.py` |
| 指定一組 | 使用者自己 | 否 | `pick_best.py` |

```bash
python photo_grouping.py batch_01.jsonl --output photo_groups_01.json
python run_bursts.py batch_01.jsonl photo_metadata.json --output burst_groups_01.json
python pick_best.py batch_01.jsonl --output best_pick.json
```

- **`photo_grouping.py`**：特徵正規化後以餘弦相似度分組（預設門檻 0.85，須與組內每一張都達門檻），
  同組綜合分最高者為建議保留。
- **`run_bursts.py`**：在上述基礎上，要求兩張照片出自同一台相機且拍攝時間相差 5 秒內。
  需要先用 [ExifTool](https://exiftool.org/) 匯出：

  ```bash
  exiftool -json -SubSecDateTimeOriginal -DateTimeOriginal -Make -Model -SerialNumber -InternalSerialNumber <資料夾> > photo_metadata.json
  ```

  `--camera-match` 決定相機識別的嚴格程度：`serial` 只認機身序號；`model`（預設）
  序號讀不到時退到廠牌＋型號；`ignore` 不比對相機，只看拍攝時間。
  以序號認定與以型號認定的照片不會被配成一組，執行時會印出各層級的張數。

  為什麼預設不是最嚴格的 `serial`：實測 Sony ILCE-7M4 的 JPG 完全沒有序號欄位、
  ARW 只有 `InternalSerialNumber`，只認 `SerialNumber` 會讓整個資料夾得到 0 組且毫無錯誤訊息。

  ⚠ 上面的 `>` 請在 cmd 或 Git Bash 執行。Windows PowerShell 5.1 的 `>` 會存成 UTF-16，
  `run_bursts.py` 以 UTF-8 讀取會直接解析失敗。
- **`pick_best.py`**：把輸入的整批照片當成同一組連拍，只做排名與挑選，不自行分組。
  使用者已經知道哪些是一組時（例如在相機或 Lightroom 裡挑好、複製到同一個資料夾）用這個，
  不依賴 EXIF 也不受相似度門檻影響；相似度只在偏低時印出提醒，不會排除照片。

四者的輸出都含照片的絕對路徑，已列入 `.gitignore`。輸出檔已存在時會拒絕覆寫。

門檻 0.85 與 5 秒是 2026-09-23 用 325 張 ARW 的人工標註校準出來的：原本的 0.9 / 2 秒，抽查 15 張未分組照片有 11 張其實有同伴被漏掉；放寬後救回 9 張，且重新標註「有變動的 35 組」沒有出現錯誤合併（連同第一輪的 105 組，共 140 組人工判斷、0 組誤分）。詳見 `photo_grouping.py` 與 `burst_metadata.py` 的常數說明。

### 桌面前台

```bash
python arw_viewer_gui.py
```

PyQt6 影像檢視器，載入照片並顯示評分結果。AI 模組載入失敗時會關閉評分功能
但程式仍可正常瀏覽照片。目前測試檔路徑寫死在 `__main__` 內，
且只對 `.arw` 做 RAW 分流（`.dng` 會被 PIL 讀成內嵌縮圖）——這支檔案由前台負責人維護。

### 訓練

```bash
python train_nima.py --mode distribution --train-csv data/ava_train.csv --val-csv data/ava_val.csv --save nima_aes_dist.pth
```

```bash
python train_tech.py --img-dir data/koniq/512x384 --save nima_tech_best.pth
```

兩者共用參數：`--epochs 30`、`--batch-size 32`、`--lr-features 1e-5`、
`--lr-head 1e-4`（骨幹與分類頭分開設定學習率）、`--patience 5`（Early Stopping）。
輸出權重若已存在會中止，需明確加 `--force`。

`train_nima.py` 的 `--mode` 決定損失函數：`single` 用 MSELoss，
`distribution` 用 EMD loss（推土機距離，對 CDF 差值的平方取 mean）。
與 NIMA 論文的定義差在沒有開根號，數值不能和論文直接比較；報表裡的 EMD 用的是論文定義
（`common/metrics.py` 的 `emd_per_sample`）。

訓練速度（美感模型 5,600 張、batch 32、RTX 5070 Ti / Ryzen 7 5800X）：2026-09-22 在電腦閒置時
以 `benchmark_gpu.py` 量測，`num_workers=0`（目前 Windows 上的設定）每批 206 ms，一個 epoch 約 36 秒；
GPU 本身只需約 33 ms，**約 84% 的時間在等 CPU 解碼與做資料增強**。
同一台機器上 `num_workers=8` 可降到每批 40 ms（一個 epoch 約 7 秒，約快 5 倍），訓練腳本尚未採用。

> 舊版本這裡寫的「CPU 104 分鐘、GPU 9 分鐘、快 11.7 倍、每批 102 ms」是 2026-09-05 臨時量的，
> 量測程式沒有保留，用現行程式碼也重現不出來，請不要再引用。
> 資料載入是 CPU 工作，受機器當下負載影響很大：同一天電腦同時有其他工作時，
> `num_workers=0` 量到 215–306 ms。要引用數字時請關掉其他程式重跑 `benchmark_gpu.py`，
> 並附上它輸出的硬體與版本。

### 評估與比較

```bash
python eval_models.py --save metrics.json
```

```bash
python compare_aesthetic_models.py --val-csv data/ava_val.csv
```

### 模型成效報表

```bash
python model_report.py            # 完整報表，約 1–2 分鐘
python model_report.py --quick    # 每個驗證集只取前 300 張，確認流程用
```

把兩個模型在驗證集上的表現整理成 `summary.md`（表格）、`metrics.json`、逐張預測 CSV
與 13 張圖表，輸出到 `reports/model_report/<時間>/`。內容包含 PLCC / SRCC / RMSE、
預測 vs 真實散佈圖、各分數區間的偏差、「警告」與「優秀」判定的混淆矩陣與門檻掃描、
美感模型的 EMD 與 ROC、新舊美感權重比較，以及影像量測細項
（被標記模糊、過曝等的照片，人工評分是否真的比較低）。

分數換算與門檻直接取自 `ai_inference.py`，不另寫一份，執行時並會抽樣與 `evaluate_photo()` 比對。
混淆矩陣「真實答案」的切點（MOS 60、AVA 群眾平均 6）是自己定的，可用參數調整。

### 效能量測

```bash
python benchmark_gpu.py           # 約 3–5 分鐘
python benchmark_gpu.py --quick   # 約 1 分鐘，數字只用來確認流程
```

照課程投影片 4-3 的測速三守則（warm-up、計時前後 synchronize、重複 20 次取中位數與 IQR）量測：
矩陣乘法 CPU vs GPU、兩個模型的推論延遲與批次吞吐、單張照片全流程的時間拆解、
訓練單步的 batch size × FP32／混合精度、DataLoader 的 `num_workers`。
只量時間，不改任何檔案；輸出到 `reports/benchmark/<時間>/`。

### 資料準備腳本

```bash
python build_ava_labels.py    # 從 AVA 原始評分分佈重建完整標籤（含 1–10 級分佈）
python split_koniq.py         # 切分 KonIQ → train_tech.csv / val_tech.csv
python split_data.py          # 切分 data/train_full.csv → train.csv / val.csv
```

`split_data.py` 原本讀 `data/train.csv` 又寫回同一個檔，重複執行會每次再砍掉 20%
且無任何錯誤訊息（3920 → 3136 → 2508 …）。2026-09-14 改成與 `split_koniq.py` 相同的結構：
來源 `data/train_full.csv` 只讀、輸出 `train.csv` / `val.csv` 只寫，腳本並會主動比對
來源與輸出是否指向同一個檔。重跑現在完全安全（同一來源、同一亂數種子，結果每次相同）。

`data/train_full.csv`（4,900 筆）是把現有的 train/val 合併重建出來的——原始檔已被舊版
就地覆寫吃掉列順序，因此**重切不會重現現有的分配**。現有的 `train.csv` / `val.csv` 才是與
`nima_best.pth` 對應的那一份，輸出已存在時需 `--force` 才會覆蓋。

### 資料庫維護

```bash
python reset_db_analysis.py <db路徑> --apply
```

美感模型由二元換成分佈後輸出尺度改變（舊：0–100 且大量卡在極值；新：實際約 21–74），
「優秀」門檻也由 85 改為 62。資料庫中兩套尺度的分數混在同一份清單裡排序，
會讓舊資料的 100 分永遠霸佔「★最佳照片」。這支腳本清空分析欄位讓系統重跑，
不動檔名與 metadata。**預設為 dry-run，須加 `--apply` 才實際寫入。**

---

## 專案結構

```
common/                 模型架構、前處理、訓練迴圈的唯一定義
├── model.py            NIMABaseline（MobileNetV2）+ emd_loss
├── transforms.py       build_transform（訓練/驗證/推論共用）
├── engine.py           run_epoch
├── metrics.py          混淆矩陣、ROC、相關係數、EMD 等指標（報表用）
└── plotting.py         報表圖表的共用樣式（不從 common 匯出，避免推論依賴 matplotlib）

ai_inference.py         推論核心：雙模型評分 + OpenCV 技術量測 + 綜合分
arw_viewer_gui.py       PyQt6 桌面前台
check_env.py            環境自檢

train_nima.py           美感模型訓練（single / distribution 雙模式）
train_tech.py           技術模型訓練
eval_models.py          輸出客觀指標（PLCC/SRCC/AUC）
compare_aesthetic_models.py  跨模式公平比較美感模型
model_report.py         模型成效報表（混淆矩陣、門檻掃描、細項分析、圖表）
benchmark_gpu.py        GPU / CPU 效能量測（測速三守則）

batch_pipeline.py       批次分析核心：平行解碼 + 單一推論消費者（宋宇宸）
run_batch.py            批次分析命令列入口
photo_grouping.py       以特徵餘弦相似度分組、挑建議保留
burst_metadata.py       連拍判定：EXIF 拍攝時間與相機識別（序號／型號／不比對）
run_bursts.py           連拍候選命令列入口
pick_best.py            使用者自己指定一組連拍，只做排名與挑選

build_ava_labels.py     重建 AVA 標籤
split_data.py           切分美感資料
split_koniq.py          切分 KonIQ 資料
reset_db_analysis.py    清除舊尺度的資料庫分析結果

tests/                  unittest 測試（見 tests/README.md）
```

### 為什麼有 `common/`

模型架構、前處理、訓練迴圈原本在多個檔案裡各有一份且**內容不一致**。
前處理的分岔尤其有實質影響：

| 前處理 | PLCC | SRCC |
|---|---|---|
| `Resize((224,224))`（推論用，壓扁長寬比） | 0.8056 | 0.7636 |
| `Resize(256)+CenterCrop(224)`（訓練驗證用） | 0.8315 | 0.7953 |

同一組權重、只換前處理，SRCC 差 0.0317。統一到 `common/` 後才消除這個落差。

---

## 測試

```bash
python -m unittest discover -s tests -t .
```

用標準庫 `unittest`，不需額外套件。每一項測試都對應一個真實修復過的缺陷，
不是為了湊覆蓋率。資料集與權重缺席時會標記 skip 並說明缺什麼，而非直接失敗。

寫完後做過變異測試：把已修好的 11 個 bug 逐一植回，**11/11 全部被攔截**。
目前共 176 個測試。
細節見 [tests/README.md](tests/README.md)。
