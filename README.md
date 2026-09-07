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

於 2026-09-05 在對應驗證集上量測：

| 模型 | 指標 | 數值 | 驗證集 |
|---|---|---|---|
| 技術（`nima_tech_best.pth`） | PLCC / SRCC | 0.8315 / 0.7953 | KonIQ 驗證集 2,015 張 |
| 美感（`nima_aes_binary.pth`，舊二元版） | AUC | 0.9791 | AVA 驗證集 |

隨時可用 `python eval_models.py` 重現。美感模型現已改用評分分佈版
（`nima_aes_dist.pth`），二元與分佈兩種模式的指標不可直接比較，
跨模式比較請用 `compare_aesthetic_models.py`（統一以 AVA 群眾平均分為對照答案）。

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
├── train.csv / val.csv              美感（二元標籤）
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

三個實用細節：

- **已解碼的影像可直接傳入**：`evaluate_photo(p, image=arr)` 省下一次解碼
  （RAW 全尺寸解碼約 700–1100 ms，遠高於推論本身的 22.7 ms）。
  ⚠ 通道順序必須是 **RGB**，傳入 BGR 不會報錯，只會安靜地算出錯誤分數。
- **權重可調**：`evaluate_photo(p, aesthetic_weight=0.8)`，另一個自動補成 0.2。
- **權重只影響 `overall_score`**：`status` 只由技術分決定、「優秀」只由美感分決定，
  調整權重會改變排序，但不會把「警告」變成「正常」。

### 桌面前台

```bash
python arw_viewer_gui.py
```

PyQt6 影像檢視器，載入照片並顯示評分結果。AI 模組載入失敗時會關閉評分功能
但程式仍可正常瀏覽照片。目前測試檔路徑寫死在 `__main__` 內。

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
`distribution` 用 EMD loss（推土機距離，對 CDF 差值取 mean，與 NIMA 論文一致）。

訓練速度實測（美感模型 5,600 張、batch 32、30 epoch）：CPU 約 104 分鐘，
GPU 約 9 分鐘（快 11.7 倍）。GPU 上有 75% 時間花在資料載入（JPEG 解碼與增強都在 CPU）。

### 評估與比較

```bash
python eval_models.py --save metrics.json
```

```bash
python compare_aesthetic_models.py --val-csv data/ava_val.csv
```

### 資料準備腳本

```bash
python build_ava_labels.py    # 從 AVA 原始評分分佈重建完整標籤（含 1–10 級分佈）
python split_koniq.py         # 切分 KonIQ → train_tech.csv / val_tech.csv
python split_data.py          # 切分 data/train.csv（⚠ 就地覆寫，見下）
```

`split_data.py` 的來源與目的地是同一個檔案，重複執行會每次再砍掉 20% 且無任何錯誤訊息
（3920 → 3136 → 2508 …）。已加保護：偵測到 `data/val.csv` 存在就中止，需 `--force` 才會重切。

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
└── engine.py           run_epoch

ai_inference.py         推論核心：雙模型評分 + OpenCV 技術量測 + 綜合分
arw_viewer_gui.py       PyQt6 桌面前台
check_env.py            環境自檢

train_nima.py           美感模型訓練（single / distribution 雙模式）
train_tech.py           技術模型訓練
eval_models.py          輸出客觀指標（PLCC/SRCC/AUC）
compare_aesthetic_models.py  跨模式公平比較美感模型

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
細節見 [tests/README.md](tests/README.md)。
