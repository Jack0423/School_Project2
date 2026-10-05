# 宋宇宸工作項目交付

## 完成與待補資料

- 已把 `main_v2.py` 的單張、批次分析接到 `qt_batch_worker.py` 的 QThread。
- 工作執行緒使用既有 `batch_pipeline.analyze_batch()`；4 個執行緒解碼，模型只由一個背景執行緒循序呼叫。
- 進度透過 Qt signal 回到主執行緒；SQLite 更新與介面操作留在主執行緒。
- 分析期間固定權重與資料夾，禁止重複分析、改權重、刪除（含快捷鍵）、寫 XMP 和換資料夾。仍可瀏覽照片。結束或失敗後恢復按鈕原狀。
- 執行中關閉視窗會提示等分析完成；沒有強制終止 GPU／RAW 工作的取消功能。（併入後已加上「取消分析」，見文末併入紀錄）
- 每次結果保存在 `reports/batches/batch-<唯一識別碼>.jsonl`，保留分組需要的特徵、成功與失敗紀錄。既有資料庫仍只保存原本欄位。
- `compare_batch.py` 可產出循序／平行的逐張三項分數、差異、耗時與加速比。
- `validate_grouping.py` 可建立人工標註表，再依真實標註計算誤分、漏分、precision、recall、F1、pair accuracy 與完全一致群組比例。
- 分組預設值沿用現有 **0.85 / 5 秒**，未修改分組演算法或既有五支管線程式。

**仍待資料才能完成的驗收：** 真實模型的分數一致性、真實照片耗時、人工分組準確率。此交付沒有把模擬測試數字當成模型實測結果。

來源資料夾沒有 `.git` 與遠端資訊，無法確認正式 repo 的最新版本。`ai_inference.py` 和 `common/model.py` 保留提供資料夾中的原檔，未另行修改；`batch_pipeline.py` 原本已有 `return_features=True`。取得正式 repo 後，應以該 repo 的這兩個核心檔為準。

## 1. 啟動介面

先依根目錄 README 安裝環境，另需 PyQt6。將下列現役模型權重放在專案根目錄：

- `nima_aes_dist.pth`
- `nima_tech_best.pth`

在專案根目錄執行：

```bash
python main_v2.py
```

選照片資料夾後，點「分析目前照片」或「開始批次分析」。批次仍沿用「只分析未分析／舊模型版本照片」規則；要建立完整分組驗證資料，使用下面的 `run_batch.py` 對整個資料夾分析一次。

「相似照片分組」按鈕仍是原有預留介面。此次完成的是背景分析串接；分組與準確率驗證由下列命令執行。

## 2. 循序與平行的分數、耗時比較

```bash
python tools/compare_batch.py "照片資料夾" --output reports/batch_compare_01 --repeats 5
```

輸出目錄必須是新的，以免混入先前結果。會產生：

- `summary.md`：兩種模式耗時中位數、張／秒、加速比、分數是否全部一致。
- `comparison.json`：測試環境、模型版本、權重、固定照片清單、每次耗時，以及逐張美感／技術／綜合分數與差異。
- `warmup-*.jsonl` 與每輪 `*-sequential.jsonl`、`*-parallel.jsonl`：完整原始結果。

循序版是解碼一張、推論一張、寫一張；平行版直接使用既有四執行緒解碼管線。兩者都使用相同解碼函式、相同權重和 `return_features=True`，模型推論都保持循序。

兩種模式各跑一個完整暖機，再交替測試先後順序，預設各量 5 次並取中位數；共 12 次完整資料夾處理。計時包含解碼、推論、特徵與 JSONL 寫入，不包含模型載入。CUDA 計時前後同步。這是暖快取測試，不宣稱冷磁碟效能。

預設三種分數容許絕對誤差 `1e-6`；可以用 `--atol` 調整。只要有失敗照片、缺少照片或分數超出誤差，就不會判定全部一致。命令失敗或比對未通過會回傳非零結束碼。

## 3. 分組準確率驗證

先對固定的驗證資料夾完整分析，輸出名稱不要重複：

```bash
python tools/run_batch.py "照片資料夾" --output batch_validation_01.jsonl
python tools/validate_grouping.py template batch_validation_01.jsonl --output human_labels_01.csv
```

打開 CSV，人工填完每張照片的 `group_id`：同一組填同一個值，獨立照片各用不同值。請依照片內容及專題定義獨立判斷，不要直接抄程式的預測。所有照片都必須標註，空白、重複路徑、標註與批次清單不一致都會被拒絕；含分析失敗照片的批次也必須先處理，避免靜默排除難例。

用 ExifTool 匯出同一份照片的拍攝時間及相機資訊（建議使用照片資料夾的絕對路徑）：

```bash
exiftool -r -json -SubSecDateTimeOriginal -DateTimeOriginal -Make -Model -SerialNumber -InternalSerialNumber "照片資料夾" > photo_metadata_01.json
python tools/validate_grouping.py evaluate batch_validation_01.jsonl human_labels_01.csv --metadata photo_metadata_01.json --output grouping_validation_01.json
```

預設是連拍分組，沿用相似度 0.85、整組跨度 5 秒及 `camera-match=model`。要只測視覺相似分組，明確指定 `--mode similarity`，不需 metadata。

報表包含所有群組（含單張）、設定、缺漏 metadata 清單、metadata 統計，以及：

- pair precision：預測同組的照片配對，有多少確實同組；誤分會降低此值。
- pair recall：人工同組的配對，有多少被找回；漏分會降低此值。
- pair F1：precision 與 recall 的綜合指標。
- pair accuracy：所有照片配對的判斷正確比例；大量不相關照片可能讓它偏高，不能只報這一項。
- exact group precision／recall：整組成員完全一致的群組比例。
- 最多 50 組誤分／漏分的照片配對，方便回看。

缺少拍攝時間或相機資訊的照片仍會留在評估內；沿用現有演算法，不能配對時成為單張，不會被偷偷剔除。沒有分母的指標為 `null`，不假裝是 100%。請把調參資料與最終驗證資料分開，避免同一批資料調完門檻又當成獨立驗證。

## 4. 自動測試

```bash
python -m unittest tests.test_batch_comparison tests.test_grouping_validation tests.test_qt_background tests.test_photo_grouping tests.test_console_encoding.TestSourcesAreCp950Encodable -v
```

2026-10-05 的相關測試結果：37 項，36 通過、1 項因缺少模型權重跳過。Qt 測試實際啟動 QThread、signals、SQLite 與介面計時器；模型以確定性的測試替身代替。驗證背景運作、單一推論執行緒、資料庫主執行緒更新、重複啟動防護、關閉與刪除防護、單張分析、失敗恢復，以及兩支命令的完整報表輸出。

測試紀錄見 `tests_batch_handoff.log`。這些測試驗證程式串接與計算邏輯，不代表真實模型的準確率或速度。

全專案測試另跑 132 項：3 個失敗子項、5 個錯誤、25 個跳過；8 個失敗／錯誤皆由缺少現役權重導致（部分既有測試與 `--help` 在載入時便匯入模型）。完整紀錄見 `tests_full_handoff.log`。不能宣稱全套測試已通過。

## 併入紀錄（2026-10-05，王凱立）

- 在正式 repo（GitHub `Jack0423/School_Project2`）上合併。`main_v2.py` 同期另有修改（分析後自動寫入 XMP、
  狀態列、結果面板），背景分析以手動移植，設計不變；分析完成後接上「分析後自動寫入 XMP」。
- `tests/test_grouping_validation.py`、`tests/test_batch_comparison.py` 讀寫檔案補上 `encoding='utf-8'`：
  Windows 繁中預設 cp950，原本會在讀 JSON 時失敗。
- Windows＋RTX 5070 Ti、現役權重下全部 234 項測試通過；325 張 ARW 背景分析 42.1 秒（原本 84 秒），
  分析中畫面最長停頓 0.24 秒。
- 上方「全專案測試另跑 132 項…不能宣稱全套測試已通過」是缺少權重時的紀錄，現在有權重時全部通過。
- 兩份測試紀錄檔（`tests_batch_handoff.log`、`tests_full_handoff.log`）是缺少權重時的結果，沒有放進 repo；
  其中 `tests_full_handoff.log` 含個人電腦路徑。
- 之後加上「取消分析」（王凱立）：`analyze_batch()` 新增 `should_stop`，每推論完一張檢查一次；
  `BatchAnalysisThread.cancel()` 設定旗標，前台進度條旁的按鈕呼叫它。手上這張推論完就停（不強制中斷 GPU／RAW 解碼），
  已完成的照常寫進資料庫與 `reports/batches/` 的結果檔，回傳值多一個 `cancelled`。
  分析中關閉視窗改成先詢問，選「是」就取消並在停下後自動關閉。測試見 `tests/test_batch_cancel.py` 與 `qt_background_check.py`。
- 分組底層（王凱立）：前台分析時特徵存進 `photos.db`、拍攝時間由 `BatchAnalysisThread` 背景讀（`capture_metadata` signal，
  在主執行緒寫入）；`main_v2.group_folder()` 從資料庫分組。前台的分組畫面接法見 [grouping_handoff.md](grouping_handoff.md)。
