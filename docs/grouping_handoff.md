# 相似照片分組接進前台（F06）

分成兩半：底層（資料進資料庫、分組函式）已經做好；前台的分組畫面由宋宇宸接。

## 分組要找的是什麼

提案書寫的是「相似度 90% 以上的聚類＋連拍挑選最佳者」：**連拍或幾乎一樣的照片**，
目的是幾張裡只留最好的一張。不是「同一個場景」——同一個地方隔幾分鐘、換了構圖的照片不算同一組。

門檻沿用 **相似度 0.85、連拍 5 秒內**，不要改。這是 2026-09-23 用 325 張 ARW 人工標註校準的
（兩輪共 140 組人工判斷，0 組誤分），細節見 `photo_grouping.DEFAULT_THRESHOLD`、`burst_metadata.DEFAULT_MAX_SECONDS` 的說明。

## 已經做好的（王凱立，2026-10-05）

- 前台分析時，1280 維特徵存進 `photos.db` 的 `feature` 欄；連拍要的拍攝時間與相機由 ExifTool
  在分析時同時讀，存進 `capture_meta` 欄。舊的 `photos.db` 啟動時自動補這兩欄。
- 「開始批次分析」會補跑沒有特徵（加入這個功能前分析的）、或還沒讀過拍攝時間的照片。
- `main_v2.group_folder(資料夾, mode)` 從資料庫分組，**只讀資料庫、不跑模型、不呼叫 ExifTool**，
  325 張約 0.15 秒，可以直接在主執行緒呼叫，不用另開執行緒。
- 分組本體與命令列的 `photo_grouping.py`、`run_bursts.py` 是同一份程式；
  325 張實測前台與命令列分出的組、每組建議保留的照片都完全相同。

```python
result = group_folder(self.current_folder, mode="similar")   # 或 "burst"（連拍：另外要同一台相機、5 秒內）

result["groups"]            # 2 張以上的組，group_id 從 1 連號，組依組內最高綜合分排、組內也是高到低
# [{"group_id": 1, "count": 3, "best_path": "...",
#   "photos": [{"path", "file_name", "overall_score", "is_best", "similarity_to_best"}, ...]}, ...]
result["singles"]           # 可以分組、但沒有相似照片的
result["not_ready"]         # 還不能分組（沒分析、舊模型版本、沒有特徵）→ 提示按「開始批次分析」
result["no_capture_time"]   # 只有連拍模式：讀不到拍攝時間或相機的照片，這些不會進連拍組
result["exiftool_missing"]  # 只有連拍模式：有照片沒讀過拍攝時間、電腦又沒有 ExifTool → 提示安裝

non_best_paths(result)      # 每組除了建議保留以外的照片，給「選取每組非最佳」用
```

`is_best` 依**目前權重**下的綜合分，所以權重滑桿動過、或刪了照片之後，要重新呼叫 `group_folder`。

## 要做的（前台）

1. 接上「相似照片分組」按鈕（`self.group_btn`，現在是停用的佔位）。可選「相似」或「連拍」兩種模式。
   分析中這顆按鈕會自動停用、結束後恢復（它已經在 `_analysis_controls` 裡）。
2. 分組顯示：清單依組列出「第 N 組（k 張）」，每組建議保留的那張標出來（`is_best`）；
   點組內照片時照現在的方式顯示預覽與分析結果。單張的不列或放在最後。
   一般清單的 ★ 是「整個資料夾最佳」，分組畫面裡的標記是「這組建議保留」，兩者畫面上要分得出來。
3. 「選取每組非最佳」：用 `non_best_paths(result)` 在清單裡選取，再交給現有的刪除功能
   （`delete_selected`，移到資源回收筒、RAW 的 `.xmp` 一起移）。不要做一鍵直接刪除。
4. 提示：`not_ready` 有照片時說明要先批次分析；連拍模式的 `no_capture_time`、`exiftool_missing` 也要說明原因。
5. 驗證：用「連拍／幾乎一樣」的定義重新標註一份，用你的 `tools/validate_grouping.py` 算 precision／recall／F1。
   前台分析的結果檔在 `reports/batches/`（最新的那一份），連拍模式另外需要 ExifTool 匯出的 metadata（見 README）。
   前台與命令列分出的組相同，所以用命令列驗證就等於驗證前台。
6. 測試：至少測「選取每組非最佳」只會選到每組非最佳的照片、刪除之後重新分組的結果正確。
   最後全套測試跑過：`python -m unittest discover -s tests -t .`

不要在 `main_v2.py` 裡另寫一套分組或去讀 JSONL：`reports/batches/` 的 JSONL 是每次分析的紀錄，不是前台的資料來源。
分組畫面的程式盡量放在新的函式或新檔案裡，`main_v2.py` 只加少量接線，和其他人的修改比較好合併。
