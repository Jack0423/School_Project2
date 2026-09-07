"""
把 data/train.csv 切成 train / val 兩份（80 / 20）。

⚠️ 這支腳本會「就地覆寫」來源檔案 —— 讀進來的和寫出去的都是 data/train.csv。
   因此重複執行會每次再砍掉 20%，而且 data/val.csv 會被新的切分蓋掉：

       現況        train=3920  val=980
       再跑一次    train=3136  val=784   ← 原本的 980 筆驗證集永久消失
       再跑一次    train=2508  val=628
       再跑一次    train=2006  val=502

   這種資料流失不會有任何錯誤訊息，只會安靜地讓資料集越來越小。
   所以下面加了一道保護：偵測到 data/val.csv 已存在（代表切分做過了）
   就直接中止。確定要重切請加 --force，並自行先備份。

（對照：split_koniq.py 的來源是 koniq10k_distributions_sets.csv、
  目的地是 train_tech.csv / val_tech.csv，來源與目的地不同，重跑是安全的。）
"""
import os
import sys

import pandas as pd

CSV_PATH = 'data/train.csv'
VAL_PATH = 'data/val.csv'

# ── 保護：避免重複執行造成資料流失 ──────────────────────────
if os.path.exists(VAL_PATH) and '--force' not in sys.argv:
    existing_train = len(pd.read_csv(CSV_PATH))
    existing_val = len(pd.read_csv(VAL_PATH))
    print(f"⛔ 偵測到 {VAL_PATH} 已存在（train={existing_train} 筆、val={existing_val} 筆），")
    print(f"   代表切分已經做過了。")
    print()
    print(f"   本腳本會就地覆寫 {CSV_PATH}，再跑一次會變成：")
    print(f"       train={int(existing_train * 0.8)} 筆、val={existing_train - int(existing_train * 0.8)} 筆")
    print(f"   目前的 {existing_val} 筆驗證集會被覆蓋且無法復原。")
    print()
    print(f"   確定要重新切分請先備份 data/train.csv 與 data/val.csv，再執行：")
    print(f"       python {os.path.basename(__file__)} --force")
    sys.exit(1)

if '--force' in sys.argv:
    print("⚠️ --force 已指定，將覆寫現有的 train.csv 與 val.csv。")

# 1. 讀取原本的訓練標籤
print(f"讀取 {CSV_PATH} 中...")
df = pd.read_csv(CSV_PATH)

# 2. 隨機打亂資料 (確保資料分佈平均)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. 計算 80% 的分界線
split_idx = int(len(df) * 0.8)

# 4. 切割成 train 與 val
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

# 5. 覆寫原檔並產生新檔 (不保留 index)
train_df.to_csv(CSV_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)

print(f"✅ 切分大功告成！")
print(f"📦 訓練集 (train.csv): {len(train_df)} 張照片")
print(f"📦 驗證集 (val.csv): {len(val_df)} 張照片")
