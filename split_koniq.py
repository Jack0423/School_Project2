import pandas as pd
import os

# 1. 修正後的讀取路徑 (指向 data/koniq/ 資料夾)
csv_path = 'data/koniq/koniq10k_distributions_sets.csv' 

if not os.path.exists(csv_path):
    print(f"❌ 找不到檔案：{csv_path}")
    print("請確認你已經把 CSV 檔搬到 data/koniq/ 資料夾底下了！")
else:
    # 2. 讀取與處理
    df = pd.read_csv(csv_path)
    
    # 將 0~100 分縮小到 0~1
    df_simple = pd.DataFrame({
        'image': df['image_name'],
        'label': df['MOS'] / 100.0  
    })

    # 3. 隨機打亂並切分 (80% 訓練, 20% 驗證)
    df_simple = df_simple.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_simple) * 0.8)

    train_df = df_simple.iloc[:split_idx]
    val_df = df_simple.iloc[split_idx:]

    # 4. 儲存到 data 資料夾下，供 train_tech.py 使用
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train_tech.csv', index=False)
    val_df.to_csv('data/val_tech.csv', index=False)

    print(f"✅ 技術資料集切分完成！")
    print(f"📂 已產生：data/train_tech.csv 與 data/val_tech.csv")