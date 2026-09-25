"""
把 KonIQ-10k 的 MOS 縮到 0~1，再切成 train / val 兩份（80 / 20）。

    python split_koniq.py

輸出 data/train_tech.csv、data/val_tech.csv，給 train_tech.py 使用。
同一個來源、同一個亂數種子（42），每次切出來的結果都完全相同。

[注意] 這不是 KonIQ 的官方切分
-------------------------------
來源 CSV 的 set 欄位本來就有官方切分（training 7,058 / validation 1,000 / test 2,015），
這支腳本沒有用它，而是自己隨機切 80/20。驗證集剛好也是 2,015 張，但組成是
官方 training 1,421 張 + test 389 張 + validation 205 張，所以量出來的指標
不能直接和論文在官方 test 上的數字比較。現役的 nima_tech_best.pth 是用這份切分訓練的，
改用官方切分就要重新訓練。

2026-09-25 改寫成 main() + argparse：原本整支是模組層級的程式碼，
連 `python split_koniq.py --help` 都會直接開始切分、寫檔。輸出內容與改寫前逐位元組相同。
"""
import argparse
import os
import sys

import pandas as pd

SOURCE = 'data/koniq/koniq10k_distributions_sets.csv'
TRAIN_OUT = 'data/train_tech.csv'
VAL_OUT = 'data/val_tech.csv'
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


def parse_args():
    p = argparse.ArgumentParser(
        description='把 KonIQ-10k 切成技術模型的 train / val（80 / 20）')
    p.add_argument('--source', default=SOURCE, help=f'KonIQ 的分數檔（預設 {SOURCE}）')
    p.add_argument('--train-out', default=TRAIN_OUT, help=f'訓練集輸出（預設 {TRAIN_OUT}）')
    p.add_argument('--val-out', default=VAL_OUT, help=f'驗證集輸出（預設 {VAL_OUT}）')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.source):
        print(f"[FAIL] 找不到檔案：{args.source}")
        print("請確認你已經把 CSV 檔搬到 data/koniq/ 資料夾底下了！")
        return 1

    df = pd.read_csv(args.source)

    # 將 0~100 分縮小到 0~1
    df_simple = pd.DataFrame({
        'image': df['image_name'],
        'label': df['MOS'] / 100.0,
    })

    # 隨機打亂並切分（80% 訓練、20% 驗證）
    df_simple = df_simple.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(df_simple) * TRAIN_RATIO)

    train_df = df_simple.iloc[:split_idx]
    val_df = df_simple.iloc[split_idx:]

    for path in (args.train_out, args.val_out):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    train_df.to_csv(args.train_out, index=False)
    val_df.to_csv(args.val_out, index=False)

    print("[ OK ] 技術資料集切分完成！")
    print(f"[ OK ] 已產生：{args.train_out}（{len(train_df)} 筆）與 {args.val_out}（{len(val_df)} 筆）")
    return 0


if __name__ == '__main__':
    sys.exit(main())
