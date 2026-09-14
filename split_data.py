"""
把美感標籤的完整檔切成 train / val 兩份（80 / 20）。

    python split_data.py
    python split_data.py --source data/train_full.csv --force

讀取源與輸出目標分離（2026-09-14 修正）
----------------------------------------
這支腳本原本讀 data/train.csv、又寫回 data/train.csv —— 來源與目的地是同一個檔。
後果是每跑一次就再砍掉 20%，而且 data/val.csv 被新的切分蓋掉：

    現況        train=3920  val=980
    再跑一次    train=3136  val=784   <- 原本的 980 筆驗證集永久消失
    再跑一次    train=2508  val=628

全程沒有任何錯誤訊息，只會安靜地讓資料集越來越小。當時只加了一道
「偵測到 val.csv 已存在就中止」的保護，擋得住誤觸，但沒有解決根本的設計問題。

現在改成與 split_koniq.py 相同的結構：

    來源  data/train_full.csv     只讀，永遠不寫
    輸出  data/train.csv          只寫
          data/val.csv            只寫

來源與輸出是不同的檔案，重跑完全安全（同一個來源、同一個亂數種子，
切出來的結果每次都一樣）。腳本並且會主動檢查來源與輸出是否指向同一個檔，
是的話直接中止。

[注意] 重新切分不會重現現有的 train.csv / val.csv
---------------------------------------------------
data/train_full.csv 是把現有的 train.csv（3,920 筆）與 val.csv（980 筆）
合併、依 index 排序重建出來的 4,900 筆。上述的就地覆寫已經把原始檔案的
列順序吃掉了，而 df.sample() 的結果取決於列順序，所以重切的分配會與現有的不同。

現有的 train.csv / val.csv 才是與 nima_best.pth（舊二元美感模型）對應的那一份。
把它們重切，等於讓舊模型的驗證集混進它訓練過的照片，跨模型比較就失去意義。
**除非你真的要重新開始一輪實驗，否則不需要跑這支腳本。**

（現行的美感模型用的是 data/ava_train.csv / ava_val.csv，
  由 build_ava_labels.py 產生，與這支腳本無關。）
"""
import argparse
import os
import sys

import pandas as pd

SOURCE = 'data/train_full.csv'
TRAIN_OUT = 'data/train.csv'
VAL_OUT = 'data/val.csv'
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


def parse_args():
    p = argparse.ArgumentParser(
        description='把美感標籤切成 train / val（80 / 20）')
    p.add_argument('--source', default=SOURCE,
                   help=f'完整標籤檔，只讀不寫（預設 {SOURCE}）')
    p.add_argument('--train-out', default=TRAIN_OUT,
                   help=f'訓練集輸出路徑（預設 {TRAIN_OUT}）')
    p.add_argument('--val-out', default=VAL_OUT,
                   help=f'驗證集輸出路徑（預設 {VAL_OUT}）')
    p.add_argument('--force', action='store_true',
                   help='允許覆寫已存在的輸出檔')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.source):
        print(f'[FAIL] 找不到來源檔：{args.source}')
        print( '       這支腳本只讀來源、不寫回來源，因此必須有一個獨立的完整標籤檔。')
        print(f'       若你手上只有切好的 {TRAIN_OUT} 與 {VAL_OUT}，可以先把它們合併回來：')
        print( '           python -c "import pandas as pd; '
               f'pd.concat([pd.read_csv(\'{TRAIN_OUT}\'), pd.read_csv(\'{VAL_OUT}\')])'
               f'.sort_values(\'index\').to_csv(\'{SOURCE}\', index=False)"')
        return 1

    # 核心保護：來源與輸出不可以是同一個檔。
    # 這正是原本那個會安靜吃掉 20% 資料的 bug，用 realpath 比對才擋得住
    # 「同一個檔但路徑寫法不同」（相對／絕對、大小寫、符號連結）。
    source_real = os.path.realpath(args.source)
    for label, out_path in (('訓練集', args.train_out), ('驗證集', args.val_out)):
        if os.path.realpath(out_path) == source_real:
            print(f'[FAIL] {label}輸出路徑與來源檔是同一個檔案：{args.source}')
            print( '       就地覆寫會讓資料每跑一次少掉 20%，且完全不會報錯。')
            print( '       請用 --train-out / --val-out 指定不同的輸出路徑。')
            return 1

    existing = [p for p in (args.train_out, args.val_out) if os.path.exists(p)]
    if existing and not args.force:
        print(f'[FAIL] 以下輸出檔已存在：{", ".join(existing)}')
        print( '       現役模型是用現有的切分訓練的，重新切分會讓驗證集')
        print( '       混進模型訓練過的照片，新舊模型的比較就失去意義。')
        print(f'       確定要重新切分請先備份，再加上 --force：')
        print(f'           python {os.path.basename(__file__)} --force')
        return 1

    print(f'[INFO] 來源 {args.source}（只讀）')
    df = pd.read_csv(args.source)
    print(f'       共 {len(df)} 筆')

    # 隨機打亂後依比例切分。random_state 固定，同一個來源每次結果都相同。
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    train_df.to_csv(args.train_out, index=False)
    val_df.to_csv(args.val_out, index=False)

    print(f'[ OK ] 切分完成')
    print(f'       {args.train_out:<22} {len(train_df):>5} 筆')
    print(f'       {args.val_out:<22} {len(val_df):>5} 筆')
    print(f'       來源 {args.source} 未被修改')
    return 0


if __name__ == '__main__':
    sys.exit(main())
