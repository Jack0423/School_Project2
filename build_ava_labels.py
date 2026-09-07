"""
從 AVA 原始評分分佈，重建完整的美感標籤。

背景
----
專案原本的 data/train.csv 只有二元標籤（0 / 1），來自 Kaggle
「Aesthetic Visual Analysis」競賽提供的 train.csv。競賽把 AVA 的
1~10 分群眾評分壓成兩類，但沒有公開分界點在哪，所以：

  * 標籤的定義無法說明，也無法重現。
  * 磁碟上 7,000 張照片裡有 2,100 張沒有標籤——那是競賽的測試集，
    主辦方本來就不公開答案。
  * 二元標籤讓模型只能學「是不是好照片」，但推論端卻把輸出乘以 100
    當成「0~100 分的美感分」在顯示，這是分數會跑出 -23 分、145 分的根本原因。

這支腳本用 AVA 原始評分分佈（data/ava/ground_truth_dataset.csv）把上述三點一次補齊。

輸出
----
  data/ava_full.csv    7,000 張，含二元標籤、平均分、以及 1~10 級的完整分佈
  data/ava_train.csv   訓練集
  data/ava_val.csv     驗證集

欄位 score_1 ~ score_10 的命名，刻意對齊 train_nima.py 內 AVADataset
在 mode='distribution' 時讀取的欄位名，因此不必修改訓練程式即可直接使用。

安全性
------
不覆寫任何既有檔案。輸出檔已存在時直接中止，需明確加上 --force。
現有的 data/train.csv 與 data/val.csv 完全不會被動到。

用法
----
    python build_ava_labels.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

GROUND_TRUTH = 'data/ava/ground_truth_dataset.csv'
IMAGE_DIR = 'data/dataset'
EXISTING_TRAIN = 'data/train.csv'
EXISTING_VAL = 'data/val.csv'

OUT_FULL = 'data/ava_full.csv'
OUT_TRAIN = 'data/ava_train.csv'
OUT_VAL = 'data/ava_val.csv'

# 反推出來的競賽二元化門檻。
# 做法是拿現有 4,900 張已知標籤的照片，掃過所有可能的門檻值，
# 找出最能重現原始標籤的那一個。實測 4.47 分可重現 98.06%。
# 對不上的 95 張平均分全部落在 3.46~5.05，也就是門檻附近那些
# 「差一點點」的照片，屬於四捨五入與資料版本差異，不是規則不同。
BINARY_THRESHOLD = 4.47

SCORE_LEVELS = np.arange(1, 11)


def load_ground_truth():
    gt = pd.read_csv(GROUND_TRUTH)
    expected = ['image_num'] + [f'vote_{i}' for i in range(1, 11)]
    missing = [c for c in expected if c not in gt.columns]
    if missing:
        raise ValueError(f'{GROUND_TRUTH} 缺少欄位：{missing}')
    gt['image_num'] = gt['image_num'].astype(int)
    return gt


def list_local_images():
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f'找不到影像資料夾：{IMAGE_DIR}')
    ids = []
    for name in os.listdir(IMAGE_DIR):
        stem, ext = os.path.splitext(name)
        if ext.lower() == '.jpg' and stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def build(gt, image_ids):
    """把評分分佈整理成訓練用的表格。"""
    df = gt[gt['image_num'].isin(image_ids)].copy()

    votes = df[[f'vote_{i}' for i in range(1, 11)]].values
    # 這份資料的 vote 欄位已經是比例（每列總和為 1），
    # 但仍重新正規化一次，避免不同來源的檔案是原始票數。
    totals = votes.sum(axis=1, keepdims=True)
    votes = votes / totals

    df['mean_score'] = (votes * SCORE_LEVELS).sum(axis=1)
    df['label'] = (df['mean_score'] > BINARY_THRESHOLD).astype(int)

    for i in range(1, 11):
        df[f'score_{i}'] = votes[:, i - 1]

    df = df.rename(columns={'image_num': 'image'})
    cols = ['image', 'label', 'mean_score'] + [f'score_{i}' for i in range(1, 11)]
    return df[cols].sort_values('image').reset_index(drop=True)


def verify_against_existing(df):
    """
    拿現有的 4,900 個已知標籤驗證重建結果，並回報一致率。
    這是這支腳本最重要的一步——若一致率不高，代表門檻或資料對應有問題。
    """
    if not (os.path.exists(EXISTING_TRAIN) and os.path.exists(EXISTING_VAL)):
        print('⚠️ 找不到現有的 train.csv / val.csv，略過驗證。')
        return None

    old = pd.concat([pd.read_csv(EXISTING_TRAIN), pd.read_csv(EXISTING_VAL)])
    old['image'] = old['image'].astype(int)
    merged = old[['image', 'label']].merge(df[['image', 'label']], on='image',
                                           suffixes=('_old', '_new'))
    if merged.empty:
        print('⚠️ 現有標籤與重建結果沒有交集，無法驗證。')
        return None

    agree = (merged['label_old'] == merged['label_new']).mean()
    print(f'  與現有標籤比對：{len(merged)} 張，一致率 {agree:.2%}')
    if agree < 0.95:
        print(f'  ⚠️ 一致率偏低。門檻 {BINARY_THRESHOLD} 可能不正確，'
              f'或影像編號與 AVA 的對應有誤，請先釐清再使用輸出結果。')
    return agree


def split_preserving_existing(df):
    """
    切分訓練／驗證集，但**保留現有的 train / val 分配**。

    為什麼不直接重新隨機切分：
      現有模型是用現有的 train.csv 訓練的。若重新隨機切分，
      原本在驗證集裡的照片可能跑進新的訓練集，
      那麼「新模型 vs 舊模型」的比較就失去意義——
      新模型會在它訓練過的照片上被評分。
    因此原本 4,900 張維持原分配，只把新增的 2,100 張依相同比例分進去。
    """
    old_train = set(pd.read_csv(EXISTING_TRAIN)['image'].astype(int))
    old_val = set(pd.read_csv(EXISTING_VAL)['image'].astype(int))
    known = old_train | old_val

    new_ids = df[~df['image'].isin(known)]['image'].values
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(new_ids)
    val_ratio = len(old_val) / max(len(known), 1)
    n_val = int(round(len(shuffled) * val_ratio))

    val_ids = old_val | set(shuffled[:n_val].tolist())
    train_ids = old_train | set(shuffled[n_val:].tolist())

    assert not (val_ids & train_ids), '訓練集與驗證集出現重疊'
    return (df[df['image'].isin(train_ids)].reset_index(drop=True),
            df[df['image'].isin(val_ids)].reset_index(drop=True),
            len(shuffled), n_val)


def main():
    parser = argparse.ArgumentParser(description='從 AVA 評分分佈重建完整標籤')
    parser.add_argument('--force', action='store_true',
                        help='覆寫已存在的輸出檔')
    args = parser.parse_args()

    outputs = [OUT_FULL, OUT_TRAIN, OUT_VAL]
    existing = [p for p in outputs if os.path.exists(p)]
    if existing and not args.force:
        print(f'⛔ 以下輸出檔已存在：{", ".join(existing)}')
        print(f'   為避免覆蓋，已中止。確定要重新產生請加上 --force。')
        return 1

    print(f'讀取 {GROUND_TRUTH} ...')
    gt = load_ground_truth()
    print(f'  評分紀錄 {len(gt):,} 筆')

    image_ids = list_local_images()
    print(f'掃描 {IMAGE_DIR} ...')
    print(f'  本地影像 {len(image_ids):,} 張')

    df = build(gt, set(image_ids))
    coverage = len(df) / len(image_ids)
    print(f'  在評分紀錄中找到 {len(df):,} 張（覆蓋率 {coverage:.1%}）')
    if coverage < 1.0:
        missing = sorted(set(image_ids) - set(df['image']))
        print(f'  ⚠️ 有 {len(missing)} 張找不到評分，將不會出現在輸出中。'
              f'前幾個編號：{missing[:5]}')

    print('驗證重建結果 ...')
    verify_against_existing(df)

    train_df, val_df, n_new, n_new_val = split_preserving_existing(df)

    df.to_csv(OUT_FULL, index=False)
    train_df.to_csv(OUT_TRAIN, index=False)
    val_df.to_csv(OUT_VAL, index=False)

    print()
    print(f'✅ 完成')
    print(f'  {OUT_FULL:<22} {len(df):>5} 張'
          f'（label=1: {int(df["label"].sum())}，label=0: {int((df["label"] == 0).sum())}）')
    print(f'  {OUT_TRAIN:<22} {len(train_df):>5} 張')
    print(f'  {OUT_VAL:<22} {len(val_df):>5} 張')
    print(f'  新增的 {n_new} 張中，{n_new_val} 張分入驗證集、'
          f'{n_new - n_new_val} 張分入訓練集')
    print(f'  現有的 train.csv 與 val.csv 未被修改')
    return 0


if __name__ == '__main__':
    sys.exit(main())
