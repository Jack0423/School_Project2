"""
在同一組驗證資料上比較多個美感模型，決定要採用哪一個。

為什麼需要這支腳本
------------------
訓練過程印出的 PLCC / SRCC 不能直接跨模式比較：

  single 模式       標籤是二元 0/1，相關係數是對二元標籤算的
  distribution 模式 標籤是 1~10 級分佈，相關係數是對評分期望值算的

兩者的「對照答案」根本不同，數字放在一起看沒有意義。
這裡改成統一用 AVA 的群眾平均評分（1~10 分連續值）當唯一的對照答案，
所有模型都跟它比，才是公平的比較。

指標選擇
--------
以 SRCC 為主。它是排序相關係數，不受輸出尺度影響——
二元模型輸出的是 0~1 附近的機率，分佈模型輸出的是 1~10 分，
尺度完全不同，只有排序相關才能公平比較兩者。
PLCC 一併列出作為參考，但跨尺度比較時不應作為主要依據。

用法
----
    python compare_aesthetic_models.py
    python compare_aesthetic_models.py --val-csv data/ava_val.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from common import NIMABaseline, build_transform

SCORE_LEVELS = np.arange(1, 11)


def load_model(path, mode, device):
    model = NIMABaseline(output_mode=mode, pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict(model, mode, df, img_dir, transform, device):
    """回傳每張照片的預測值。distribution 模式取評分期望值。"""
    names = [f'{n}.jpg' if not str(n).lower().endswith('.jpg') else str(n)
             for n in df['image'].tolist()]
    preds = []
    levels = torch.tensor(SCORE_LEVELS, dtype=torch.float32, device=device)

    with torch.no_grad():
        for name in names:
            path = os.path.join(img_dir, name)
            with Image.open(path) as im:
                x = transform(im.convert('RGB')).unsqueeze(0).to(device)
            out = model(x)
            preds.append((out * levels).sum().item() if mode == 'distribution'
                         else out.item())
    return np.array(preds)


def main():
    p = argparse.ArgumentParser(description='比較多個美感模型')
    p.add_argument('--val-csv', default='data/ava_val.csv')
    p.add_argument('--img-dir', default='data/dataset')
    args = p.parse_args()

    df = pd.read_csv(args.val_csv)
    if 'mean_score' not in df.columns:
        print(f'⛔ {args.val_csv} 沒有 mean_score 欄位。')
        print('   需要含 AVA 群眾平均評分的驗證集才能做公平比較，')
        print('   請先執行 python build_ava_labels.py 產生 data/ava_val.csv。')
        return 1

    truth = df['mean_score'].values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = build_transform('eval')

    # (顯示名稱, 權重檔, 模式, 說明)
    candidates = [
        # 「現行」指的是 ai_inference.py 的 AES_WEIGHTS 目前指向誰。
        # 這個標記換過一次：比較做完後系統即改用分佈模型，
        # 而本清單仍把舊的 nima_best.pth 標為「現行模型」，與實際情況相反。
        ('舊版（已淘汰）', 'nima_best.pth', 'single', '舊資料 3,920 張、二元標籤'),
        ('二元 + 新資料', 'nima_aes_binary.pth', 'single', '5,600 張、二元標籤'),
        ('分佈 + emd_loss（現行）', 'nima_aes_dist.pth', 'distribution', '5,600 張、1~10 級分佈'),
    ]

    print(f'驗證集：{args.val_csv}（{len(df)} 張）')
    print(f'對照答案：AVA 群眾平均評分，範圍 {truth.min():.2f} ~ {truth.max():.2f}')
    print(f'裝置：{device}\n')
    print(f'{"模型":<18}{"說明":<26}{"SRCC":>9}{"PLCC":>9}{"輸出範圍":>18}  範圍合理?')
    print('-' * 92)

    results = []
    for label, path, mode, note in candidates:
        if not os.path.exists(path):
            print(f'{label:<18}{note:<26}{"(權重檔不存在，略過)":>40}')
            continue
        model = load_model(path, mode, device)
        preds = predict(model, mode, df, args.img_dir, transform, device)
        srcc = spearmanr(preds, truth)[0]
        plcc = pearsonr(preds, truth)[0]

        # distribution 模式輸出天然落在 1~10；single 模式沒有任何界限
        in_range = (1.0 <= preds.min() and preds.max() <= 10.0) if mode == 'distribution' \
            else (0.0 <= preds.min() and preds.max() <= 1.0)
        results.append((label, srcc, plcc, mode, in_range))
        print(f'{label:<18}{note:<26}{srcc:>9.4f}{plcc:>9.4f}'
              f'{f"{preds.min():.3f} ~ {preds.max():.3f}":>18}  '
              f'{"是" if in_range else "否（需裁切）"}')

    if len(results) >= 2:
        best = max(results, key=lambda r: r[1])
        print(f'\n以 SRCC 為準，表現最佳：{best[0]}（SRCC {best[1]:.4f}）')
        print('提醒：SRCC 只說明「排序」的準確度。')
        print('      若兩者 SRCC 接近，應優先選擇輸出範圍天然合理的分佈模型——')
        print('      二元模型的輸出沒有界限，必須靠裁切才能當分數顯示，')
        print('      而裁切會把大量照片壓在 0 與 100 兩端、失去彼此的排序資訊。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
