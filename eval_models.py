"""
評估目前的模型權重，輸出可比較的客觀指標。

用途有三個，都是「改動前後要拿來對照」的場合：
  1. 升級 PyTorch 前後——確認換版本沒有改變模型輸出
  2. 重新訓練前後——確認新模型真的比較好才採用
  3. 修改前處理等推論設定前後——量化影響

指標的選擇
----------
  技術模型  PLCC / SRCC。標籤是連續的 KonIQ MOS 分數，
            而影像品質評估在意的是「排序」而非絕對數值，
            相關係數比單純的 loss 更貼近人類評分行為。
  美感模型  AUC / 準確率。目前標籤是二元 0/1，
            用 PLCC / SRCC 意義有限。
            若改用 1~10 級分佈重新訓練，本腳本會自動改算 PLCC / SRCC。

用法
----
    python eval_models.py                    # 用現有的驗證集
    python eval_models.py --save metrics.json
    python eval_models.py --aes-csv data/ava_val.csv   # 重訓後改用新驗證集
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

import ai_inference as ai


def _predict(model, csv_path, img_dir, add_suffix):
    """
    回傳 (預測值, 對照答案, 略過清單, 說明)。

    略過清單刻意往上傳而不是就地吞掉——見下方 except 的說明。
    """
    df = pd.read_csv(csv_path)

    # 逐「欄」取檔名，不使用 df.iterrows()。
    #
    # iterrows() 會把整列轉成單一 dtype 的 Series。CSV 只要含任何浮點欄位
    # （ava_val.csv 的 mean_score、score_1~10 就是），int64 的 image 欄會被
    # 一起升成 float64：53 變成 53.0，組出來的檔名成了 '53.0.jpg'。
    # 實測 ava_val.csv 的 1,400 張因此「全部」找不到檔案。
    # 舊的 val.csv 只有整數欄位不會觸發，所以這個 bug 是在換用新驗證集後才出現。
    names = [str(v) for v in df['image'].tolist()]

    # 對照答案優先取 mean_score（AVA 群眾平均評分，1~10 連續值）。
    # 只有舊的二元標籤資料集才退回 label。
    # 二元標籤無法評估連續輸出的模型——分佈模型預測的是 1~10 分，
    # 拿 0/1 當答案只能算 AUC，量不出評分的準確度。
    truth_col = 'mean_score' if 'mean_score' in df.columns else 'label'
    truth = df[truth_col].astype(float).tolist()

    # 輸出模式直接問模型本人，不從外部設定推測——兩者不同步會安靜地算錯分數。
    mode = getattr(model, 'output_mode', 'single')
    levels = torch.arange(1, 11, dtype=torch.float32, device=ai.DEVICE)

    preds, labels, skipped = [], [], []
    with torch.no_grad():
        for name, y in zip(names, truth):
            if add_suffix and not name.lower().endswith('.jpg'):
                name += '.jpg'
            path = os.path.join(img_dir, name)
            try:
                with Image.open(path) as im:
                    tensor = ai.TRANSFORM(im.convert('RGB')).unsqueeze(0).to(ai.DEVICE)
            except Exception as e:
                # 記下來往上傳，不要靜靜地 continue。
                # 原本這裡直接吞掉，導致「全部 1,400 張都讀失敗」與
                # 「資料夾是空的」印出完全相同的訊息，無從分辨。
                skipped.append(f'{name} -> {type(e).__name__}: {e}')
                continue

            out = model(tensor)
            # distribution 模式輸出 10 維機率，取評分期望值換算成 1~10 分。
            # 對 (1, 10) 的張量呼叫 .item() 會直接拋 RuntimeError。
            preds.append((out * levels).sum().item() if mode == 'distribution'
                         else out.item())
            labels.append(y)

    return (np.array(preds), np.array(labels), skipped,
            {'truth_col': truth_col, 'output_mode': mode})


def _auc(scores, labels):
    """以排名法計算 AUC，避免為此引入 scikit-learn 依賴。"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(name, model, csv_path, img_dir, add_suffix):
    if not os.path.exists(csv_path):
        print(f'  {name}：找不到 {csv_path}，略過')
        return None

    started = time.perf_counter()
    preds, labels, skipped, info = _predict(model, csv_path, img_dir, add_suffix)
    elapsed = time.perf_counter() - started

    if len(preds) == 0:
        print(f'  {name}：沒有任何影像可評估——{len(skipped)} 張全部讀取失敗')
        if skipped:
            print(f'      首例：{skipped[0]}')
            print(f'      （檔名若出現多餘的 .0，代表 CSV 的整數欄位被讀成浮點數）')
        return None

    if skipped:
        print(f'  {name}：略過 {len(skipped)} 張讀取失敗的影像，首例 {skipped[0]}')

    result = {'csv': csv_path, 'n': int(len(preds)), 'skipped': len(skipped),
              'seconds': round(elapsed, 1),
              'truth_col': info['truth_col'], 'output_mode': info['output_mode'],
              'pred_min': round(float(preds.min()), 4),
              'pred_max': round(float(preds.max()), 4)}

    binary = set(np.unique(labels)) <= {0.0, 1.0}
    if binary:
        result['auc'] = round(float(_auc(preds, labels)), 4)
        result['accuracy'] = round(float(((preds > 0.5).astype(int) == labels).mean()), 4)
        summary = f"AUC {result['auc']:.4f}   準確率 {result['accuracy']:.4f}"
    else:
        result['plcc'] = round(float(pearsonr(preds, labels)[0]), 4)
        result['srcc'] = round(float(spearmanr(preds, labels)[0]), 4)
        summary = f"PLCC {result['plcc']:.4f}   SRCC {result['srcc']:.4f}"

    print(f'  {name:<10} {result["n"]:>5} 張   {summary}   ({elapsed:.0f}s)')
    return result


def main():
    parser = argparse.ArgumentParser(description='評估模型權重')
    # 預設改為 ava_val.csv：現行美感模型是用 ava_train.csv 訓練的，
    # 對應的驗證集就是 ava_val.csv，且它帶有 mean_score（1~10 連續值）可算 SRCC。
    # 舊的 val.csv 仍是乾淨的保留集（其 980 張完全包含於 ava_val 的 1,400 張內，
    # 與 ava_train 交集為 0），但只有二元標籤，量不出連續評分的準確度。
    parser.add_argument('--aes-csv', default='data/ava_val.csv')
    parser.add_argument('--aes-dir', default='data/dataset')
    parser.add_argument('--tech-csv', default='data/val_tech.csv')
    parser.add_argument('--tech-dir', default='data/koniq/512x384')
    parser.add_argument('--save', metavar='PATH', help='另存為 JSON')
    args = parser.parse_args()

    print(f'裝置：{ai.DEVICE}    torch {torch.__version__}')
    print()
    metrics = {
        'torch': torch.__version__,
        'device': str(ai.DEVICE),
        'weights': {
            'aesthetic': str(ai.AES_WEIGHTS.name),
            'technical': str(ai.TECH_WEIGHTS.name),
        },
        'aesthetic': evaluate('美感模型', ai.MODEL_AES, args.aes_csv, args.aes_dir,
                              add_suffix=True),
        'technical': evaluate('技術模型', ai.MODEL_TECH, args.tech_csv, args.tech_dir,
                              add_suffix=False),
    }

    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f'\n已寫入 {args.save}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
