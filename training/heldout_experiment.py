"""
補充實驗：用「訓練／驗證／測試」三份切分重訓兩個模型，量測沒有參與選權重的測試集成績。

    python training/heldout_experiment.py                 # 3 個種子，RTX 5070 Ti 全部跑完約 4 小時
    python training/heldout_experiment.py --seeds 1       # 只跑一個種子試試
    python training/heldout_experiment.py --summarize-only reports/heldout/<時間>

為什麼要做
----------
現役權重（nima_aes_dist.pth、nima_tech_best.pth）挑最佳 epoch 用的驗證集，
也是模型報表量 PLCC／SRCC 的同一份資料，成績可能偏樂觀，而且只訓練過一次、沒有變異數。
這支程式**不動現役權重**，只用同一套訓練程式與預設超參數另外重訓：

  美感  訓練 4,480／驗證 1,120（由 ava_train.csv 依種子 42 切出，固定不變）
        測試 1,400 = ava_val.csv，只在最後量一次，不參與選 epoch
  技術  KonIQ-10k 官方切分：訓練 7,058／驗證 1,000／測試 2,015

每個模型用 3 個亂數種子各訓練一次，報告測試集 PLCC／SRCC 的平均與標準差。
輸出（權重、逐 epoch 訓練紀錄、逐張預測、summary）都在 reports/heldout/<時間>/。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from common import NIMABaseline, build_transform, to_mean_score

SPLIT_SEED = 42


def make_splits(out):
    """產生三份切分的 CSV，回傳每個模型的 (train, val, test, img_dir, mode)。"""
    out.mkdir(parents=True, exist_ok=True)

    ava = pd.read_csv(ROOT / 'data/ava_train.csv').sample(frac=1, random_state=SPLIT_SEED)
    cut = int(len(ava) * 0.8)
    ava.iloc[:cut].to_csv(out / 'aes_train.csv', index=False)
    ava.iloc[cut:].to_csv(out / 'aes_val.csv', index=False)
    test = pd.read_csv(ROOT / 'data/ava_val.csv')
    test.to_csv(out / 'aes_test.csv', index=False)
    assert not set(ava['image']) & set(test['image']), 'AVA 測試集與訓練資料重疊'

    src = pd.read_csv(ROOT / 'data/koniq/koniq10k_distributions_sets.csv')
    for name, part in (('train', 'training'), ('val', 'validation'), ('test', 'test')):
        df = src[src['set'] == part]
        pd.DataFrame({'image': df['image_name'], 'label': df['MOS'] / 100.0}).to_csv(
            out / f'tech_{name}.csv', index=False)

    sizes = {f.stem: len(pd.read_csv(f)) for f in sorted(out.glob('*.csv'))}
    print('[INFO] 切分：', sizes)
    return sizes


def train(kind, seed, out, workers):
    weights = out / f'{kind}_seed{seed}.pth'
    log = out / f'{kind}_seed{seed}.log'
    if kind == 'aes':
        cmd = [sys.executable, 'training/train_nima.py', '--mode', 'distribution',
               '--train-csv', str(out / 'aes_train.csv'), '--val-csv', str(out / 'aes_val.csv')]
    else:
        cmd = [sys.executable, 'training/train_tech.py',
               '--train-csv', str(out / 'tech_train.csv'), '--val-csv', str(out / 'tech_val.csv')]
    cmd += ['--save', str(weights), '--seed', str(seed), '--num-workers', str(workers), '--force']
    print(f'[INFO] 訓練 {kind} seed={seed}，紀錄寫到 {log.name}')
    start = time.time()
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    with open(log, 'w', encoding='utf-8') as f:
        f.write(' '.join(cmd) + '\n')
        f.flush()
        code = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    if code != 0:
        raise RuntimeError(f'{kind} seed={seed} 訓練失敗，見 {log}')
    return weights, log, time.time() - start


def parse_log(log):
    """從訓練紀錄讀出每個 epoch 的驗證 PLCC／SRCC，回傳 (總 epoch 數, 最佳 epoch, 最佳驗證 PLCC, SRCC)。"""
    rows = []
    for line in Path(log).read_text(encoding='utf-8').splitlines():
        if line.startswith('Epoch ['):
            parts = dict(p.strip().split(': ', 1) for p in line.split('|')[1:] if ': ' in p)
            rows.append((float(parts['PLCC']), float(parts['SRCC'])))
    best = max(range(len(rows)), key=lambda i: sum(rows[i]))
    return len(rows), best + 1, rows[best][0], rows[best][1]


@torch.no_grad()
def evaluate(kind, weights, test_csv, device):
    mode = 'distribution' if kind == 'aes' else 'single'
    img_dir = ROOT / ('data/dataset' if kind == 'aes' else 'data/koniq/512x384')
    model = NIMABaseline(output_mode=mode)
    model.load_state_dict(torch.load(weights, map_location='cpu'))
    model.to(device).eval()
    tf = build_transform('eval')
    df = pd.read_csv(test_csv)
    preds, truths = [], []
    for i in range(0, len(df), 64):
        chunk = df.iloc[i:i + 64]
        names = [str(n) if str(n).lower().endswith('.jpg') else f'{n}.jpg' for n in chunk['image']]
        x = torch.stack([tf(Image.open(img_dir / n).convert('RGB')) for n in names]).to(device)
        preds.append(np.atleast_1d(to_mean_score(model(x), mode).cpu().numpy()))
    pred = np.concatenate(preds)
    if kind == 'aes':
        truth = df['mean_score'].to_numpy(float)
    else:
        truth = df['label'].to_numpy(float) * 100
        pred = pred * 100
    pd.DataFrame({'image': df['image'], 'truth': truth, 'pred': pred}).to_csv(
        Path(weights).with_suffix('.test.csv'), index=False)
    return float(pearsonr(pred, truth)[0]), float(spearmanr(pred, truth)[0])


def summarize(out):
    runs = json.loads((out / 'runs.json').read_text(encoding='utf-8'))
    lines = ['# 補充實驗：三份切分、多個種子', '',
             f'產生時間 {datetime.now():%Y-%m-%d %H:%M}，切分見 `splits` 欄。', '',
             '| 模型 | 種子 | 訓練 epoch | 最佳 epoch | 驗證 PLCC | 驗證 SRCC | 測試 PLCC | 測試 SRCC | 耗時（分） |',
             '|---|---|---|---|---|---|---|---|---|']
    stats = {}
    for r in runs['runs']:
        lines.append(f"| {r['kind']} | {r['seed']} | {r['epochs']} | {r['best_epoch']} | "
                     f"{r['val_plcc']:.4f} | {r['val_srcc']:.4f} | {r['test_plcc']:.4f} | "
                     f"{r['test_srcc']:.4f} | {r['minutes']:.1f} |")
        stats.setdefault(r['kind'], []).append(r)
    lines += ['', '| 模型 | n | 測試 PLCC 平均 ± 標準差 | 測試 SRCC 平均 ± 標準差 | 驗證減測試 SRCC 差（平均） |',
              '|---|---|---|---|---|']
    for kind, rs in stats.items():
        tp = np.array([r['test_plcc'] for r in rs])
        ts = np.array([r['test_srcc'] for r in rs])
        gap = np.mean([r['val_srcc'] - r['test_srcc'] for r in rs])
        sd = (lambda a: a.std(ddof=1) if len(a) > 1 else float('nan'))
        lines.append(f'| {kind} | {len(rs)} | {tp.mean():.4f} ± {sd(tp):.4f} | '
                     f'{ts.mean():.4f} ± {sd(ts):.4f} | {gap:+.4f} |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description='三份切分、多種子的補充實驗')
    p.add_argument('--seeds', type=int, default=3, help='每個模型訓練幾個種子（預設 3）')
    p.add_argument('--models', default='aes,tech', help='要跑的模型，逗號分隔（aes、tech）')
    p.add_argument('--workers', type=int, default=8, help='DataLoader worker 數（預設 8）')
    p.add_argument('--summarize-only', help='只重新產生某次輸出的 summary.md')
    args = p.parse_args()

    if args.summarize_only:
        summarize(Path(args.summarize_only))
        return 0

    out = ROOT / 'reports/heldout' / datetime.now().strftime('%Y%m%d-%H%M%S')
    sizes = make_splits(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    record = {'splits': sizes, 'split_seed': SPLIT_SEED, 'device': str(device),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'runs': []}
    for kind in args.models.split(','):
        for seed in range(1, args.seeds + 1):
            weights, log, secs = train(kind, seed, out, args.workers)
            epochs, best, vp, vs = parse_log(log)
            tp, ts = evaluate(kind, weights, out / f'{kind}_test.csv', device)
            record['runs'].append({'kind': kind, 'seed': seed, 'epochs': epochs, 'best_epoch': best,
                                   'val_plcc': vp, 'val_srcc': vs, 'test_plcc': tp, 'test_srcc': ts,
                                   'minutes': secs / 60})
            print(f'[ OK ] {kind} seed={seed}：最佳 epoch {best}/{epochs}，'
                  f'驗證 SRCC {vs:.4f}，測試 PLCC {tp:.4f}／SRCC {ts:.4f}')
            (out / 'runs.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    summarize(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
