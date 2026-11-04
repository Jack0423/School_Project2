"""
重算書面報告第四章與附錄 B 的統計數字（不需要重新推論，也不需要影像）。

    python training/report_stats.py
    python training/report_stats.py --save docs/evidence/report_stats.json

用到的資料都在專案裡：
  docs/evidence/model_report_20260922/   2026-09-22 模型報表的逐張預測（model_report.py 產生）
  docs/evidence/burst_labeling/          連拍人工選擇：系統答案與兩位評選者（已去名，A、B）
  docs/heldout_20261104/runs.json        補充實驗（heldout_experiment.py）的結果
  data/*.csv                             AVA、KonIQ 的切分

輸出對照報告：
  1. 資料統計     2.3 的平均分、標準差、4.5 至 6.5 分比例；4.1 的 KonIQ 驗證集組成與重疊檢查
  2. bootstrap    4.3 的 95% 信賴區間（5000 次重抽，亂數種子 20261104，與報告數字相同）
  3. 人工選擇     4.7 的一致率、隨機基準、Wilson 95% 區間、精確檢定（各組命中機率 1/n）
  4. 補充實驗     表 4.6 的平均與標準差（含驗證 SRCC 的標準差）
  5. 訓練輪數     B.2 由 BatchNorm 計數反推；沒有的權重檔會略過

精確檢定的意思：若系統每組都是隨機挑一張，第 i 組命中的機率是 1/n_i，
命中組數的分佈是這些機率的卷積；p 值是「隨機挑也能命中這麼多組以上」的機率（單尾）。
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/evidence/model_report_20260922'
LABELS = ROOT / 'docs/evidence/burst_labeling'
HELDOUT = ROOT / 'docs/heldout_20261104/runs.json'
BOOT_SEED = 20261104
BOOT_N = 5000


def data_stats():
    full = pd.read_csv(ROOT / 'data/ava_full.csv')
    train = pd.read_csv(ROOT / 'data/ava_train.csv')
    val = pd.read_csv(ROOT / 'data/ava_val.csv')
    mid = lambda df: float(((df.mean_score >= 4.5) & (df.mean_score <= 6.5)).mean())
    out = {
        'ava_n': len(full), 'ava_mean': float(full.mean_score.mean()), 'ava_sd': float(full.mean_score.std()),
        'ava_mid_share_all': mid(full), 'ava_mid_share_val': mid(val),
        'ava_val_above6': float((val.mean_score > 6).mean()),
        'ava_train_val_overlap': len(set(train.image) & set(val.image)),
    }
    src = pd.read_csv(ROOT / 'data/koniq/koniq10k_distributions_sets.csv')
    tt = pd.read_csv(ROOT / 'data/train_tech.csv')
    vt = pd.read_csv(ROOT / 'data/val_tech.csv')
    out['koniq_train_val'] = [len(tt), len(vt)]
    out['koniq_val_composition'] = src[src.image_name.isin(vt.image)]['set'].value_counts().to_dict()
    out['koniq_train_val_overlap'] = len(set(tt.image) & set(vt.image))
    return out


def bootstrap():
    """與報告相同的重抽方式：先美感、後技術，共用同一個亂數產生器。"""
    rng = np.random.default_rng(BOOT_SEED)
    aes = pd.read_csv(REPORT / 'predictions_aesthetic.csv', encoding='utf-8-sig')
    tech = pd.read_csv(REPORT / 'predictions_technical.csv', encoding='utf-8-sig')
    out = {}
    for name, df, truth, pred in (('aesthetic', aes, 'true_mean', 'pred_mean'),
                                  ('technical', tech, 'true_mos', 'technical_score')):
        x, y = df[pred].to_numpy(float), df[truth].to_numpy(float)
        n = len(x)
        p, s = np.empty(BOOT_N), np.empty(BOOT_N)
        for b in range(BOOT_N):
            i = rng.integers(0, n, n)
            p[b] = stats.pearsonr(x[i], y[i])[0]
            s[b] = stats.spearmanr(x[i], y[i])[0]
        out[name] = {
            'n': n, 'plcc': float(stats.pearsonr(x, y)[0]), 'srcc': float(stats.spearmanr(x, y)[0]),
            'plcc_ci95': [float(v) for v in np.percentile(p, [2.5, 97.5])],
            'srcc_ci95': [float(v) for v in np.percentile(s, [2.5, 97.5])],
            'srcc_boot_se': float(s.std(ddof=1)),
        }
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def exact_p(probs, hits):
    """各組命中機率不同時，「命中組數 >= hits」的精確機率（單尾）。"""
    dist = np.array([1.0])
    for q in probs:
        dist = np.convolve(dist, [1 - q, q])
    return float(dist[hits:].sum())


def human_agreement():
    key = json.loads((LABELS / 'answer_key.json').read_text(encoding='utf-8'))['groups']
    picks, notes = {}, {}
    for tag in 'AB':
        with open(LABELS / f'rater_{tag}.csv', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        # 有效：pick 是該組裡的檔名
        picks[tag] = {r['group_id']: r['pick'] for r in rows
                      if r['pick'] and r['pick'] in key[r['group_id']]['photos']}
        notes[tag] = {r['group_id']: r['note'] for r in rows if r['note']}

    def summary(groups, chosen):
        hits = sum(1 for g in groups if chosen[g] == key[g]['system_best'])
        probs = [1 / len(key[g]['photos']) for g in groups]
        return {'valid': len(groups), 'hits': hits, 'rate': hits / len(groups),
                'chance': float(np.mean(probs)), 'wilson95': wilson(hits, len(groups)),
                'exact_p_one_sided': exact_p(probs, hits)}

    out = {tag: summary(list(picks[tag]), picks[tag]) for tag in 'AB'}
    common = sorted(set(picks['A']) & set(picks['B']), key=int)
    agree = [g for g in common if picks['A'][g] == picks['B'][g]]
    out['between_raters'] = {'common': len(common), 'agree': len(agree), 'rate': len(agree) / len(common),
                             'chance': float(np.mean([1 / len(key[g]['photos']) for g in common]))}
    out['consensus_subset'] = summary(agree, picks['A'])
    out['notes'] = {tag: dict(sorted(notes[tag].items(), key=lambda kv: int(kv[0]))) for tag in 'AB'}
    return out


def heldout():
    runs = json.loads(HELDOUT.read_text(encoding='utf-8'))['runs']
    out = {}
    for kind in ('aes', 'tech'):
        rs = [r for r in runs if r['kind'] == kind]
        col = lambda k: np.array([r[k] for r in rs])
        out[kind] = {k: [float(col(k).mean()), float(col(k).std(ddof=1))]
                     for k in ('val_srcc', 'test_plcc', 'test_srcc')}
        out[kind]['best_epoch'] = [r['best_epoch'] for r in rs]
        out[kind]['epochs'] = [r['epochs'] for r in rs]
    return out


def trained_epochs():
    """BatchNorm 的 num_batches_tracked 只在訓練模式前向時加 1，扣掉 ImageNet 初始值即為訓練批次數。"""
    try:
        import torch
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    except ImportError:
        return {'skipped': '沒有 torch／torchvision'}
    base_sd = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).state_dict()
    base = int(next(v for k, v in base_sd.items() if k.endswith('num_batches_tracked')))
    out = {'imagenet_base': base}
    for name, n_train in (('nima_aes_dist.pth', 5600), ('nima_tech_best.pth', 8058),
                          ('nima_aes_binary.pth', 5600), ('nima_best.pth', 3920)):
        path = ROOT / name
        if not path.exists():
            out[name] = '找不到權重檔，略過'
            continue
        sd = torch.load(path, map_location='cpu')
        counts = {int(v) for k, v in sd.items() if k.endswith('num_batches_tracked')}
        steps = counts.pop() - base
        per_epoch = math.ceil(n_train / 32)
        out[name] = {'steps': steps, 'batches_per_epoch': per_epoch, 'epochs': steps / per_epoch}
    return out


def main():
    p = argparse.ArgumentParser(description='重算書面報告的統計數字')
    p.add_argument('--save', help='另存成 JSON')
    p.add_argument('--skip-weights', action='store_true', help='不反推訓練輪數（不必載入 torch）')
    args = p.parse_args()

    result = {'data': data_stats(), 'bootstrap': bootstrap(), 'human': human_agreement(),
              'heldout': heldout()}
    if not args.skip_weights:
        result['epochs'] = trained_epochs()

    d = result['data']
    print(f"[INFO] AVA {d['ava_n']} 張：平均 {d['ava_mean']:.2f}、標準差 {d['ava_sd']:.2f}；"
          f"4.5 至 6.5 分占 {d['ava_mid_share_all']:.1%}（驗證集 {d['ava_mid_share_val']:.1%}）；"
          f"驗證集平均分 > 6 占 {d['ava_val_above6']:.1%}")
    print(f"[INFO] KonIQ 訓練／驗證 {d['koniq_train_val']}，驗證集官方組成 {d['koniq_val_composition']}；"
          f"重疊 AVA {d['ava_train_val_overlap']}、KonIQ {d['koniq_train_val_overlap']}")
    for name, b in result['bootstrap'].items():
        print(f"[INFO] {name} n={b['n']} PLCC {b['plcc']:.4f} [{b['plcc_ci95'][0]:.3f}, {b['plcc_ci95'][1]:.3f}]  "
              f"SRCC {b['srcc']:.4f} [{b['srcc_ci95'][0]:.3f}, {b['srcc_ci95'][1]:.3f}]  "
              f"SRCC 標準誤 {b['srcc_boot_se']:.4f}")
    h = result['human']
    for tag in ('A', 'B', 'consensus_subset'):
        r = h[tag]
        print(f"[INFO] {tag}: {r['hits']}/{r['valid']} = {r['rate']:.1%}，隨機 {r['chance']:.1%}，"
              f"Wilson [{r['wilson95'][0]:.1%}, {r['wilson95'][1]:.1%}]，精確檢定單尾 p = {r['exact_p_one_sided']:.3f}")
    br = h['between_raters']
    print(f"[INFO] 兩位評選者彼此：{br['agree']}/{br['common']} = {br['rate']:.1%}，兩人都隨機時 {br['chance']:.1%}")
    print(f"[INFO] 評選備註：{h['notes']}")
    for kind, r in result['heldout'].items():
        print(f"[INFO] 補充實驗 {kind}：驗證 SRCC {r['val_srcc'][0]:.4f} +/- {r['val_srcc'][1]:.4f}，"
              f"測試 PLCC {r['test_plcc'][0]:.4f} +/- {r['test_plcc'][1]:.4f}，"
              f"測試 SRCC {r['test_srcc'][0]:.4f} +/- {r['test_srcc'][1]:.4f}，最佳 epoch {r['best_epoch']}")
    if 'epochs' in result:
        print(f"[INFO] 訓練輪數：{result['epochs']}")

    if args.save:
        Path(args.save).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'[ OK ] 已存到 {args.save}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
