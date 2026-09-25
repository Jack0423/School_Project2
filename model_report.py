"""
模型成效報表：把兩個模型在驗證集上的表現整理成數據、表格與圖表。

    python model_report.py              # 完整報表（約 1~2 分鐘）
    python model_report.py --quick      # 每個驗證集只取前 300 張，確認流程用

會產生什麼
----------
  技術模型（KonIQ 驗證集 2,015 張）
      PLCC、SRCC、KRCC、RMSE、MAE
      預測 vs 真實散佈圖、分數分佈、各分數區間的偏差
      「警告」判定的混淆矩陣、Precision / Recall / F1、門檻掃描
  美感模型（AVA 驗證集 1,400 張）
      同上，另外加上 EMD（評分分佈的差距）、ROC / AUC、評分分佈的預測範例
      「優秀」判定的混淆矩陣、Precision / Recall / F1、門檻掃描
  美感模型新舊版比較（舊權重存在時）
  影像量測細項：被標記模糊、過曝、欠曝、對比不足的照片，人工評分是否真的比較低

分數一律照系統的算法
--------------------
分數換算與門檻直接取自 ai_inference.py（_output_to_score、_clamp_score、
TECH_ISSUE_THRESHOLD、AESTHETIC_EXCELLENT_THRESHOLD），不另外寫一份——
另寫一份的話，報表量的就不一定是系統實際的行為。
執行時會抽前幾張，和 evaluate_photo() 逐張算出的分數比對，確認兩邊一致。

混淆矩陣的「真實答案」怎麼定
----------------------------
模型輸出的是連續分數，混淆矩陣需要分成兩類。預測端用系統實際的門檻，
真實端的切點則要自己定：
    警告  KonIQ 人工評分 MOS < 60（和系統門檻同一個尺度）
    優秀  AVA 群眾平均分 > 6（和 ai_inference.py 校準門檻時用的定義相同）
切點換了矩陣就會變，可用 --tech-truth-cut / --aes-truth-cut 調整。

沒有做「優秀／正常／警告」三類的混淆矩陣：沒有任何資料集同時具備
美感與技術兩種人工評分（KonIQ 只有技術、AVA 只有美感）。

也沒有訓練曲線：訓練腳本沒有把每個 epoch 的 loss 存下來。

只讀不寫
--------
不修改任何權重、CSV 或程式。輸出寫到 reports/model_report/<時間>/：
    summary.md                 表格版摘要（每張圖的數字都在這裡）
    metrics.json               所有指標
    predictions_*.csv          每張照片的真實分數與預測分數
    *.png                      圖表（需要 matplotlib）
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

QUICK_LIMIT = 300
MIN_POSITIVES_FOR_PRECISION = 20
# AVA 驗證集的群眾評分集中在兩端，這個區間是「最難分辨」的中段
AES_MIDDLE_RANGE = (4.5, 6.5)
TECH_BIN_EDGES = (0, 30, 40, 50, 60, 70, 80, 100)
AES_BIN_EDGES = (1, 3, 4, 5, 6, 7, 10)

# 新舊美感權重。「現行」是誰由 ai_inference.AES_WEIGHTS 決定，不寫在這裡——
# compare_aesthetic_models.py 就發生過換了模型、清單上的「現行」標記卻沒跟著改。
COMPARE_CANDIDATES = (
    ('二元標籤 3,920 張（舊版）', 'nima_best.pth'),
    ('二元標籤 5,600 張', 'nima_aes_binary.pth'),
    ('評分分佈 + EMD 5,600 張', 'nima_aes_dist.pth'),
)

# technical_issues 是給人看的句子，這裡依關鍵字歸類。順序即圖表順序。
ISSUE_KEYWORDS = (('模糊', '模糊'), ('過曝', '過曝'), ('曝光不足', '欠曝'),
                  ('對比度不足', '對比不足'), ('雜訊', '雜訊'))


# ==========================================
# 推論
# ==========================================
def run_models(ai, models, paths, batch_size, analyze=False):
    """
    每張照片只解碼一次，同時餵給所有模型；analyze=True 時順便跑影像量測。

    回傳：
        raws    每個模型的原始輸出（numpy）
        scores  每個模型換算後的 0~100 分（未裁切）
        issues  每張照片的 technical_issues（analyze=False 時為空）
    """
    import torch
    from PIL import Image

    outs = [[] for _ in models]
    issues = []
    total = len(paths)
    started = time.perf_counter()
    with torch.no_grad():
        for i in range(0, total, batch_size):
            # 讀檔走 evaluate_photo 同一個函式（依副檔名分流、一律 RGB）
            arrays = [ai._load_image_array(str(p)) for p in paths[i:i + batch_size]]
            x = torch.stack([ai.TRANSFORM(Image.fromarray(a)) for a in arrays]).to(ai.DEVICE)
            for k, (model, _) in enumerate(models):
                outs[k].append(model(x))
            if analyze:
                issues += [ai._analyze_technical_issues(a) for a in arrays]
            done = min(i + batch_size, total)
            print(f'\r    {done:,}/{total:,} 張（{time.perf_counter() - started:.0f} 秒）',
                  end='', flush=True)
    print()

    raws, scores = [], []
    for chunks, (_, mode) in zip(outs, models):
        raw = torch.cat(chunks)
        # 逐張呼叫系統自己的換算函式，確保與 evaluate_photo 同一條公式
        scores.append(np.array([ai._output_to_score(r, mode) for r in raw]))
        raws.append(raw.float().cpu().numpy())
    return raws, scores, issues


def check_consistency(ai, paths, key, batch_scores, n=3):
    """抽前 n 張，用 evaluate_photo() 逐張重算，確認批次算法和前台看到的分數相同。"""
    worst = 0.0
    for path, score in zip(paths[:n], batch_scores[:n]):
        result = ai.evaluate_photo(str(path))
        if result is None:
            return {'checked': 0, 'error': ai.LAST_ERROR}
        # evaluate_photo 回傳值四捨五入到小數 2 位
        worst = max(worst, abs(result[key] - round(ai._clamp_score(score), 2)))
    # 容差 0.2 分：GPU 上 batch 64 與逐張推論會選到不同的卷積演算法（cuDNN 預設開 TF32），
    # 實測最多差約 0.1 分。公式錯誤造成的差距遠大於此；要逐位元一致請用 --batch-size 1。
    ok = worst <= 0.2
    print(f"    {'[ OK ]' if ok else '[WARN]'} 與 evaluate_photo() 逐張比對 {n} 張，最大差 {worst:.3f} 分")
    return {'checked': n, 'max_abs_diff': worst, 'ok': ok}


def classify_issues(texts):
    found = set()
    for text in texts:
        for keyword, name in ISSUE_KEYWORDS:
            if keyword in text:
                found.add(name)
                break
        else:
            found.add('其他')
    return found


def display_path(path):
    """報表裡顯示相對於專案的路徑，不把個人磁碟路徑寫進要分享出去的檔案。"""
    try:
        return Path(path).relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path)


def image_paths(names, img_dir, add_suffix):
    # 逐「欄」取檔名：iterrows() 會把整數檔名升成浮點數，53 變成 '53.0.jpg'
    out = []
    for name in names:
        name = str(name)
        if add_suffix and not name.lower().endswith('.jpg'):
            name += '.jpg'
        out.append(Path(img_dir) / name)
    return out


# ==========================================
# 技術模型
# ==========================================
def technical_section(ai, args):
    import pandas as pd
    from common import metrics as M

    print('\n=== 技術模型（KonIQ 驗證集）===')
    df = pd.read_csv(args.tech_csv)
    if args.quick:
        df = df.head(QUICK_LIMIT)
    paths = image_paths(df['image'].tolist(), args.tech_dir, add_suffix=False)
    true = df['label'].to_numpy(dtype=float) * 100   # split_koniq.py 把 MOS 縮成 0~1

    _, scores, issues = run_models(ai, [(ai.MODEL_TECH, ai.TECH_MODE)], paths,
                                   args.batch_size, analyze=not args.no_issues)
    pred = np.array([ai._clamp_score(s) for s in scores[0]])
    consistency = check_consistency(ai, paths, 'technical_score', scores[0])

    thr = ai.TECH_ISSUE_THRESHOLD
    pred_pos = M.predict_positive(pred, thr, 'below')
    true_pos = true < args.tech_truth_cut
    counts = M.confusion_counts(pred_pos, true_pos)
    res = {
        'dataset': display_path(args.tech_csv), 'n': int(len(df)),
        'regression': M.regression_summary(pred, true),
        'warning': {'rule': f'技術分 < {thr}', 'truth': f'KonIQ MOS < {args.tech_truth_cut:g}',
                    'threshold': thr, 'truth_cut': args.tech_truth_cut,
                    'counts': counts, 'summary': M.classification_summary(counts)},
        'bins': M.binned_error(pred, true, TECH_BIN_EDGES),
        'sweep': M.threshold_sweep(pred, true_pos, np.arange(30, 80.5, 0.5), 'below'),
        'consistency': consistency,
    }
    reg, s = res['regression'], res['warning']['summary']
    print(f"    PLCC {reg['plcc']:.4f}  SRCC {reg['srcc']:.4f}  RMSE {reg['rmse']:.2f} 分  MAE {reg['mae']:.2f} 分")
    print(f"    警告判定  Precision {s['precision']:.1%}  Recall {s['recall']:.1%}  "
          f"F1 {s['f1']:.3f}  Accuracy {s['accuracy']:.1%}")

    arrays = {'true': true, 'pred': pred, 'names': df['image'].astype(str).tolist(),
              'pred_pos': pred_pos, 'true_pos': true_pos, 'issues': issues}
    return res, arrays


def issues_section(arrays):
    """影像量測細項：被標記的照片，人工評分（MOS）是否真的比較低。"""
    from scipy.stats import mannwhitneyu

    mos = arrays['true']
    tags = [classify_issues(t) for t in arrays['issues']]
    names = [name for _, name in ISSUE_KEYWORDS] + ['其他']
    rows = []
    noise = None
    for name in names:
        flagged = np.array([name in t for t in tags])
        n_flag = int(flagged.sum())
        if name == '雜訊':
            # 雜訊不列入表格與圖表，只在說明裡交代數字。
            # 門檻是用 Sony ARW 原檔校準的；KonIQ 是縮小到 512x384 的網路 JPG，
            # 縮小會把細節擠進每個像素，而這個估計量本來就分不開細節與雜訊。
            # 2026-09-25 實測：驗證集 61% 被標記，被標記者的 MOS 中位數反而較高（67 vs 49）——
            # 量到的是「畫面清晰」而不是雜訊，放進表格只會誤導讀者。
            noise = {'flagged': n_flag, 'rate': n_flag / len(mos),
                     'mos_flagged_median': float(np.median(mos[flagged])) if n_flag else None,
                     'mos_rest_median': float(np.median(mos[~flagged])) if (~flagged).any() else None}
            continue
        if n_flag == 0 and name == '其他':
            continue
        row = {'issue': name, 'flagged': n_flag, 'rate': n_flag / len(mos),
               'mos_flagged_median': float(np.median(mos[flagged])) if n_flag else None,
               'mos_rest_median': float(np.median(mos[~flagged])),
               'mos_flagged_mean': float(mos[flagged].mean()) if n_flag else None,
               'mos_rest_mean': float(mos[~flagged].mean()),
               'p_value': None}
        # 兩組都要有一定數量，檢定才有意義
        if n_flag >= 5 and (~flagged).sum() >= 5:
            row['p_value'] = float(mannwhitneyu(mos[flagged], mos[~flagged]).pvalue)
        rows.append(row)
    # 「至少被標記一項」與表格一致，不含雜訊
    any_flag = np.array([bool(t - {'雜訊'}) for t in tags])
    print('\n=== 影像量測細項（KonIQ 驗證集）===')
    for r in rows:
        if r['flagged']:
            print(f"    {r['issue']:<5}  標記 {r['flagged']:>4} 張（{r['rate']:5.1%}）"
                  f"  MOS 中位數 {r['mos_flagged_median']:5.1f} vs 其餘 {r['mos_rest_median']:5.1f}")
        else:
            print(f"    {r['issue']:<5}  標記    0 張")

    note = ('雜訊不列入：門檻是用相機 RAW 原檔校準的，KonIQ 是縮小到 512x384 的網路 JPG，'
            '縮小後的細節會被這個估計量當成雜訊')
    if noise and noise['flagged'] and noise['mos_flagged_median'] is not None \
            and noise['mos_rest_median'] is not None:
        note += (f"（本次標記 {noise['flagged']:,} 張、{noise['rate']:.0%}，被標記者 MOS 中位數 "
                 f"{noise['mos_flagged_median']:.1f}、其餘 {noise['mos_rest_median']:.1f}）")
        print(f"    雜訊   不列入（標記 {noise['flagged']} 張，量到的主要是細節，見 summary.md）")
    return {'n': int(len(mos)), 'any_flagged': int(any_flag.sum()), 'rows': rows,
            'noise_excluded': noise, 'note': note}, tags


# ==========================================
# 美感模型
# ==========================================
def aesthetic_section(ai, args):
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from common import metrics as M

    print('\n=== 美感模型（AVA 驗證集）===')
    df = pd.read_csv(args.aes_csv)
    if args.quick:
        df = df.head(QUICK_LIMIT)
    if 'mean_score' not in df.columns:
        print(f'    [WARN] {args.aes_csv} 沒有 mean_score 欄位，略過（請先執行 build_ava_labels.py）')
        return {'skipped': '驗證集沒有 mean_score'}, None
    paths = image_paths(df['image'].tolist(), args.aes_dir, add_suffix=True)
    true_mean = df['mean_score'].to_numpy(dtype=float)

    # 現行模型用 ai_inference 已載入的那一顆；其餘候選另外載入，缺檔就略過
    current = ai.AES_WEIGHTS.name
    models = [(ai.MODEL_AES, ai.AES_MODE)]
    labels = [next((l for l, f in COMPARE_CANDIDATES if f == current), current)]
    files = [current]
    if not args.no_compare:
        for label, filename in COMPARE_CANDIDATES:
            if filename == current:
                continue
            try:
                model, mode = ai._load_model(BASE_DIR / filename, label)
            except FileNotFoundError:
                print(f'    [INFO] 找不到 {filename}，比較時略過')
                continue
            models.append((model, mode))
            labels.append(label)
            files.append(filename)

    raws, scores, _ = run_models(ai, models, paths, args.batch_size)
    pred = np.array([ai._clamp_score(s) for s in scores[0]])
    consistency = check_consistency(ai, paths, 'aesthetic_score', scores[0])

    thr = ai.AESTHETIC_EXCELLENT_THRESHOLD
    pred_pos = M.predict_positive(pred, thr, 'above')
    true_pos = true_mean > args.aes_truth_cut
    counts = M.confusion_counts(pred_pos, true_pos)
    res = {
        'dataset': display_path(args.aes_csv), 'n': int(len(df)), 'weights': current, 'mode': ai.AES_MODE,
        'excellent': {'rule': f'美感分 > {thr}', 'truth': f'AVA 群眾平均 > {args.aes_truth_cut:g}',
                      'threshold': thr, 'truth_cut': args.aes_truth_cut,
                      'counts': counts, 'summary': M.classification_summary(counts)},
        'sweep': M.threshold_sweep(pred, true_pos, np.arange(30, 80.5, 0.5), 'above'),
        'consistency': consistency,
    }
    # 這份驗證集的群眾評分集中在兩端（Kaggle 競賽的抽樣方式），最難分辨的中段照片很少。
    # 這會讓 Accuracy、AUC 看起來比在一般照片上高，報表必須把這個比例一起列出來。
    lo, hi = AES_MIDDLE_RANGE
    res['middle_share'] = {'range': [lo, hi],
                           'share': float(np.mean((true_mean >= lo) & (true_mean <= hi)))}
    arrays = {'names': df['image'].astype(str).tolist(), 'true_mean': true_mean, 'pred': pred,
              'pred_pos': pred_pos, 'true_pos': true_pos}

    if ai.AES_MODE == 'distribution':
        # 0~100 換回 1~10，和群眾平均分同一個尺度，RMSE / MAE 才有意義
        pred_mean = pred / 100 * 9 + 1
        score_cols = [f'score_{i}' for i in range(1, 11)]
        true_dist = df[score_cols].to_numpy(dtype=float)
        true_dist = true_dist / true_dist.sum(axis=1, keepdims=True)
        emd = M.emd_per_sample(raws[0], true_dist)
        res['regression'] = M.regression_summary(pred_mean, true_mean)
        res['bins'] = M.binned_error(pred_mean, true_mean, AES_BIN_EDGES)
        res['emd'] = {'mean': float(emd.mean()), 'median': float(np.median(emd)),
                      'definition': 'NIMA 論文定義（r=2，含開根號），與訓練用 emd_loss 的數值不同'}
        arrays.update(pred_mean=pred_mean, pred_dist=raws[0], true_dist=true_dist, emd=emd)
    else:
        # 退回二元權重時輸出沒有 1~10 的尺度，只剩排序相關係數有意義
        reg = M.regression_summary(pred, true_mean)
        res['regression'] = {k: reg[k] for k in ('n', 'plcc', 'srcc', 'krcc')}

    if 'label' in df.columns:
        labels01 = df['label'].to_numpy() == 1
        fpr, tpr = M.roc_curve(pred, labels01)
        res['binary'] = {'auc': M.auc_from_roc(fpr, tpr), 'positive_rate': float(labels01.mean()),
                         'definition': 'AVA 二元標籤（build_ava_labels.py，以群眾平均約 4.47 分為界）'}
        arrays.update(fpr=fpr, tpr=tpr)

    compare = []
    for label, filename, (_, mode), s in zip(labels, files, models, scores):
        # 比較用未裁切的分數：舊二元模型有大量照片會被裁到 0 或 100，裁切會抹掉排序資訊
        compare.append({'label': label, 'weights': filename, 'mode': mode,
                        'current': filename == current,
                        'srcc': float(spearmanr(s, true_mean)[0]),
                        'plcc': float(pearsonr(s, true_mean)[0]),
                        'min': float(s.min()), 'max': float(s.max()),
                        'clipped_share': float(np.mean((s < 0) | (s > 100)))})
    res['compare'] = compare

    reg, s = res['regression'], res['excellent']['summary']
    line = f"    PLCC {reg['plcc']:.4f}  SRCC {reg['srcc']:.4f}"
    if 'rmse' in reg:
        line += f"  RMSE {reg['rmse']:.3f}  MAE {reg['mae']:.3f}（1~10 分）  EMD {res['emd']['mean']:.4f}"
    if 'binary' in res:
        line += f"  AUC {res['binary']['auc']:.4f}"
    print(line)
    print(f"    優秀判定  Precision {s['precision']:.1%}  Recall {s['recall']:.1%}  "
          f"F1 {s['f1']:.3f}  Accuracy {s['accuracy']:.1%}")
    print(f"    [INFO] 群眾平均分落在 {lo:g}~{hi:g} 的照片只佔 {res['middle_share']['share']:.1%}"
          f"，評分集中在兩端，Accuracy 與 AUC 會偏高")
    if len(compare) > 1:
        for c in compare:
            print(f"    {'*' if c['current'] else ' '} {c['label']:<22} SRCC {c['srcc']:.4f}  PLCC {c['plcc']:.4f}")
    return res, arrays


# ==========================================
# 圖表
# ==========================================
def make_charts(res, arr, out):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from common import plotting as P
    except ImportError as e:
        print(f'[WARN] 沒有 matplotlib，略過圖表：{e}（pip install matplotlib）')
        return []
    if P.setup() is None:
        print('[WARN] 找不到中文字型，圖表裡的中文可能會顯示成方框')
    made = []

    def save(fig, name):
        P.save(fig, out / name)
        made.append(name)

    def scatter(true, pred, cut, thr, direction, lim, xlabel, ylabel, title, sub, counts, name):
        fig, ax = plt.subplots(figsize=(6.4, 6))
        ax.scatter(true, pred, s=9, alpha=0.35, color=P.BLUE, edgecolors='none')
        ax.plot(lim, lim, color=P.MUTED, linewidth=1)
        ax.axvline(cut, color=P.INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.axhline(thr, color=P.INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.grid(axis='both')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # 四個象限就是混淆矩陣的四格
        box = dict(boxstyle='round,pad=0.25', facecolor=P.SURFACE, edgecolor='none', alpha=0.85)
        lo, hi = lim
        pad = (hi - lo) * 0.02
        if direction == 'below':   # 警告：真實低、預測低 = TP（左下）
            spots = {'tp': (lo + pad, lo + pad, 'left', 'bottom'), 'fn': (lo + pad, hi - pad, 'left', 'top'),
                     'fp': (hi - pad, lo + pad, 'right', 'bottom'), 'tn': (hi - pad, hi - pad, 'right', 'top')}
        else:                      # 優秀：真實高、預測高 = TP（右上）
            spots = {'tp': (hi - pad, hi - pad, 'right', 'top'), 'fn': (hi - pad, lo + pad, 'right', 'bottom'),
                     'fp': (lo + pad, hi - pad, 'left', 'top'), 'tn': (lo + pad, lo + pad, 'left', 'bottom')}
        for key, (x, y, ha, va) in spots.items():
            ax.text(x, y, f'{key.upper()} {counts[key]:,}', ha=ha, va=va, fontsize=9,
                    fontweight='bold', color=P.INK_2, bbox=box)
        P.title(ax, title, sub)
        save(fig, name)

    def confusion(counts, summary, pos, neg, title, sub, name):
        mat = np.array([[counts['tp'], counts['fn']], [counts['fp'], counts['tn']]])
        share = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        ax.imshow(share, cmap=P.SEQUENTIAL, vmin=0, vmax=1)
        ax.grid(False)
        ax.axhline(0.5, color=P.SURFACE, linewidth=3)
        ax.axvline(0.5, color=P.SURFACE, linewidth=3)
        tags = [['TP', 'FN'], ['FP', 'TN']]
        for i in range(2):
            for j in range(2):
                color = 'white' if share[i, j] > 0.6 else P.INK
                ax.text(j, i - 0.1, f'{mat[i, j]:,}', ha='center', va='center',
                        fontsize=18, fontweight='bold', color=color)
                ax.text(j, i + 0.2, f'{tags[i][j]}，佔這一列 {share[i, j]:.0%}',
                        ha='center', va='center', fontsize=9, color=color)
        ax.set_xticks([0, 1], [f'預測：{pos}', f'預測：{neg}'])
        ax.set_yticks([0, 1], [f'實際：{pos}', f'實際：{neg}'])
        ax.tick_params(length=0, labelsize=10, labelcolor=P.INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        P.title(ax, title, sub)
        ax.text(0, -0.11, f"Precision {summary['precision']:.1%}    Recall {summary['recall']:.1%}    "
                          f"F1 {summary['f1']:.3f}    Accuracy {summary['accuracy']:.1%}",
                transform=ax.transAxes, ha='left', va='top', fontsize=10, color=P.INK)
        ax.text(0, -0.18, f"實際為「{pos}」的照片佔 {summary['positive_rate']:.1%}"
                          "（接近一半，Accuracy 不受類別不平衡影響）"
                if 0.3 <= summary['positive_rate'] <= 0.7 else
                f"實際為「{pos}」的照片佔 {summary['positive_rate']:.1%}（類別不平衡，Accuracy 參考價值低）",
                transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=P.MUTED)
        save(fig, name)

    def sweep(sw, current, xlabel, title, sub, name):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        # 判定為正的照片太少時，Precision 只是幾張照片的運氣，不畫那一段
        precision = np.where(sw['predicted_positive'] >= MIN_POSITIVES_FOR_PRECISION,
                             sw['precision'], np.nan)
        for values, label, color in ((precision, 'Precision', P.BLUE), (sw['recall'], 'Recall', P.ORANGE),
                                     (sw['f1'], 'F1', P.AQUA)):
            ax.plot(sw['threshold'], values, color=color, label=label)
        ax.axvline(current, color=P.INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f'目前門檻 {current:g}', (current, 0.02), xytext=(4, 0), textcoords='offset points',
                    ha='left', va='bottom', fontsize=9, color=P.INK_2)
        if np.isnan(precision).any() and not np.isnan(sw['precision']).all():
            # 放在 x 軸標題下方，圖內右下角會壓到門檻線與曲線
            P.note(ax, f'判定為正少於 {MIN_POSITIVES_FOR_PRECISION} 張的門檻不畫 Precision（樣本太少）',
                   x=1.0, y=-0.2, va='top')
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.set_xlabel(xlabel)
        ax.legend(loc='lower left', ncol=3)
        P.title(ax, title, sub)
        save(fig, name)

    def bins_chart(rows, unit, title, sub, name):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        labels = [f"{r['low']:g}-{r['high']:g}\nn={r['count']}" for r in rows]
        vals = [r['mean_error'] if r['count'] else 0 for r in rows]
        bars = ax.bar(labels, vals, width=0.5, color=P.BLUE)
        for bar, v, r in zip(bars, vals, rows):
            if not r['count']:
                continue
            up = v >= 0
            ax.annotate(f'{v:+.1f}' if unit == '分' else f'{v:+.2f}',
                        (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3 if up else -3),
                        textcoords='offset points', ha='center', va='bottom' if up else 'top',
                        fontsize=9, fontweight='bold', color=P.INK)
        ax.axhline(0, color=P.INK_2, linewidth=1)
        ax.set_ylabel(f'平均誤差（預測 - 真實，{unit}）')
        ax.margins(y=0.2)
        P.title(ax, title, sub)
        save(fig, name)

    tech = res.get('technical')
    ta = arr.get('technical')
    if tech and ta:
        reg = tech['regression']
        c = tech['warning']
        scatter(ta['true'], ta['pred'], cut=c['truth_cut'], thr=c['threshold'],
                direction='below', lim=(0, 100),
                xlabel='真實：KonIQ 人工評分（MOS）', ylabel='預測：技術分',
                title='技術模型：預測 vs 人工評分',
                sub=f"KonIQ 驗證集 {tech['n']:,} 張｜PLCC {reg['plcc']:.3f}  SRCC {reg['srcc']:.3f}",
                counts=c['counts'], name='tech_scatter.png')
        confusion(c['counts'], c['summary'], '差', '不差', '「警告」判定的混淆矩陣',
                  f"預測：{c['rule']}｜實際：{c['truth']}", 'tech_confusion.png')
        sweep(tech['sweep'], c['threshold'], '「警告」門檻（技術分低於多少就警告）',
              '「警告」門檻怎麼設：Precision / Recall 的取捨',
              '門檻往右：抓到更多差照片（Recall 上升），但誤報也變多', 'tech_threshold.png')
        bins_chart(tech['bins'], '分', '技術模型：各分數區間的預測偏差',
                   '正值 = 高估、負值 = 低估；依真實 MOS 分區間', 'tech_bins.png')

    aes = res.get('aesthetic')
    aa = arr.get('aesthetic')
    if aes and aa and 'skipped' not in aes:
        c = aes['excellent']
        reg = aes['regression']
        thr = c['threshold']
        cut = c['truth_cut']
        if 'pred_mean' in aa:
            scatter(aa['true_mean'], aa['pred_mean'], cut=cut, thr=thr / 100 * 9 + 1,
                    direction='above', lim=(1, 10),
                    xlabel='真實：AVA 群眾平均分（1~10）', ylabel='預測：評分期望值（1~10）',
                    title='美感模型：預測 vs 群眾評分',
                    sub=f"AVA 驗證集 {aes['n']:,} 張｜PLCC {reg['plcc']:.3f}  SRCC {reg['srcc']:.3f}"
                        f"｜門檻 {thr:g} 分 = {thr / 100 * 9 + 1:.2f}",
                    counts=c['counts'], name='aes_scatter.png')
        confusion(c['counts'], c['summary'], '優秀', '非優秀', '「優秀」判定的混淆矩陣',
                  f"預測：{c['rule']}｜實際：{c['truth']}", 'aes_confusion.png')
        sweep(aes['sweep'], thr, '「優秀」門檻（美感分高於多少算優秀）',
              '「優秀」門檻怎麼設：Precision / Recall 的取捨',
              '門檻往右：評為優秀的更可靠（Precision 上升），但漏掉的好照片更多', 'aes_threshold.png')
        if 'bins' in aes:
            bins_chart(aes['bins'], '1~10 分', '美感模型：各分數區間的預測偏差',
                       '正值 = 高估、負值 = 低估；依群眾平均分分區間', 'aes_bins.png')

        if 'fpr' in aa:
            fig, ax = plt.subplots(figsize=(5.6, 5.4))
            ax.plot([0, 1], [0, 1], color=P.MUTED, linewidth=1)
            ax.plot(aa['fpr'], aa['tpr'], color=P.BLUE)
            ax.fill_between(aa['fpr'], aa['tpr'], color=P.BLUE, alpha=0.1, linewidth=0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.grid(axis='both')
            ax.set_xlabel('False Positive Rate（壞照片被當成好的比例）')
            ax.set_ylabel('True Positive Rate（好照片被認出的比例）')
            P.note(ax, '對角線 = 亂猜（AUC 0.5）')
            P.title(ax, f"ROC 曲線：AUC {aes['binary']['auc']:.4f}",
                    '以 AVA 二元標籤為答案（群眾平均約 4.47 分為界）')
            save(fig, 'aes_roc.png')

        if 'emd' in aa:
            emd = aa['emd']
            order = np.argsort(emd)
            picks = [('EMD 最小：預測最準', order[0]), ('EMD 中位數：典型情況', order[len(order) // 2]),
                     ('EMD 最大：預測最差', order[-1])]
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, gridspec_kw={'wspace': 0.12})
            xs = np.arange(1, 11)
            for ax, (head, i) in zip(axes, picks):
                ax.bar(xs - 0.2, aa['true_dist'][i], width=0.38, color=P.BLUE, label='真實（群眾投票）')
                ax.bar(xs + 0.2, aa['pred_dist'][i], width=0.38, color=P.ORANGE, label='預測')
                ax.set_xticks(xs)
                ax.set_xlabel('評分（1~10）')
                P.title(ax, head, f"#{aa['names'][i]}｜平均 {aa['true_mean'][i]:.2f} vs "
                                  f"{aa['pred_mean'][i]:.2f}｜EMD {emd[i]:.3f}")
            axes[0].set_ylabel('機率')
            axes[0].legend(loc='upper left')
            save(fig, 'aes_distributions.png')

        comp = aes.get('compare') or []
        if len(comp) > 1:
            fig, ax = plt.subplots(figsize=(8.5, 1.4 + 0.7 * len(comp)))
            names = [c['label'] + ('（現行）' if c['current'] else '') for c in comp]
            colors = [P.BLUE if c['current'] else P.MUTED for c in comp]
            bars = ax.barh(names, [c['srcc'] for c in comp], height=0.32, color=colors)
            P.label_bars(ax, bars, [f"SRCC {c['srcc']:.4f}" for c in comp], horizontal=True)
            P.category_axis(ax, 'y')
            ax.invert_yaxis()
            ax.set_xlim(0, 1.15)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(axis='y', visible=False)
            ax.grid(axis='x', visible=True)
            ax.set_xlabel('SRCC（排序相關，越高越好）')
            P.title(ax, '美感模型新舊版比較',
                    f"同一份 AVA 驗證集 {aes['n']:,} 張，對照答案都是群眾平均分")
            save(fig, 'aes_compare.png')

    # 分數分佈：模型有沒有把分數擠在中間
    hist_panels = []
    if ta:
        hist_panels.append(('技術分（0~100）', ta['true'], ta['pred'], np.arange(0, 101, 4)))
    if aa and 'pred_mean' in aa:
        hist_panels.append(('美感評分（1~10）', aa['true_mean'], aa['pred_mean'], np.arange(1, 10.01, 0.25)))
    if hist_panels:
        fig, axes = plt.subplots(1, len(hist_panels), figsize=(6 * len(hist_panels), 4), squeeze=False)
        for ax, (head, true, pred, bins) in zip(axes[0], hist_panels):
            ax.hist(true, bins=bins, histtype='step', linewidth=2, color=P.BLUE, label='真實（人工評分）')
            ax.hist(pred, bins=bins, histtype='step', linewidth=2, color=P.ORANGE, label='預測')
            ax.set_xlabel(head)
            ax.set_ylabel('張數')
            P.title(ax, f'分數分佈：{head[:2]}',
                    f'真實標準差 {np.std(true):.2f}，預測標準差 {np.std(pred):.2f}')
        axes[0][0].legend(loc='upper left')
        save(fig, 'score_hist.png')

    iss = res.get('issues')
    if iss and ta and arr.get('issue_tags') is not None:
        rows = iss['rows']
        mos = ta['true']
        tags = arr['issue_tags']
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ticks = []
        for i, r in enumerate(rows):
            flagged = np.array([r['issue'] in t for t in tags])
            common_kw = dict(widths=0.3, patch_artist=True, showfliers=False,
                             medianprops=dict(color=P.INK, linewidth=1.5),
                             whiskerprops=dict(color=P.MUTED), capprops=dict(color=P.MUTED))
            if flagged.any():
                b = ax.boxplot([mos[flagged]], positions=[i - 0.18], **common_kw)
                b['boxes'][0].set(facecolor=P.BLUE, edgecolor=P.SURFACE)
            else:
                ax.text(i - 0.18, np.median(mos), '0 張', ha='center', va='center', fontsize=9, color=P.MUTED)
            b = ax.boxplot([mos[~flagged]], positions=[i + 0.18], **common_kw)
            b['boxes'][0].set(facecolor=P.DEEMPH, edgecolor=P.SURFACE)
            ticks.append(f"{r['issue']}\n標記 {r['flagged']} 張（{r['rate']:.0%}）")
        ax.set_xticks(range(len(rows)), ticks)
        P.category_axis(ax, 'x')
        ax.set_ylabel('KonIQ 人工評分（MOS）')
        # 上方留空給圖例，避免壓到盒鬚（MOS 最高約 90）
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
        ax.legend(handles=[Patch(color=P.BLUE, label='有被標記'), Patch(color=P.DEEMPH, label='沒被標記')],
                  loc='upper left', ncol=2)
        P.title(ax, '影像量測細項：被標記的照片，人工評分真的比較低嗎？',
                f"KonIQ 驗證集 {iss['n']:,} 張；盒中橫線是中位數；雜訊不列入（原因見 summary.md）")
        save(fig, 'issues_mos.png')

    return made


# ==========================================
# 摘要與輸出
# ==========================================
def _clean(obj):
    """轉成標準 JSON：numpy 型別轉成 Python 型別，nan 轉成 null（標準 JSON 沒有 NaN）。"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    return obj


def pct(v):
    return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.1%}'


def confusion_md(c, pos, neg):
    k, s = c['counts'], c['summary']
    return [f"預測：{c['rule']}；實際：{c['truth']}", '',
            f'| | 預測：{pos} | 預測：{neg} |', '|---|---|---|',
            f"| **實際：{pos}** | TP {k['tp']:,} | FN {k['fn']:,} |",
            f"| **實際：{neg}** | FP {k['fp']:,} | TN {k['tn']:,} |", '',
            f"Accuracy {pct(s['accuracy'])}｜Precision {pct(s['precision'])}｜Recall {pct(s['recall'])}｜"
            f"F1 {s['f1']:.3f}｜實際為「{pos}」的比例 {pct(s['positive_rate'])}", '']


def write_summary(res, out, charts):
    L = [f"# 模型成效報表（{res['generated']}）", '']
    if res['quick']:
        L += [f'> **快速模式**：每個驗證集只取前 {QUICK_LIMIT} 張，數字只能用來確認流程，不可寫進報告。', '']
    L += [f"裝置 {res['device']}｜PyTorch {res['torch']}｜權重：美感 `{res['weights']['aesthetic']}`、"
          f"技術 `{res['weights']['technical']}`", '']

    t = res.get('technical')
    if t:
        r = t['regression']
        L += ['## 技術模型', '', f"驗證集 `{t['dataset']}`，{t['n']:,} 張。", '',
              '| PLCC | SRCC | KRCC | RMSE | MAE | 平均誤差 |', '|---|---|---|---|---|---|',
              f"| {r['plcc']:.4f} | {r['srcc']:.4f} | {r['krcc']:.4f} | {r['rmse']:.2f} 分 | "
              f"{r['mae']:.2f} 分 | {r['mean_error']:+.2f} 分 |", '',
              '### 「警告」判定', ''] + confusion_md(t['warning'], '差', '不差')
        L += ['### 各分數區間的偏差', '', '| 真實 MOS 區間 | 張數 | 平均誤差 | MAE |', '|---|---|---|---|']
        L += [f"| {b['low']:g}-{b['high']:g} | {b['count']} | "
              + (f"{b['mean_error']:+.2f} | {b['mae']:.2f} |" if b['count'] else '- | - |') for b in t['bins']]
        L.append('')

    a = res.get('aesthetic')
    if a and 'skipped' not in a:
        r = a['regression']
        L += ['## 美感模型', '', f"驗證集 `{a['dataset']}`，{a['n']:,} 張，權重 `{a['weights']}`（{a['mode']} 模式）。", '']
        head = ['PLCC', 'SRCC', 'KRCC']
        vals = [f"{r['plcc']:.4f}", f"{r['srcc']:.4f}", f"{r['krcc']:.4f}"]
        if 'rmse' in r:
            head += ['RMSE', 'MAE', '平均誤差', 'EMD']
            vals += [f"{r['rmse']:.3f}", f"{r['mae']:.3f}", f"{r['mean_error']:+.3f}", f"{a['emd']['mean']:.4f}"]
        if 'binary' in a:
            head.append('AUC')
            vals.append(f"{a['binary']['auc']:.4f}")
        L += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head), '| ' + ' | '.join(vals) + ' |', '']
        if 'rmse' in r:
            L += ['RMSE、MAE 為 1~10 分尺度。EMD 採' + a['emd']['definition'] + '。', '']
        if 'binary' in a:
            L += [f"AUC 以{a['binary']['definition']}為答案。", '']
        L += ['### 「優秀」判定', ''] + confusion_md(a['excellent'], '優秀', '非優秀')
        if 'bins' in a:
            L += ['### 各分數區間的偏差', '', '| 群眾平均分區間 | 張數 | 平均誤差 | MAE |', '|---|---|---|---|']
            L += [f"| {b['low']:g}-{b['high']:g} | {b['count']} | "
                  + (f"{b['mean_error']:+.3f} | {b['mae']:.3f} |" if b['count'] else '- | - |') for b in a['bins']]
            L.append('')
        if len(a.get('compare') or []) > 1:
            L += ['### 新舊版比較', '', '同一份驗證集、同一個對照答案（群眾平均分）。比較用未裁切的原始分數。', '',
                  '| 模型 | 權重 | SRCC | PLCC | 超出 0~100 的比例 |', '|---|---|---|---|---|']
            L += [f"| {c['label']}{'（現行）' if c['current'] else ''} | `{c['weights']}` | {c['srcc']:.4f} | "
                  f"{c['plcc']:.4f} | {pct(c['clipped_share'])} |" for c in a['compare']]
            L.append('')

    i = res.get('issues')
    if i:
        L += ['## 影像量測細項', '', f"KonIQ 驗證集 {i['n']:,} 張，至少被標記一項的有 {i['any_flagged']:,} 張。"
              f"{i['note']}。", '',
              '| 項目 | 被標記 | MOS 中位數（被標記 / 其餘） | MOS 平均（被標記 / 其餘） | p 值 | 被標記的照片 |',
              '|---|---|---|---|---|---|']
        for r in i['rows']:
            if r['flagged']:
                p = r['p_value']
                if p is None or p >= 0.05:
                    verdict = '無顯著差異'
                elif r['mos_flagged_median'] < r['mos_rest_median']:
                    verdict = '評分較低'
                else:
                    verdict = '**評分反而較高**'
                L.append(f"| {r['issue']} | {r['flagged']} 張（{pct(r['rate'])}） | "
                         f"{r['mos_flagged_median']:.1f} / {r['mos_rest_median']:.1f} | "
                         f"{r['mos_flagged_mean']:.1f} / {r['mos_rest_mean']:.1f} | "
                         f"{'-' if p is None else f'{p:.2g}'} | {verdict} |")
            else:
                L.append(f"| {r['issue']} | 0 張 | - | - | - | - |")
        L += ['', 'p 值為 Mann-Whitney U 檢定（雙尾），小於 0.05 代表兩組的人工評分有顯著差異。'
              '「評分反而較高」代表這項量測標記的不是品質問題，不宜當成缺陷提示。', '']

    L += ['## 限制', '']
    if a and 'middle_share' in a:
        lo, hi = a['middle_share']['range']
        L.append(f"- AVA 驗證集的群眾評分集中在兩端（約 3~4 分與 7 分），平均分落在 {lo:g}~{hi:g} 的照片"
                 f"只佔 {pct(a['middle_share']['share'])}。最難分辨的中段照片很少，美感模型的 Accuracy、"
                 'AUC 會比在一般照片上高；SRCC 受影響較小，但也不能直接和論文在完整 AVA 上的數字比較。')
    L += ['- 混淆矩陣的「真實答案」切點是自己定的（MOS 60、群眾平均 6），換切點矩陣就會變。',
          '- 沒有「優秀／正常／警告」三類的混淆矩陣：沒有資料集同時有美感與技術兩種人工評分。',
          '- 兩個驗證集都是網路照片，不是自己相機拍的原檔。',
          '- 沒有訓練曲線：訓練腳本沒有保存每個 epoch 的 loss。', '']
    if charts:
        L += ['## 圖表', ''] + [f'![{c}]({c})' for c in charts] + ['']
    (out / 'summary.md').write_text('\n'.join(L), encoding='utf-8')


def write_predictions(arr, out):
    import pandas as pd
    ta = arr.get('technical')
    if ta:
        df = pd.DataFrame({'image': ta['names'], 'true_mos': ta['true'].round(2),
                           'technical_score': ta['pred'].round(2),
                           'warning': ta['pred_pos'], 'truly_bad': ta['true_pos']})
        if ta['issues']:
            df['technical_issues'] = ['；'.join(x) for x in ta['issues']]
        df.to_csv(out / 'predictions_technical.csv', index=False, encoding='utf-8-sig')
    aa = arr.get('aesthetic')
    if aa:
        df = pd.DataFrame({'image': aa['names'], 'true_mean': aa['true_mean'].round(3),
                           'aesthetic_score': aa['pred'].round(2),
                           'excellent': aa['pred_pos'], 'truly_excellent': aa['true_pos']})
        if 'pred_mean' in aa:
            df['pred_mean'] = aa['pred_mean'].round(3)
            df['emd'] = aa['emd'].round(4)
        df.to_csv(out / 'predictions_aesthetic.csv', index=False, encoding='utf-8-sig')


# ==========================================
# 主程式
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description='模型成效報表：準確率、混淆矩陣、細項分析與圖表')
    p.add_argument('--quick', action='store_true',
                   help=f'每個驗證集只取前 {QUICK_LIMIT} 張，確認流程用，數字不可寫進報告')
    p.add_argument('--out', metavar='資料夾', help='輸出資料夾（預設 reports/model_report/<時間>）')
    p.add_argument('--tech-csv', default='data/val_tech.csv')
    p.add_argument('--tech-dir', default='data/koniq/512x384')
    p.add_argument('--aes-csv', default='data/ava_val.csv')
    p.add_argument('--aes-dir', default='data/dataset')
    p.add_argument('--tech-truth-cut', type=float, default=60.0,
                   help='技術：人工評分 MOS 低於多少算「真的差」（預設 60）')
    p.add_argument('--aes-truth-cut', type=float, default=6.0,
                   help='美感：群眾平均分高於多少算「真的優秀」（預設 6）')
    p.add_argument('--batch-size', type=int, default=64,
                   help='一次推論幾張（預設 64）。GPU 批次運算與逐張會有約 0.1 分的數值差；'
                        '設為 1 可與 evaluate_photo() 逐位元一致，但比較慢')
    p.add_argument('--no-compare', action='store_true', help='不比較新舊美感權重')
    p.add_argument('--no-issues', action='store_true', help='不做影像量測細項分析')
    p.add_argument('--no-charts', action='store_true', help='不產生圖表')
    return p.parse_args()


def main():
    args = parse_args()
    for attr in ('tech_csv', 'tech_dir', 'aes_csv', 'aes_dir'):
        path = Path(getattr(args, attr))
        setattr(args, attr, path if path.is_absolute() else BASE_DIR / path)
    have_tech = args.tech_csv.exists() and args.tech_dir.is_dir()
    have_aes = args.aes_csv.exists() and args.aes_dir.is_dir()
    if not (have_tech or have_aes):
        print('[FAIL] 找不到任何驗證集。需要以下其中一組：')
        print(f'       技術：{args.tech_csv} 與 {args.tech_dir}')
        print(f'       美感：{args.aes_csv} 與 {args.aes_dir}')
        return 1

    try:
        import ai_inference as ai
    except FileNotFoundError as e:
        print(f'[FAIL] {e}')
        return 1
    import torch

    out = Path(args.out) if args.out else (
        BASE_DIR / 'reports' / 'model_report' / datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    if args.quick:
        print(f'[INFO] 快速模式：每個驗證集只取前 {QUICK_LIMIT} 張，數字不可寫進報告')

    res = {'generated': datetime.now().isoformat(timespec='seconds'), 'quick': args.quick,
           'device': str(ai.DEVICE), 'torch': torch.__version__,
           'weights': {'aesthetic': ai.AES_WEIGHTS.name, 'technical': ai.TECH_WEIGHTS.name},
           'thresholds': {'tech_warning': ai.TECH_ISSUE_THRESHOLD,
                          'aes_excellent': ai.AESTHETIC_EXCELLENT_THRESHOLD,
                          'tech_truth_cut': args.tech_truth_cut, 'aes_truth_cut': args.aes_truth_cut}}
    arr = {}

    if have_tech:
        res['technical'], arr['technical'] = technical_section(ai, args)
        if arr['technical']['issues']:
            res['issues'], arr['issue_tags'] = issues_section(arr['technical'])
    else:
        print(f'[WARN] 找不到技術驗證集（{args.tech_csv}），略過')
    if have_aes:
        res['aesthetic'], arr['aesthetic'] = aesthetic_section(ai, args)
    else:
        print(f'[WARN] 找不到美感驗證集（{args.aes_csv}），略過')

    charts = [] if args.no_charts else make_charts(res, arr, out)
    write_summary(res, out, charts)
    write_predictions(arr, out)
    with open(out / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(_clean(res), f, ensure_ascii=False, indent=2)

    print(f'\n[ OK ] 已寫入 {out}')
    print('       summary.md、metrics.json、predictions_*.csv'
          + (f'、{len(charts)} 張圖表' if charts else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
