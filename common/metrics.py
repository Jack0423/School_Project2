"""
評估指標的唯一定義。只依賴 numpy / scipy，不載入模型。

model_report.py 用它把模型分數整理成準確率、混淆矩陣等數據。
獨立成一個模組是為了能在沒有權重、沒有資料集的機器上單獨測試
（tests/test_metrics.py）——指標算錯不會報錯，只會安靜地給出錯的數字。

判定方向必須與 ai_inference.py 完全一致，否則報表量的就不是系統實際的行為：

    警告  技術分 <  TECH_ISSUE_THRESHOLD           -> direction='below'
    優秀  美感分 >  AESTHETIC_EXCELLENT_THRESHOLD  -> direction='above'

兩者都是「嚴格」不等式，分數剛好等於門檻時不算。
"""
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def predict_positive(scores, threshold, direction):
    """依門檻把連續分數切成「判定為正」的布林陣列。方向見模組說明。"""
    scores = np.asarray(scores, dtype=float)
    if direction == 'above':
        return scores > threshold
    if direction == 'below':
        return scores < threshold
    raise ValueError(f"direction 只能是 'above' 或 'below'，收到 {direction!r}")


def confusion_counts(pred_pos, true_pos):
    """2x2 混淆矩陣的四格。回傳 dict：tp / fn / fp / tn。"""
    pred_pos = np.asarray(pred_pos, dtype=bool)
    true_pos = np.asarray(true_pos, dtype=bool)
    if pred_pos.shape != true_pos.shape:
        raise ValueError(f'預測與答案的形狀不同：{pred_pos.shape} vs {true_pos.shape}')
    return {
        'tp': int(np.sum(pred_pos & true_pos)),
        'fn': int(np.sum(~pred_pos & true_pos)),
        'fp': int(np.sum(pred_pos & ~true_pos)),
        'tn': int(np.sum(~pred_pos & ~true_pos)),
    }


def _ratio(num, den):
    # 分母為 0 時回 nan 而不是 0：「沒有任何樣本可算」和「算出來是 0%」是兩回事
    return num / den if den else float('nan')


def classification_summary(counts):
    """
    由混淆矩陣四格算出常用指標。

    positive_rate 是「實際為正」的比例（基準率）。一併回傳是因為投影片 2-8
    的提醒：類別不平衡時 Accuracy 會騙人（99% 都是負樣本，全猜負也有 99%），
    看 Accuracy 之前要先看這個數字。
    """
    tp, fn, fp, tn = counts['tp'], counts['fn'], counts['fp'], counts['tn']
    n = tp + fn + fp + tn
    return {
        'n': n,
        'accuracy': _ratio(tp + tn, n),
        'precision': _ratio(tp, tp + fp),
        'recall': _ratio(tp, tp + fn),
        # 直接用 2TP / (2TP+FP+FN)，不經過 precision 與 recall：
        # 兩者之一為 nan 時仍能給出正確結果（例如完全沒有判定為正的樣本）
        'f1': _ratio(2 * tp, 2 * tp + fp + fn),
        'specificity': _ratio(tn, tn + fp),
        'positive_rate': _ratio(tp + fn, n),
        'predicted_positive_rate': _ratio(tp + fp, n),
    }


def threshold_sweep(scores, true_pos, thresholds, direction):
    """
    把門檻從頭掃到尾，看 Precision / Recall / F1 怎麼變。
    用來回答「門檻設在這裡合不合理、往哪邊調會發生什麼事」。

    另外回傳每個門檻下「判定為正」的張數：門檻推到極端時只剩幾張，
    Precision 會因為樣本太少而劇烈跳動，畫圖時要據此略過那一段。
    """
    rows = {'threshold': [], 'precision': [], 'recall': [], 'f1': [], 'predicted_positive': []}
    for t in thresholds:
        counts = confusion_counts(predict_positive(scores, t, direction), true_pos)
        s = classification_summary(counts)
        rows['threshold'].append(float(t))
        rows['predicted_positive'].append(counts['tp'] + counts['fp'])
        for key in ('precision', 'recall', 'f1'):
            rows[key].append(s[key])
    return {k: np.array(v, dtype=float) for k, v in rows.items()}


def roc_curve(scores, labels):
    """
    ROC 曲線，回傳 (fpr, tpr)，兩端各含 (0, 0) 與 (1, 1)。

    門檻由高往低移，同分的樣本視為同一個門檻、一起跨過去——
    若逐一跨過，同分樣本的排列順序就會影響曲線形狀與 AUC。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError('ROC 需要正負樣本都至少一個')

    order = np.argsort(-scores, kind='mergesort')
    s = scores[order]
    y = labels[order]
    tps = np.cumsum(y)
    fps = np.cumsum(~y)
    # 只在分數改變的位置取點（每一段同分的最後一個樣本）
    last_of_tie = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0.0, tps[last_of_tie] / n_pos]
    fpr = np.r_[0.0, fps[last_of_tie] / n_neg]
    return fpr, tpr


def auc_from_roc(fpr, tpr):
    """梯形法積分。不用 np.trapezoid：它在 numpy 2.0 才出現，舊版叫 np.trapz。"""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def regression_summary(pred, true):
    """
    連續分數的指標。影像品質評估以 SRCC（排序相關）為主，
    PLCC（線性相關）為輔；RMSE / MAE 與分數尺度有關，只能在同一尺度內比較。
    mean_error 是 預測 - 真實 的平均：正值代表整體高估，負值代表低估。
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    err = pred - true
    return {
        'n': int(len(pred)),
        'plcc': float(pearsonr(pred, true)[0]),
        'srcc': float(spearmanr(pred, true)[0]),
        'krcc': float(kendalltau(pred, true)[0]),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'mean_error': float(np.mean(err)),
    }


def emd_per_sample(pred_dist, true_dist, r=2):
    """
    逐張計算評分分佈之間的 EMD（NIMA 論文的定義）：

        EMD = ( mean_k |CDF_pred(k) - CDF_true(k)|^r ) ^ (1/r)

    注意與 common/model.py 的 emd_loss 不同：那是訓練用的 loss，
    沒有開 1/r 次方，而且對整個 batch 一起取平均。兩者的數值不能直接比較；
    要和論文或其他專案比較時用這一個。
    """
    pred = np.asarray(pred_dist, dtype=float)
    true = np.asarray(true_dist, dtype=float)
    diff = np.abs(np.cumsum(pred, axis=1) - np.cumsum(true, axis=1))
    return np.mean(diff ** r, axis=1) ** (1.0 / r)


def binned_error(pred, true, edges):
    """
    依「真實分數」分區間，看每個區間的預測偏差。

    回歸模型常見的毛病是往平均值縮：低分高估、高分低估。
    整體的相關係數看不出這件事，分區間的平均誤差一眼就看得出來。
    區間為左閉右開，最後一個區間包含右端點。
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    edges = np.asarray(edges, dtype=float)
    idx = np.clip(np.searchsorted(edges, true, side='right') - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        err = pred[mask] - true[mask]
        rows.append({
            'low': float(edges[b]),
            'high': float(edges[b + 1]),
            'count': int(mask.sum()),
            'mean_error': float(err.mean()) if mask.any() else float('nan'),
            'mae': float(np.abs(err).mean()) if mask.any() else float('nan'),
        })
    return rows
