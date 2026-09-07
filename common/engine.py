"""
訓練／驗證迴圈的唯一定義。

稽核前 train_nima.py 與 train_tech.py 各有一份 run_epoch，約 90% 相同。
兩份分岔的後果實際發生過：train_tech.py 用 np.atleast_1d 去修
batch size = 1 時張量被壓成 0 維的問題，train_nima.py 卻沒有跟著修。
（根因已在 common/model.py 以 squeeze(-1) 解決。）
"""
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def to_mean_score(tensor, output_mode):
    """
    把模型輸出／標籤轉成單一分數，方便計算相關係數。
    distribution 模式下取期望值 sum(i * prob)，i 從 1 到 10。
    """
    if output_mode == 'distribution':
        scores = torch.arange(1, 11, device=tensor.device, dtype=tensor.dtype)
        return (tensor * scores).sum(dim=1)
    return tensor


def run_epoch(model, loader, criterion, device, optimizer=None, is_train=True,
              output_mode='single', collect_metrics=False):
    """
    統一的 epoch 執行函式，train / val 共用。

    - is_train=True  需傳入 optimizer，會執行反向傳播。
    - is_train=False 不需 optimizer，純推論模式。
    - collect_metrics=True 額外回傳 PLCC / SRCC（建議只在驗證時開啟）。
      用相關係數而非單純的 loss 來挑選模型，是因為影像品質評估的目標是
      「排序」而非絕對數值，相關係數更貼近人類評分行為。
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()

    all_preds, all_labels = [], []

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if collect_metrics:
                all_preds.append(
                    np.atleast_1d(to_mean_score(outputs.detach(), output_mode).cpu().numpy())
                )
                all_labels.append(
                    np.atleast_1d(to_mean_score(labels.detach(), output_mode).cpu().numpy())
                )

    avg_loss = total_loss / len(loader)

    if collect_metrics:
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_labels)
        plcc, _ = pearsonr(preds, targets)
        srcc, _ = spearmanr(preds, targets)
        return avg_loss, plcc, srcc

    return avg_loss
