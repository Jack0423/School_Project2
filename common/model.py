"""
NIMA 模型架構的唯一定義。

稽核前這個 class 在三個檔案裡各有一份（ai_inference.py、train_nima.py、
train_tech.py），而且內容不一致：訓練用的兩份載入 ImageNet 預訓練權重，
推論那份沒有。目前因為推論時會完整載入 .pth 而不影響結果，
但只要有人改動其中一份架構，另外兩份就會默默不同步。
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class NIMABaseline(nn.Module):
    """
    以 MobileNetV2 為骨幹的 NIMA 模型。

    參數:
        output_mode:
            'single'       輸出單一數值（搭配 MSELoss）
            'distribution' 輸出 10 級分數的 Softmax 機率（搭配 emd_loss）
        pretrained:
            True  載入 ImageNet 預訓練權重。訓練時使用。
            False 不載入。推論時使用——反正稍後會用 load_state_dict()
                  完整覆蓋所有權重，先下載預訓練權重只是浪費時間與流量。
    """

    def __init__(self, output_mode='single', pretrained=False):
        super().__init__()
        self.output_mode = output_mode

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features

        if output_mode == 'distribution':
            self.classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(1280, 10),
                nn.Softmax(dim=1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(1280, 1),
            )

    def forward(self, x, return_features=False):
        """
        return_features=True 時回傳 (輸出, 特徵)。

        特徵是 Global Average Pooling 之後、分類頭之前的 1280 維向量，
        給相似照片比對用（batch_pipeline.py / photo_grouping.py）。
        同一次前向運算順便取出，不必為了特徵再跑一次骨幹。

        預設 False，回傳值與過去完全相同——訓練、評估、推論的既有呼叫都不受影響。
        """
        features = self.features(x).mean([2, 3])    # Global Average Pooling
        x = self.classifier(features)
        if self.output_mode == 'single':
            # 用 squeeze(-1) 而非 squeeze()：
            # squeeze() 會把「所有」長度為 1 的維度都壓掉，當 batch size 剛好是 1 時
            # 形狀 (1, 1) 會變成 0 維純量，與形狀 (1,) 的標籤做 MSELoss 會觸發廣播，
            # loss 算錯且只會跳出 UserWarning。
            # squeeze(-1) 只壓最後一維，(N, 1) -> (N,)，batch size 為 1 時也安全。
            x = x.squeeze(-1)
        if return_features:
            return x, features
        return x


def emd_loss(pred, target, r=2):
    """
    Earth Mover's Distance（推土機距離），訓練用的 loss。
    對 CDF 差值的 r 次方直接取 mean，整個 batch 一起平均。
    僅在 output_mode='distribution' 時使用。

    與 NIMA 論文的定義差在最後沒有開 1/r 次方（論文逐張開根號再平均）。
    兩者縮小的是同一個 CDF 差距，只是各張照片的加權方式不同；
    數值不能和論文或其他專案直接比較，報表要用論文定義時請用
    common/metrics.py 的 emd_per_sample。
    """
    cdf_pred = torch.cumsum(pred, dim=1)
    cdf_target = torch.cumsum(target, dim=1)
    return torch.mean(torch.pow(torch.abs(cdf_pred - cdf_target), r))
