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

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])          # Global Average Pooling
        x = self.classifier(x)
        if self.output_mode == 'single':
            # 用 squeeze(-1) 而非 squeeze()：
            # squeeze() 會把「所有」長度為 1 的維度都壓掉，當 batch size 剛好是 1 時
            # 形狀 (1, 1) 會變成 0 維純量，與形狀 (1,) 的標籤做 MSELoss 會觸發廣播，
            # loss 算錯且只會跳出 UserWarning。
            # squeeze(-1) 只壓最後一維，(N, 1) -> (N,)，batch size 為 1 時也安全。
            x = x.squeeze(-1)
        return x


def emd_loss(pred, target, r=2):
    """
    Earth Mover's Distance（推土機距離）。
    對 CDF 差值直接取 mean，與 NIMA 論文定義一致。
    僅在 output_mode='distribution' 時使用。
    """
    cdf_pred = torch.cumsum(pred, dim=1)
    cdf_target = torch.cumsum(target, dim=1)
    return torch.mean(torch.pow(torch.abs(cdf_pred - cdf_target), r))
