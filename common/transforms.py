"""
影像前處理的唯一定義。

稽核前這些設定散在專案的 7 個位置各自寫死，而且**訓練與推論不一致**：

    訓練驗證  Resize(256) + CenterCrop(224)   保持長寬比
    實際推論  Resize((224, 224))              直接壓扁長寬比

兩者的輸入分佈不同，代表線上分數會系統性偏離驗證集測得的指標。
實測（技術模型，KonIQ 驗證集 2,015 張，同一組權重、只換前處理）：

    Resize((224,224))           PLCC 0.8056   SRCC 0.7636
    Resize(256)+CenterCrop(224) PLCC 0.8315   SRCC 0.7953   <- 統一後
    差異                             +0.0258       +0.0316

也就是說，光是把推論的前處理對齊訓練驗證，不需重新訓練就能拿回
約 0.03 的相關係數。這是統一管理這些設定的直接理由。
"""
from torchvision import transforms

# ImageNet 統計值。骨幹是 ImageNet 預訓練的 MobileNetV2，必須沿用同一組。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 模型輸入尺寸，以及縮放時的中間尺寸
INPUT_SIZE = 224
RESIZE_SIZE = 256

_NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def build_transform(mode):
    """
    參數:
        mode:
            'eval'            驗證與推論共用。保持長寬比縮放後置中裁切。
            'train_aesthetic' 美感模型訓練用。較強的增強：隨機裁切範圍大、
                              加上色彩抖動——構圖與色調的變化正是美感模型
                              該學會容忍的。
            'train_technical' 技術模型訓練用。較保守：用 RandomCrop 而非
                              RandomResizedCrop 以保留原始像素的清晰度，
                              且不做色彩抖動——技術品質評估對這些很敏感，
                              增強過頭會把要偵測的失真本身給洗掉。
    """
    if mode == 'eval':
        return transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            _NORMALIZE,
        ])

    if mode == 'train_aesthetic':
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            _NORMALIZE,
        ])

    if mode == 'train_technical':
        return transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _NORMALIZE,
        ])

    raise ValueError(
        f"未知的前處理模式：{mode!r}"
        f"｜可用的模式：'eval'、'train_aesthetic'、'train_technical'"
    )
