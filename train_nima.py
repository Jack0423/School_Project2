import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 模型架構、前處理、訓練迴圈一律來自 common/，
# 不再在每支檔案各寫一份（稽核前這三者已經彼此不一致）。
from common import NIMABaseline, emd_loss, build_transform, run_epoch

# ==========================================
# 1. 資料集定義
# ==========================================
class AVADataset(Dataset):
    """
    支援兩種 CSV 格式：
      * 單一標籤：欄位為 [image, label]
      * 分佈標籤：欄位為 [image, score_1, ..., score_10]（AVA 原始格式）
    透過 mode 參數切換。
    """
    def __init__(self, csv_file, img_dir, transform=None, mode='single'):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

        # 檔名在這裡就從「欄位」算好，不在 __getitem__ 裡用 row['image'] 組。
        #
        # 原因：df.iloc[i] 回傳的 Series 會把整列轉成共同型別。
        # 當 CSV 同時含有整數欄與浮點欄（例如加上 mean_score 與 score_1~10 之後），
        # 整數的 image 欄會被轉成 float64，f"{row['image']}.jpg" 就變成
        # "53.0.jpg" 而不是 "53.jpg"，於是每一張都找不到檔案。
        # 直接取用 self.df['image'] 這一欄則能保留原本的型別。
        self.filenames = [
            f'{name}.jpg' if not str(name).lower().endswith('.jpg') else str(name)
            for name in self.df['image'].tolist()
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, self.filenames[idx])

        # 讀不到就直接拋錯，不用全黑圖代替。
        #
        # 原本的寫法是 `except Exception: image = Image.new('RGB', (224, 224))`，
        # 讀取失敗時靜默塞一張全黑圖。這會讓整個訓練在完全錯誤的資料上跑完
        # 而毫無徵兆——實測就發生過：因為上面那個 float64 檔名問題，
        # 5,600 張全部讀不到、全部變成全黑圖，訓練照樣跑完 5 個 epoch、
        # 印出看似合理的 loss，只有相關係數變成 nan 才透露出異常。
        # 檔案讀不到幾乎都是系統性問題（路徑錯、型別錯、資料沒下載完），
        # 早點大聲失敗遠比安靜地訓練出一個垃圾模型好。
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(
                f'無法讀取訓練影像：{img_path}'
                f'｜{type(e).__name__}: {e}'
                f'｜請確認 {self.img_dir} 內有這個檔案，且 CSV 的 image 欄位正確'
            ) from e

        if self.mode == 'distribution':
            scores = torch.tensor(
                [float(row[f'score_{i}']) for i in range(1, 11)], dtype=torch.float32
            )
            label = scores / scores.sum()  # 正規化成機率分佈
        else:
            label = torch.tensor(float(row['label']), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# 2. 主程式
# ==========================================
def parse_args():
    """
    超參數與檔案路徑改為命令列參數。

    原本全部寫死在 main() 裡，導致要跑兩組對照實驗（例如二元標籤 vs
    評分分佈）就得改原始碼，改完還可能忘了改回去，而且 SAVE_PATH 寫死成
    nima_best.pth——任何一次實驗都會直接覆蓋正在服役的權重。
    """
    p = argparse.ArgumentParser(description='訓練美感評分模型')
    p.add_argument('--mode', choices=['single', 'distribution'], default='single',
                   help="single：二元標籤配 MSELoss。"
                        "distribution：1~10 級評分分佈配 emd_loss，"
                        "輸出天然落在 1~10，不需裁切")
    p.add_argument('--train-csv', default='data/train.csv')
    p.add_argument('--val-csv', default='data/val.csv')
    p.add_argument('--img-dir', default='data/dataset')
    p.add_argument('--save', default='nima_best.pth', help='權重輸出路徑')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr-features', type=float, default=1e-5,
                   help='預訓練層學習率（較小，避免破壞預訓練特徵）')
    p.add_argument('--lr-head', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=5, help='Early Stopping 容忍 epoch 數')
    p.add_argument('--force', action='store_true',
                   help='允許覆寫已存在的權重檔')
    return p.parse_args()


def main():
    args = parse_args()
    EPOCHS       = args.epochs
    BATCH_SIZE   = args.batch_size
    LR_FEATURES  = args.lr_features
    LR_HEAD      = args.lr_head
    WEIGHT_DECAY = args.weight_decay
    PATIENCE     = args.patience
    OUTPUT_MODE  = args.mode
    SAVE_PATH    = args.save

    # 權重檔是訓練數小時的產物，覆寫掉就沒了，因此預設拒絕覆蓋。
    if os.path.exists(SAVE_PATH) and not args.force:
        print(f"⛔ 權重檔已存在：{SAVE_PATH}")
        print(f"   訓練會覆蓋它且無法復原。請改用 --save 指定其他檔名，"
              f"或確認後加上 --force。")
        return 1

    print(f"📋 模式={OUTPUT_MODE}  訓練集={args.train_csv}  驗證集={args.val_csv}")
    print(f"   輸出={SAVE_PATH}  epochs={EPOCHS}  batch={BATCH_SIZE}")

    # 硬體設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"✅ 設備確認: {device_name}")

    # pin_memory 只在有 GPU 時才有意義
    pin_memory = torch.cuda.is_available()

    # Windows 環境 num_workers > 0 容易觸發多進程問題
    num_workers = 0 if os.name == 'nt' else 2

    # 前處理一律由 common/transforms.py 提供，確保與推論端完全一致
    train_transform = build_transform('train_aesthetic')
    val_transform = build_transform('eval')

    # 資料集與 DataLoader
    train_ds = AVADataset(args.train_csv, args.img_dir, train_transform, mode=OUTPUT_MODE)
    val_ds   = AVADataset(args.val_csv,   args.img_dir, val_transform,   mode=OUTPUT_MODE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # 模型
    # pretrained=True：訓練時從 ImageNet 預訓練權重出發
    model = NIMABaseline(output_mode=OUTPUT_MODE, pretrained=True).to(device)

    # 分層學習率：預訓練層與分類頭使用不同 LR；改用 AdamW 加入 weight decay 抑制過擬合
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(),   'lr': LR_FEATURES},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    # 損失函數
    criterion = emd_loss if OUTPUT_MODE == 'distribution' else nn.MSELoss()

    # ── 訓練迴圈 ────────────────────────────────
    print("🚀 開始訓練...\n")
    # 模型選擇改用 PLCC+SRCC（越高越好），比單看 MSE 更貼近人類評分排序
    best_val_metric  = -float('inf')
    patience_counter = 0

    # Cosine Annealing：放在迴圈外，T_max 對應總 epoch 數
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer,
                                is_train=True, output_mode=OUTPUT_MODE)
        val_loss, plcc, srcc = run_epoch(model, val_loader, criterion, device, is_train=False,
                                          output_mode=OUTPUT_MODE, collect_metrics=True)
        val_metric = (plcc + srcc) / 2

        current_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch [{epoch:>2}/{EPOCHS}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PLCC: {plcc:.4f} | SRCC: {srcc:.4f} | LR: {current_lr:.2e}")

        # 儲存最佳模型（依相關係數挑選，而非單純 val_loss 最低）
        if val_metric > best_val_metric:
            best_val_metric  = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  💾 PLCC/SRCC 改善，模型已存至 {SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  ⏳ 未改善 ({patience_counter}/{PATIENCE})")

        # Early Stopping：先判斷再 step，確保 break 時 scheduler 不多走一步
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early Stopping 觸發（連續 {PATIENCE} 個 epoch 未改善）")
            break

        scheduler.step()

    print(f"\n🎉 訓練完成！最佳 PLCC/SRCC 平均: {best_val_metric:.4f}，權重儲存於 {SAVE_PATH}")
    return 0


if __name__ == '__main__':
    sys.exit(main())