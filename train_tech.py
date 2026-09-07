import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 模型架構、前處理、訓練迴圈一律來自 common/
from common import NIMABaseline, build_transform, run_epoch

# ==========================================
# 1. 資料集定義 (針對 KonIQ-10k 優化)
# ==========================================
class KonIQDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 檔名在這裡就從「欄位」算好。理由與 train_nima.py 的 AVADataset 相同：
        # df.iloc[i] 會把整列轉成共同型別，CSV 一旦同時有整數欄與浮點欄，
        # 檔名就可能變成 "12345.0.jpg" 而全部讀不到。
        # KonIQ 的 CSV 檔名通常沒帶 .jpg，這裡一併補上。
        self.filenames = [
            str(name) if str(name).lower().endswith('.jpg') else f'{name}.jpg'
            for name in self.df['image'].tolist()
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, self.filenames[idx])

        # 讀不到就直接拋錯，不用全黑圖代替——原本的靜默 fallback
        # 會讓訓練在完全錯誤的資料上跑完而毫無徵兆（詳見 AVADataset 的說明）。
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(
                f'無法讀取訓練影像：{img_path}'
                f'｜{type(e).__name__}: {e}'
                f'｜請確認 {self.img_dir} 內有這個檔案，且 CSV 的 image 欄位正確'
            ) from e

        # 這裡的 label 是 split_koniq.py 縮小到 0~1 之間的分數
        label = torch.tensor(float(row['label']), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================================
# 2. 主程式
# ==========================================
def parse_args():
    """
    超參數與檔案路徑改為命令列參數，與 train_nima.py 保持一致。

    原本全部寫死在 main() 裡，其中 SAVE_PATH 直接指向正在服役的
    nima_tech_best.pth——任何人只要執行一次 `python train_tech.py`
    （即使只是想看看訓練長什麼樣子），第一個有改善的 epoch 就會把現役權重
    覆蓋掉，而且蓋上去的是只訓練了 1 個 epoch 的模型。
    分數看起來仍然「正常」，不會有任何錯誤訊息。

    技術模型沒有 --mode 參數：它的標籤是 KonIQ 的連續 MOS 分數，
    只有 single 模式適用。美感模型才需要在二元與分佈之間切換。
    """
    p = argparse.ArgumentParser(description='訓練技術品質評分模型（KonIQ-10k）')
    p.add_argument('--train-csv', default='data/train_tech.csv')
    p.add_argument('--val-csv', default='data/val_tech.csv')
    p.add_argument('--img-dir', default='data/koniq/512x384')
    p.add_argument('--save', default='nima_tech_best.pth', help='權重輸出路徑')
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
    SAVE_PATH    = args.save

    # 權重檔是訓練數小時的產物，覆寫掉就沒了，因此預設拒絕覆蓋。
    if os.path.exists(SAVE_PATH) and not args.force:
        print(f"⛔ 權重檔已存在：{SAVE_PATH}")
        print(f"   訓練會覆蓋它且無法復原。請改用 --save 指定其他檔名，"
              f"或確認後加上 --force。")
        return 1

    print(f"📋 訓練集={args.train_csv}  驗證集={args.val_csv}  影像={args.img_dir}")
    print(f"   輸出={SAVE_PATH}  epochs={EPOCHS}  batch={BATCH_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"✅ 設備確認: {device_name}")

    # pin_memory 只在有 GPU 時才有意義（純 CPU 環境開著會噴警告）
    pin_memory = torch.cuda.is_available()

    # Windows 環境 num_workers > 0 容易觸發多進程問題
    num_workers = 0 if os.name == 'nt' else 2

    # 前處理一律由 common/transforms.py 提供，確保與推論端完全一致
    train_transform = build_transform('train_technical')
    val_transform = build_transform('eval')

    # 載入路徑 (請確保你已經跑過 split_koniq.py)
    train_ds = KonIQDataset(args.train_csv, args.img_dir, train_transform)
    val_ds   = KonIQDataset(args.val_csv,   args.img_dir, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # pretrained=True：訓練時從 ImageNet 預訓練權重出發
    model = NIMABaseline(pretrained=True).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(),   'lr': LR_FEATURES},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"🚀 開始訓練技術品質模型 (目標檔案: {SAVE_PATH})...\n")
    # 模型選擇改用 PLCC+SRCC（越高越好），比單看 MSE 更貼近人類評分排序
    best_val_metric = -float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, is_train=True)
        val_loss, plcc, srcc = run_epoch(model, val_loader, criterion, device, is_train=False, collect_metrics=True)
        val_metric = (plcc + srcc) / 2

        current_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch [{epoch:>2}/{EPOCHS}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PLCC: {plcc:.4f} | SRCC: {srcc:.4f} | LR: {current_lr:.2e}")

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  💾 PLCC/SRCC 改善，技術品質模型已存至 {SAVE_PATH}")
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
