import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets.seg_dataset import TongueSegDataset
from src.models.unet import UNet
from src.losses.dice_loss import DiceLoss

# -------------------------
# Config
# -------------------------
IMG_DIR = "../data/images/tongue"
MASK_DIR = "../data/masks/tongue"

TRAIN_SPLIT = "../data/splits/train.txt"
VAL_SPLIT = "../data/splits/val.txt"

BATCH_SIZE = 2          # CPU 安全值
EPOCHS = 3              # 第一次只跑 3 轮，确认无误后再改 50
LR = 1e-4
MODEL_SAVE_PATH = "unet_tongue_best.pth"

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Load splits
# -------------------------
with open(TRAIN_SPLIT) as f:
    train_ids = [l.strip() for l in f if l.strip()]

with open(VAL_SPLIT) as f:
    val_ids = [l.strip() for l in f if l.strip()]

# -------------------------
# Dataset & Loader
# -------------------------
train_ds = TongueSegDataset(IMG_DIR, MASK_DIR, train_ids)
val_ds = TongueSegDataset(IMG_DIR, MASK_DIR, val_ids)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

assert len(train_ds) > 0, "❌ Train dataset is empty"
assert len(val_ds) > 0, "❌ Val dataset is empty"

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4  # 设置为 4，或更高，来加速数据加载
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4  # 设置为 4，或更高，来加速数据加载
)

# -------------------------
# Model / Loss / Optimizer
# -------------------------
model = UNet().to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Validation function
# -------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0

    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)

        pred = model(img)
        loss = criterion(pred, mask)
        total_loss += loss.item()

    return total_loss / len(loader)

# -------------------------
# Training loop
# -------------------------
def train():
    best_val_loss = 1e9

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for img, mask in pbar:
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader)

        print(
            f"[Epoch {epoch}] "
            f"Train DiceLoss: {train_loss:.4f} | "
            f"Val DiceLoss: {val_loss:.4f}"
        )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved best model (val loss = {val_loss:.4f})")

    print("Training finished.")
    print("Best model saved as:", MODEL_SAVE_PATH)


# -------------------------
# Main function for Windows compatibility
# -------------------------
if __name__ == '__main__':
    train()
