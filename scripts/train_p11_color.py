import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.p11_color_dataset import P11ColorDataset
from src.models.p11_color_regressor import P11ColorRegressor


LABEL_CSV = "../data/labels/p11_tg_color.csv"
CKPT_DIR = "../checkpoints/p11"
CKPT_PATH = os.path.join(CKPT_DIR, "p11_color_best.pth")


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2).item()
            total += loss
            n += 1
    return total / max(n, 1)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # datasets
    train_ds = P11ColorDataset("../outputs/roi/train", "../data/splits/train.txt", LABEL_CSV, normalize_y=True)
    val_ds = P11ColorDataset("../outputs/roi/val", "../data/splits/val.txt", LABEL_CSV, normalize_y=True)
    test_ds = P11ColorDataset("../outputs/roi/test", "../data/splits/test.txt", LABEL_CSV, normalize_y=True)

    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    out_dim = len(train_ds.label_cols)
    model = P11ColorRegressor(out_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = 1e9
    EPOCHS = 15

    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}")
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            pbar.set_postfix(train_mse=loss.item())

        train_mse = total / len(train_loader)
        val_mse = evaluate(model, val_loader, device)

        print(f"[Epoch {ep}] Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "label_cols": train_ds.label_cols,
                    "y_mean": train_ds.y_mean,
                    "y_std": train_ds.y_std,
                },
                CKPT_PATH
            )
            print(f"✅ Saved best: {CKPT_PATH} (val_mse={best_val:.6f})")

    # final test
    print("\n=== Final Test Evaluation ===")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_mse = evaluate(model, test_loader, device)
    print(f"Test MSE (normalized): {test_mse:.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
