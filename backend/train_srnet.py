# backend/train_srnet.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.srnet import SRNet
from models.srnet_dataset import LSBStegoDataset

ROOT = r"E:\last\dataset"
COVER_DIR = os.path.join(ROOT, "cover")
STEGO_DIR = os.path.join(ROOT, "stego")
SPLIT_DIR = os.path.join(ROOT, "splits")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets
    train_ds = LSBStegoDataset(
        os.path.join(SPLIT_DIR, "train.txt"), COVER_DIR, STEGO_DIR, augment=True
    )
    val_ds = LSBStegoDataset(
        os.path.join(SPLIT_DIR, "val.txt"), COVER_DIR, STEGO_DIR, augment=False
    )

    # Dataloaders
    # use fewer workers and slightly smaller batch to avoid worker memory pressure
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # Model, loss, optimizer
    model = SRNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(1, 101):  # increased to 100 epochs
        model.train()
        running_loss = 0.0

        print(f"\nStarting epoch {epoch}...")

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  epoch {epoch}, batch {batch_idx+1}/{len(train_loader)}")

        # average training loss for this epoch
        train_loss = running_loss / len(train_loader.dataset)

        # ===== validation =====
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        # Step the scheduler
        scheduler.step()

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            weights_dir = r"E:\last\backend\weights"
            os.makedirs(weights_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(weights_dir, "srnet_best.pth"))
            print("  Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training finished. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
