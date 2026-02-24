# Test SRNet learning on small batch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from models.srnet import SRNet
from models.srnet_dataset import LSBStegoDataset
import os

def test_small_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ROOT = r"E:\last\dataset"
    COVER_DIR = os.path.join(ROOT, "cover")
    STEGO_DIR = os.path.join(ROOT, "stego")
    SPLIT_DIR = os.path.join(ROOT, "splits")

    # Create small dataset (first 100 samples from train)
    train_ds = LSBStegoDataset(
        os.path.join(SPLIT_DIR, "train.txt"), COVER_DIR, STEGO_DIR, augment=False
    )
    small_ds = Subset(train_ds, range(100))  # 100 samples
    loader = DataLoader(small_ds, batch_size=10, shuffle=True, num_workers=0)

    model = SRNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("Testing on small batch...")
    for epoch in range(20):  # 20 epochs
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, acc={acc:.4f}")

        if acc > 0.8:  # Good accuracy
            print("Model learned successfully!")
            break

if __name__ == "__main__":
    test_small_batch()