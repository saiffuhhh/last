import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lsb_srnet import LSBSRNet
from models.srnet_dataset import LSBStegoDataset

ROOT = r"E:\last\dataset"
COVER_DIR = os.path.join(ROOT, "cover")
STEGO_DIR = os.path.join(ROOT, "stego")
SPLIT_DIR = os.path.join(ROOT, "splits")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    train_txt = os.path.join(SPLIT_DIR, "train.txt")
    val_txt = os.path.join(SPLIT_DIR, "val.txt")
    
    train_ds = LSBStegoDataset(train_txt, COVER_DIR, STEGO_DIR, augment=True)
    val_ds = LSBStegoDataset(val_txt, COVER_DIR, STEGO_DIR, augment=False)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    
    model = LSBSRNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    
    best_val_acc = 0.0
    num_epochs = 50
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Train")
        for batch_idx, (imgs, labels) in enumerate(train_pbar):
            imgs = imgs.to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar with current loss
            train_pbar.set_postfix(loss=f"{train_loss/train_total:.4f}")
        
        train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} Val")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs = imgs.to(device)
                labels = torch.tensor(labels, dtype=torch.long, device=device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch}/{num_epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights_dir = r"E:\last\backend\weights"
            os.makedirs(weights_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(weights_dir, "lsb_best.pth"))
            print(f"  ✓ Saved best model (Val Acc: {best_val_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
