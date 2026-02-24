import os
import torch
from torch.utils.data import DataLoader
from models.srnet import SRNet
from models.srnet_dataset import BossbaseStegoDataset

def test_model():
    ROOT = r"E:\last\dataset"
    COVER_DIR = os.path.join(ROOT, "cover")
    STEGO_DIR = os.path.join(ROOT, "stego")
    SPLIT_DIR = os.path.join(ROOT, "splits")
    WEIGHTS_DIR = r"E:\last\backend\weights"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    test_ds = BossbaseStegoDataset(
        os.path.join(SPLIT_DIR, "test.txt"), COVER_DIR, STEGO_DIR, augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    # Load model
    model = SRNet(num_classes=2).to(device)
    weights_path = os.path.join(WEIGHTS_DIR, "srnet_best.pth")
    
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded model from {weights_path}")
    
    # Test
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    test_acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    test_model()
