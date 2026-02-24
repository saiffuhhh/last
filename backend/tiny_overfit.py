import sys
ROOT = r"e:\last"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch
from torch.utils.data import DataLoader, Subset
from models.srnet import SRNet
from models.srnet_dataset import BossbaseStegoDataset
import torch.nn as nn

device = torch.device('cpu')
print('device', device)
full_ds = BossbaseStegoDataset(r'e:\last\dataset\splits\train.txt', r'e:\last\dataset\cover', r'e:\last\dataset\stego', augment=False)
K = 16
small_len = 2 * K
subset = Subset(full_ds, list(range(small_len)))
loader = DataLoader(subset, batch_size=8, shuffle=True)
model = SRNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 201):
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    acc = correct / total
    train_loss = running_loss / total
    if epoch % 10 == 0 or acc > 0.95:
        print(f'epoch {epoch}: loss={train_loss:.4f}, acc={acc:.4f}')
    if acc > 0.98:
        print('Reached high training accuracy, stopping')
        break
print('Done')
