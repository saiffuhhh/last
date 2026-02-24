import sys
ROOT = r"e:\last"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch
from models.srnet import SRNet
from models.srnet_dataset import BossbaseStegoDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# small reproducible step
device = torch.device('cpu')
print('device', device)
train_ds = BossbaseStegoDataset(r'e:\last\dataset\splits\train.txt', r'e:\last\dataset\cover', r'e:\last\dataset\stego', augment=False)
loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
model = SRNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

imgs, labels = next(iter(loader))
print('batch imgs shape, labels shape:', imgs.shape, labels.shape)
imgs = imgs.to(device)
labels = labels.to(device)

model.train()
outputs = model(imgs)
loss1 = criterion(outputs, labels)
print('initial loss', loss1.item())
loss1.backward()
optimizer.step()

with torch.no_grad():
    outputs2 = model(imgs)
    loss2 = criterion(outputs2, labels)
print('post-step loss', loss2.item())
print('loss change', loss2.item() - loss1.item())
print('done')
