import os
import sys
# ensure project root is on sys.path
ROOT = r"e:\last"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from backend.models.srnet_dataset import BossbaseStegoDataset

split_file = r"e:\last\dataset\splits\train.txt"
cover_dir = r"e:\last\dataset\cover"
stego_dir = r"e:\last\dataset\stego"

dset = BossbaseStegoDataset(split_file, cover_dir, stego_dir, augment=False)

print('Total names in split:', len(dset.names))

N = 10
for i in range(N):
    fname = dset.names[i]
    cover_path = os.path.join(cover_dir, fname)
    name_root, _ = os.path.splitext(fname)
    stego_name = name_root + ".png"
    stego_path = os.path.join(stego_dir, stego_name)
    cover_exists = os.path.exists(cover_path)
    stego_exists = os.path.exists(stego_path)
    print(f"[{i}] {fname}: cover exists={cover_exists}, stego exists={stego_exists}")
    if cover_exists:
        import cv2
        c = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
        print('  cover shape/min/max/mean:', c.shape, c.min(), c.max(), c.mean())
    if stego_exists:
        import cv2
        s = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
        print('  stego shape/min/max/mean:', s.shape, s.min(), s.max(), s.mean())
    # check if identical
    if cover_exists and stego_exists:
        same = (c.shape == s.shape) and (c.flatten().tolist() == s.flatten().tolist())
        print('  identical arrays:', same)

# Also try dataset __getitem__ for first few indices (cover and stego)
print('\nSampling dataset __getitem__ outputs:')
for i in range(6):
    img, label = dset[i]
    print(f'idx {i}: img.shape={img.shape}, label={label}, min={img.min().item()}, max={img.max().item():.6f}')
for i in range(len(dset.names), len(dset.names)+6):
    img, label = dset[i]
    print(f'idx {i}: img.shape={img.shape}, label={label}, min={img.min().item()}, max={img.max().item():.6f}')
print('Done')
