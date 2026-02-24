import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class LSBStegoDataset(Dataset):
    def __init__(self, split_file, cover_dir, stego_dir, augment=False):
        with open(split_file, "r") as f:
            self.names = [line.strip() for line in f]
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.augment = augment
    
    def __len__(self):
        return 2 * len(self.names)
    
    def __getitem__(self, idx):
        if idx < len(self.names):
            fname = self.names[idx]
            path = os.path.join(self.cover_dir, fname)
            label = 0
        else:
            fname = self.names[idx - len(self.names)]
            name_root, _ = os.path.splitext(fname)
            stego_name = name_root + ".png"
            path = os.path.join(self.stego_dir, stego_name)
            label = 1
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read: {path}")
        
        if self.augment:
            img = self._augment(img)
        
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        return img, label
    
    def _augment(self, img):
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 0)
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k)
        return img
