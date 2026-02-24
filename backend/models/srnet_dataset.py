import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pywt

class LSBStegoDataset(Dataset):
    def __init__(self, split_file, cover_dir, stego_dir, augment=False):
        with open(split_file, "r") as f:
            self.names = [line.strip() for line in f]

        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.augment = augment

    def __len__(self):
        return 2 * len(self.names)  # cover + stego

    def _apply_high_pass_filter(self, img):
        """Apply high-pass filter using Laplacian kernel"""
        # Normalize to 0-255 range for filtering
        img_uint8 = ((img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8))
        
        # Apply Laplacian high-pass filter
        laplacian = cv2.Laplacian(img_uint8, cv2.CV_32F)
        
        # Normalize high-pass components
        if laplacian.max() > 0:
            laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
        
        return laplacian.astype(np.float32)

    def _extract_srm_like_features(self, img):
        """Create a SRM-like set of high-pass residuals (30 channels).
        This computes a variety of neighbor differences (residuals) using shifts
        and absolute differences to produce a 30-channel residual stack.
        """
        # ensure img is float32 and scaled to 0-1
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        h, w = img.shape
        # define a set of offsets to produce residuals
        offsets = [
            (0, 1), (1, 0), (1, 1), (-1, 1),
            (0, 2), (2, 0), (2, 2), (-2, 2),
            (0, 3), (3, 0), (3, 3), (-3, 3),
            (1, 2), (2, 1), (-1, 2)
        ]

        n_off = len(offsets)
        channels = np.empty((n_off * 2, h, w), dtype=np.float32)
        idx = 0
        for dx, dy in offsets:
            # shift using np.roll (wrap-around) then pad edge by copying border to avoid wrap artefacts
            shifted = np.roll(img, shift=dy, axis=0)
            shifted = np.roll(shifted, shift=dx, axis=1)

            # fix borders by copying nearest rows/cols for the rolled areas
            if dy > 0:
                shifted[:dy, :] = img[:dy, :]
            elif dy < 0:
                shifted[dy:, :] = img[dy:, :]
            if dx > 0:
                shifted[:, :dx] = img[:, :dx]
            elif dx < 0:
                shifted[:, dx:] = img[:, dx:]

            resid = img - shifted
            channels[idx] = resid
            channels[idx + 1] = np.abs(resid)
            idx += 2

        # channels list length should be 30 (15 offsets * 2)
        feat = channels

        # simple per-channel normalization to 0-1 (in-place)
        for i in range(feat.shape[0]):
            ch = feat[i]
            mmin = ch.min()
            mmax = ch.max()
            if mmax > mmin:
                feat[i] = (ch - mmin) / (mmax - mmin)

        # resize if needed (keep HxW 256x256 already)
        if feat.shape[1] != 256 or feat.shape[2] != 256:
            stacked = feat.transpose(1, 2, 0)
            stacked = cv2.resize(stacked, (256, 256))
            feat = stacked.transpose(2, 0, 1)

        return feat

    def __getitem__(self, idx):
        if idx < len(self.names):
            fname = self.names[idx]
            path = os.path.join(self.cover_dir, fname)
            label = 0  # cover
        else:
            fname = self.names[idx - len(self.names)]
            # stegos are saved as PNG, keep same base name
            name_root, _ = os.path.splitext(fname)
            stego_name = name_root + ".png"
            path = os.path.join(self.stego_dir, stego_name)
            label = 1  # stego

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        # Resize to 256x256 if necessary
        if img.shape != (256, 256):
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img = img.astype("float32") / 255.0

        # return raw grayscale image tensor (1 x H x W); SRM filters applied in model on GPU
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        return img_tensor, label
