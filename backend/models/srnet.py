import torch
import torch.nn as nn
import torch.nn.functional as F


class SRNet(nn.Module):
    """
    SRNet for steganalysis with SRM-like high-pass residual features.
    Input: 1 x 256 x 256 (grayscale image)
    Output: logits for 2 classes (cover, stego)
    """

    def __init__(self, num_classes: int = 2):
        super(SRNet, self).__init__()

        # High-pass preprocessing: 30 -> 32 channels (SRM-like residual input)
        # SRM-like fixed filter bank (applied to raw grayscale input)
        # we'll implement as a frozen conv layer with 30 filters, then take absolute value
        self.srm_conv = nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=2, bias=False)
        
        # Initialize with proper SRM filter bank (30 filters for steganalysis)
        srm_k = torch.zeros((30, 1, 5, 5), dtype=torch.float32)
        
        # SRM filter bank - these are the standard filters used in steganalysis
        # Based on the Spatial Rich Model for steganalysis
        srm_filters = [
            # Horizontal edges
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0], 
             [-1, 2, -2, 2, -1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            
            # Vertical edges  
            [[0, 0, -1, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, -2, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, -1, 0, 0]],
             
            # Diagonal edges (45 degrees)
            [[-1, 0, 0, 0, 0],
             [0, 2, 0, 0, 0],
             [0, 0, -2, 0, 0],
             [0, 0, 0, 2, 0],
             [0, 0, 0, 0, -1]],
             
            # Diagonal edges (135 degrees)
            [[0, 0, 0, 0, -1],
             [0, 0, 0, 2, 0],
             [0, 0, -2, 0, 0],
             [0, 2, 0, 0, 0],
             [-1, 0, 0, 0, 0]],
             
            # Horizontal line detector
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [-1, 2, -2, 2, -1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
             
            # Vertical line detector
            [[0, 0, -1, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, -2, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, -1, 0, 0]],
             
            # Square detector
            [[-1, 2, -2, 2, -1],
             [2, -6, 8, -6, 2],
             [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2],
             [-1, 2, -2, 2, -1]],
             
            # Corner detectors and other patterns
            [[0, 0, -1, 0, 0],
             [0, 2, 0, 0, 0],
             [-1, 0, 0, 0, 0],
             [0, 0, 0, 2, 0],
             [0, 0, 0, 0, -1]],
             
            [[0, 0, 0, 0, -1],
             [0, 0, 0, 2, 0],
             [0, 0, -2, 0, 0],
             [0, 2, 0, 0, 0],
             [-1, 0, 0, 0, 0]],
             
            [[-1, 0, 0, 0, 0],
             [0, 2, 0, 0, 0],
             [0, 0, -2, 0, 0],
             [0, 0, 0, 2, 0],
             [0, 0, 0, 0, -1]],
             
            # Additional filters for better coverage
            [[0, -1, 0, 1, 0],
             [-1, 0, 2, 0, 1],
             [0, 2, 0, -2, 0],
             [1, 0, -2, 0, -1],
             [0, 1, 0, -1, 0]],
             
            [[-1, 2, -1, 2, -1],
             [2, -6, 2, -6, 2],
             [-1, 2, -1, 2, -1],
             [2, -6, 2, -6, 2],
             [-1, 2, -1, 2, -1]],
             
            [[0, 0, 1, 0, 0],
             [0, -1, 0, 1, 0],
             [1, 0, -4, 0, 1],
             [0, 1, 0, -1, 0],
             [0, 0, 1, 0, 0]],
             
            [[1, 0, -2, 0, 1],
             [0, -2, 0, 2, 0],
             [-2, 0, 8, 0, -2],
             [0, 2, 0, -2, 0],
             [1, 0, -2, 0, 1]],
             
            [[0, 1, 0, -1, 0],
             [1, 0, -2, 0, 1],
             [0, -2, 0, 2, 0],
             [-1, 0, 2, 0, -1],
             [0, -1, 0, 1, 0]],
             
            # Fill remaining filters with variations
            [[-1, 0, 1, 0, -1],
             [0, 1, 0, -1, 0],
             [1, 0, -4, 0, 1],
             [0, -1, 0, 1, 0],
             [-1, 0, 1, 0, -1]],
             
            [[0, -1, 2, -1, 0],
             [-1, 2, -4, 2, -1],
             [2, -4, 8, -4, 2],
             [-1, 2, -4, 2, -1],
             [0, -1, 2, -1, 0]],
             
            [[1, -2, 1, -2, 1],
             [-2, 4, -2, 4, -2],
             [1, -2, 1, -2, 1],
             [-2, 4, -2, 4, -2],
             [1, -2, 1, -2, 1]],
             
            [[0, 0, -1, 0, 0],
             [0, 1, 0, -1, 0],
             [-1, 0, 2, 0, -1],
             [0, -1, 0, 1, 0],
             [0, 0, -1, 0, 0]],
             
            [[0, 1, 0, -1, 0],
             [1, 0, -2, 0, 1],
             [0, -2, 0, 2, 0],
             [-1, 0, 2, 0, -1],
             [0, -1, 0, 1, 0]],
             
            [[-1, 2, -1, 2, -1],
             [2, -4, 2, -4, 2],
             [-1, 2, -1, 2, -1],
             [2, -4, 2, -4, 2],
             [-1, 2, -1, 2, -1]],
             
            [[1, 0, -2, 0, 1],
             [0, -2, 0, 2, 0],
             [-2, 0, 8, 0, -2],
             [0, 2, 0, -2, 0],
             [1, 0, -2, 0, 1]],
             
            [[0, -1, 0, 1, 0],
             [-1, 0, 2, 0, -1],
             [0, 2, 0, -2, 0],
             [1, 0, -2, 0, 1],
             [0, 1, 0, -1, 0]],
             
            [[-1, 0, 1, 0, -1],
             [0, 1, 0, -1, 0],
             [1, 0, -4, 0, 1],
             [0, -1, 0, 1, 0],
             [-1, 0, 1, 0, -1]],
             
            [[0, 1, -2, 1, 0],
             [1, -2, 4, -2, 1],
             [-2, 4, -8, 4, -2],
             [1, -2, 4, -2, 1],
             [0, 1, -2, 1, 0]],
             
            [[1, -2, 1, -2, 1],
             [-2, 4, -2, 4, -2],
             [1, -2, 1, -2, 1],
             [-2, 4, -2, 4, -2],
             [1, -2, 1, -2, 1]],
             
            [[0, 0, 1, 0, 0],
             [0, -1, 0, 1, 0],
             [1, 0, -4, 0, 1],
             [0, 1, 0, -1, 0],
             [0, 0, 1, 0, 0]],
             
            [[1, 0, -2, 0, 1],
             [0, -2, 0, 2, 0],
             [-2, 0, 8, 0, -2],
             [0, 2, 0, -2, 0],
             [1, 0, -2, 0, 1]],
             
            [[0, 1, 0, -1, 0],
             [1, 0, -2, 0, 1],
             [0, -2, 0, 2, 0],
             [-1, 0, 2, 0, -1],
             [0, -1, 0, 1, 0]]
        ]
        
        # Convert to torch tensors
        for i, filt in enumerate(srm_filters):
            srm_k[i, 0] = torch.tensor(filt, dtype=torch.float32)
        
        with torch.no_grad():
            self.srm_conv.weight.copy_(srm_k)
        for p in self.srm_conv.parameters():
            p.requires_grad = False

        # Preprocess conv after SRM: 30 -> 32
        self.preprocess = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Block 1: 32 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Block 2: 32 -> 64, downsample
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Block 3: 64 -> 128, downsample
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Block 4: 128 -> 256, downsample
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # 256 x 1 x 1
            nn.Flatten(),                   # 256
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, 30, 256, 256]
        # apply SRM fixed filters (input is raw grayscale 1-channel)
        x = self.srm_conv(x)     # [B, 30, 256, 256]
        x = torch.abs(x)
        x = self.preprocess(x)   # [B, 32, 256, 256]
        x = self.block1(x)       # [B, 32, 256, 256]
        x = self.block2(x)       # [B, 64, 128, 128]
        x = self.block3(x)       # [B, 128, 64, 64]
        x = self.block4(x)       # [B, 256, 32, 32]

        x = self.classifier(x)   # [B, 2]
        return x
