import torch
import torch.nn as nn

class NativeVoxelBackbone(nn.Module):
    """
    Pure PyTorch 3D point/voxel segmentation backbone.
    Zero C++ compilation needed; runs cleanly on Python 3.12 and RTX 3050 (6GB).
    """
    def __init__(self, in_channels=4, num_classes=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(256 + 128 + 64, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, in_channels, N)
        Returns logits of shape (B, num_classes, N)
        """
        f1 = self.encoder[0:3](x)
        f2 = self.encoder[3:6](f1)
        f3 = self.encoder[6:9](f2)
        
        feat = torch.cat([f1, f2, f3], dim=1)
        out = self.decoder(feat)
        return out