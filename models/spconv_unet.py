import torch
import torch.nn as nn
import spconv.pytorch as spconv

class SpConvUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()
        
        algo = spconv.ConvAlgo.Native

        # Stem
        self.conv_in = spconv.SubMConv3d(
            in_channels, 32, kernel_size=3, padding=1, bias=False, indice_key="subm0", algo=algo
        )
        self.bn_in = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)

        # Down 1 (32 -> 64)
        self.down1 = spconv.SparseConv3d(
            32, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key="down1", algo=algo
        )
        self.bn_d1 = nn.BatchNorm1d(64)
        self.subm1 = spconv.SubMConv3d(
            64, 64, kernel_size=3, padding=1, bias=False, indice_key="subm1", algo=algo
        )
        self.bn_s1 = nn.BatchNorm1d(64)

        # Down 2 (64 -> 128)
        self.down2 = spconv.SparseConv3d(
            64, 128, kernel_size=3, stride=2, padding=1, bias=False, indice_key="down2", algo=algo
        )
        self.bn_d2 = nn.BatchNorm1d(128)
        self.subm2 = spconv.SubMConv3d(
            128, 128, kernel_size=3, padding=1, bias=False, indice_key="subm2", algo=algo
        )
        self.bn_s2 = nn.BatchNorm1d(128)

        # Up 2 (128 -> 64)
        self.up2 = spconv.SparseInverseConv3d(
            128, 64, kernel_size=3, bias=False, indice_key="down2", algo=algo
        )
        self.bn_u2 = nn.BatchNorm1d(64)
        self.dec_conv2 = spconv.SubMConv3d(
            128, 64, kernel_size=3, padding=1, bias=False, indice_key="dec_subm2", algo=algo
        )
        self.bn_dc2 = nn.BatchNorm1d(64)

        # Up 1 (64 -> 32)
        self.up1 = spconv.SparseInverseConv3d(
            64, 32, kernel_size=3, bias=False, indice_key="down1", algo=algo
        )
        self.bn_u1 = nn.BatchNorm1d(32)
        self.dec_conv1 = spconv.SubMConv3d(
            64, 32, kernel_size=3, padding=1, bias=False, indice_key="dec_subm1", algo=algo
        )
        self.bn_dc1 = nn.BatchNorm1d(32)

        # Classifier head
        self.classifier = spconv.SubMConv3d(
            32, num_classes, kernel_size=1, bias=True, indice_key="classifier", algo=algo
        )

    def forward(self, x_sp):
        # Stem
        x0 = self.relu(self.bn_in(self.conv_in(x_sp).features))
        x0_sp = x_sp.replace_feature(x0)

        # Down 1
        d1 = self.relu(self.bn_d1(self.down1(x0_sp).features))
        d1_sp = self.down1(x0_sp).replace_feature(d1)
        s1 = self.relu(self.bn_s1(self.subm1(d1_sp).features))
        x1_sp = d1_sp.replace_feature(s1)

        # Down 2
        d2 = self.relu(self.bn_d2(self.down2(x1_sp).features))
        d2_sp = self.down2(x1_sp).replace_feature(d2)
        s2 = self.relu(self.bn_s2(self.subm2(d2_sp).features))
        x2_sp = d2_sp.replace_feature(s2)

        # Up 2
        up2_feat = self.relu(self.bn_u2(self.up2(x2_sp).features))
        cat2 = torch.cat([up2_feat, x1_sp.features], dim=1)
        dec2 = self.relu(self.bn_dc2(self.dec_conv2(x1_sp.replace_feature(cat2)).features))
        x_dec2 = x1_sp.replace_feature(dec2)

        # Up 1
        up1_feat = self.relu(self.bn_u1(self.up1(x_dec2).features))
        cat1 = torch.cat([up1_feat, x0_sp.features], dim=1)
        dec1 = self.relu(self.bn_dc1(self.dec_conv1(x0_sp.replace_feature(cat1)).features))
        x_dec1 = x0_sp.replace_feature(dec1)

        out_sp = self.classifier(x_dec1)
        return out_sp.features