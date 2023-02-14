import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 --> C, H, W
        # x2 --> C/2, H*2, W*2

        x1 = self.up(x1)  # C, H, W --> C/2, H*2, W*2
        # Pad x1 before concatination
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        # C/2, H*2, W*2 and C/2, H*2, W*2 --> C,  H*2, W*2
        x = torch.cat([x2, x1], dim=1)
        # DoubleConv C, H*2, W*2 --> C/2, H*2, W*2
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, initial_feature_map=64, usa_cuda=True):
        super(UNet, self).__init__()
        self.usa_cuda = usa_cuda
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.initial_feature_map = initial_feature_map

        #Encoder
        self.inc = DoubleConv(n_channels, self.initial_feature_map)
        self.inc.apply(self.weights_init)
        self.down1 = Down(self.initial_feature_map, self.initial_feature_map*2)
        self.down1.apply(self.weights_init)
        self.down2 = Down(self.initial_feature_map*2, self.initial_feature_map*4)
        self.down2.apply(self.weights_init)
        self.down3 = Down(self.initial_feature_map*4, self.initial_feature_map*8)
        self.down3.apply(self.weights_init)
        self.down4 = Down(self.initial_feature_map*8, self.initial_feature_map*16)
        self.down4.apply(self.weights_init)

        #decoder
        self.up1 = Up(self.initial_feature_map*16, self.initial_feature_map*8)
        self.up1.apply(self.weights_init)
        self.up2 = Up(self.initial_feature_map*8, self.initial_feature_map*4)
        self.up2.apply(self.weights_init)
        self.up3 = Up(self.initial_feature_map*4, self.initial_feature_map*2)
        self.up3.apply(self.weights_init)
        self.up4 = Up(self.initial_feature_map*2, self.initial_feature_map)
        self.up4.apply(self.weights_init)
        self.outc = OutConv(self.initial_feature_map, n_classes)
        self.outc.apply(self.weights_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class UNet_multitask(nn.Module):
    def __init__(self, n_channels, n_classes, initial_feature_map=64, usa_cuda=True):
        super(UNet_multitask, self).__init__()
        self.usa_cuda = usa_cuda
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.initial_feature_map = initial_feature_map

        # Encoder
        self.inc = DoubleConv(n_channels, self.initial_feature_map)
        self.inc.apply(self.weights_init)
        self.down1 = Down(self.initial_feature_map, self.initial_feature_map*2)
        self.down1.apply(self.weights_init)
        self.down2 = Down(self.initial_feature_map*2, self.initial_feature_map*4)
        self.down2.apply(self.weights_init)
        self.down3 = Down(self.initial_feature_map*4, self.initial_feature_map*8)
        self.down3.apply(self.weights_init)
        self.down4 = Down(self.initial_feature_map*8, self.initial_feature_map*16)
        self.down4.apply(self.weights_init)

        # Decoder Binary
        self.up1_bin = Up(self.initial_feature_map*16, self.initial_feature_map*8)
        self.up1_bin.apply(self.weights_init)
        self.up2_bin = Up(self.initial_feature_map*8, self.initial_feature_map*4)
        self.up2_bin.apply(self.weights_init)
        self.up3_bin = Up(self.initial_feature_map*4, self.initial_feature_map*2)
        self.up3_bin.apply(self.weights_init)
        self.up4_bin = Up(self.initial_feature_map*2, self.initial_feature_map)
        self.up4_bin.apply(self.weights_init)
        self.outc_bin = OutConv(self.initial_feature_map, n_classes)
        self.outc_bin.apply(self.weights_init)

        # Decoder Regression
        self.up1_reg = Up(self.initial_feature_map*16, self.initial_feature_map*8)
        self.up1_reg.apply(self.weights_init)
        self.up2_reg = Up(self.initial_feature_map*8, self.initial_feature_map*4)
        self.up2_reg.apply(self.weights_init)
        self.up3_reg = Up(self.initial_feature_map*4, self.initial_feature_map*2)
        self.up3_reg.apply(self.weights_init)
        self.up4_reg = Up(self.initial_feature_map*2, self.initial_feature_map)
        self.up4_reg.apply(self.weights_init)
        self.outc_reg = OutConv(self.initial_feature_map, n_classes)
        self.outc_reg.apply(self.weights_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_bin = self.up1_bin(x5, x4)
        x_bin = self.up2_bin(x_bin, x3)
        x_bin = self.up3_bin(x_bin, x2)
        x_bin = self.up4_bin(x_bin, x1)
        logits_bin = self.outc_bin(x_bin)

        x_reg = self.up1_reg(x5, x4)
        x_reg = self.up2_reg(x_reg, x3)
        x_reg = self.up3_reg(x_reg, x2)
        x_reg = self.up4_reg(x_reg, x1)
        logits_reg = self.outc_reg(x_bin)

        return logits_bin, logits_reg

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class Attention_block(nn.Module):
    def __init__(self, C_q, C_x, C_hidden):
        super(Attention_block, self).__init__()
        self.W_q = nn.Sequential(
            nn.Conv2d(C_q, C_hidden, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(C_hidden)
        )

        self.up = nn.ConvTranspose2d(
            C_q, C_q, kernel_size=2, stride=2)

        self.W_x = nn.Sequential(
            nn.Conv2d(C_x, C_hidden, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(C_hidden)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(C_hidden, 1, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, q, x):
        q = self.up(q)
        Q1 = self.W_q(q)
        X1 = self.W_x(x)

        E = self.relu(Q1+X1)
        A = self.psi(E)
        return x*A


class UNet_attention(nn.Module):
    def __init__(self, n_channels, n_classes, initial_feature_map=64, usa_cuda=True):
        super(UNet_attention, self).__init__()
        self.usa_cuda = usa_cuda
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.initial_feature_map = initial_feature_map

        # Encoder
        self.inc = DoubleConv(n_channels, self.initial_feature_map)
        self.inc.apply(self.weights_init)
        self.down1 = Down(self.initial_feature_map, self.initial_feature_map*2)
        self.down1.apply(self.weights_init)
        self.down2 = Down(self.initial_feature_map*2,
                          self.initial_feature_map*4)
        self.down2.apply(self.weights_init)
        self.down3 = Down(self.initial_feature_map*4,
                          self.initial_feature_map*8)
        self.down3.apply(self.weights_init)
        self.down4 = Down(self.initial_feature_map*8,
                          self.initial_feature_map*16)
        self.down4.apply(self.weights_init)

        #attention gates
        self.attenion4 = Attention_block(
            C_q=self.initial_feature_map*16,
            C_x=self.initial_feature_map*8,
            C_hidden=self.initial_feature_map*4)
        self.attenion3 = Attention_block(
            C_q=self.initial_feature_map*8,
            C_x=self.initial_feature_map*4,
            C_hidden=self.initial_feature_map*2)
        self.attenion2 = Attention_block(
            C_q=self.initial_feature_map*4,
            C_x=self.initial_feature_map*2,
            C_hidden=self.initial_feature_map)
        self.attenion1 = Attention_block(
            C_q=self.initial_feature_map*2,
            C_x=self.initial_feature_map,
            C_hidden=int(self.initial_feature_map/2))
        
        # decoder
        self.up1 = Up(self.initial_feature_map*16, self.initial_feature_map*8)
        self.up1.apply(self.weights_init)
        self.up2 = Up(self.initial_feature_map*8, self.initial_feature_map*4)
        self.up2.apply(self.weights_init)
        self.up3 = Up(self.initial_feature_map*4, self.initial_feature_map*2)
        self.up3.apply(self.weights_init)
        self.up4 = Up(self.initial_feature_map*2, self.initial_feature_map)
        self.up4.apply(self.weights_init)
        self.outc = OutConv(self.initial_feature_map, n_classes)
        self.outc.apply(self.weights_init)

    def forward(self, x):
        #encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #decoder

        x4_attention = self.attenion4(q=x5, x=x4)
        x = self.up1(x5, x4_attention)

        x3_attention = self.attenion3(q=x, x=x3)
        x = self.up2(x, x3_attention)

        x2_attention = self.attenion2(q=x, x=x2)
        x = self.up3(x, x2_attention)

        x1_attention = self.attenion1(q=x, x=x1)
        x = self.up4(x, x1_attention)

        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class UNet_fourier1_2(nn.Module):
    def __init__(self, n_channels, n_classes, initial_feature_map=64, usa_cuda=True):
        super(UNet_fourier1_2, self).__init__()
        self.usa_cuda = usa_cuda
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.initial_feature_map = initial_feature_map

        # Encoder
        self.inc = DoubleConv(n_channels, self.initial_feature_map)
        self.inc.apply(self.weights_init)
        self.down1 = Down(self.initial_feature_map, self.initial_feature_map*2)
        self.down1.apply(self.weights_init)
        self.down2 = Down(self.initial_feature_map*2,
                          self.initial_feature_map*4)
        self.down2.apply(self.weights_init)
        self.down3 = Down(self.initial_feature_map*4,
                          self.initial_feature_map*8)
        self.down3.apply(self.weights_init)
        self.down4 = Down(self.initial_feature_map*8,
                          self.initial_feature_map*16)
        self.down4.apply(self.weights_init)

        # Decoder Binary
        self.up1_bin = Up(self.initial_feature_map*16,
                          self.initial_feature_map*8)
        self.up1_bin.apply(self.weights_init)
        self.up2_bin = Up(self.initial_feature_map*8,
                          self.initial_feature_map*4)
        self.up2_bin.apply(self.weights_init)
        self.up3_bin = Up(self.initial_feature_map*4,
                          self.initial_feature_map*2)
        self.up3_bin.apply(self.weights_init)
        self.up4_bin = Up(self.initial_feature_map*2, self.initial_feature_map)
        self.up4_bin.apply(self.weights_init)
        self.outc_bin = OutConv(self.initial_feature_map, n_classes)
        self.outc_bin.apply(self.weights_init)

        # Decoder Fourier Outer 1
        self.up1_fouter1 = Up(self.initial_feature_map*16,
                              self.initial_feature_map*8)
        self.up1_fouter1.apply(self.weights_init)
        self.up2_fouter1 = Up(self.initial_feature_map*8,
                              self.initial_feature_map*4)
        self.up2_fouter1.apply(self.weights_init)
        self.up3_fouter1 = Up(self.initial_feature_map*4,
                              self.initial_feature_map*2)
        self.up3_fouter1.apply(self.weights_init)
        self.up4_fouter1 = Up(self.initial_feature_map *
                              2, self.initial_feature_map)
        self.up4_fouter1.apply(self.weights_init)
        self.outc_fouter1 = OutConv(self.initial_feature_map, n_classes)
        self.outc_fouter1.apply(self.weights_init)


        # Decoder Fourier Outer 2
        self.up1_fouter2 = Up(self.initial_feature_map*16,
                          self.initial_feature_map*8)
        self.up1_fouter2.apply(self.weights_init)
        self.up2_fouter2 = Up(self.initial_feature_map*8,
                          self.initial_feature_map*4)
        self.up2_fouter2.apply(self.weights_init)
        self.up3_fouter2 = Up(self.initial_feature_map*4,
                          self.initial_feature_map*2)
        self.up3_fouter2.apply(self.weights_init)
        self.up4_fouter2 = Up(self.initial_feature_map *
                              2, self.initial_feature_map)
        self.up4_fouter2.apply(self.weights_init)
        self.outc_fouter2 = OutConv(self.initial_feature_map, n_classes)
        self.outc_fouter2.apply(self.weights_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_bin = self.up1_bin(x5, x4)
        x_bin = self.up2_bin(x_bin, x3)
        x_bin = self.up3_bin(x_bin, x2)
        x_bin = self.up4_bin(x_bin, x1)
        logits_bin = self.outc_bin(x_bin)

        x_fouter1 = self.up1_fouter1(x5, x4)
        x_fouter1 = self.up2_fouter1(x_fouter1, x3)
        x_fouter1 = self.up3_fouter1(x_fouter1, x2)
        x_fouter1 = self.up4_fouter1(x_fouter1, x1)
        logits_fouter1 = self.outc_fouter1(x_fouter1)

        x_fouter2 = self.up1_fouter2(x5, x4)
        x_fouter2 = self.up2_fouter2(x_fouter2, x3)
        x_fouter2 = self.up3_fouter2(x_fouter2, x2)
        x_fouter2 = self.up4_fouter2(x_fouter2, x1)
        logits_fouter2 = self.outc_fouter2(x_fouter2)

        return logits_bin, logits_fouter1, logits_fouter2

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet_fourier1(nn.Module):
    def __init__(self, n_channels, n_classes, initial_feature_map=64, usa_cuda=True):
        super(UNet_fourier1, self).__init__()
        self.usa_cuda = usa_cuda
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.initial_feature_map = initial_feature_map

        # Encoder
        self.inc = DoubleConv(n_channels, self.initial_feature_map)
        self.inc.apply(self.weights_init)
        self.down1 = Down(self.initial_feature_map, self.initial_feature_map*2)
        self.down1.apply(self.weights_init)
        self.down2 = Down(self.initial_feature_map*2,
                          self.initial_feature_map*4)
        self.down2.apply(self.weights_init)
        self.down3 = Down(self.initial_feature_map*4,
                          self.initial_feature_map*8)
        self.down3.apply(self.weights_init)
        self.down4 = Down(self.initial_feature_map*8,
                          self.initial_feature_map*16)
        self.down4.apply(self.weights_init)

        # Decoder Binary
        self.up1_bin = Up(self.initial_feature_map*16,
                          self.initial_feature_map*8)
        self.up1_bin.apply(self.weights_init)
        self.up2_bin = Up(self.initial_feature_map*8,
                          self.initial_feature_map*4)
        self.up2_bin.apply(self.weights_init)
        self.up3_bin = Up(self.initial_feature_map*4,
                          self.initial_feature_map*2)
        self.up3_bin.apply(self.weights_init)
        self.up4_bin = Up(self.initial_feature_map*2, self.initial_feature_map)
        self.up4_bin.apply(self.weights_init)
        self.outc_bin = OutConv(self.initial_feature_map, n_classes)
        self.outc_bin.apply(self.weights_init)

        # Decoder Fourier Outer 1
        self.up1_fouter1 = Up(self.initial_feature_map*16,
                              self.initial_feature_map*8)
        self.up1_fouter1.apply(self.weights_init)
        self.up2_fouter1 = Up(self.initial_feature_map*8,
                              self.initial_feature_map*4)
        self.up2_fouter1.apply(self.weights_init)
        self.up3_fouter1 = Up(self.initial_feature_map*4,
                              self.initial_feature_map*2)
        self.up3_fouter1.apply(self.weights_init)
        self.up4_fouter1 = Up(self.initial_feature_map *
                              2, self.initial_feature_map)
        self.up4_fouter1.apply(self.weights_init)
        self.outc_fouter1 = OutConv(self.initial_feature_map, n_classes)
        self.outc_fouter1.apply(self.weights_init)

    def forward(self, x):
        x1 = self.inc(x)     # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 1024

        x_bin = self.up1_bin(x5, x4)  # 1024(x5) --> ConvT --> 512 concat with x4(512) --> 1024 --> DoubleConv --> 512 
        x_bin = self.up2_bin(x_bin, x3)
        x_bin = self.up3_bin(x_bin, x2)
        x_bin = self.up4_bin(x_bin, x1)
        logits_bin = self.outc_bin(x_bin)

        x_fouter1 = self.up1_fouter1(x5, x4)
        x_fouter1 = self.up2_fouter1(x_fouter1, x3)
        x_fouter1 = self.up3_fouter1(x_fouter1, x2)
        x_fouter1 = self.up4_fouter1(x_fouter1, x1)
        logits_fouter1 = self.outc_fouter1(x_fouter1)

        return logits_bin, logits_fouter1

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
