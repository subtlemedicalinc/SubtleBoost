import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        skip = x
        x = self.model(x)
        return torch.cat((x, skip), dim=1)



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, act='relu', use_bn=False):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True) if act=='relu' else nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True) if act=='relu' else nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, base_filter=32, act='leakyrelu'):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # encoding path
        self.enc_block1 = DoubleConv(in_channels, base_filter, act=act)
        self.down1 = nn.MaxPool2d(2)
        self.enc_block2 = DoubleConv(base_filter, base_filter, act=act)
        self.down2 = nn.MaxPool2d(2)
        self.enc_block3 = DoubleConv(base_filter, base_filter, act=act)

        # decoding path
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(base_filter, 64, kernel_size=3, padding=1)
        self.dec_block1 = DoubleConv(base_filter+64, base_filter, act=act)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3, padding=1)
        self.dec_block2 = DoubleConv(base_filter*2, base_filter, act=act)

        self.conv3 = nn.Conv2d(base_filter, out_channels, kernel_size=3, padding=1)
        self.out = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        x = self.enc_block1(x)
        skip1 = x
        x = self.enc_block2(self.down1(x))
        skip2 = x
        x = self.conv1(self.up1(self.enc_block3(self.down2(x))))
        x = torch.cat((x, skip2), dim=1)
        x = self.conv2(self.up2(self.dec_block1(x)))
        x = torch.cat((x, skip1), dim=1)
        x = self.out(self.conv3(self.dec_block2(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            ResBlock(in_channels, 32),
            ResBlock(in_channels + 32, 32),
            nn.Conv2d(in_channels + 64, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.model(x)


class MMT_baseline_agis(nn.Module):
    def __init__(self, n_contrast):
        super(MMT_baseline_agis, self).__init__()
        self.n_contrast = n_contrast
        self.encoders = nn.ModuleList([Encoder(1, 16) for _ in range(n_contrast)])
        self.decoders = nn.ModuleList([Decoder(16, 1) for _ in range(n_contrast)])

    def encode_imgs(self, img_list, contrasts):
        return [self.encoders[contrast](img) for img, contrast in zip(img_list, contrasts)]

    def decode_imgs(self, img_code, contrasts):
        return [self.decoders[contrast](img_code) for contrast in contrasts]

    def forward(self, x, inputs, outputs):
        img_codes = self.encode_imgs(x, inputs)
        # latent code fusion
        z = torch.stack(img_codes, dim=1)
        z, _ = torch.max(z, dim=1)
        return self.decode_imgs(z, outputs)





    