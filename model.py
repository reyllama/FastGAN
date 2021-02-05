import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.models import inception_v3
from torchvision.utils import makegrid
import numpy as np
import random
import os
import time
from PIL import Image
import glob
from IPython.display import HTML
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

device = "cuda" if torch.cuda.is_available() else "cpu"
inception = inception_v3(pretrained=True) # For computation of FID score

class SLE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=4),
            nn.Conv2d(in_channel, in_channel, 4, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channel, in_channel//8, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, high, low):
        x = self.block(low)
        return high * x

def make_noise(z_dim=256):
    noise = torch.randn(256)
    return noise[:,None,None]

class Generator(nn.Module):
    def __init__(self, z_dim=256, out_res=256):
        super().__init__()
        assert out_res == 256, "Only Output Resolution of 256x256 Implemented, got {}".format(out_res)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, z_dim, 4, 1, 0),
            nn.BatchNorm2d(z_dim),
            nn.GLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(z_dim, 2*z_dim, 3, 1, 1),
            nn.BatchNorm2d(2*z_dim),
            nn.GLU()
        )
        self.block2 = make_block(2*z_dim, z_dim)
        self.block3 = make_block(z_dim, z_dim//2)
        self.block4 = make_block(z_dim//2, z_dim//4)
        self.block5 = make_block(z_dim//4, z_dim//4)
        self.block6 = make_block(z_dim//4, z_dim//8)
        self.out = nn.Sequential(
            nn.Conv2d(z_dim//8, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.SLE1 = SLE(512)
        self.SLE2 = SLE(256)


    def make_block(self, in_channel, out_channel):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.GLU()
        )
        return block

    def forward(self, x):
        h1 = self.block1(x)     # 512 x 8 x 8
        h2 = self.block2(h1)    # 256 x 16 x 16
        x = self.block3(h2)     # 128 x 32 x 32
        x = self.block4(x)      # 64 x 64 x 64
        x = self.block5(x)      # 64 x 128 x 128
        x = self.SLE1(x, h1)    # 64 x 128 x 128
        x = self.block6(x)      # 32 x 256 x 256
        x = self.SLE2(x, h2)    # 32 x 256 x 256
        x = self.out(x)         # 3 x 256 x 256
        return x

class Decoder(nn.Module):
    def __init__(self, in_feature=32):
        super().__init__()
        self.in_feature = in_feature
        g = []
        for _ in range(3):
            g += [make_block(in_feature)]
        g += [make_block(3)]
        self.decoder = nn.Sequential(*g)

    def make_block(self, out_feature):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.in_feature, out_feature),
            nn.BatchNorm2d(out_feature),
            nn.GLU()
        )

    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim=64, in_res=256):
        super().__init__()
        assert in_res == 256, "Only Output Resolution of 256x256 Implemented, got {}".format(in_res)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, hidden_dim//2, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim//2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.block2 = make_block(hidden_dim, hidden_dim)
        self.skip2 = down_sample(hidden_dim, hidden_dim)
        self.block3 = make_block(hidden_dim, hidden_dim//2)
        self.skip3 = down_sample(hidden_dim, hidden_dim//2)
        self.block4 = make_block(hidden_dim//2, hidden_dim//2)
        self.skip4 = down_sample(hidden_dim//2, hidden_dim//2)
        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim//2, hidden_dim//4, 1, 1, 0),
            nn.BatchNorm2d(hidden_dim//4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim//4, 1, 4, 1, 0)
        )
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

    def make_recon(self, recon=True):
        self.recon = recon

    def forward(self, x):
        y = self.block1(x)
        y1 = self.block2(y)
        y2 = self.skip2(y)
        y = y1 + y2
        y1 = self.block3(y)
        y2 = self.skip3(y)
        h1 = y1 + y2        # 32 x 16 x 16 : For cropping
        # Simply center crop for now, where the literature implemented random crop
        if len(h1.shape)==4:
            h1 = h1[:,:,4:12, 4:12]
        elif len(h1.shape)==3:
            h1 = h1[:, 4:12, 4:12]
        else:
            raise ValueError, "invalid shape for feature map to be cropped, {}".format(h1.shape)
        y1 = self.block4(h1)
        y2 = self.skip4(h1)
        h2 = y1 + y2        # 32 x 8 x 8
        y = self.out(h2)    # 1 x 5 x 5
        if self.recon is True:
            y_part = self.decoder1(h1)
            y_recon = self.decoder2(h2)
            # y: 5 x 5 true/false
            # y_part: reconstructed image from center y_part
            # y_recon: reconstructed image from whole feature map
            return y, y_part, y_recon
        else:
            return y

    def make_block(self, in_channel, out_channel):
        block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 2, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )
        return block

    def down_sample(self, in_channel, out_channel):
        block = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )
        return block


class encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Conv2d()

    def forward(self, x):
        return self.block(x)
