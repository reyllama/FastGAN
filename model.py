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
