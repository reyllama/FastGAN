import torch
import numpy as np
import glob
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.utils as vutils
from torchvision.utils import make_grid
import pandas as pd

class ImageSet(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.imgs = sorted(glob.glob(os.path.join(root, "*.*")))

    def __getitem__(self, index):
        img = Image.open(self.imgs[index % len(self.imgs)])
        img = to_rgb(img)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
