import numpy as np
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
import pandas as pd
from torchvision.models import inception_v3
from torch.distributions import MultivariateNormal
import scipy
from scipy import linalg

def is_cuda():
    if torch.cuda.is_available():
        print("CUDA available")
        return "cuda"
    else:
        print("No CUDA. Working on CPU.")
        return "cpu"

def show_tensor_images(image_tensor, num_images=8, size=(3, 64, 64), nrow=4, figsize=8):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def to_rgb(img):
    rgb_img = Image.new("RGB", img.size)
    rgb_img.paste(img)
    return rgb_img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def make_noise(n_samples=batch_size, z_dim=1024, device="cuda"):
    noise = torch.randn(n_samples, z_dim, device=device)
    return noise[:,:,None,None]

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def hinge_loss(output, real=True):
    return -torch.mean(torch.min(torch.zeros_like(output), -1+output)) if real else -torch.mean(torch.min(torch.zeros_like(output), -1-output))

def gen_loss(output):
    return -torch.mean(output)

def matrix_sqrt(x):
    y = x.cpu().detach().numpy()
    y = linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)

def frechet_distance(mu_x, mu_y, sig_x, sig_y):
    return torch.norm(mu_x-mu_y).pow(2) + torch.trace(sig_x+sig_y-2*matrix_sqrt(torch.matmul(sig_x, sig_y)))

def preprocess(img):
    return F.interpolate(img, size=(299,299), mode='bilinear', align_corners=False)

def get_cov(x):
    return torch.Tensor(np.cov(x.detach().numpy(), rowvar=False))
