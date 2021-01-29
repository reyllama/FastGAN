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

def hinge_loss(output, real=True):
    return -torch.mean(torch.min(0, -1+output)) if real else -torch.mean(torch.min(0, -1-output))

def recon_loss(output, target):
    return torch.mean(torch.norm(output, target))

def gen_loss(output):
    return -torch.mean(output)

def show_tensor_images(image_tensor, num_images=25, size=(3, 64, 64)):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.figure(figsize=(8,8))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

G = Generator()
D = Discriminator()
G_optim = optim.Adam(G.parameters())
D_optim = optim.Adam(D.parameters())

D_l, G_l = [], []
imgs_list = []
fixed_noise = make_noise()
cur_iter = 0
num_iters = 50000
while cur_iter < num_iters:
    for real in tqdm(dataloader):

        D.zero_grad()
        D.make_recon(True)
        D_real_pred, I_part, I_glob = D(real)
        D_real_loss = hinge_loss(D_real_pred)

        noise = make_noise()
        fake = G(noise)
        D.make_recon(False)
        D_fake_pred = D(fake)
        D_fake_loss = hinge_loss(D_fake_pred, real=False)

        if len(real.shape)==4:
            real_part = real[:,:,64:192, 64:192]
        elif len(real.shape)==3:
            real_part = [:,64:192, 64:192]
        else:
            raise ValueError, "Invalid real shape, {}".format(real.shape)
        real_glob = F.interpolate(real, scale_factor=0.5)
        D_recon_loss = recon_loss(I_part, real_part) + recon_loss(I_glob, real_glob)
        D_loss = D_real_loss + D_fake_loss + D_recon_loss
        D_l.append(D_loss.item())
        D_loss.backward()
        D_optim.step()

        G.zero_grad()
        noise = make_noise()
        fake = G(noise)
        D_fake_pred = D(fake)
        G_loss = gen_loss(D_fake_pred)
        G_l.append(G_loss.item())
        G_loss.backward()
        G_optim.step()

    if (cur_iter+1) % 1000 == 0:
        print("{} / {}, D_loss: {:.4f}, G_loss: {:.4f}".format(cur_iter+1, num_iters, D_loss.item(), G_loss.item()))
        show_tensor_images(fake)
        imgs_list.append(G(fixed_noise))
