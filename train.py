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
from torch.distributions import MultivariateNormal
import scipy
from scipy import linalg

def hinge_loss(output, real=True):
    return -torch.mean(torch.min(torch.zeros_like(output), -1+output)) if real else -torch.mean(torch.min(torch.zeros_like(output), -1-output))

def recon_loss(output, target):
    return torch.mean(torch.norm(output-target))

def gen_loss(output):
    return -torch.mean(output)

G = Generator()
G.apply(weights_init)
G.to(device)
D = Discriminator()
D.apply(weights_init)
D.to(device)
G_optim = optim.Adam(G.parameters())
D_optim = optim.Adam(D.parameters())
scaler1 = GradScaler()
scaler2 = GradScaler()

D_l, G_l = [], []
imgs_list = []
fixed_noise = make_noise()
cur_iter = 0
num_iters = 50000

while cur_iter < num_iters:
    for real in tqdm(dataloader):

        real = real.to(device)
        D_optim.zero_grad()
        D.make_recon(True)

        with autocast():

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
                real_part = real[:,64:192, 64:192]
            else:
                print("Invalid real shape, {}".format(real.shape))
            real_glob = F.interpolate(real, scale_factor=0.5)
            D_recon_loss = recon_loss(I_part, real_part) + recon_loss(I_glob, real_glob)
            D_loss = D_real_loss + D_fake_loss + D_recon_loss

        D_l.append(D_loss.item())
        scaler1.scale(D_loss).backward()
        scaler1.step(D_optim)
        scaler1.update()

        G_optim.zero_grad()

        with autocast():

            noise = make_noise()
            fake = G(noise)
            D_fake_pred = D(fake)
            G_loss = gen_loss(D_fake_pred)

        G_l.append(G_loss.item())
        scaler2.scale(G_loss).backward()
        scaler2.step(G_optim)
        scaler2.update()

        cur_iter += 1


        if (cur_iter) % 2243 == 0:
            print("{} / {}, D_loss: {:.4f}, G_loss: {:.4f}".format(cur_iter, num_iters, D_loss.item(), G_loss.item()))
            noise = make_noise()
            fake = G(noise)
            show_tensor_images(fake)
            imgs_list.append(G(fixed_noise).detach().cpu())

            torch.save(G.state_dict(), "G.pt")
            torch.save(D.state_dict(), "D.pt")

        del D_fake_pred, D_real_pred, I_part, I_glob, fake
        torch.cuda.empty_cache()
