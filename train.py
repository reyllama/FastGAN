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
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import *
from utils import *
from dataset import *
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', required=True, , help="Data Path")
parser.add_argument("--model_path", required=False, help="Model Checkpoint Path")
parser.add_argument("--batch_size", required=False, default=8, type=int, help="Batch Size")
parser.add_argument("--learning_rate", required=False, default=2e-4, help='Learning Rate')
parser.add_argument("--beta1", required=False, default=0.5, help="Beta1 of Adam")
parser.add_argument("--beta2", required=False, default=0.999, help="Beta2 of Adam")
parser.add_argument("--continued", required=False, default=False, type=Bool)
parser.add_argument("--num_iters", required=False, default=5e4, type=int, help="# of iterations")
parser.add_argument("--show_iters", required=False, default=1e3, type=int, help='Intervals at which show snapshots')

args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root = args.data_path

dataset = ImageSet(transform=transform, root=root)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = is_cuda()

inception = inception_v3(pretrained=True) # For computation of FID score

inception.fc = nn.Identity()
inception.to(device)

G = Generator()
D = Discriminator()

if args.continued == True:

    model_path = args.model_path

    G.load_state_dict(torch.load(model_path + "/G.pt"))
    D.load_state_dict(torch.load(model_path + "/D.pt"))

G.to(device)
D.to(device)

lr = args.learning_rate
beta1 = args.beta1
beta2 = args.beta2
batch_size = args.batch_size

G_optim = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optim = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
scaler1 = GradScaler()
scaler2 = GradScaler()

D_l, G_l = [], []
imgs_list = []
fixed_noise = make_noise()

def train(num_iters, show_iter=1000):

    cur_iter = 0

    while cur_iter < num_iters:
        for real in tqdm(dataloader):

            real = real.to(device)
            D_optim.zero_grad()
            D.make_recon(True)

            with autocast():

                D_real_pred, I_part, I_glob = D(real)
                D_real_loss = hinge_loss(D_real_pred)

                z = make_noise()

                fake = G(z)
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

                z = make_noise()
                fake = G(z)
                D_fake_pred = D(fake)
                G_adv_loss = gen_loss(D_fake_pred)
                ### For Sequence Generation Only
#                 bound_fake = torch.stack([fake[0], fake[-1]])
#                 bound_real = torch.stack([real[0], real[-1]])
#                 G_recon_loss = recon_loss(bound_fake, bound_real)
                G_loss = G_adv_loss

            G_l.append(G_loss.item())
            scaler2.scale(G_loss).backward()
            scaler2.step(G_optim)
            scaler2.update()

            cur_iter += 1


            if (cur_iter) % show_iter == 0:
                print("{} / {}, D_loss: {:.4f}, G_loss: {:.4f}".format(cur_iter, num_iters, D_loss.item(), G_loss.item()))
                # show_tensor_images(fake.float())
                imgs_list.append(G(fixed_noise).detach().cpu())

                torch.save(G.state_dict(), "G.pt")
                torch.save(D.state_dict(), "D.pt")
                np.save(D_l, "D_loss.npy")
                np.save(G_l, "G_loss.npy")
                with open("gen_images.txt", "wb") as f:
                    pickle.dump(imgs_list, f)

            del D_fake_pred, D_real_pred, I_part, I_glob, fake
            torch.cuda.empty_cache()

num_iters = args.num_iters
show_iter = args.show_iters

if __name__ == "__main__":

    train(num_iters=num_iters, show_iter=show_iter)
