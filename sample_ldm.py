import numpy as np
import os

import torch
import torch.optim as optim
import torchvision.utils as vutils

from models.vqvae import vqvae
import utils

from models.diffusion.modules.UNet import UNet
from models.diffusion.core import DDPMSampler, DDIMSampler
from models.diffusion.utils import load_yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VQVAE Model Parameters
h_dim = 128
n_embeddings = 1024#256
embedding_dim = 8#3

# VQVAE Model
vq_net = vqvae.VQVAE(h_dim, n_embeddings, embedding_dim).to(device)
vq_net.load_state_dict(torch.load(os.path.join("checkpoints","vqvae_new.pt")))

# Diffusion Model
config = load_yaml("config_ldm2.yml", encoding="utf-8")
diff_net = UNet(**config["Model"]).to(device)
diff_net.load_state_dict(torch.load(os.path.join("checkpoints","ldm.pt")))
ddpm_sampler = DDPMSampler(diff_net, beta=[0.0001, 0.02], T=1000).to(device)
ddim_sampler = DDIMSampler(diff_net, beta=[0.0001, 0.02], T=1000).to(device)

samp_size = 64

print("DDPM_sample")
z_t = torch.randn((samp_size, embedding_dim, 16, 16), device=device)
z_0 = ddpm_sampler(z_t, only_return_x_0=True, interval=50)
z_q, _, _ = vq_net.quantizer(z_0)
x_samp = vq_net.decoder(z_q)
x_fig = (x_samp.flip(1).cpu() + 1) / 2
path = os.path.join("output_ldm", "ddpm_sample.jpg")
vutils.save_image(x_fig, path, padding=2, normalize=False)

print("DDPM_sample")
z_t = torch.randn((samp_size, embedding_dim, 16, 16), device=device)
z_0 = ddim_sampler(z_t, only_return_x_0=True, interval=50, steps=100)
z_q, _, _ = vq_net.quantizer(z_0)
x_samp = vq_net.decoder(z_q)
x_fig = (x_samp.flip(1).cpu() + 1) / 2
path = os.path.join("output_ldm", "ddim_sample.jpg")
vutils.save_image(x_fig, path, padding=2, normalize=False)
