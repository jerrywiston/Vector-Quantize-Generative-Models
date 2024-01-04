import numpy as np
import os

import torch
import torchvision.utils as vutils

from models.vqvae import vqvae
import utils

from models.diffusion.unet import UNet
from models.diffusion.core import DDPMSampler, DDIMSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VQVAE Model Parameters
h_dim = 128
n_embeddings = 1024#256
embedding_dim = 8#3

# VQVAE Model
vq_net = vqvae.VQVAE(h_dim, n_embeddings, embedding_dim).to(device)
vq_net.load_state_dict(torch.load(os.path.join("checkpoints","vqvae_new.pt")))

# Diffusion Model
unet_config = {
    "in_channels": 8,
    "out_channels": 8,
    "model_channels": 128,
    "attention_resolutions": [1, 2, ],
    "num_res_blocks": 2,
    "dropout": 0.1,
    "channel_mult": [1, 2, 2, 2],
    "conv_resample": True,
    "num_heads": 4
}
diff_config = {"T": 1000, "beta": [0.0001, 0.02]}

diff_net = UNet(**unet_config).to(device)
diff_net.load_state_dict(torch.load(os.path.join("checkpoints","ldm.pt")))
ddpm_sampler = DDPMSampler(diff_net, **diff_config).to(device)
ddim_sampler = DDIMSampler(diff_net, **diff_config).to(device)

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
