import numpy as np
import os

from maze3d import maze
from maze3d import maze_env
from maze3d.gen_maze_dataset_new import gen_dataset

import torch
import torch.optim as optim
import torchvision.utils as vutils

from models.vqvae import vqvae
import utils

from models.diffusion.unet import UNet
from models.diffusion.core import GaussianDiffusionTrainer, DDPMSampler, DDIMSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Environment
maze_obj = maze.MazeGridRandom2(obj_prob=0.3)
env = maze_env.MazeBaseEnv(maze_obj, render_res=(64,64), fov=80*np.pi/180)

# VQVAE Model 
h_dim = 128
n_embeddings = 1024#256
embedding_dim = 8#3
vqmodel_path = "vqvae.pt"
vq_net = vqvae.VQVAE(h_dim, n_embeddings, embedding_dim).to(device)
vq_net.load_state_dict(torch.load(os.path.join("checkpoints", vqmodel_path)))

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
optimizer = optim.AdamW(diff_net.parameters(), lr=0.0002, weight_decay=1e-4)
trainer = GaussianDiffusionTrainer(diff_net, **diff_config).to(device)
trainer.train()
sampler = DDIMSampler(diff_net, **diff_config).to(device)
#sampler = DDPMSampler(diff_net, **diff_config).to(device)

# Training Parameters
max_training_iter = 100000
gen_data_size = 80
gen_dataset_iter = 1000
samp_field = 3.0
batch_size = 32

save_path = "checkpoints"
exp_path = "experiments"
model_name = "ldm"
results_path = os.path.join(exp_path, model_name)

if not os.path.exists(exp_path):
    os.makedirs(exp_path)
if not os.path.exists(results_path):
    os.mkdir(os.path.join(results_path))
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Training Iteration
for iter in range(max_training_iter):
    if iter % gen_dataset_iter == 0:
        color_data, depth_data, pose_data = gen_dataset(env, gen_data_size, samp_range=samp_field, samp_size=10)
        color_data = (color_data.astype(float) / 255.)*2-1
        print("Done")
    # Get Latent Feature
    x_obs, pose_obs, _, _ = utils.get_batch(color_data, pose_data, 1, batch_size)
    z = vq_net.encoder(x_obs).detach()

    # Train Diffusion
    optimizer.zero_grad()
    loss = trainer(z)
    loss.backward()
    optimizer.step()

    if iter % 1000 == 0:
        print("Iter " + str(iter).zfill(5) + " | diffusion_loss: " + str(loss.item()))

        # Generate
        with torch.no_grad():
            z_t = torch.randn((batch_size, embedding_dim, 16, 16), device=device)
            z_0 = sampler(z_t, only_return_x_0=True, interval=50, steps=100)
            z_q, _, _ = vq_net.quantizer(z_0)
            x_samp = vq_net.decoder(z_q)
            x_fig = (x_samp.flip(1).cpu() + 1) / 2
            path = os.path.join(results_path, str(iter).zfill(4)+".jpg")
            vutils.save_image(x_fig, path, padding=2, normalize=False)

            # Save model
            torch.save(diff_net.state_dict(), os.path.join(save_path, model_name+".pt"))
        