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

from models.diffusion.modules.CondUNet import UNet, CondUNet
from models.diffusion.modules.CondEncoder import Encoder
from models.diffusion.engine import CondGaussianDiffusionTrainer, CondDDPMSampler, CondDDIMSampler
from models.diffusion.utils import load_yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Environment
maze_obj = maze.MazeGridRandom2(obj_prob=0.3)
env = maze_env.MazeBaseEnv(maze_obj, render_res=(64,64), fov=80*np.pi/180)

# VQVAE Model Parameters
h_dim = 128
n_embeddings = 256
embedding_dim = 3

# VQVAE Model
vq_net = vqvae.VQVAE(h_dim, n_embeddings, embedding_dim).to(device)
vq_net.load_state_dict(torch.load(os.path.join("checkpoints","vqvae2.pt")))

# Conditional Encoder
cond_dim = 64
cond_net = Encoder(input_channels=1, embedding_dim=cond_dim).to(device)
# Diffusion Model
config = load_yaml("config_ldm.yml", encoding="utf-8")
diff_net = CondUNet(**config["Model"], context_channels=cond_dim).to(device)
optimizer = optim.AdamW(list(diff_net.parameters())+list(cond_net.parameters()), lr=0.0002, weight_decay=1e-4)
trainer = CondGaussianDiffusionTrainer(diff_net, **config["Trainer"]).to(device)
trainer.train()
sampler = CondDDIMSampler(diff_net, beta=[0.0001, 0.02], T=1000).to(device)

# Training Parameters
max_training_iter = 200000
gen_data_size = 40
gen_dataset_iter = 1000
samp_field = 3.0
batch_size = 32

output_path = "output_ldm_cond/"
save_path = "checkpoints"
if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Training Iteration
for iter in range(max_training_iter):
    if iter % gen_dataset_iter == 0:
        color_data, depth_data, pose_data = gen_dataset(env, gen_data_size, samp_range=samp_field, samp_size=10)
        color_data = (color_data.astype(float) / 255.)*2-1
        depth_data = depth_data.astype(float) / 5.0
        print("Done")
    # Get Latent Feature
    x_obs, pose_obs, depth_obs, _, _, _ = utils.get_batch_depth(color_data, pose_data, depth_data, 1, batch_size)
    z = vq_net.encoder(x_obs).detach()
    c = cond_net(depth_obs)

    # Train Diffusion
    optimizer.zero_grad()
    loss = trainer(z, c)
    loss.backward()
    optimizer.step()

    if iter % 1000 == 0:
        print("Iter " + str(iter).zfill(5) + " | diffusion_loss: " + str(loss.item()))

        # Generate
        with torch.no_grad():
            z_t = torch.randn((batch_size, embedding_dim, 16, 16), device=device)
            z_0 = sampler(z_t, c, only_return_x_0=True, interval=50, steps=100)
            z_q, _, _ = vq_net.quantizer(z_0)
            x_samp = vq_net.decoder(z_q)
            x_fig = (x_samp.flip(1).cpu() + 1) / 2
            gt_fig = (x_obs.flip(1).cpu() + 1) / 2
            depth_fig = depth_obs.repeat(1,3,1,1).cpu()
            out_fig = torch.cat([depth_fig[0:8], x_fig[0:8], gt_fig[0:8], depth_fig[8:16], x_fig[8:16], gt_fig[8:16]], 0)
            path = os.path.join(output_path, str(iter).zfill(4)+".jpg")
            vutils.save_image(out_fig, path, padding=2, normalize=False)

            # Save model
            torch.save(diff_net.state_dict(), os.path.join(save_path,"ldm_cond.pt"))
        