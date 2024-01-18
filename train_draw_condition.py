import numpy as np
import os

from maze3d import maze
from maze3d import maze_env
from maze3d.gen_maze_dataset_new import gen_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from models.vqvae import vqvae
from models.draw import generator
from models.diffusion.cond_encoder import Encoder

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Environment
maze_obj = maze.MazeGridRandom2(obj_prob=0.3)
env = maze_env.MazeBaseEnv(maze_obj, render_res=(64,64), fov=80*np.pi/180)

# VQVAE Model
h_dim = 128
n_embeddings = 1024
embedding_dim = 8
vqmodel_path = "vqvae.pt"
vq_net = vqvae.VQVAE(h_dim, n_embeddings, embedding_dim).to(device)
vq_net.load_state_dict(torch.load(os.path.join("checkpoints",vqmodel_path)))

# Conditional Encoder
cond_dim = 64
cond_net = Encoder(input_channels=1, embedding_dim=cond_dim, add_pos=False).to(device)

# DRAW Model
draw_net = generator.CondGeneratorNetwork(x_dim=embedding_dim, z_dim=32, h_dim=128, c_dim=cond_dim, L=6, share=True).to(device)
optimizer = optim.AdamW(draw_net.parameters(), lr=0.0002, weight_decay=1e-4)

# Training Parameters
max_training_iter = 100001
gen_data_size = 80
gen_dataset_iter = 1000
samp_field = 3.0
batch_size = 32

save_path = "checkpoints"
exp_path = "experiments"
model_name = "draw_cond"
results_path = os.path.join(exp_path, model_name)

if not os.path.exists(exp_path):
    os.makedirs(exp_path)
if not os.path.exists(results_path):
    os.mkdir(os.path.join(results_path))
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Load trained weight
if os.path.exists(os.path.join(save_path, model_name+".pt")):
    print("Load trained weights ...")
    draw_net.load_state_dict(torch.load(os.path.join(save_path, model_name+".pt")))
    cond_net.load_state_dict(torch.load(os.path.join(save_path, model_name+"_condnet.pt")))

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

    # Train DRAW
    optimizer.zero_grad()
    z_rec, kl = draw_net(z, c)
    loss = nn.MSELoss()(z, z_rec) + 0.001*kl
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print("Iter " + str(iter).zfill(5) + " | draw_loss: " + str(loss.item())+ " | kl: " + str(kl.item()))

        # Generate
        with torch.no_grad():
            z_samp = draw_net.sample(z_shape=(16,16), c=c)
            z_q, _, _ = vq_net.quantizer(z_samp)
            x_samp = vq_net.decoder(z_q)
            x_fig = (x_samp.flip(1).cpu() + 1) / 2
            gt_fig = (x_obs.flip(1).cpu() + 1) / 2
            depth_fig = depth_obs.repeat(1,3,1,1).cpu()
            out_fig = torch.cat([depth_fig[0:8], x_fig[0:8], gt_fig[0:8], depth_fig[8:16], x_fig[8:16], gt_fig[8:16]], 0)
            path = os.path.join(results_path, str(iter).zfill(4)+".jpg")
            vutils.save_image(out_fig, path, padding=2, normalize=False)

            # Save model
            torch.save(draw_net.state_dict(), os.path.join(save_path, model_name+".pt"))
            torch.save(cond_net.state_dict(), os.path.join(save_path, model_name+"_condnet.pt"))