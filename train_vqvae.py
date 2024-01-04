import numpy as np
import os

from maze3d import maze
from maze3d import maze_env
from maze3d.gen_maze_dataset_new import gen_dataset

import torch
import torch.optim as optim
import torchvision.utils as vutils

from models.vqvae import losses, vqvae
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Environment
maze_obj = maze.MazeGridRandom2(obj_prob=0.3)
env = maze_env.MazeBaseEnv(maze_obj, render_res=(64,64), fov=80*np.pi/180)

# Training Parameters
max_training_iter = 100001
gen_data_size = 80
gen_dataset_iter = 1000
samp_field = 3.0
batch_size = 32

# Model Parameters
config = {
    "h_dim": 128,
    "n_embeddings": 1024, #256
    "embedding_dim": 8, #3
}

# Model and Optimizer
net = vqvae.VQVAE(**config).to(device)
vggnet = losses.Vgg16().to(device)
disc = losses.Discriminator().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4, amsgrad=True)
optimizer_disc = optim.Adam(disc.parameters(), lr=1e-4, amsgrad=True)

save_path = "checkpoints"
exp_path = "experiments"
model_name = "vqvae"
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
    x_obs, pose_obs, _, _ = utils.get_batch(color_data, pose_data, 1, batch_size)
    optimizer.zero_grad()
    optimizer_disc.zero_grad()

    x_hat, latent_loss, ind = net(x_obs)
    recon_loss = torch.mean((x_hat - x_obs)**2)
    perceptual_loss = losses.perceptual_loss(vggnet, x_hat, x_obs)
    d_loss, g_loss = losses.patch_discriminator_loss(disc, x_hat, x_obs)
    loss = recon_loss + perceptual_loss + latent_loss + g_loss

    loss.backward()
    optimizer.step()

    d_loss.backward()
    optimizer_disc.step()

    if iter % 100 == 0:
        print("Iter " + str(iter).zfill(5) + " | Rec_loss: " + str(recon_loss.item()) + " | Emb_loss: " + str(latent_loss.item()) \
            + " | Perceptual_loss: " + str(perceptual_loss.item()) + " | G_loss: " + str(g_loss.item()) + " | D_loss: " + str(d_loss.item()) )

        # Generate reconstructed samples
        x_rec = x_hat.detach()
        x_fig = torch.cat([x_obs[0:8], x_rec[0:8], x_obs[8:16], x_rec[8:16]], 0)
        x_fig = (x_fig.flip(1).cpu() + 1) / 2
        path = os.path.join(results_path, str(iter).zfill(4)+".jpg")
        vutils.save_image(x_fig, path, padding=2, normalize=False)

        # Save model
        torch.save(net.state_dict(), os.path.join(save_path, model_name+".pt"))
