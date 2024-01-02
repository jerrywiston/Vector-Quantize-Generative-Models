import numpy as np
import os

from maze3d import maze
from maze3d import maze_env
from maze3d.gen_maze_dataset_new import gen_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils

from models.vqvae import vqvae
import utils

from models.transformer.transformer_re import ImgTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Environment
maze_obj = maze.MazeGridRandom2(obj_prob=0.3)
env = maze_env.MazeBaseEnv(maze_obj, render_res=(64,64), fov=80*np.pi/180)

# VQVAE Model Parameters
h_dim = 128

#n_embeddings = 256
#embedding_dim = 3
#vqmodel_path = "vqvae2.pt"

n_embeddings = 1024
embedding_dim = 8
vqmodel_path = "vqvae_new.pt"

# VQVAE Model
vq_net = vqvae.VQVAE(h_dim, n_embeddings, embedding_dim).to(device)
vq_net.load_state_dict(torch.load(os.path.join("checkpoints",vqmodel_path)))

# Transformer model
transformer_config = {
    "vocab_size": n_embeddings,
    "block_size": 256, 
    "n_layer": 8,
    "n_head": 16,
    "n_embd": 512
}
vqtransformer = ImgTransformer(transformer_config, vq_net, n_embeddings=n_embeddings, embedding_dim=embedding_dim).to(device)
optimizer = vqtransformer.transformer.configure_optimizers(weight_decay=0.01, learning_rate=4.5e-06, betas=(0.9, 0.95), device_type=device)

# Training Parameters
max_training_iter = 200001
gen_data_size = 40
gen_dataset_iter = 1000
samp_field = 3.0
batch_size = 32

output_path = "output_transformer_re/"
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
        print("Done")
    # Get Latent Feature
    x_obs, pose_obs, _, _ = utils.get_batch(color_data, pose_data, 1, batch_size)

    # Train Transformer
    optimizer.zero_grad()
    logits, loss = vqtransformer(x_obs)
    #loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print("Iter " + str(iter).zfill(5) + " | loss: " + str(loss.item()))

        # Generate
        with torch.no_grad():
            x_samp = vqtransformer.sample(samp_size=32)
            x_fig = (x_samp.flip(1).cpu() + 1) / 2
            path = os.path.join(output_path, str(iter).zfill(4)+".jpg")
            vutils.save_image(x_fig, path, padding=2, normalize=False)

            # Save model
            torch.save(vqtransformer.state_dict(), os.path.join(save_path,"transformer.pt"))
        