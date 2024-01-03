"""
taken from: https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/deepmind_enc_dec.py
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from codebook import Codebook

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=3, n_hid=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
        )
        self.proj = nn.Conv2d(2*n_hid, embedding_dim, 1)

    def forward(self, x):
        h = self.net(x)
        z = self.proj(h)
        return z

class Decoder(nn.Module):
    def __init__(self, n_init=32, n_hid=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------------------------

class VQVAE(nn.Module):
    def __init__(self, h_dim, n_embeddings, embedding_dim, input_channels=3):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, embedding_dim=embedding_dim, n_hid=h_dim)
        self.quantizer = Codebook(n_embeddings, embedding_dim)
        self.decoder = Decoder(n_init=embedding_dim, n_hid=h_dim, output_channels=input_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind
