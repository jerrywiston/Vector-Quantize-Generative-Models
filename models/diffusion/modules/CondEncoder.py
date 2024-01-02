import torch
from torch import nn, einsum
import torch.nn.functional as F

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

class AddPosEmb(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, f):
        with torch.no_grad():
            H, W = f.shape[-2], f.shape[-1]
            h_idx = torch.linspace(-1, 1, H)   #(h)
            w_idx = torch.linspace(-1, 1, W)   #(w)
            h_grid, w_grid = torch.meshgrid(h_idx, w_idx, indexing="ij")   
            pos_emb = torch.cat((h_grid.unsqueeze(-1), w_grid.unsqueeze(-1)), dim=-1).reshape(-1,2).to(f.get_device()) #(h*w, 2)
            pos_emb = pos_emb.transpose(1,0).reshape(1,2,H,W).repeat(f.shape[0],1,1,1)
        out = torch.cat([f, pos_emb],1)
        return out

class Encoder(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=64, n_hid=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(n_hid, n_hid),
        )
        self.proj = nn.Conv2d(n_hid, embedding_dim-2, 1)

    def forward(self, x):
        h = self.net(x)
        z = self.proj(h)
        out = AddPosEmb()(z)
        return out

class EncoderX(nn.Module):
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