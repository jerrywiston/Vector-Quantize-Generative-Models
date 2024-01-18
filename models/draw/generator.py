"""
Modified by https://github.com/wohlert/generative-query-network-pytorch/blob/master/gqn/generator.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, ch=64):
        super().__init__()
        # Down
        self.conv1 = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch*2, 3, padding=1)
        self.conv3 = nn.Conv2d(ch*2, ch*2, 3, padding=1)
        # Up
        self.conv4 = nn.Conv2d(ch*2+ch*2, ch*2, 3, padding=1)
        self.conv5 = nn.Conv2d(ch*2+ch, ch, 3, padding=1)
        self.conv6 = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, x):
        # 16
        h1 = torch.relu(self.conv1(x))
        # 16
        h2 = nn.MaxPool2d(2, stride=2)(h1)
        h2 = torch.relu(self.conv2(h2))
        # 8
        h3 = nn.MaxPool2d(2, stride=2)(h2)
        h3 = torch.relu(self.conv3(h3))
        # 4
        h4 = nn.Upsample(scale_factor=2)(h3)
        h4 = torch.relu(self.conv4(torch.cat((h4,h2), 1)))
        # 8
        h5 = nn.Upsample(scale_factor=2)(h4)
        h5 = torch.relu(self.conv5(torch.cat((h5,h1), 1)))
        # 16
        out = self.conv6(h5)
        return out

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvGRUCell, self).__init__()
        in_channels += out_channels
        
        self.reset_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.update_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.state_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input, state):
        input_cat1 = torch.cat((state, input), dim=1)
        reset_gate = torch.sigmoid(self.reset_conv(input_cat1))
        update_gate = torch.sigmoid(self.update_conv(input_cat1))

        state_reset = reset_gate * state
        input_cat2 = torch.cat((state_reset, input), dim=1)
        state_update = (1-update_gate)*state + update_gate*torch.tanh(self.state_conv(input_cat2))
        return state_update 

class UNetGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels += out_channels
        self.unet_gate = UNet(in_channels, out_channels*2)
        self.unet_state = UNet(in_channels, out_channels)

        self.reset_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.update_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.state_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input, state):
        input_cat1 = torch.cat((state, input), dim=1)
        gates = self.unet_gate(input_cat1)
        reset_gate, update_gate = torch.chunk(gates, 2, 1)        
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        state_reset = reset_gate * state
        input_cat2 = torch.cat((state_reset, input), dim=1)
        state_update = (1-update_gate)*state + update_gate*torch.tanh(self.unet_state(input_cat2))
        return state_update 

class GeneratorNetwork(nn.Module):
    def __init__(self, x_dim, z_dim=32, h_dim=128, L=6, share=True):
        super(GeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.share = share

        # Core computational units
        inference_args = dict(in_channels=h_dim + x_dim, out_channels=h_dim)
        generator_args = dict(in_channels=z_dim, out_channels=h_dim)
        if self.share:
            self.inference_core = UNetGRUCell(**inference_args)
            self.generator_core = UNetGRUCell(**generator_args)
        else:
            self.inference_core = nn.ModuleList([UNetGRUCell(**inference_args) for _ in range(L)])
            self.generator_core = nn.ModuleList([UNetGRUCell(**generator_args) for _ in range(L)])

        # Inference, prior
        self.posterior_density = nn.Conv2d(h_dim, 2*z_dim, kernel_size=3, padding=1)
        self.prior_density     = nn.Conv2d(h_dim, 2*z_dim, kernel_size=3, padding=1)

        # Generative density
        self.in_proj = nn.Conv2d(x_dim, x_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(h_dim, x_dim, kernel_size=1)
        self.draw_proj = nn.Conv2d(h_dim, h_dim, kernel_size=1)

    def forward(self, x):
        batch_size, _, h, w = x.shape
        kl = 0
        x = self.in_proj(x)

        # Reset hidden and cell state
        hidden_i = x.new_zeros((batch_size, self.h_dim, h, w))
        hidden_g = x.new_zeros((batch_size, self.h_dim, h, w))

        # Canvas for updating
        u = x.new_zeros((batch_size, self.h_dim, h, w))

        for l in range(self.L):
            # Prior factor (eta π network)
            p_mu, p_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            # Inference state update
            inference = self.inference_core if self.share else self.inference_core[l]
            hidden_i = inference(torch.cat([hidden_g, x], dim=1), hidden_i)

            # Posterior factor (eta e network)
            q_mu, q_std = torch.chunk(self.posterior_density(hidden_i), 2, dim=1)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            # Posterior sample
            z = posterior_distribution.rsample()

            # Generator state update
            generator = self.generator_core if self.share else self.generator_core[l]
            hidden_g = generator(z, hidden_g)

            # Calculate u
            u = self.draw_proj(hidden_g) + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.out_proj(u)
        kl = torch.mean(torch.sum(kl, dim=[1,2,3]))
        return x_mu, kl

    def sample(self, z_shape, batch_size, noise=True):
        h, w = z_shape

        # Reset hidden and cell state for generator
        device = next(self.parameters()).device
        hidden_g = torch.zeros((batch_size, self.h_dim, h, w), device=device)

        u = torch.zeros((batch_size, self.h_dim, h, w), device=device)

        for l in range(self.L):
            p_mu, p_log_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))

            # Prior sample
            if noise:
                z = prior_distribution.sample()
            else:
                z = p_mu

            # Calculate u
            generator = self.generator_core if self.share else self.generator_core[l]
            hidden_g = generator(z, hidden_g)
            u = self.draw_proj(hidden_g) + u

        x_mu = self.out_proj(u)
        return x_mu

class CondGeneratorNetwork(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=32, h_dim=128, L=6, scale=1, share=True):
        super(CondGeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.share = share
        self.scale = scale

        # Core computational units
        inference_args = dict(in_channels=c_dim + h_dim + x_dim, out_channels=h_dim)
        generator_args = dict(in_channels=c_dim + z_dim, out_channels=h_dim)
        if self.share:
            self.inference_core = UNetGRUCell(**inference_args)
            self.generator_core = UNetGRUCell(**generator_args)
        else:
            self.inference_core = nn.ModuleList([UNetGRUCell(**inference_args) for _ in range(L)])
            self.generator_core = nn.ModuleList([UNetGRUCell(**generator_args) for _ in range(L)])

        # Inference, prior
        self.posterior_density = nn.Conv2d(h_dim, 2*z_dim, kernel_size=3, padding=1)
        self.prior_density     = nn.Conv2d(h_dim, 2*z_dim, kernel_size=3, padding=1)

        # Generative density
        self.in_proj = nn.Conv2d(x_dim, x_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(h_dim, x_dim, kernel_size=1)
        self.draw_proj = nn.Conv2d(h_dim, h_dim, kernel_size=1)
        
    def forward(self, x, c):
        batch_size, _, h, w = x.shape
        kl = 0

        # Downsample x, upsample v and r
        x = self.in_proj(x)

        # Reset hidden and cell state
        hidden_i = x.new_zeros((batch_size, self.h_dim, h, w))
        hidden_g = x.new_zeros((batch_size, self.h_dim, h, w))

        # Canvas for updating
        u = x.new_zeros((batch_size, self.h_dim, h, w))

        for l in range(self.L):
            # Prior factor (eta π network)
            p_mu, p_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            # Inference state update
            inference = self.inference_core if self.share else self.inference_core[l]
            cond_inf = F.interpolate(c, size=x.shape[-2:], mode='nearest')
            hidden_i = inference(torch.cat([hidden_g, x, cond_inf], dim=1), hidden_i)

            # Posterior factor (eta e network)
            q_mu, q_std = torch.chunk(self.posterior_density(hidden_i), 2, dim=1)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            # Posterior sample
            z = posterior_distribution.rsample()

            # Generator state update
            generator = self.generator_core if self.share else self.generator_core[l]
            cond_gen = F.interpolate(c, size=z.shape[-2:], mode='nearest')
            hidden_g = generator(torch.cat([z, cond_gen], dim=1), hidden_g)

            # Calculate u
            u = self.draw_proj(hidden_g) + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.out_proj(u)
        kl = torch.mean(torch.sum(kl, dim=[1,2,3]))
        return x_mu, kl

    def sample(self, z_shape, c, noise=True):
        h, w = z_shape
        batch_size = c.size(0)

        # Reset hidden and cell state for generator
        hidden_g = c.new_zeros((batch_size, self.h_dim, h, w))

        u = c.new_zeros((batch_size, self.h_dim, h, w))

        for l in range(self.L):
            p_mu, p_log_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))

            # Prior sample
            if noise:
                z = prior_distribution.sample()
            else:
                z = p_mu

            # Calculate u
            generator = self.generator_core if self.share else self.generator_core[l]
            cond_gen = F.interpolate(c, size=z.shape[-2:], mode='nearest')
            hidden_g = generator(torch.cat([z, cond_gen], dim=1), hidden_g)
            u = self.draw_proj(hidden_g) + u

        x_mu = self.out_proj(u)
        return x_mu
    