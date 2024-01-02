"""
The inference-generator architecture is conceptually
similar to the encoder-decoder pair seen in variational
autoencoders. The difference here is that the model
must infer latents from a cascade of time-dependent inputs
using convolutional and recurrent networks.
Additionally, a representation vector is shared between
the networks.
Modified by https://github.com/wohlert/generative-query-network-pytorch/blob/master/gqn/generator.py
"""
#SCALE = 4 # Scale of image generation process

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from padding_same_conv import Conv2d

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvGRUCell, self).__init__()
        kwargs = dict(kernel_size=kernel_size, stride=stride)
        in_channels += out_channels
        
        self.reset_conv = Conv2d(in_channels, out_channels, **kwargs)
        self.update_conv = Conv2d(in_channels, out_channels, **kwargs)
        self.state_conv = Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, state):
        input_cat1 = torch.cat((state, input), dim=1)
        reset_gate = torch.sigmoid(self.reset_conv(input_cat1))
        update_gate = torch.sigmoid(self.update_conv(input_cat1))

        state_reset = reset_gate * state
        input_cat2 = torch.cat((state_reset, input), dim=1)
        state_update = (1-update_gate)*state + update_gate*torch.tanh(self.state_conv(input_cat2))
        return state_update 

class GeneratorNetwork(nn.Module):
    """
    Network similar to a convolutional variational
    autoencoder that refines the generated image
    over a number of iterations.
    :param x_dim: number of channels in input
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: number of density refinements
    :param share: whether to share cores across refinements
    """
    def __init__(self, x_dim,  z_dim=32, h_dim=128, L=6, scale=2, share=True):
        super(GeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.share = share
        self.scale = scale

        # Core computational units
        kwargs = dict(kernel_size=5, stride=1)
        inference_args = dict(in_channels=h_dim + x_dim, out_channels=h_dim, **kwargs)
        generator_args = dict(in_channels=z_dim, out_channels=h_dim, **kwargs)
        if self.share:
            self.inference_core = ConvGRUCell(**inference_args)
            self.generator_core = ConvGRUCell(**generator_args)
        else:
            self.inference_core = nn.ModuleList([ConvGRUCell(**inference_args) for _ in range(L)])
            self.generator_core = nn.ModuleList([ConvGRUCell(**generator_args) for _ in range(L)])

        # Inference, prior
        self.posterior_density = Conv2d(h_dim, 2*z_dim, **kwargs)
        self.prior_density     = Conv2d(h_dim, 2*z_dim, **kwargs)

        # Generative density
        self.observation_density = Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)

        self.upsample = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=self.scale, stride=self.scale, padding=0, bias=False)
        self.downsample = Conv2d(x_dim, x_dim, kernel_size=self.scale, stride=self.scale, padding=0, bias=False)

    def forward(self, x):
        """
        Attempt to reconstruct x with corresponding
        viewpoint v and context representation r.
        :param x: image to send through
        :param v: viewpoint of image
        :return reconstruction of x and kl-divergence
        """
        batch_size, _, h, w = x.shape
        kl = 0
        x = self.downsample(x)

        # Reset hidden and cell state
        hidden_i = x.new_zeros((batch_size, self.h_dim, h // self.scale, w // self.scale))
        hidden_g = x.new_zeros((batch_size, self.h_dim, h // self.scale, w // self.scale))

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
            u = self.upsample(hidden_g) + u
            #u = hidden_g + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.observation_density(u)
        kl = torch.mean(torch.sum(kl, dim=[1,2,3]))
        return x_mu, kl

    def sample(self, z_shape, batch_size, noise=True):
        """
        Sample from the prior distribution to generate
        a new image given a viewpoint and representation
        :param x_shape: (height, width) of image
        :param v: viewpoint
        :param r: representation (context)
        """
        h, w = z_shape

        # Reset hidden and cell state for generator
        device = next(self.parameters()).device
        hidden_g = torch.zeros((batch_size, self.h_dim, h // self.scale, w // self.scale), device=device)

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
            u = self.upsample(hidden_g) + u
            #u = hidden_g + u

        x_mu = self.observation_density(u)
        return x_mu

class CondGeneratorNetwork(nn.Module):
    """
    Network similar to a convolutional variational
    autoencoder that refines the generated image
    over a number of iterations.
    :param x_dim: number of channels in input
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: number of density refinements
    :param share: whether to share cores across refinements
    """
    def __init__(self, x_dim, r_dim, z_dim=32, h_dim=128, L=6, scale=4, share=True):
        super(CondGeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.share = share
        self.scale = scale

        # Core computational units
        kwargs = dict(kernel_size=5, stride=1)
        inference_args = dict(in_channels=r_dim + h_dim + x_dim, out_channels=h_dim, **kwargs)
        generator_args = dict(in_channels=r_dim + z_dim, out_channels=h_dim, **kwargs)
        if self.share:
            self.inference_core = ConvGRUCell(**inference_args)
            self.generator_core = ConvGRUCell(**generator_args)
        else:
            self.inference_core = nn.ModuleList([ConvGRUCell(**inference_args) for _ in range(L)])
            self.generator_core = nn.ModuleList([ConvGRUCell(**generator_args) for _ in range(L)])

        # Inference, prior
        self.posterior_density = Conv2d(h_dim, 2*z_dim, **kwargs)
        self.prior_density     = Conv2d(h_dim, 2*z_dim, **kwargs)

        # Generative density
        self.observation_density = Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)

        # Up/down-sampling primitives
        #self.upsample   = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=self.scale, stride=self.scale, padding=0, bias=False)
        self.downsample = Conv2d(x_dim, x_dim, kernel_size=self.scale, stride=self.scale, padding=0, bias=False)

        self.upsample = nn.Sequential(
            Conv2d(h_dim, h_dim, 3, stride=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2d(h_dim, h_dim, 3, stride=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2d(h_dim, h_dim, 3, stride=1),
        )

        #self.downsample = nn.Sequential(
        #    Conv2d(x_dim, x_dim, 3, stride=2),
        #    nn.LeakyReLU(),
        #    Conv2d(x_dim, x_dim, 3, stride=1),
        #    nn.LeakyReLU(),
        #    nn.BatchNorm2d(x_dim),
        #    Conv2d(x_dim, x_dim, 3, stride=2),
        #)

    def forward(self, x, r):
        """
        Attempt to reconstruct x with corresponding
        viewpoint v and context representation r.
        :param x: image to send through
        :param v: viewpoint of image
        :param r: representation for image
        :return reconstruction of x and kl-divergence
        """
        batch_size, _, h, w = x.shape
        kl = 0

        # Downsample x, upsample v and r
        x = self.downsample(x)

        # Reset hidden and cell state
        hidden_i = x.new_zeros((batch_size, self.h_dim, h // self.scale, w // self.scale))
        hidden_g = x.new_zeros((batch_size, self.h_dim, h // self.scale, w // self.scale))

        # Canvas for updating
        u = x.new_zeros((batch_size, self.h_dim, h, w))

        for l in range(self.L):
            # Prior factor (eta π network)
            p_mu, p_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            # Inference state update
            inference = self.inference_core if self.share else self.inference_core[l]
            hidden_i = inference(torch.cat([hidden_g, x, r], dim=1), hidden_i)

            # Posterior factor (eta e network)
            q_mu, q_std = torch.chunk(self.posterior_density(hidden_i), 2, dim=1)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            # Posterior sample
            z = posterior_distribution.rsample()

            # Generator state update
            generator = self.generator_core if self.share else self.generator_core[l]
            hidden_g = generator(torch.cat([z, r], dim=1), hidden_g)

            # Calculate u
            u = self.upsample(hidden_g) + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.observation_density(u)
        return torch.sigmoid(x_mu), kl

    def sample(self, x_shape, r, noise=False):
        """
        Sample from the prior distribution to generate
        a new image given a viewpoint and representation
        :param x_shape: (height, width) of image
        :param v: viewpoint
        :param r: representation (context)
        """
        h, w = x_shape
        batch_size = r.size(0)

        # Reset hidden and cell state for generator
        hidden_g = r.new_zeros((batch_size, self.h_dim, h // self.scale, w // self.scale))

        u = r.new_zeros((batch_size, self.h_dim, h, w))

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
            hidden_g = generator(torch.cat([z, r], dim=1), hidden_g)
            u = self.upsample(hidden_g) + u

        x_mu = self.observation_density(u)
        
        return torch.sigmoid(x_mu)
    