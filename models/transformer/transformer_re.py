import torch
import torch.nn as nn
import torch.nn.functional as F
from nanogpt import GPT, GPTConfig
from transformer_encdec import TransformerEncoder, TransformerDecoder
from vqvae.vqvae import VQVAE

# Can be conditioned or unconditioned, the condition part also utilize causal attention
class ImgTransformer(nn.Module):
    def __init__(self, transformer_config, vqmodel, vqmodel_cond=None, vq_size=(16,16), sos_token=0, n_embeddings=256, embedding_dim=3, pkeep=0.5):
        super().__init__()
        self.sos_token = sos_token
        self.vqmodel = vqmodel.eval()
        if vqmodel_cond is not None:
            self.vqmodel_cond = vqmodel_cond.eval()
        else:
            vqmodel_cond = None

        self.transformer_config = transformer_config
        self.transformer = GPT(GPTConfig(**transformer_config))

        self.vq_size = vq_size
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.pkeep = pkeep
    
    @torch.no_grad()
    def encode_to_z(self, x):
        z_e = self.vqmodel.encoder(x)
        z_q, _, indices = self.vqmodel.quantizer(z_e)
        indices = indices.view(z_q.shape[0], -1)
        return z_q, indices

    @torch.no_grad()
    def encode_to_zc(self, c):
        z_e = self.vqmodel_cond.encoder(c)
        z_q, _, indices = self.vqmodel_cond.quantizer(z_e)
        indices = indices.view(z_q.shape[0], -1)
        return z_q, indices
    
    @torch.no_grad()
    def z_to_image(self, indices):
        ix_to_vectors = self.vqmodel.quantizer.embedding(indices)
        ix_to_vectors= ix_to_vectors.reshape(indices.shape[0], self.vq_size[0], self.vq_size[1], self.embedding_dim)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqmodel.decoder(ix_to_vectors)
        return image

    def _construct_sos_token(self, batch_size):
        sos_tokens = torch.ones(batch_size, 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(next(self.transformer.parameters()).get_device())
        return sos_tokens

    def forward(self, x, c=None):
        _, indices = self.encode_to_z(x)
        
        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        if c is None:
            indices_c = self._construct_sos_token(x.shape[0])
        else:
            _, indices_c = self.encode_to_zc(c)
        new_indices = torch.cat((indices_c, new_indices), dim=1)

        targets = indices

        if c == None:
            logits, loss = self.transformer(new_indices[:, :-1], targets)
        else:
            logits, loss = self.transformer(new_indices[:, :-1], targets, indices_c.shape[1])
        return logits, loss

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    def sample(self, c=None, samp_size=64, temperature=1.0, top_k=100):
        self.transformer.eval()
        if c is None:
            indices_c = self._construct_sos_token(samp_size)
        else:
            _, indices_c = self.encode_to_zc(c)

        n_tokens = self.vq_size[0] * self.vq_size[1]
        z_indices = self.transformer.generate(indices_c, n_tokens, temperature, top_k)[:,indices_c.shape[1]:]
        x = self.z_to_image(z_indices)
        self.transformer.train()
        return x

# Conditional Generation based on Transformer Encoder-Decoder Architecture
class ImgTransformerEncDec(nn.Module):
    def __init__(self, transformer_enc_config, vqmodel_cond, transformer_dec_config, vqmodel, vq_size=(16,16), sos_token=0, n_embeddings=256, embedding_dim=3, pkeep=0.5):
        super().__init__()
        self.sos_token = sos_token
        self.vqmodel = vqmodel.eval()
        self.vqmodel_cond = vqmodel_cond.eval()

        self.transformer_enc_config = transformer_enc_config
        self.transformer_dec_config = transformer_dec_config
        self.transformer_enc = TransformerEncoder(GPTConfig(**transformer_enc_config))
        self.transformer_dec = TransformerDecoder(GPTConfig(**transformer_dec_config))

        self.vq_size = vq_size
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.pkeep = pkeep
    
    @torch.no_grad()
    def encode_to_z(self, x):
        z_e = self.vqmodel.encoder(x)
        z_q, _, indices = self.vqmodel.quantizer(z_e)
        indices = indices.view(z_q.shape[0], -1)
        return z_q, indices

    @torch.no_grad()
    def encode_to_zc(self, c):
        z_e = self.vqmodel_cond.encoder(c)
        z_q, _, indices = self.vqmodel_cond.quantizer(z_e)
        indices = indices.view(z_q.shape[0], -1)
        return z_q, indices
    
    @torch.no_grad()
    def z_to_image(self, indices):
        ix_to_vectors = self.vqmodel.quantizer.embedding(indices)
        ix_to_vectors= ix_to_vectors.reshape(indices.shape[0], self.vq_size[0], self.vq_size[1], self.embedding_dim)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqmodel.decoder(ix_to_vectors)
        return image

    def _construct_sos_token(self, batch_size):
        sos_tokens = torch.ones(batch_size, 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(next(self.transformer_dec.parameters()).get_device())
        return sos_tokens

    def forward(self, x, c):
        _, indices_c = self.encode_to_zc(c)
        _, indices = self.encode_to_z(x)
        
        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer_dec.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices
        
        sos = self._construct_sos_token(x.shape[0])
        new_indices = torch.cat((sos, new_indices), dim=1)

        targets = indices
        out_layers = self.transformer_enc(indices_c)
        logits, loss = self.transformer_dec(new_indices[:, :-1], out_layers, targets)
        return logits, loss

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    def sample(self, c, samp_size=64, temperature=1.0, top_k=100):
        self.transformer_enc.eval()
        self.transformer_dec.eval()

        _, indices_c = self.encode_to_zc(c)
        out_layers = self.transformer_enc(indices_c)
        sos = self._construct_sos_token(samp_size)

        n_tokens = self.vq_size[0] * self.vq_size[1]
        z_indices = self.transformer_dec.generate(sos, out_layers, n_tokens, temperature, top_k)[:,1:]
        x = self.z_to_image(z_indices)
        
        self.transformer_enc.train()
        self.transformer_dec.train()
        return x