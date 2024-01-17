# Vector-Quantize-Generative-Models
Implementation of vector quantize-based generative models with good readability and minimal dependencies, including vector quantize model, transformer generator (VQGAN) and diffusion model (LDM). All models are tested under the pc with RTX 2070 GPU.

## Dependancy
Dpendencies of the models:
- pytorch = 2.0.1
- torchvision = 0.15.2

The training data is based on Maze3D environments, with the following dependancies:
- pyglm = 2.7.0
- pyrender = 0.1.39
- pyopengl = 3.1.0

## Models
### Vector Quantize Model
<img src="assets/vqvae.jpg" width="512"/>

```
python train_vqvae.py

# Optional, used for conditional transformer
python train_vqvae_depth.py 
```

Image Reconstruction Results of VQVAE

<img src="assets/results_vqvae.jpg" width="300"/>

Depth Reconstruction Results of VQVAE

<img src="assets/results_vqvae_depth.jpg" width="300"/>

### Latent Diffusion Model 
<img src="assets/ldm.jpg" width="400"/>

```
python train_ldm.py

# Depth conditional generation (cross-attention)
python train_ldm_condition.py

# Depth conditional generation (concatenation)
python train_ldm_condition_cat.py
```

Random Generation Results of LDM

<img src="assets/results_ldm.jpg" width="300"/>

Conditional Generation Results of LDM (Cross-Attention)

<img src="assets/results_ldm_condition.jpg" width="300"/>

Conditional Generation Results of LDM (Concatenation)

<img src="assets/results_ldm_condition_cat.jpg" width="300"/>

### Transformer Generation Model
<img src="assets/transformer.jpg" width="512"/>

```
python train_transformer.py

# Depth conditional generation (decoder-only)
python train_transformer_condition.py

# Depth conditional generation (encoder-decoder)
python train_transformer_condition_encdec.py
```

Random Generation Results of Transformer

<img src="assets/results_transformer.jpg" width="300"/>

Conditional Generation Results of Transformer (Decoder-Only)

<img src="assets/results_transformer_condition.jpg" width="300"/>

Conditional Generation Results of Transformer (Encoder-Decoder)

<img src="assets/results_transformer_condition_encdec.jpg" width="300"/>

### Latent DRAW Model (Experimental)
```
python train_draw.py

# Depth conditional generation
python train_draw_condition.py
```

Random Generation Results of DRAW

<img src="assets/results_draw.jpg" width="300"/>

Conditional Generation Results of DRAW

<img src="assets/results_draw_condition.jpg" width="300"/>

## References
- Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).
- Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
- Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
- Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
- https://github.com/karpathy/nanoGPT
- https://github.com/dome272/VQGAN-pytorch
- https://github.com/Alokia/diffusion-DDIM-pytorch