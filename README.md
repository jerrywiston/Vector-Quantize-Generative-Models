# Vector-Quantize-Generative-Models
Implementation of vector quantize-based models, including vector quantization, transformer generator (VQGAN) and diffusion model (LDM).

## Dependancy
Dpendencies of the models:
- pytorch = 2.0.1
- torchvision = 0.15.2

The training data is based on Maze3D environments, with the following dependancies:
- pyglm = 2.7.0
- pyrender = 0.1.39
- pyopengl = 3.1.0

## Training
### Vector Quantize Model
```
python train_vqvae.py

# Optional, used for conditional transformer
python train_vqvae_depth.py 
```

### Latent Diffusion Model 
```
python train_ldm.py

# Depth conditional generation (cross-attention)
python train_ldm_condition.py

# Depth conditional generation (concatenation)
python train_ldm_condition_cat.py
```

### Transformer Generation Model
```
python train_transformer.py

# Depth conditional generation (decoder-only)
python train_transformer_condition.py

# Depth conditional generation (encoder-decoder)
python train_transformer_condition_encdec.py
```

### DRAW Model (Experimental)
```
python train_draw.py
```