import torch
import torch.nn as nn
from mamba3_mimo_builder import Mamba3MIMORLF

model = Mamba3MIMORLF(vocab_size=100, d_model=768, n_layers=2)
count = 0
for name, param in model.mimo_reasoning_blocks.named_parameters():
    if 'weight' in name and param.dim() >= 2:
        print(f"Applying orthogonal to {name} with shape {param.shape}")
        count += 1
print(f"Total parameters modified: {count}")
