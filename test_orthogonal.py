import torch
from mamba3_mimo_builder import Mamba3MIMORLF

model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=4)
model.initialize_asymmetric_arms()

dummy_input = torch.randint(0, 50304, (1, 128))
model.train()
model(dummy_input)

print(model.last_telemetry)
