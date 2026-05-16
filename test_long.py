import torch
from mamba3_mimo_builder import Mamba3MIMORLF

model = Mamba3MIMORLF(vocab_size=100, d_model=768, n_layers=24).cuda()
model.initialize_asymmetric_arms()

dummy_input = torch.randint(0, 100, (1, 1024)).cuda()
model.train()
out = model(dummy_input)
print(f"Collapse Metric after init on 1024 length: {model.last_telemetry['arm_collapse_metric']}")
