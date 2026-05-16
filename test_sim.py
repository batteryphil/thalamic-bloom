import torch
import torch.nn.functional as F
from mamba3_mimo_builder import Mamba3MIMORLF

model = Mamba3MIMORLF(vocab_size=100, d_model=768, n_layers=2).cuda()
model.initialize_asymmetric_arms()
dummy_input = torch.randint(0, 100, (1, 128)).cuda()
out = model(dummy_input)
print(f"Collapse Metric after init: {model.last_telemetry['arm_collapse_metric']}")
