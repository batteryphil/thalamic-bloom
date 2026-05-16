import torch
import torch.nn.functional as F
from mamba3_mimo_builder import Mamba3MIMORLF

model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24).cuda()
checkpoint = torch.load("jarvis_v4.pth", map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.initialize_asymmetric_arms()

dummy_input = torch.randint(0, 50304, (1, 128)).cuda()
model.train()
out = model(dummy_input)
print(f"Collapse Metric after load+init: {model.last_telemetry['arm_collapse_metric']}")
