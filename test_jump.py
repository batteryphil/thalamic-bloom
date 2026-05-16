import torch
import torch.nn.functional as F
from mamba3_mimo_builder import Mamba3MIMORLF
from torch.optim import AdamW

model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24).cuda()
checkpoint = torch.load("jarvis_v4.pth", map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.initialize_asymmetric_arms()

dummy_input = torch.randint(0, 50304, (4, 128)).cuda()
target = torch.randint(0, 50304, (4, 128)).cuda()
criterion = torch.nn.CrossEntropyLoss()

model.train()
for i in range(5):
    out = model(dummy_input)
    loss = criterion(out.view(-1, 50304), target.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {i} Collapse Metric: {model.last_telemetry['arm_collapse_metric']}")
