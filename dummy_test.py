import torch
import time
from mamba3_mimo_builder import Mamba3MIMORLF

print("Init model...")
model = Mamba3MIMORLF().cuda()
x = torch.randint(0, 50000, (4, 1024)).cuda()
print("Forward pass...")
start = time.time()
with torch.amp.autocast('cuda'):
    y = model(x)
print(f"Done in {time.time()-start:.2f}s")
