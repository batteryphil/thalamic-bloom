import torch
from mamba3_mimo_builder import Mamba3MIMORLF
from transformers import AutoTokenizer

device = torch.device('cuda')
model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24).to(device)
checkpoint = torch.load("/hdd_data/mamba_checkpoints/jarvis_v4.pth", map_location=device, weights_only=True)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
prompt = "Write a poem about space."
input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=40, temperature=0.3, top_k=5)

print(tokenizer.decode(out[0].tolist()))
