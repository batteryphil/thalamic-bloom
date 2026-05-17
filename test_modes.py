import torch
from mamba3_mimo_builder import Mamba3MIMORLF
from transformers import AutoTokenizer

device = torch.device('cuda')
model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
model.load_state_dict(torch.load("jarvis_v4.pth", map_location=device)['model_state_dict'], strict=False)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
prompt = "User: Who are you?\nAssistant: "
inputs = torch.tensor([tokenizer.encode(prompt)]).to(device)

print("\n--- Testing Neural Agent Standards ---")
print("1. Identity Mode (T=0.05, top_k=1)")
model.eval()
with torch.no_grad():
    out = model.generate(inputs, max_new_tokens=40, temperature=0.05, top_k=1)
print(tokenizer.decode(out[0].tolist()))

print("\n2. Creative Mode (T=0.3, top_k=5)")
prompt2 = "User: Write a poem about space.\nAssistant: "
inputs2 = torch.tensor([tokenizer.encode(prompt2)]).to(device)
with torch.no_grad():
    out2 = model.generate(inputs2, max_new_tokens=40, temperature=0.3, top_k=5)
print(tokenizer.decode(out2[0].tolist()))
