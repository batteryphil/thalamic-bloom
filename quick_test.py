import torch
import torch.nn as nn
from torch.optim import AdamW
from mamba3_mimo_builder import Mamba3MIMORLF
from transformers import AutoTokenizer

def quick_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Setting up the 1-Batch Crucible...")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # "The 1-Batch Crucible" dataset: 4 paragraphs
    texts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter in the English alphabet.",
        "Water boils at 100 degrees Celsius under standard atmospheric pressure, which is roughly sea level.",
        "The Great Wall of China is one of the most recognizable structures in the world, spanning thousands of miles.",
        "Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability."
    ]
    
    # Encode and pad to a uniform length
    input_ids_list = []
    max_len = max(len(tokenizer.encode(t)) for t in texts) + 1
    
    for t in texts:
        encoded = tokenizer.encode(t)
        if len(encoded) < max_len:
            encoded += [tokenizer.pad_token_id or 0] * (max_len - len(encoded))
        input_ids_list.append(encoded[:max_len])
        
    x = torch.tensor(input_ids_list, dtype=torch.long).to(device)
    # Target is simply x shifted by 1
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    y[:, -1] = tokenizer.pad_token_id or 0
    
    print("Instantiating patched Mamba3MIMORLF model...")
    model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
    model.to(device)
    model.initialize_asymmetric_arms()
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id or 0)
    scaler = torch.amp.GradScaler('cuda')
    
    print("Starting 500 steps of single-batch overfitting...")
    
    for step in range(1, 501):
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits = model(x, loop_idx=0)
            # Shift logits to match shifted targets
            loss = criterion(logits[:, :-1, :].reshape(-1, model.vocab_size), y[:, :-1].reshape(-1))
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if step % 50 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f}")
            
    print("Overfit test complete. Final loss:", loss.item())
    
    if loss.item() < 0.05:
        print("SUCCESS! Model successfully saturated the batch.")
    else:
        print("WARNING! Model failed to memorize the batch.")

if __name__ == "__main__":
    quick_test()
