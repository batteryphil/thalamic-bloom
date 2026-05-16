import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from mamba3_mimo_builder import Mamba3MIMORLF
from mamba3_mimo_generator import get_hybrid_dataloader
from transformers import AutoTokenizer

def train() -> None:
    """
    Execute the from-scratch pre-training loop with resume capability.
    Incorporates AdamW optimizer strictly, manages gradient clipping, and commits checkpoints.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Mamba 3 MIMO Pre-training on: {device}")
    
    # Initialize 150M model with the entire multi-path architecture
    model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
    model.to(device)
    
    # Standards: AdamW, 1e-5 recovery LR, 0.01 weight decay
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    dataloader = get_hybrid_dataloader(batch_size=4, seq_len=1024)
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler('cuda')
    step = 0
    max_steps = 500000  # Full convergence pre-training run
    checkpoint_path = "jarvis_v3.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Attempting to resume...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if it's the old format (just model state_dict) or new format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            step = checkpoint['step']
            print(f"Resumed from step {step} with saved optimizer state.")
        else:
            # Old format: just the model weights
            model.load_state_dict(checkpoint)
            print("Loaded previous model weights (old checkpoint format). Optimizer state and step count reset.")
            
    # Asymmetric Initialization for Octopoda-OS
    print("Applying Orthogonal Weights (Decentralized Ganglionic Processing)...")
    model.initialize_asymmetric_arms()
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    dummy_input = torch.tensor([tokenizer.encode("System Context: ")]).to(device)
            
    model.train()
    print("Starting training loop...")
    
    start_time = time.time()
    tokens_per_step = 4 * 1024  # batch_size * seq_len
    accumulation_steps = 4 # Gradient accumulation parameter
    
    smoothed_loss = None
    alpha = 0.05
    current_lr = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    print(f"Starting training with dynamic LR initialized to {current_lr}")
    
    optimizer.zero_grad()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        try:
            # Forward pass through Backbone + CP + Bridge + Parallel MIMO Loops
            with torch.amp.autocast('cuda'):
                logits = model(x, loop_idx=0) 
                loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
                loss = loss / accumulation_steps
                
            if torch.isnan(loss):
                print(f"CRITICAL: NaN Loss detected at step {step}! Skipping step to prevent corruption.")
                optimizer.zero_grad()
                continue
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                # Mandatory strict gradient norm clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                
                # Dynamic LR Scheduling based on smoothed loss
                actual_loss = loss.item() * accumulation_steps
                if smoothed_loss is None:
                    smoothed_loss = actual_loss
                else:
                    smoothed_loss = (1 - alpha) * smoothed_loss + alpha * actual_loss
                
                if smoothed_loss >= 15.0:
                    target_lr = 1e-4
                elif smoothed_loss <= 8.0:
                    target_lr = 1e-5
                else:
                    ratio = (smoothed_loss - 8.0) / (15.0 - 8.0)
                    target_lr = 1e-5 + ratio * (1e-4 - 1e-5)
                    
                if abs(current_lr - target_lr) > 1e-7:
                    current_lr = target_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"WARNING: CUDA OOM caught. Graceful recovery via Autotomic Ink backup...")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            else:
                raise e
        
        # Only log/save if a full optimizer step was taken
        if (i + 1) % accumulation_steps != 0:
            continue
        
        if step % 50 == 0:
            elapsed = time.time() - start_time
            tps = (50 * tokens_per_step) / elapsed if elapsed > 0 else 0
            loss_val = loss.item() * accumulation_steps # Report unscaled loss
            print(f"Step {step:04d} | Loss: {loss_val:.4f} (Smoothed: {smoothed_loss:.4f}) | LR: {current_lr} | TPS: {tps:.2f} | Time/50 steps: {elapsed:.2f}s")
            
            salad = None
            if step % 1000 == 0 and step > 0:
                model.eval()
                try:
                    with torch.no_grad():
                        out = model.generate(dummy_input, max_new_tokens=30, temperature=0.5, top_k=5, stop_sequences={0, 198, 200})
                    salad = tokenizer.decode(out[0].tolist()[len(dummy_input[0]):]).replace('\n', ' ')
                except Exception as e:
                    salad = f"Error: {e}"
                model.train()
            
            # Export metrics for dashboard
            os.makedirs("dashboard", exist_ok=True)
            with open("dashboard/metrics.jsonl", "a") as f:
                import json
                if salad:
                    f.write(json.dumps({"step": step, "loss": loss_val, "tps": tps, "elapsed": elapsed, "lr": current_lr, "salad": salad}) + "\n")
                else:
                    f.write(json.dumps({"step": step, "loss": loss_val, "tps": tps, "elapsed": elapsed, "lr": current_lr}) + "\n")
                
            start_time = time.time()
            
        if step % 200 == 0:
            print(f"Saving checkpoint at step {step}...")
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step
            }
            torch.save(save_dict, checkpoint_path)
            
        if step >= max_steps: 
            break

    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(save_dict, checkpoint_path)
    print(f"Training Complete. Final artifact saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
