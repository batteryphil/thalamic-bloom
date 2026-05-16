import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from mamba3_mimo_builder import Mamba3MIMORLF
from mamba3_sft_generator import get_sft_dataloader
from transformers import AutoTokenizer

def train() -> None:
    """
    Execute Phase 3 (Jarvis v4) True Cognitive Routing Training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Jarvis v4 (Phase 3) on: {device}")
    
    model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
    model.to(device)
    
    checkpoint_path = "jarvis_v4.pth"
    base_checkpoint_path = "jarvis_v3_sft.pth"
    
    step = 0
    # Check for existing Phase 3 checkpoint first
    if os.path.exists(checkpoint_path):
        print(f"Found Phase 3 checkpoint at {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            step = checkpoint['step']
            optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
            # FORCE WIPE OPTIMIZER MOMENTUM to prevent clone history from overriding orthogonal init!
            # try:
            #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #     print(f"Resumed from Phase 3 step {step}.")
            # except ValueError:
            print(f"Resumed from Phase 3 step {step}. Optimizer wiped intentionally to break clone momentum.")
        else:
            model.load_state_dict(checkpoint, strict=False)
            optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    elif os.path.exists(base_checkpoint_path):
        print(f"Loading Phase 2 Baseline from {base_checkpoint_path}...")
        checkpoint = torch.load(base_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # We must load with strict=False because we added self.thalamic_primer
            # which does not exist in the Phase 2 weights!
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Missing keys (expected for new Thalamic Primer): {missing_keys}")
        else:
            model.load_state_dict(checkpoint, strict=False)
        # RESET OPTIMIZER: We explicitly wipe Phase 2 momentum for the new Phase 3 routing!
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        print("Model loaded. Optimizer state mathematically WIPED for Phase 3.")
    else:
        print("No checkpoints found. Starting from scratch!")
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        
    print("Freezing Cognitive Backbone and MIMO Arms for Phase 4 IPC Calibration...")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "ipc_mixer" in name or "domain_router" in name or "thalamic_primer" in name or "bridge" in name:
            param.requires_grad = True
    dataloader = get_sft_dataloader(batch_size=4, seq_len=1024)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    scaler = torch.amp.GradScaler('cuda')
    max_steps = 50000
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    dummy_input = torch.tensor([tokenizer.encode("User: Hello!\nAssistant: ")]).to(device)
            
    model.train()
    print("Starting Phase 3 (Jarvis v4) Loop...")
    
    start_time = time.time()
    tokens_per_step = 4 * 1024
    accumulation_steps = 4 
    
    smoothed_loss = None
    alpha = 0.05
    current_lr = 5e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    optimizer.zero_grad()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        try:
            with torch.amp.autocast('cuda'):
                logits = model(x, loop_idx=0) 
                loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
                
                loss = loss / accumulation_steps
                
            if torch.isnan(loss):
                print(f"CRITICAL: NaN Loss detected at step {step}! Skipping step.")
                optimizer.zero_grad()
                continue
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                
                actual_loss = loss.item() * accumulation_steps
                if smoothed_loss is None:
                    smoothed_loss = actual_loss
                else:
                    smoothed_loss = (1 - alpha) * smoothed_loss + alpha * actual_loss
                
                if smoothed_loss >= 5.0:
                    target_lr = 5e-5
                elif smoothed_loss <= 1.5:
                    target_lr = 1e-6
                else:
                    ratio = (smoothed_loss - 1.5) / (5.0 - 1.5)
                    target_lr = 1e-6 + ratio * (5e-5 - 1e-6)
                    
                if abs(current_lr - target_lr) > 1e-7:
                    current_lr = target_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM Caught! Skipping batch...")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            else:
                raise e
        
        if (i + 1) % accumulation_steps != 0:
            continue
        
        if step % 50 == 0:
            elapsed = time.time() - start_time
            tps = (50 * tokens_per_step) / elapsed if elapsed > 0 else 0
            loss_val = loss.item() * accumulation_steps
            print(f"Phase3 Step {step:04d} | Loss: {loss_val:.4f} (Smoothed: {smoothed_loss:.4f}) | LR: {current_lr:.7f} | TPS: {tps:.2f} | Time/50: {elapsed:.2f}s")
            
            salad = None
            if step % 1000 == 0 and step > 0:
                model.eval()
                try:
                    with torch.no_grad():
                        out = model.generate(dummy_input, max_new_tokens=30, temperature=0.3, top_k=5, stop_sequences={0, 198, 200})
                    salad = tokenizer.decode(out[0].tolist()[len(dummy_input[0]):]).replace('\n', ' ')
                except Exception as e:
                    salad = f"Error: {e}"
                model.train()
            
            os.makedirs("dashboard", exist_ok=True)
            with open("dashboard/metrics.jsonl", "a") as f:
                import json
                
                # Fetch live telemetry from the biological routing logic
                telemetry = model.last_telemetry if hasattr(model, 'last_telemetry') else {}
                if step % 50 == 0:
                    print(f"DEBUG TELEMETRY: {telemetry}")
                
                gate = telemetry.get('gate_score', 0)
                entropy = telemetry.get('entropy', 0)
                collapse_metric = telemetry.get('arm_collapse_metric', 0.0)
                energy_metric = telemetry.get('latent_energy', 0.0)
                
                payload = {
                    "step": step, "loss": loss_val, "tps": tps, "elapsed": elapsed, 
                    "lr": current_lr, "gate_score": gate, "entropy": entropy,
                    "collapse_metric": collapse_metric, "latent_energy": energy_metric
                }
                if salad:
                    payload["salad"] = salad
                f.write(json.dumps(payload) + "\n")
                
            # --- AUTO-STOP LOGIC (Phase 4 Target) ---
            if smoothed_loss < 1.0:
                print(f"\n*** PHASE 4 TARGET REACHED ***")
                print(f"Smoothed Loss: {smoothed_loss:.4f}")
                print("IPC Mixer has successfully synthesized language. Halting training.")
                
                # Save final checkpoint
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Phase 4 Checkpoint saved to {checkpoint_path}")
                
                # Run automated benchmark
                print("\nRunning Automated Benchmark (oo_benchmark.py)...")
                import subprocess
                subprocess.run(["python3", "oo_benchmark.py"])
                
                break
                
            start_time = time.time()

            
        if step % 200 == 0:
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step
            }
            torch.save(save_dict, checkpoint_path)
            
        if step >= max_steps: 
            break

if __name__ == "__main__":
    train()
