"""
Titan Phase 4 — Supervised Fine-Tuning (SFT) Trainer
Turns the pre-trained language model into a chat/reasoning assistant.

Data: 70% OpenHermes-2.5 (chat) + 30% GSM8K (math reasoning)
Format: "User: {prompt}\nAssistant: {response}<|endoftext|>"
Key: -100 masking on prompt tokens so loss is computed only on completions.
"""
import torch
import torch.nn as nn
import argparse
import os
import sys
import json
import time
import math
import signal
import collections

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def get_gpu_temp():
        return pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
except Exception:
    def get_gpu_temp(): return None

try:
    from huggingface_hub import login
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─────────────────────────────────────────────────────────────────────────────
# Signal handler
# ─────────────────────────────────────────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\n[SIGNAL] Received — will save and exit after this step.")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# SFT Data Pipeline — quality-filtered, picky chain-of-thought reasoning mix
# ─────────────────────────────────────────────────────────────────────────────
IGNORE_INDEX = -100

# Quality filter: reject completions shorter than this (avoids "Answer: 4" junk)
MIN_COMPLETION_CHARS = 60

def _is_quality(completion: str) -> bool:
    """Returns True if the completion is substantive enough to train on."""
    c = completion.strip()
    if len(c) < MIN_COMPLETION_CHARS:
        return False
    if c.count('\n') == 0 and len(c.split()) < 8:
        return False   # single short line — too terse
    return True


def format_hermes_sample(sample: dict) -> tuple[str, str]:
    convs = sample.get('conversations', [])
    if not convs:
        return None, None
    turns = []
    for t in convs:
        role  = t.get('from', '')
        value = t.get('value', '').strip()
        if role in ('human','user'):      turns.append(f"User: {value}")
        elif role in ('gpt','assistant'): turns.append(f"Assistant: {value}")
    if len(turns) < 2:
        return None, None
    prompt     = "\n".join(turns[:-1]) + "\nAssistant: "
    completion = turns[-1].replace("Assistant: ","",1) + "<|endoftext|>\n"
    if not _is_quality(completion):
        return None, None
    return prompt, completion


def format_gsm8k_sample(sample: dict) -> tuple[str, str]:
    """GSM8K: require full step-by-step explanation, not just the final number."""
    q = sample.get('question','').strip()
    a = sample.get('answer','').strip()
    if not _is_quality(a):
        return None, None
    prompt     = f"User: Solve step by step: {q}\nAssistant: "
    completion = f"Let me work through this carefully.\n\n{a}<|endoftext|>\n"
    return prompt, completion


def format_openr1_sample(sample: dict) -> tuple[str, str]:
    """OpenR1-Math: competition math with full chain-of-thought reasoning."""
    problem  = sample.get('problem', sample.get('question', '')).strip()
    solution = sample.get('solution', sample.get('answer', '')).strip()
    if not problem or not _is_quality(solution):
        return None, None
    prompt     = f"User: {problem}\nAssistant: "
    completion = f"<think>\n{solution}\n</think>\n{solution.split(chr(10))[-1]}<|endoftext|>\n"
    return prompt, completion


def sft_batch_generator(tokenizer, seq_len=1024, hermes_ratio=0.50,
                         gsm_ratio=0.30, openr1_ratio=0.20):
    """
    Yields (input_ids, labels) with -100 masking on prompt tokens.
    Data mix: 50% Hermes chat + 30% GSM8K step-by-step + 20% OpenR1 CoT.
    Picky: skips any completion that fails _is_quality().
    """
    import random
    login(token=HF_TOKEN)

    print("[SFT] Loading OpenHermes-2.5 (50%)...")
    ds_hermes = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    print("[SFT] Loading GSM8K (30%)...")
    ds_gsm = load_dataset("gsm8k", "main", split="train", streaming=True)
    print("[SFT] Loading OpenR1-Math-220k (20%)...")
    try:
        ds_r1 = load_dataset("openr1/OpenR1-Math-220k", split="train", streaming=True)
        print("[SFT] ✅ OpenR1 loaded.")
    except Exception as e:
        print(f"[SFT] OpenR1 unavailable ({e}) — using GSM8K as fallback.")
        ds_r1 = load_dataset("gsm8k", "main", split="train", streaming=True)

    iter_h  = iter(ds_hermes)
    iter_g  = iter(ds_gsm)
    iter_r1 = iter(ds_r1)

    skipped  = 0
    accepted = 0

    while True:
        rng = random.random()
        try:
            if rng < hermes_ratio:
                s = next(iter_h)
                prompt, completion = format_hermes_sample(s)
            elif rng < hermes_ratio + gsm_ratio:
                s = next(iter_g)
                prompt, completion = format_gsm8k_sample(s)
            else:
                s = next(iter_r1)
                prompt, completion = format_openr1_sample(s)
        except StopIteration:
            iter_h  = iter(load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True))
            iter_g  = iter(load_dataset("gsm8k","main", split="train", streaming=True))
            continue

        if prompt is None:
            skipped += 1
            continue

        accepted += 1
        if accepted % 500 == 0:
            print(f"[SFT] Accepted {accepted} | Skipped {skipped} ({skipped/(accepted+skipped):.0%} filtered)")

        prompt_ids     = tokenizer.encode(prompt)
        completion_ids = tokenizer.encode(completion)
        input_ids      = prompt_ids + completion_ids
        labels         = [IGNORE_INDEX] * len(prompt_ids) + completion_ids

        for i in range(0, len(input_ids), seq_len):
            chunk_in  = input_ids[i:i + seq_len]
            chunk_lbl = labels[i:i + seq_len]
            if len(chunk_in) < seq_len:
                pad = seq_len - len(chunk_in)
                chunk_in  = chunk_in  + [tokenizer.eos_token_id] * pad
                chunk_lbl = chunk_lbl + [IGNORE_INDEX] * pad
            yield (
                torch.tensor([chunk_in],  dtype=torch.long),
                torch.tensor([chunk_lbl], dtype=torch.long),
            )


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule (lower than pre-training — preserves pre-trained weights)
# ─────────────────────────────────────────────────────────────────────────────
def sft_lr(step: int, total: int, base_lr=5e-5, warmup=200, min_lr=1e-6) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_ckpt(save_dir, step, model, optimizer):
    path = os.path.join(save_dir, "phase_sft.pt")
    tmp  = path + ".tmp"
    print(f"[CKPT] Saving SFT checkpoint at step {step}...")
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'phase': 'sft'}, tmp)
    os.replace(tmp, path)
    print(f"[CKPT] Saved → {path}")


def load_ckpt(path, model, optimizer, device):
    if not os.path.exists(path):
        return 0
    print(f"[CKPT] Loading {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'], strict=False)
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("[CKPT] Optimizer state restored.")
    except Exception as e:
        print(f"[CKPT] Optimizer restore failed ({e}) — fresh optimizer.")
    return int(ckpt.get('step', 0))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10_000)
    parser.add_argument('--save_every', type=int, default=250)
    args = parser.parse_args()

    from mamba3_titan_builder import Mamba3Titan

    project_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir    = os.path.join(project_dir, "titan_checkpoints")
    telem_path  = os.path.join(project_dir, "monitor_ui", "telemetry.json")
    salad_path  = os.path.join(project_dir, "monitor_ui", "word_salad.json")
    log_path    = os.path.join(project_dir, "training_log.txt")
    os.makedirs(save_dir, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"\n[SFT TRAINER] Start UTC={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n")

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 0

    # Load model — start from phase_3j checkpoint
    model = Mamba3Titan(vocab_size=50304, d_model=2048, n_layers=80,
                        mimo_paths=16, use_gradient_checkpointing=True)
    model.set_phase('sft')
    model = model.to(torch.bfloat16).to(device)

    # Optimizer — lower LR than pre-training
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=5e-5, weight_decay=0.01)
        print("Using Adam8bit.")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Load SFT checkpoint if resuming, else load 3j weights
    sft_ckpt = os.path.join(save_dir, "phase_sft.pt")
    if os.path.exists(sft_ckpt):
        resume_step = load_ckpt(sft_ckpt, model, optimizer, device)
    else:
        phase3j_ckpt = os.path.join(save_dir, "phase_3j.pt")
        if os.path.exists(phase3j_ckpt):
            print(f"[SFT] Cold start from Phase 3j weights: {phase3j_ckpt}")
            ckpt = torch.load(phase3j_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            print("[SFT] WARNING: No Phase 3j checkpoint found — starting from scratch.")
        resume_step = 0

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    data_gen  = sft_batch_generator(tokenizer, seq_len=1024)
    model.train()

    print(f"\n{'='*68}")
    print(f"  TITAN SFT — Phase 4 Reasoning Fine-Tune")
    print(f"  Resume step: {resume_step} / {args.steps}")
    print(f"  Data: 70% OpenHermes + 30% GSM8K  |  Prompt-masked labels")
    print(f"{'='*68}\n")

    step_start = time.time()

    for step, (input_ids, labels) in enumerate(data_gen):
        if step < resume_step:
            continue
        if step >= args.steps:
            print(f"\n[SFT COMPLETE] Reached {args.steps} steps.")
            save_ckpt(save_dir, step, model, optimizer)
            break

        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        # LR update
        lr = sft_lr(step, args.steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(input_ids, loop_idx=0)
            # SFT loss: only on completion tokens (-100 masked prompt tokens are ignored)
            loss = criterion(logits.view(-1, 50304), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 5 == 0:
            torch.cuda.empty_cache()

        elapsed  = time.time() - step_start
        tps      = 1024 / elapsed if elapsed > 0 else 0.0
        step_start = time.time()
        gpu_temp = get_gpu_temp()
        temp_str = f" | GPU: {gpu_temp}°C" if gpu_temp else ""
        t        = model.last_telemetry

        log_line = (
            f"SFT | Step {step+1:05d} | Loss: {loss.item():.4f} | "
            f"TPS: {tps:.1f} | LR: {lr:.2e} | "
            f"Entropy: {t.get('entropy',0):.3f}{temp_str}"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # Telemetry
        telem = {
            "phase": "sft", "step": step + 1,
            "lm_loss": round(loss.item(), 4), "domain_loss": 0.0,
            "gate_score": t.get('gate_score', 0.0),
            "entropy": t.get('entropy', 0.0),
            "tps": round(tps, 1), "gpu_temp": gpu_temp,
            "lr": round(lr, 8),
            "arm_weights": t.get('arm_weights', []),
            "top_arms": t.get('top_arms', []),
        }
        with open(telem_path, "w") as f:
            json.dump(telem, f)

        if (step + 1) % args.save_every == 0:
            save_ckpt(save_dir, step + 1, model, optimizer)

        if _shutdown_requested:
            save_ckpt(save_dir, step + 1, model, optimizer)
            sys.exit(0)


if __name__ == "__main__":
    train()
