"""
Phase 5b: Operating Organism Engram Burn
Targeted memorization of the 5 OO architecture concepts.
Freezes MIMO arms, trains everything else on 100% OO data for 300 steps.
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from mamba3_mimo_builder import Mamba3MIMORLF
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader


class OOEngramDataset(IterableDataset):
    """Streams 100% OO Gold Samples from oo_engrams.jsonl indefinitely."""

    def __init__(self, path: str, tokenizer, seq_len: int = 1024) -> None:
        """
        Initialize the OO Engram Dataset.

        Args:
            path: Path to the oo_engrams.jsonl file.
            tokenizer: HuggingFace tokenizer.
            seq_len: Sequence length for packing.
        """
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        """Pack OO samples into fixed-length sequences with prompt masking."""
        buffer_tokens = []
        buffer_targets = []

        while True:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                        prompt = f"User: {s['prompt']}\nAssistant: "
                        answer = f"{s['answer']}<|endoftext|>\n"

                        u_toks = self.tokenizer.encode(prompt)
                        a_toks = self.tokenizer.encode(answer)

                        buffer_tokens.extend(u_toks + a_toks)
                        buffer_targets.extend([-100] * len(u_toks) + a_toks)

                        while len(buffer_tokens) >= self.seq_len + 1:
                            seq = buffer_tokens[: self.seq_len + 1]
                            tgt = buffer_targets[: self.seq_len + 1]

                            x = torch.tensor(seq[:-1], dtype=torch.long)
                            y = torch.tensor(tgt[1:], dtype=torch.long)
                            yield x, y

                            buffer_tokens = buffer_tokens[self.seq_len :]
                            buffer_targets = buffer_targets[self.seq_len :]
                    except Exception as e:
                        print(f"  [Engram Dataset] Error: {e}")
                        continue


def train() -> None:
    """Execute Phase 5b: Targeted OO Engram Burn (300 steps)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 5b Engram Burn on: {device}")

    model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
    model.to(device)

    checkpoint_path = "jarvis_v4_oo.pth"  # Start from previous engram burn
    engram_path = "jarvis_v4_oo_prose.pth"

    step = 0
    if os.path.exists(engram_path):
        print(f"Resuming from existing prose engram checkpoint: {engram_path}")
        ckpt = torch.load(engram_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        step = ckpt.get("step", 0)
    elif os.path.exists(checkpoint_path):
        print(f"Loading JSON-engram checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print("Loaded. Starting prose format-fix pass.")
    else:
        print("No checkpoint found — aborting.")
        return

    # Freeze ONLY the MIMO arms to protect reasoning orthogonality
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "mimo_reasoning_blocks" in name:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1

    print(f"Trainable params: {trainable_count} | Frozen (MIMO arms): {frozen_count}")

    # Low LR for surgical injection — don't blow up prior knowledge
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
        weight_decay=0.01,
    )

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    dataset = OOEngramDataset(
        path="/hdd_data/mamba_training_data/oo_engrams_prose.jsonl",
        tokenizer=tokenizer,
        seq_len=1024,
    )
    dataloader = DataLoader(dataset, batch_size=4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler("cuda")

    # Probe prompt — we'll check this every 50 steps
    probe_prompt = "User: Explain the 5 Organic Laws of the Operating Organism D+ Policy Engine and how they affect module actions.\nAssistant:"
    probe_ids = torch.tensor([tokenizer.encode(probe_prompt)]).to(device)

    TARGET_STEPS = 200  # Shorter — just a format correction on top of burned weights
    accumulation_steps = 4
    smoothed_loss = None
    alpha = 0.1
    start_time = time.time()

    model.train()
    print(f"Starting Engram Burn — target: {TARGET_STEPS} steps")
    print("=" * 65)

    optimizer.zero_grad()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        try:
            with torch.amp.autocast("cuda"):
                logits = model(x, loop_idx=0)
                loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
                loss = loss / accumulation_steps

            if torch.isnan(loss):
                print(f"NaN at step {step} — skipping")
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
                smoothed_loss = (
                    actual_loss
                    if smoothed_loss is None
                    else (1 - alpha) * smoothed_loss + alpha * actual_loss
                )

                if step % 50 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Engram Step {step:03d}/{TARGET_STEPS} | "
                        f"Loss: {actual_loss:.4f} (Smoothed: {smoothed_loss:.4f}) | "
                        f"Time: {elapsed:.1f}s"
                    )
                    # Quick probe
                    model.eval()
                    with torch.no_grad():
                        out = model.generate(
                            probe_ids, max_new_tokens=60, temperature=0.05, top_k=1
                        )
                    probe_out = tokenizer.decode(
                        out[0].tolist()[len(probe_ids[0]):]
                    ).replace("\n", " ")
                    print(f"  PROBE: {probe_out[:120]}")
                    model.train()
                    start_time = time.time()

                if step % 100 == 0:
                    torch.save(
                        {"step": step, "model_state_dict": model.state_dict()},
                        engram_path,
                    )
                    print(f"  Checkpoint saved → {engram_path}")

                if step >= TARGET_STEPS:
                    break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM — skipping batch")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            raise

    print("\n*** ENGRAM BURN COMPLETE ***")
    torch.save({"step": step, "model_state_dict": model.state_dict()}, engram_path)
    print(f"Final checkpoint saved → {engram_path}")

    # Auto-run benchmark
    print("\n" + "=" * 65)
    print("  AUTO-BENCHMARK: OO Bare-Metal Knowledge Test")
    print("=" * 65)
    model.eval()
    prompts = [
        "User: Explain the 5 Organic Laws of the Operating Organism D+ Policy Engine and how they affect module actions.\nAssistant:",
        "User: Implement a module that allocates a buffer for KV Cache in the bare-metal environment.\nAssistant:",
        "User: Integrate a new inference engine using the official OO Mamba Bridge interface.\nAssistant:",
        "User: Save a new memory state to disk using the bare-metal NeuralFS.\nAssistant:",
        "User: List the commands used to evaluate and apply the Halt Policy in the OO Runtime REPL.\nAssistant:",
    ]
    tokenizer_bench = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    for idx, prompt in enumerate(prompts):
        ids = torch.tensor([tokenizer_bench.encode(prompt)]).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=120, temperature=0.05, top_k=1)
        response = tokenizer_bench.decode(out[0].tolist()[len(ids[0]):])
        print(f"\n[Test {idx + 1}] {prompt.split(chr(10))[0]}")
        print(f"OUTPUT: {response.strip()[:400]}")
        print("-" * 65)
    print("\n*** BENCHMARK COMPLETE — Check results above ***")


if __name__ == "__main__":
    train()
