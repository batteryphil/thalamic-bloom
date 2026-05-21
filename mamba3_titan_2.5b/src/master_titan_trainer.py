import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba3_titan_builder import Mamba3Titan
import argparse
import os
import sys
import json
import time
import math
import signal
import collections
import traceback
import threading
import queue

# Reduce CUDA memory fragmentation — must be set before any CUDA initialization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# GPU temperature via NVML
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def get_gpu_temp():
        return pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
except Exception:
    def get_gpu_temp():
        return None

try:
    from huggingface_hub import login
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("WARNING: huggingface_hub, datasets, or transformers not found.")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN — catches SIGTERM (kill) and SIGINT (Ctrl+C)
# Sets a flag that the training loop checks after every step.
# ─────────────────────────────────────────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[SIGNAL] {sig_name} received — finishing current step then saving checkpoint...")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# Cosine decay with linear warmup. On resume: brief re-warmup from 10% of the
# scheduled LR back to full, to prevent the optimizer-reset gradient spike.
# ─────────────────────────────────────────────────────────────────────────────
def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 500) -> float:
    """Warmup then cosine decay to 5% of base_lr."""
    if step < warmup_steps:
        return base_lr * max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def get_lr(step: int, total_steps: int, base_lr: float,
           warmup_steps: int = 500,
           resume_step: int = 0, resume_warmup: int = 300) -> float:
    """
    Returns the LR for `step`.
    If we just resumed (step near resume_step), we re-warm from 10% → scheduled.
    """
    scheduled = cosine_lr(step, total_steps, base_lr, warmup_steps)
    steps_since_resume = step - resume_step
    if resume_step > 0 and steps_since_resume < resume_warmup:
        frac = steps_since_resume / resume_warmup
        return scheduled * (0.10 + 0.90 * frac)
    return scheduled


def apply_lr(optimizer, lr_core: float, lr_head: float):
    for i, pg in enumerate(optimizer.param_groups):
        pg["lr"] = lr_head if i == 1 else lr_core


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-STOP POLICY
# Tracks a rolling window of recent losses. Triggers stop if:
#   • Catastrophic divergence: rolling mean > DIVERGE_THRESHOLD for
#     DIVERGE_PATIENCE consecutive steps (model has exploded, no recovery)
#   • Plateau stop is intentionally NOT triggered — we let it keep training.
# ─────────────────────────────────────────────────────────────────────────────
DIVERGE_THRESHOLD  = 150.0   # if rolling-1000 mean exceeds this → diverged
DIVERGE_PATIENCE   = 1500    # must stay above threshold this many steps

class AutoStop:
    def __init__(self):
        self._window    = collections.deque(maxlen=1000)
        self._bad_steps = 0

    def update(self, loss: float) -> tuple[bool, str]:
        """Returns (should_stop, reason). Call every step."""
        self._window.append(loss)
        mean = sum(self._window) / len(self._window)

        if mean > DIVERGE_THRESHOLD:
            self._bad_steps += 1
            if self._bad_steps >= DIVERGE_PATIENCE:
                return True, (
                    f"DIVERGENCE: rolling-1000 mean loss {mean:.2f} > "
                    f"{DIVERGE_THRESHOLD} for {self._bad_steps} steps. "
                    f"Training has collapsed — stopping to protect checkpoint."
                )
        else:
            self._bad_steps = 0   # reset if it recovers

        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE — Fixed
# ─────────────────────────────────────────────────────────────────────────────

def get_text_from_item(item, ds_name):
    """Extract clean plain text from each dataset's schema. No raw dicts."""
    n = ds_name.lower()

    if 'fineweb' in n or 'c4' in n or 'pile' in n:
        # FineWeb-Edu: {'text': '...', 'score': ...}
        return item.get('text', '')

    elif 'wikipedia' in n:
        # wikimedia/wikipedia: {'title': '...', 'text': '...'}
        title = item.get('title', '')
        text  = item.get('text', '')
        return f"{title}\n\n{text}" if title else text

    elif 'openhermes' in n or 'hermes' in n:
        # OpenHermes-2.5: format as readable chat text, stripping system prompts
        # System prompts are training artifacts — skip them entirely
        convs = item.get('conversations', [])
        turns = []
        for t in convs:
            role  = (t.get('from') or '').lower()
            value = (t.get('value') or '').strip()
            if not value:
                continue
            if role == 'system':
                continue  # ← Key fix: system prompt was causing "world class trivia AI" memorization
            elif role in ('human', 'user'):
                turns.append(f"User: {value}")
            elif role in ('gpt', 'assistant'):
                turns.append(f"Assistant: {value}")
        return '\n'.join(turns)

    elif 'gsm8k' in n:
        # GSM8K: {'question': '...', 'answer': '...'} — answer has step-by-step
        q = item.get('question', '').strip()
        a = item.get('answer',   '').strip()
        return f"Problem: {q}\nSolution: {a}"

    elif 'metamath' in n:
        # MetaMathQA: {'query': '...', 'response': '...'} — augmented math with CoT
        q = item.get('query',    item.get('question', '')).strip()
        a = item.get('response', item.get('answer',   '')).strip()
        return f"Problem: {q}\nSolution: {a}"

    elif 'code' in n or 'alpaca' in n:
        # CodeAlpaca: {'instruction': '...', 'input': '...', 'output': '...'}
        inst = item.get('instruction', '').strip()
        inp  = item.get('input',       '').strip()
        out  = item.get('output',      '').strip()
        body = f"{inst}\n{inp}".strip() if inp else inst
        return f"Task: {body}\nSolution:\n{out}"

    elif 'cnn' in n or 'dailymail' in n:
        # CNN/DailyMail: {'article': '...', 'highlights': '...'}
        art  = item.get('article',    '').strip()
        summ = item.get('highlights', '').strip()
        return f"{art}\n\nSummary: {summ}"

    elif 'arc' in n:
        # AI2-ARC: {'question': '...', 'choices': {...}, 'answerKey': '...'}
        q = item.get('question', '').strip()
        choices = item.get('choices', {})
        opts = ''
        if choices:
            for lbl, txt in zip(choices.get('label', []), choices.get('text', [])):
                opts += f"\n  {lbl}. {txt}"
        key = item.get('answerKey', '')
        return f"Question: {q}{opts}\nAnswer: {key}"

    # Fallback: try common text keys
    for key in ('text', 'content', 'document', 'passage', 'body'):
        if key in item and isinstance(item[key], str):
            return item[key]
    return ''


def hf_streaming_generator(datasets_mix, tokenizer, seq_len=1024, start_step=0):
    """
    Weighted random sampling across datasets with auto-restart on exhaustion.
    datasets_mix: list of (hf_dataset, weight_int, ds_name_str, domain_id_int)
    start_step:   number of batches to skip at startup (text-only, no GPU tensors)
    """
    import random
    MAX_CHUNKS_PER_DOC = 3

    sources = []
    for ds, weight, name, domain_id in datasets_mix:
        sources.append({
            'iter':      iter(ds),
            'ds':        ds,
            'weight':    weight,
            'name':      name,
            'domain_id': domain_id,
        })

    weights   = [s['weight'] for s in sources]
    exhausted = [False] * len(sources)
    buf_ids, buf_dom = [], []
    emitted = 0   # total batches yielded — used for start_step skip

    # ── Fast-forward: skip to start_step WITHOUT tokenizing or creating tensors ──
    # This avoids re-downloading and GPU-allocating 29K batches on every restart.
    if start_step > 0:
        print(f"[DATA] Fast-forwarding stream to step {start_step:,} (text-only, no GPU)...")
        skipped = 0
        while skipped < start_step:
            active = [i for i, e in enumerate(exhausted) if not e]
            if not active:
                for s in sources: s['iter'] = iter(s['ds'])
                exhausted = [False] * len(sources)
                active    = list(range(len(sources)))
            active_weights = [weights[i] for i in active]
            idx = random.choices(active, weights=active_weights, k=1)[0]
            src = sources[idx]
            try:
                item = next(src['iter'])
                text = get_text_from_item(item, src['name'])
                if not text or len(text.strip()) < 20:
                    continue
                # Estimate how many chunks this doc would have produced
                word_count  = len(text.split())
                est_tokens  = int(word_count * 1.3)
                est_chunks  = min(max(1, est_tokens // seq_len), MAX_CHUNKS_PER_DOC)
                skipped    += est_chunks
                if skipped % 5000 < est_chunks:
                    print(f"[DATA] Skip progress: {min(skipped, start_step):,}/{start_step:,}")
            except StopIteration:
                exhausted[idx] = True
            except Exception:
                src['iter'] = iter(src['ds'])
        emitted = start_step
        print(f"[DATA] Fast-forward complete — resuming from step {start_step:,}")

    while True:
        # Pick source by weight; skip exhausted
        active = [i for i, e in enumerate(exhausted) if not e]
        if not active:
            # All exhausted — restart all
            print("[DATA] All streams exhausted — restarting.")
            for s in sources:
                s['iter'] = iter(s['ds'])
            exhausted = [False] * len(sources)
            active    = list(range(len(sources)))

        active_weights = [weights[i] for i in active]
        idx = random.choices(active, weights=active_weights, k=1)[0]
        src = sources[idx]

        try:
            item    = next(src['iter'])
            text    = get_text_from_item(item, src['name'])
            if not text or len(text.strip()) < 30:
                continue   # skip empty/trivial samples

            # ── Quality filter: character-level OOD detection ─────────────────
            # Word-splitting fails on dense garbage (hex, base64, minified code)
            # that has very few spaces. Character-level ratios work on everything.
            n = len(text)
            # Signal 1: non-ASCII ratio — English prose < 1%, French/accented > 3%
            non_ascii_r = sum(1 for c in text if ord(c) > 127) / n
            if non_ascii_r > 0.03:
                continue   # >3% non-ASCII chars = non-English
            # Signal 2: dense alphanum with no spaces = base64/hex blob
            alnum_r = sum(1 for c in text if c.isalnum()) / n
            space_r = text.count(' ') / n
            if alnum_r > 0.85 and space_r < 0.05:
                continue   # dense blob, no whitespace = encoded garbage
            # Signal 3: heavy special-punct = markup/template/binary
            special = sum(1 for c in text if c in '{}[]<>|\\=;@#$%^&*`~_')
            if special / n > 0.08:
                continue
            # Signal 4: code-syntax density = minified JS/CSS/code
            # Parens+semicolons+dots in code: ~18% | in prose: <3%
            code_syn = sum(1 for c in text if c in "().;")
            if code_syn / n > 0.10:
                continue   # >10% code-syntax chars = not prose
            # ─────────────────────────────────────────────────────────────────


            tokens  = tokenizer.encode(text)
            if len(tokens) < 8:
                continue   # skip too-short token sequences

            # Cap per-document chunks: prevents long FineWeb docs from flooding
            # 10+ consecutive batches with correlated text (root cause of spike clusters)
            MAX_CHUNKS_PER_DOC = 3
            chunk_count = 0
            for i in range(0, len(tokens), seq_len):
                if chunk_count >= MAX_CHUNKS_PER_DOC:
                    break   # discard remaining chunks of this doc; pick a new doc
                chunk = tokens[i:i + seq_len]
                if len(chunk) < seq_len:
                    chunk = chunk + [tokenizer.eos_token_id] * (seq_len - len(chunk))
                buf_ids.append(chunk)
                buf_dom.append(src['domain_id'])
                chunk_count += 1
                if len(buf_ids) >= 1:
                    yield (
                        torch.tensor(buf_ids[:1],  dtype=torch.long),
                        torch.tensor(buf_ids[:1],  dtype=torch.long),
                        torch.tensor(buf_dom[:1],  dtype=torch.long),
                    )
                    buf_ids, buf_dom = buf_ids[1:], buf_dom[1:]

        except StopIteration:
            exhausted[idx] = True
            print(f"[DATA] Stream '{src['name']}' exhausted — will restart when all done.")
        except Exception as e:
            # Catch shard download errors / socket timeouts gracefully
            print(f"[DATA] Stream '{src['name']}' error ({type(e).__name__}: {e}) — restarting stream.")
            src['iter'] = iter(src['ds'])  # restart this one stream immediately



def get_dataloader_for_phase(phase, tokenizer, resume_step=0, seq_len=512):
    """
    Load only the datasets needed for this phase, with correct weights and text extraction.

    Phase 1 — Web pre-training (foundation):
        70% FineWeb-Edu (10B token educational web)  ← was missing entirely
        30% OpenHermes-2.5 (chat text, system prompts stripped)

    Phase 2 — Domain injection (math + code + facts):
        40% MetaMathQA (augmented reasoning chains — much larger/richer than GSM8K alone)
        30% Wikipedia (structured encyclopedic facts)
        20% CodeAlpaca (code reasoning)
        10% GSM8K (grade-school math word problems)

    Phase 3 — Cognitive Bloom (chat format + domain mix):
        50% OpenHermes-2.5 (chat format — arms learn conversational structure)
        25% MetaMathQA (preserve math)
        25% CodeAlpaca (preserve code)

    Phase 3j — Arm Specialization (4-domain routing):
        30% OpenHermes (conversation — arm 0 anchor)
        25% MetaMathQA (math arm)
        25% CodeAlpaca (code arm)
        20% CNN/DailyMail (summarization arm)
    """
    if not HAS_HF:
        print("ERROR: HuggingFace libraries not found.")
        sys.exit(1)

    login(token=HF_TOKEN)
    print(f"\n[DATA] Loading Phase {phase} datasets (streaming)...")

    # ── Phase 1: Web pre-training ─────────────────────────────────────────────
    if phase == '1':
        print("[DATA] Phase 1: 70% FineWeb-Edu + 30% OpenHermes (system prompts stripped)")
        import socket
        socket.setdefaulttimeout(30)  # 30s socket timeout — stalled HF shards fail fast

        ds_fineweb = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        ds_hermes = load_dataset(
            "teknium/OpenHermes-2.5",
            split="train",
            streaming=True,
        )
        mix = [
            (ds_fineweb, 70, 'fineweb-edu',  0),  # domain 0 = general language
            (ds_hermes,  30, 'openhermes',   0),
        ]

    # ── Phase 2: Domain injection ─────────────────────────────────────────────
    elif phase == '2':
        print("[DATA] Phase 2: 40% MetaMath + 30% Wikipedia + 20% CodeAlpaca + 10% GSM8K")
        ds_metamath = load_dataset(
            "meta-math/MetaMathQA",
            split="train",
            streaming=True,
        )
        ds_wiki = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )
        ds_code = load_dataset(
            "HuggingFaceH4/CodeAlpaca_20K",
            split="train",
            streaming=True,
        )
        ds_gsm8k = load_dataset(
            "gsm8k", "main",
            split="train",
            streaming=True,
        )
        mix = [
            (ds_metamath, 40, 'metamath',    1),  # domain 1 = symbolic math
            (ds_wiki,     30, 'wikipedia',   4),  # domain 4 = factual recall
            (ds_code,     20, 'codealpaca',  3),  # domain 3 = code syntax
            (ds_gsm8k,    10, 'gsm8k',       1),
        ]

    # ── Phase 3: Cognitive Bloom ──────────────────────────────────────────────
    elif phase == '3':
        print("[DATA] Phase 3: 50% OpenHermes + 25% MetaMath + 25% CodeAlpaca")
        ds_hermes = load_dataset(
            "teknium/OpenHermes-2.5",
            split="train",
            streaming=True,
        )
        ds_metamath = load_dataset(
            "meta-math/MetaMathQA",
            split="train",
            streaming=True,
        )
        ds_code = load_dataset(
            "HuggingFaceH4/CodeAlpaca_20K",
            split="train",
            streaming=True,
        )
        mix = [
            (ds_hermes,   50, 'openhermes', 7),  # domain 7 = instruction following
            (ds_metamath, 25, 'metamath',   1),
            (ds_code,     25, 'codealpaca', 3),
        ]

    # ── Phase 3j: Arm Specialization (4 distinct domains) ─────────────────────
    elif phase == '3j':
        print("[DATA] Phase 3j: 30% Hermes + 25% MetaMath + 25% CodeAlpaca + 20% CNN")
        ds_hermes = load_dataset(
            "teknium/OpenHermes-2.5",
            split="train",
            streaming=True,
        )
        ds_metamath = load_dataset(
            "meta-math/MetaMathQA",
            split="train",
            streaming=True,
        )
        ds_code = load_dataset(
            "HuggingFaceH4/CodeAlpaca_20K",
            split="train",
            streaming=True,
        )
        ds_cnn = load_dataset(
            "cnn_dailymail", "3.0.0",
            split="train",
            streaming=True,
        )
        mix = [
            (ds_hermes,   30, 'openhermes',   7),   # instruction following
            (ds_metamath, 25, 'metamath',      1),   # math reasoning
            (ds_code,     25, 'codealpaca',    3),   # code syntax
            (ds_cnn,      20, 'cnn_dailymail', 5),   # summarization
        ]

    else:
        print(f"ERROR: Unknown phase '{phase}'")
        sys.exit(1)

    print(f"[DATA] Mix: {[(w, n) for _, w, n, _ in mix]}")
    return hf_streaming_generator(mix, tokenizer, seq_len=seq_len, start_step=resume_step)



def get_previous_phase(phase):
    return {'1': None, '2': '1', '3': '2', '3j': '3'}.get(phase)


# ───────────────────────────────────────────────────────────────────────────
# WORD SALAD — inline generation sanity check every 250 steps
# Uses temperature + top-p + n-gram repetition penalty to get genuine output.
# Picky prompts with known answers so we can tell if the model is learning.
# ───────────────────────────────────────────────────────────────────────────

# Specific factual prompts — the model should know these after pre-training
WORD_SALAD_PROMPTS = [
    # Phase 3: OpenHermes chat format — must match training distribution
    "<|im_start|>user\nWhat is the capital of Japan?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is the chemical formula for water?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a Python function that returns the square of a number.<|im_end|>\n<|im_start|>assistant\n",
]
WORD_SALAD_TOKENS     = 120    # longer — chat answers need more tokens to form
SALAD_TEMPERATURE     = 0.75   # slightly lower — more focused sampling
SALAD_TOP_P           = 0.90
SALAD_REP_PENALTY     = 1.10   # softened from 1.35 — model not diverse enough yet
                                # 1.35 was pushing it off probable tokens → random output
SALAD_NGRAM_BLOCK     = 3      # 3-gram block (was 4) — still kills loops, less aggressive


def _ngram_block_mask(generated_ids: list, ngram: int, vocab_size: int, device) -> torch.Tensor:
    """Returns a logit mask (-inf on banned tokens) to prevent n-gram repeats."""
    mask = torch.zeros(vocab_size, device=device)
    n = len(generated_ids)
    if n < ngram:
        return mask
    prefix = tuple(generated_ids[-(ngram - 1):])
    for i in range(n - ngram + 1):
        if tuple(generated_ids[i:i + ngram - 1]) == prefix:
            banned = generated_ids[i + ngram - 1]
            mask[banned] = float('-inf')
    return mask


def _sample_next(logits_1d: torch.Tensor, generated_ids: list,
                 temperature: float, top_p: float,
                 rep_penalty: float, ngram: int) -> int:
    """Rep penalty → temperature → n-gram block → top-p nucleus → sample."""
    vocab = logits_1d.shape[0]
    logits = logits_1d.float().clone()

    # Repetition penalty
    for tok in set(generated_ids):
        if logits[tok] > 0:
            logits[tok] /= rep_penalty
        else:
            logits[tok] *= rep_penalty

    logits /= max(temperature, 1e-8)
    logits += _ngram_block_mask(generated_ids, ngram, vocab, logits.device)

    # Top-p nucleus
    sorted_l, sorted_i = torch.sort(logits, descending=True)
    cum = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
    remove = (cum - torch.softmax(sorted_l, dim=-1)) > top_p
    sorted_l[remove] = float('-inf')
    probs = torch.softmax(sorted_l, dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        probs = torch.ones_like(probs) / vocab
    return int(sorted_i[torch.multinomial(probs, 1)].item())


@torch.no_grad()
def run_word_salad(model, tokenizer, device, step, phase,
                   save_dir, optimizer, salad_path):
    """Checkpoint → eval → sample with full pipeline → write monitor → train."""
    print(f"\n[SALAD] Step {step}: checkpoint → generate → resume...")
    save_checkpoint(save_dir, phase, step, model, optimizer,
                    reason=f"pre_word_salad_step_{step}")

    model.eval()
    samples   = []
    gen_start = time.time()
    vocab_size = getattr(model, 'vocab_size', 50304)

    for prompt in WORD_SALAD_PROMPTS:
        try:
            prompt_ids = tokenizer.encode(prompt)
            generated  = list(prompt_ids)
            id_tensor  = torch.tensor([generated], dtype=torch.long, device=device)

            for _ in range(WORD_SALAD_TOKENS):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits, _ = model(id_tensor, loop_idx=0)
                next_id = _sample_next(
                    logits[0, -1, :], generated,
                    SALAD_TEMPERATURE, SALAD_TOP_P,
                    SALAD_REP_PENALTY, SALAD_NGRAM_BLOCK,
                )
                generated.append(next_id)
                id_tensor = torch.tensor([generated], dtype=torch.long, device=device)
                if next_id == tokenizer.eos_token_id:
                    break

            output    = tokenizer.decode(generated[len(prompt_ids):], skip_special_tokens=True).strip()
            words     = output.split()
            rep_rate  = round(1.0 - len(set(words)) / max(len(words), 1), 3)

        except Exception as e:
            output, rep_rate = f"[error: {e}]", 1.0
            traceback.print_exc()

        quality = "✅" if rep_rate < 0.30 else ("🟡" if rep_rate < 0.60 else "🔴")
        print(f"  {quality} rep={rep_rate:.0%} | {prompt[:40]}")
        print(f"     → {output[:120]}")
        samples.append({"prompt": prompt, "output": output, "rep_rate": rep_rate})

    elapsed = time.time() - gen_start
    avg_rep = round(sum(s['rep_rate'] for s in samples) / len(samples), 3)
    quality = "good" if avg_rep < 0.30 else ("fair" if avg_rep < 0.60 else "poor")

    with open(salad_path, "w") as f:
        json.dump({
            "step": step, "phase": phase,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(elapsed, 2),
            "avg_rep": avg_rep, "quality": quality,
            "samples": samples,
        }, f, indent=2)

    print(f"[SALAD] Done in {elapsed:.1f}s | avg_rep={avg_rep:.0%} | {quality}\n")
    model.train()




# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS — always saves model + optimizer + step + LR state
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(save_dir, phase, step, model, optimizer, reason="periodic"):
    ckpt_path = os.path.join(save_dir, f"phase_{phase}.pt")
    tmp_path  = ckpt_path + ".tmp"
    print(f"\n[CKPT] Saving checkpoint at step {step} ({reason})...")
    state = {
        'model': model.state_dict(),
        'step':  step,
        'phase': phase,
    }
    torch.save(state, tmp_path)
    # Save optimizer state separately in CPU pinned format (PagedAdam8bit)
    # This keeps the main checkpoint small and avoids VRAM OOM on restore.
    opt_path = ckpt_path.replace('.pt', '_optim.pt')
    try:
        opt_state = {k: {sk: sv.cpu() if hasattr(sv, 'cpu') else sv
                         for sk, sv in v.items()} if isinstance(v, dict) else v
                     for k, v in optimizer.state_dict().items()}
        torch.save(opt_state, opt_path + '.tmp')
        os.replace(opt_path + '.tmp', opt_path)
    except Exception as _oe:
        pass  # optimizer save failure never blocks training
    os.replace(tmp_path, ckpt_path)   # atomic swap — no corrupt file on kill
    print(f"[CKPT] Saved → {ckpt_path}")
    return ckpt_path


def load_checkpoint(ckpt_path, model, optimizer, device):
    """Load model + optimizer state. Returns resume_step (0 if not found)."""
    if not os.path.exists(ckpt_path):
        return 0
    print(f"[CKPT] Loading checkpoint: {ckpt_path}")
    # Load entire checkpoint to CPU first — avoids VRAM OOM on optimizer restore.
    # PagedAdam8bit keeps moments in CPU RAM anyway, so this is the natural path.
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'], strict=False)
    opt_path = ckpt_path.replace('.pt', '_optim.pt')
    if os.path.exists(opt_path):
        try:
            opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
            optimizer.load_state_dict(opt_state)
            print("[CKPT] Optimizer state restored ✅ (from CPU sidecar file)")
        except Exception as e:
            print(f"[CKPT] Optimizer sidecar load failed ({type(e).__name__}) — fresh optimizer")
    else:
        print("[CKPT] No optimizer sidecar — fresh PagedAdam8bit (300-step re-ramp active)")
    resume_step = int(ckpt.get('step', 0))
    print(f"[CKPT] Resuming from step {resume_step}")
    return resume_step


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--phase', type=str, required=True, choices=['1', '2', '3', '3j'])
    try:
        import deepspeed
        parser = deepspeed.add_config_arguments(parser)
    except ImportError:
        pass
    cmd_args = parser.parse_args()
    phase    = cmd_args.phase

    # ── Paths ───────────────────────────────────────────────────────────────────────────
    project_dir  = os.path.dirname(os.path.abspath(__file__))
    save_dir     = os.path.join(project_dir, "titan_checkpoints")
    telem_path   = os.path.join(project_dir, "monitor_ui", "telemetry.json")
    salad_path   = os.path.join(project_dir, "monitor_ui", "word_salad.json")
    log_path     = os.path.join(project_dir, "training_log.txt")
    os.makedirs(save_dir, exist_ok=True)

    # ── Continuance log banner ───────────────────────────────────────────────
    # Always append; the bash launcher wrote the === RESTART === line already.
    # We write a Python-level header here too for clarity.
    with open(log_path, "a") as lf:
        lf.write(f"\n[TRAINER] Process start  UTC={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}  phase={phase}\n")

    # ── Device / Model ───────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if HAS_HF:
        print("Loading Tokenizer (EleutherAI/gpt-neox-20b)...")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = 0
    else:
        tokenizer = None

    model = Mamba3Titan(vocab_size=50304, d_model=2048, n_layers=80,
                        mimo_paths=16, use_gradient_checkpointing=True)
    if phase == '1':
        model.initialize_asymmetric_arms()
    model.set_phase(phase)
    model = model.to(torch.bfloat16).to(device)
    print(f"Model loaded in BF16 on {device}. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Phase-conditional LR ─────────────────────────────────────────────────
    if phase == '1':
        BASE_LR_CORE  = 3e-5
        BASE_LR_HEAD  = 6e-5
    elif phase == '2':
        BASE_LR_CORE  = 1e-5
        BASE_LR_HEAD  = 2e-5
    elif phase in ('3', '3j'):
        BASE_LR_CORE  = 5e-6
        BASE_LR_HEAD  = 1e-5
    else:
        BASE_LR_CORE  = 1e-5
        BASE_LR_HEAD  = 2e-5

    # ── Optimizer — BEFORE torch.compile ─────────────────────────────────────
    # Param collection MUST happen on the unwrapped model. After torch.compile,
    # model.parameters() returns wrapped tensors; id() comparison and
    # requires_grad filtering both break on the compiled object.
    # The compiled model still writes gradients to these same underlying tensors.
    head_params_set  = set(id(p) for p in model.lm_head.parameters())
    core_params_list = [p for p in model.parameters()
                        if id(p) not in head_params_set and p.requires_grad]
    head_params_list = [p for p in model.lm_head.parameters() if p.requires_grad]

    try:
        import bitsandbytes as bnb
        # PagedAdam8bit: optimizer moments live in PINNED CPU RAM.
        # Automatically paged to GPU only for the update step, then evicted.
        # Frees ~6.4 GB VRAM vs standard Adam, enabling full optimizer state
        # restore without OOM — critical for arm momentum accumulation.
        optimizer = bnb.optim.PagedAdam8bit([
            {'params': core_params_list, 'lr': BASE_LR_CORE},
            {'params': head_params_list, 'lr': BASE_LR_HEAD},
        ], weight_decay=0.01)
        print("Using bitsandbytes PagedAdam8bit optimizer (moments in CPU RAM).")
    except ImportError:
        print("WARNING: bitsandbytes not found — falling back to AdamW.")
        optimizer = torch.optim.AdamW([
            {'params': core_params_list, 'lr': BASE_LR_CORE},
            {'params': head_params_list, 'lr': BASE_LR_HEAD},
        ], weight_decay=0.01)

    # torch.compile disabled: builder has inplace ops (cp_gate telemetry, buffer updates)
    # that trigger "tensor modified by inplace op" in compiled autograd graph.
    # Prefetch + non_blocking H2D are the larger wins and remain active.

    criterion = nn.CrossEntropyLoss()

    # ── Checkpoint: load phase checkpoint (if exists) for resume ─────────────
    ckpt_path   = os.path.join(save_dir, f"phase_{phase}.pt")
    resume_step = 0

    # If this phase has a checkpoint, load it (model + optimizer)
    if os.path.exists(ckpt_path):
        resume_step = load_checkpoint(ckpt_path, model, optimizer, device)
    else:
        # Cold start: load previous phase model weights only (no optimizer)
        prev_phase = get_previous_phase(phase)
        if prev_phase:
            prev_ckpt = os.path.join(save_dir, f"phase_{prev_phase}.pt")
            if os.path.exists(prev_ckpt):
                print(f"Loading Phase {prev_phase} weights for cold-start of Phase {phase}...")
                # Load to CPU first — map_location=device would put the full 6GB ckpt
                # (model + optimizer) onto CUDA, exhausting VRAM before training starts
                _ckpt = torch.load(prev_ckpt, map_location='cpu', weights_only=True)
                _model_state = _ckpt['model']
                del _ckpt  # immediately free optimizer tensors from CPU RAM
                model.load_state_dict(_model_state, strict=False)
                del _model_state
                import gc; gc.collect()
                print("Phase weights loaded (optimizer state NOT carried over).")
            else:
                print(f"ERROR: Phase {prev_phase} checkpoint not found at {prev_ckpt}.")
                sys.exit(1)

    RESUME_WARMUP_STEPS = 300  # full warmup for new phase

    # ── Training config (MUST be before dataloader so seq_len is defined) ────
    if phase == '1':
        target_steps = 50_000
        seq_len      = 1024
        WARMUP_STEPS = 500
    elif phase == '2':
        target_steps = 30_000
        seq_len      = 512
        WARMUP_STEPS = 300
    elif phase in ('3', '3j'):
        target_steps = 30_000
        seq_len      = 768   # safe to bump: fresh optimizer, cold-start from phase_2.pt
        WARMUP_STEPS = 500   # longer warmup — Blackboard unfreezes, arms begin specializing
    else:
        target_steps = 30_000
        seq_len      = 512
        WARMUP_STEPS = 300

    SAVE_EVERY    = 500
    DATA_TIMEOUT  = 300
    GRAD_ACCUM    = 1

    # ── Data pipeline ────────────────────────────────────────────────────────
    dataloader_generator = get_dataloader_for_phase(
        phase, tokenizer, resume_step=resume_step, seq_len=seq_len
    )

    # OPT 4: Prefetch thread — keeps a 2-batch lookahead queue so the GPU never
    # idles waiting for the CPU to tokenize+assemble the next batch.
    _PREFETCH_SIZE = 2
    _prefetch_q    = queue.Queue(maxsize=_PREFETCH_SIZE)
    _sentinel      = object()  # signals generator exhaustion

    def _prefetch_worker(gen, q):
        try:
            for item in gen:
                # Pin tensors to page-locked memory for faster H2D DMA (OPT 4b)
                pinned = tuple(
                    t.pin_memory() if isinstance(t, torch.Tensor) else t
                    for t in item
                )
                q.put(pinned)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_sentinel)

    _prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(dataloader_generator, _prefetch_q),
        daemon=True,
        name='DataPrefetch'
    )
    _prefetch_thread.start()

    def _iter_prefetch():
        while True:
            item = _prefetch_q.get()
            if item is _sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    dataloader_generator = _iter_prefetch()


    # ── LR cycle restart on new-data resume ──────────────────────────────────
    # If we're resuming mid-phase on a different dataset (FineWeb-Edu injection),
    # the cosine schedule at step 25K+ gives near-zero LR. Instead, treat LR as
    # a fresh cosine cycle anchored to the resume_step so the model gets a
    # proper learning rate for the new data distribution.
    # Effective LR reference step = 0 for the purposes of cosine scheduling,
    # but we shift all step indices by resume_step when computing LR.
    # LR schedule: track actual step, no offset reset on restart.
    # The resume_warmup (300 steps) already provides a gentle re-ramp after
    # each restart. Cosine at step 6K-10K gives 4-5e-6 — no need to reset.
    lr_step_offset = 0
    lr_cycle_steps = max(target_steps, 20_000)  # full cosine span

    auto_stop = AutoStop()
    model.train()

    header = (
        f"\n{'='*72}\n"
        f"  MAMBA3 TITAN 2.5B  |  Phase {phase}  |  Resume step {resume_step}\n"
        f"  UTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n"
        f"  Target: {target_steps:,} steps  |  Remaining: {target_steps - resume_step:,}\n"
        f"{'='*72}"
    )
    print(header)
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    start_time      = time.time()
    step_start_time = time.time()
    last_lr_core    = BASE_LR_CORE

    for step, batch in enumerate(dataloader_generator, start=resume_step):

        if step >= target_steps:
            print(f"\n[PHASE COMPLETE] Reached {target_steps:,} steps for Phase {phase}.")
            save_checkpoint(save_dir, phase, step, model, optimizer, reason="phase_complete")
            break

        # ── LR update: cosine schedule on actual step, gentle resume re-ramp ──
        lr_core = get_lr(step, lr_cycle_steps, BASE_LR_CORE,
                         warmup_steps=WARMUP_STEPS,
                         resume_step=resume_step,
                         resume_warmup=RESUME_WARMUP_STEPS)
        lr_head = get_lr(step, lr_cycle_steps, BASE_LR_HEAD,
                         warmup_steps=WARMUP_STEPS,
                         resume_step=resume_step,
                         resume_warmup=RESUME_WARMUP_STEPS)
        apply_lr(optimizer, lr_core, lr_head)
        last_lr_core = lr_core


        # ── Forward / backward ────────────────────────────────────────
        input_ids, labels, domain_ids = batch
        # OPT 5: non_blocking=True — H2D DMA overlaps with previous step's optimizer.step()
        input_ids  = input_ids.to(device,  non_blocking=True)
        labels     = labels.to(device,     non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, domain_loss = model(input_ids, loop_idx=0, domain_ids=domain_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = criterion(shift_logits.view(-1, 50304), shift_labels.view(-1))
            # domain_loss: load-balance entropy in Phase 3, routing supervision in 3j
            loss    = lm_loss + (0.1 * domain_loss if phase in ('3', '3j') else 0.0)

        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        is_update_step = True
        if (step + 1) % 20 == 0:
            import gc; gc.collect()   # defrag every 20 steps — not every step (was -35% TPS)
            torch.cuda.empty_cache()


        # ── Metrics ─────────────────────────────────────────────────────────
        step_elapsed    = time.time() - step_start_time
        tps             = seq_len / step_elapsed if step_elapsed > 0 else 0.0
        step_start_time = time.time()
        gpu_temp        = get_gpu_temp()

        # Stall watchdog
        if step_elapsed > DATA_TIMEOUT:
            print(f"[WATCHDOG] Step {step+1} took {step_elapsed:.1f}s > {DATA_TIMEOUT}s. Saving and aborting.")
            save_checkpoint(save_dir, phase, step + 1, model, optimizer, reason="watchdog_stall")
            sys.exit(1)

        telem     = model.last_telemetry
        dom_l     = domain_loss.item() if isinstance(domain_loss, torch.Tensor) else domain_loss
        gate      = telem.get('gate_score', 0.0)
        entropy   = telem.get('entropy', 0.0)
        temp_str  = f" | GPU: {gpu_temp}°C" if gpu_temp is not None else ""
        lr_str    = f" | LR: {lr_core:.2e}"

        grad_norm_val = float(grad_norm) if is_update_step else 0.0
        log_line = (
            f"Phase {phase} | Step {step+1:05d} | LM Loss: {lm_loss.item():.4f} | "
            f"Dom Loss: {dom_l:.4f} | Gate: {gate:.4f} | Entropy: {entropy:.4f} | "
            f"GNorm: {grad_norm_val:.2f} | TPS: {tps:.1f}{temp_str}{lr_str}"
        )
        print(log_line)

        # Append to training log (continuance — no overwrite)
        with open(log_path, "a") as lf:
            lf.write(log_line + "\n")

        # ── Arm Divergence: weight-based cosine similarity (every 10 steps) ────
        # Uses normalized flattened in_proj weight vectors per arm.
        # Confirmed: arm weight L2 diffs exist (norm~3.2) — weight cosine reveals it.
        # Collapse score = (cosine_sim + 1) / 2  →  0=orthogonal, 1=clone
        arm_sims_live, col_mean_live, col_max_live = [], 1.0, 1.0
        if (step + 1) % 10 == 0:
            try:
                with torch.no_grad():
                    arm_vecs = []
                    for i in range(16):
                        # Use ssm.proj OR ssm.in_proj weight — NOT norm.weight
                        # (LayerNorm weight is nearly identical across arms → always ~1.0)
                        # Move to CPU to avoid OOM — this is a small tensor (~512K floats)
                        arm_mod = model.mimo_reasoning_blocks[i]
                        # Skip LayerNorm params — get the first SSM weight
                        w = None
                        for name, p in arm_mod.named_parameters():
                            if 'ssm' in name and 'weight' in name and p.dim() >= 2:
                                w = p.detach().float().cpu().view(-1)
                                break
                        if w is None:  # fallback
                            w = list(arm_mod.parameters())[-1].detach().float().cpu().view(-1)
                        arm_vecs.append(torch.nn.functional.normalize(w, dim=0))
                    arm_mat  = torch.stack(arm_vecs, dim=0)  # [16, D] on CPU
                    sim_mat  = arm_mat @ arm_mat.T            # [16, 16] in [-1, 1] — tiny, safe
                    # Convert to collapse score [0,1]: 1=clone, 0=diverse
                    collapse_mat = (sim_mat + 1.0) / 2.0
                    off_diag = ~torch.eye(16, dtype=torch.bool)
                    per_arm  = (collapse_mat * off_diag.float()).sum(dim=1) / 15.0
                    arm_sims_live = [round(v, 4) for v in per_arm.tolist()]
                    col_mean_live = round(per_arm.mean().item(), 4)
                    col_max_live  = round(per_arm.max().item(), 4)
            except Exception:
                pass  # never crash training over telemetry

        # ── Telemetry for monitor UI ─────────────────────────────────────────
        telemetry_data = {
            "phase":               phase,
            "step":                step + 1,
            "lm_loss":             round(lm_loss.item(), 4),
            "domain_loss":         round(dom_l, 4),
            "gate_score":          round(gate, 4),
            "entropy":             round(entropy, 4),
            "grad_norm":           round(grad_norm_val, 4),
            "tps":                 round(tps, 1),
            "gpu_temp":            gpu_temp,
            "lr":                  round(lr_core, 8),
            "resume_step":         resume_step,
            # Weight-based arm divergence (trainer-side, bypasses grad ckpt)
            "arm_collapse_metric": col_mean_live if arm_sims_live else round(telem.get('arm_collapse_mean', 1.0), 4),
            "arm_collapse_mean":   col_mean_live,
            "arm_collapse_max":    col_max_live,
            "arm_sims":            arm_sims_live if arm_sims_live else telem.get('arm_sims', []),
            "latent_energy":       round(telem.get('latent_energy', 0.0), 4),
            "arm_weights":         telem.get('arm_weights', []),
        }
        with open(telem_path, "w") as f:
            json.dump(telemetry_data, f)


        # ── Auto-stop check ──────────────────────────────────────────────────
        should_stop, reason = auto_stop.update(lm_loss.item())
        if should_stop:
            print(f"\n[AUTO-STOP] {reason}")
            save_checkpoint(save_dir, phase, step + 1, model, optimizer, reason="auto_stop_divergence")
            with open(log_path, "a") as lf:
                lf.write(f"[AUTO-STOP] {reason}\n")
            sys.exit(2)  # exit code 2 = diverged; run_titan.sh skips eval

        # ── Word salad every 250 steps ────────────────────────────────────────
        # Checkpoint is saved inside run_word_salad before switching eval mode.
        if (step + 1) % 250 == 0 and tokenizer is not None:
            run_word_salad(model, tokenizer, device,
                           step + 1, phase, save_dir, optimizer, salad_path)
            # model.train() is called inside run_word_salad on exit

        # ── Periodic checkpoint every 500 steps ───────────────────────────────
        # Word salad at step%250 already checkpoints, so skip on those steps.
        elif (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(save_dir, phase, step + 1, model, optimizer, reason="periodic")

        # ── Graceful shutdown (SIGTERM / SIGINT) ─────────────────────────────
        if _shutdown_requested:
            print(f"\n[SHUTDOWN] Saving checkpoint at step {step+1} and exiting cleanly.")
            save_checkpoint(save_dir, phase, step + 1, model, optimizer, reason="graceful_shutdown")
            with open(log_path, "a") as lf:
                lf.write(f"[SHUTDOWN] Graceful save at step {step+1}  "
                         f"UTC={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n")
            sys.exit(0)

    # End of loop
    elapsed = time.time() - start_time
    print(f"\nExecution Time: {elapsed:.2f}s")
    print("\n" + "="*72)
    print(f"  Phase {phase} COMPLETE — run auto_eval.py --phase {phase} to verify.")
    print("="*72)


if __name__ == "__main__":
    train()
