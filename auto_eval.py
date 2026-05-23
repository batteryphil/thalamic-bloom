"""
auto_eval.py — JARVIS V5 // Titan Phase Evaluation
Loads the phase checkpoint and runs real inference probes.
Tests: perplexity on held-out text, coherence, basic Q&A, repetition.
"""
import argparse
import os
import sys
import json
import math
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# ── Model ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
from mamba3_titan_builder import Mamba3Titan

CKPT_DIR   = os.path.join(PROJECT_DIR, "titan_checkpoints")
VOCAB_SIZE = 50304
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Held-out eval sentences (not in HF OpenHermes stream) ────────────────────
PERPLEXITY_CORPUS = [
    "The mitochondria is the powerhouse of the cell.",
    "In 1969, Neil Armstrong became the first human to walk on the Moon.",
    "Python is a high-level programming language known for its readability.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The French Revolution began in 1789 and fundamentally transformed France.",
    "Machine learning models learn patterns from data without explicit programming.",
    "The human genome contains approximately three billion base pairs of DNA.",
    "Shakespeare wrote thirty-seven plays and one hundred and fifty-four sonnets.",
]

COHERENCE_PROMPTS = [
    ("The capital of France is", ["Paris"]),
    ("Two plus two equals", ["4", "four"]),
    ("Water is made of hydrogen and", ["oxygen"]),
    ("The sun rises in the", ["east"]),
    ("A triangle has", ["3", "three"]),
]

REPETITION_PROMPT = "Tell me about the history of computing. "


def load_model(phase: str):
    ckpt_path = os.path.join(CKPT_DIR, f"phase_{phase}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"  Loading checkpoint: {ckpt_path}")
    model = Mamba3Titan(
        vocab_size=VOCAB_SIZE, d_model=2048, n_layers=80,
        mimo_paths=16, use_gradient_checkpointing=False
    )
    model.set_phase(phase)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(torch.bfloat16).to(DEVICE)
    model.eval()
    return model


def load_tokenizer():
    print("  Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tok.eos_token_id is None:
        tok.eos_token_id = 0
    return tok


@torch.no_grad()
def compute_perplexity(model, tokenizer, sentences):
    """Average per-token cross-entropy loss → perplexity on held-out sentences."""
    total_loss = 0.0
    total_tokens = 0
    for sent in sentences:
        ids = tokenizer.encode(sent, return_tensors="pt").to(DEVICE)
        if ids.shape[1] < 2:
            continue
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            logits, _ = model(ids, loop_idx=0)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, VOCAB_SIZE),
            shift_labels.view(-1),
            reduction="sum"
        )
        total_loss   += loss.item()
        total_tokens += shift_labels.numel()
    avg_loss   = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))   # cap at e^20 to avoid inf display
    return perplexity, avg_loss


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int = 60) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated = ids.clone()
    for _ in range(max_new):
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            logits, _ = model(generated, loop_idx=0)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)


def repetition_score(text: str) -> float:
    """Fraction of tokens that are repeats of the immediately preceding token.
    Lower is better. Random model ≈ 1/vocab ≈ 0.002; degenerate model ≈ 1.0."""
    words = text.split()
    if len(words) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
    return repeats / (len(words) - 1)


def coherence_test(model, tokenizer, prompts):
    hits = 0
    results = []
    for prompt, expected_any in prompts:
        output = generate(model, tokenizer, prompt, max_new=10).strip().lower()
        hit = any(e.lower() in output for e in expected_any)
        hits += int(hit)
        results.append((prompt, output, hit))
    return hits, len(prompts), results


def evaluate(phase: str):
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  JARVIS V5 // TITAN 2.5B — PHASE {phase} AUTO-EVAL")
    print(bar)

    model     = load_model(phase)
    tokenizer = load_tokenizer()

    # ── 1. Perplexity ─────────────────────────────────────────────────────────
    print("\n[ TEST 1 ] Perplexity on held-out corpus")
    t0 = time.time()
    ppl, avg_nll = compute_perplexity(model, tokenizer, PERPLEXITY_CORPUS)
    print(f"  Avg NLL:    {avg_nll:.4f}")
    print(f"  Perplexity: {ppl:.2f}  ({time.time()-t0:.1f}s)")
    # Benchmarks:  random init ≈ 50,000 | early training ≈ 500–5000 | good ≈ <200
    if ppl < 200:
        ppl_verdict = "✅ STRONG — language model is coherent"
    elif ppl < 2000:
        ppl_verdict = "🟡 LEARNING — meaningful signal, not yet coherent"
    else:
        ppl_verdict = "🔴 EARLY — still largely random"
    print(f"  Verdict:    {ppl_verdict}")

    # ── 2. Coherence spot-check ────────────────────────────────────────────────
    print("\n[ TEST 2 ] Coherence spot-check (greedy decode)")
    hits, total, results = coherence_test(model, tokenizer, COHERENCE_PROMPTS)
    for prompt, output, hit in results:
        icon = "✅" if hit else "❌"
        print(f"  {icon}  '{prompt}' → '{output[:60]}'")
    accuracy = hits / total * 100
    print(f"  Score: {hits}/{total}  ({accuracy:.0f}%)")
    if accuracy >= 60:
        coh_verdict = "✅ COHERENT — basic factual associations forming"
    elif accuracy >= 20:
        coh_verdict = "🟡 PARTIAL — some signal, mostly noise"
    else:
        coh_verdict = "🔴 INCOHERENT — expected at this stage"
    print(f"  Verdict:  {coh_verdict}")

    # ── 3. Free generation + repetition ──────────────────────────────────────
    print("\n[ TEST 3 ] Free generation (60 tokens)")
    output = generate(model, tokenizer, REPETITION_PROMPT, max_new=60)
    rep    = repetition_score(output)
    print(f"  Prompt:  '{REPETITION_PROMPT}'")
    print(f"  Output:  '{output[:200]}'")
    print(f"  Repetition rate: {rep:.3f}  (0=none, 1=fully stuck)")
    if rep < 0.05:
        rep_verdict = "✅ FLUID — no repetition loops"
    elif rep < 0.3:
        rep_verdict = "🟡 MINOR repetition — normal early training"
    else:
        rep_verdict = "🔴 HIGH repetition — model is looping"
    print(f"  Verdict: {rep_verdict}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print("  SUMMARY")
    print(bar)
    print(f"  Perplexity:       {ppl:.2f}   {ppl_verdict}")
    print(f"  Coherence:        {hits}/{total} ({accuracy:.0f}%)  {coh_verdict}")
    print(f"  Repetition rate:  {rep:.3f}   {rep_verdict}")

    # Save results to JSON for monitor
    results_path = os.path.join(PROJECT_DIR, "monitor_ui", "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "phase": phase,
            "perplexity": round(ppl, 2),
            "avg_nll": round(avg_nll, 4),
            "coherence_score": f"{hits}/{total}",
            "coherence_pct": round(accuracy, 1),
            "repetition_rate": round(rep, 4),
            "ppl_verdict": ppl_verdict,
            "coh_verdict": coh_verdict,
            "rep_verdict": rep_verdict,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    if ppl < 2000 and accuracy >= 20:
        print("  ✅ Model is learning. Safe to proceed to Phase 2.")
        print("     Run: ./run_titan.sh --phase 2")
    else:
        print("  🟡 Continue Phase 1 training for more steps.")
        print("     Model is in early convergence — not a failure.")
    print(bar + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True,
                        choices=["1", "2", "3", "3j"])
    args = parser.parse_args()
    evaluate(args.phase)
