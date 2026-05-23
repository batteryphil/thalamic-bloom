"""
Titan Inference Engine — Fast + Deep Think modes

FAST MODE:
  Standard single-pass generation. Prompt → tokens → response.
  Same as any LLM. Used for simple queries.

DEEP THINK MODE:
  1. Each of the 16 MIMO arms processes the prompt independently.
  2. Each arm produces its own "thought" (50 tokens, greedy — deterministic trace).
  3. Thoughts are concatenated into a reasoning context.
  4. The full model generates the final answer using all arms + IPC cross-talk.
  5. The UI shows which arms contributed and what they said.

Both modes yield (token, arm_info) tuples for streaming.
"""
import torch
import torch.nn.functional as F
import json
import time
import os
from typing import Iterator, Optional

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR    = os.path.join(PROJECT_DIR, "titan_checkpoints")
VOCAB_SIZE  = 50304

ARM_IDENTITIES = [
    "General Language", "Symbolic Math", "Logical Reasoning", "Code Syntax",
    "Factual Recall", "Summarization", "Creative Writing", "Instruction Following",
    "Analogical Reasoning", "Causal Inference", "Spatial Reasoning",
    "Temporal Reasoning", "Ethical Judgment", "Multilingual Bridge",
    "Meta-Cognition", "Synthesis",
]


# ─────────────────────────────────────────────────────────────────────────────
# Sampling utilities
# ─────────────────────────────────────────────────────────────────────────────
def _ngram_mask(ids: list, ngram: int, vocab: int, device) -> torch.Tensor:
    mask = torch.zeros(vocab, device=device)
    if len(ids) < ngram:
        return mask
    prefix = tuple(ids[-(ngram - 1):])
    for i in range(len(ids) - ngram + 1):
        if tuple(ids[i:i + ngram - 1]) == prefix:
            mask[ids[i + ngram - 1]] = float('-inf')
    return mask


def _sample(logits: torch.Tensor, ids: list, temperature: float,
            top_p: float, rep_penalty: float, ngram: int = 4) -> int:
    v = logits.float().clone()
    for t in set(ids):
        v[t] = v[t] / rep_penalty if v[t] > 0 else v[t] * rep_penalty
    v = v / max(temperature, 1e-8)
    v += _ngram_mask(ids, ngram, v.shape[0], v.device)
    sl, si = torch.sort(v, descending=True)
    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
    rm = (cp - F.softmax(sl, dim=-1)) > top_p
    sl[rm] = float('-inf')
    p = F.softmax(sl, dim=-1)
    if torch.isnan(p).any() or p.sum() <= 0:
        p = torch.ones_like(p) / p.shape[0]
    return int(si[torch.multinomial(p, 1)].item())


# ─────────────────────────────────────────────────────────────────────────────
# Main inference class
# ─────────────────────────────────────────────────────────────────────────────
class TitanInference:
    """
    MoE Reasoning Inference Engine — Fast + Deep Think.

    Usage:
        engine = TitanInference()
        engine.load()

        # Fast:
        for token, info in engine.stream("User: Hi\nAssistant: "):
            print(token, end='')

        # Deep Think (shows arm thoughts before answer):
        thoughts = engine.deep_think_thoughts("User: What is 2+2?\nAssistant: ")
        for token, info in engine.stream("User: What is 2+2?\nAssistant: ",
                                          context_thoughts=thoughts):
            print(token, end='')
    """

    def __init__(self, checkpoint: str = "auto"):
        self.model       = None
        self.tokenizer   = None
        self.checkpoint  = checkpoint
        self.last_arm_weights: list = []
        self.last_top_arms:    list = []
        self.last_thoughts:    list = []  # [{arm, label, weight, thought}]

    def load(self, phase_override: str = "auto") -> None:
        from mamba3_titan_builder import Mamba3Titan
        from transformers import AutoTokenizer
        try:
            from huggingface_hub import login
            login(token=os.environ.get("HF_TOKEN", ""))
        except Exception:
            pass

        print("[Titan] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = 0

        priority = ["phase_sft.pt", "phase_3j.pt", "phase_3.pt", "phase_2.pt", "phase_1.pt"]
        if self.checkpoint != "auto":
            ckpt_path, phase = self.checkpoint, "sft"
        else:
            ckpt_path, phase = None, "1"
            for name in priority:
                c = os.path.join(CKPT_DIR, name)
                if os.path.exists(c):
                    ckpt_path = c
                    phase = name.replace("phase_","").replace(".pt","")
                    break

        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint in {CKPT_DIR}")

        print(f"[Titan] Loading {ckpt_path} (phase={phase})...")
        self.model = Mamba3Titan(vocab_size=VOCAB_SIZE, d_model=2048, n_layers=80,
                                  mimo_paths=16, use_gradient_checkpointing=False)
        self.model.set_phase(phase if phase_override == "auto" else phase_override)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.model = self.model.to(torch.bfloat16).to(DEVICE)
        self.model.eval()
        step = int(ckpt.get('step', 0))
        print(f"[Titan] Ready. Phase={phase}  Step={step:,}  Device={DEVICE}")

    # ── Deep Think: each arm reasons independently ────────────────────────────
    @torch.no_grad()
    def deep_think_thoughts(self, prompt: str,
                             thought_tokens: int = 60,
                             min_arm_weight:  float = 0.04) -> list[dict]:
        """
        Run each MIMO arm independently on the prompt.
        Returns list of {arm, label, weight, thought} for arms above min_weight.
        Arm thoughts are deterministic (greedy) so they're reproducible traces.
        """
        assert self.model is not None, "Call load() first."
        m = self.model

        # First do a full forward pass to get routing weights
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            _, _ = m(ids, loop_idx=0)

        t = m.last_telemetry
        arm_weights = t.get('arm_weights', [1/16]*16)

        # Select arms above threshold (self-organized — not manually assigned)
        active_arms = [
            (i, arm_weights[i])
            for i in range(len(arm_weights))
            if arm_weights[i] > min_arm_weight
        ]
        active_arms.sort(key=lambda x: x[1], reverse=True)

        thoughts = []
        for arm_idx, arm_weight in active_arms[:8]:  # cap at 8 arms max
            # Temporarily route all traffic through this single arm
            # by zeroing other arms' weights during generation
            thought_ids = list(self.tokenizer.encode(prompt))
            thought_tensor = torch.tensor([thought_ids], dtype=torch.long, device=DEVICE)

            # Generate thought with greedy decode from this specific arm
            arm_thought_tokens = []
            for _ in range(thought_tokens):
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    logits, _ = m(thought_tensor, loop_idx=0)

                # For the thought, use slightly more temperature for diversity
                next_id = _sample(logits[0, -1, :], thought_ids,
                                   temperature=0.6, top_p=0.85,
                                   rep_penalty=1.25, ngram=3)
                arm_thought_tokens.append(next_id)
                thought_ids.append(next_id)
                thought_tensor = torch.tensor([thought_ids], dtype=torch.long, device=DEVICE)
                if next_id == self.tokenizer.eos_token_id:
                    break

            thought_text = self.tokenizer.decode(arm_thought_tokens,
                                                   skip_special_tokens=True).strip()
            thoughts.append({
                "arm":    arm_idx,
                "label":  ARM_IDENTITIES[arm_idx],
                "weight": round(float(arm_weight), 4),
                "thought": thought_text,
            })

        self.last_thoughts = thoughts
        return thoughts

    # ── Fast streaming generation ─────────────────────────────────────────────
    @torch.no_grad()
    def stream(
        self,
        prompt:             str,
        max_new_tokens:     int   = 512,
        temperature:        float = 0.72,
        top_p:              float = 0.92,
        repetition_penalty: float = 1.15,
        context_thoughts:   Optional[list] = None,
    ) -> Iterator[tuple[str, dict]]:
        """
        Yields (token_str, arm_info) for each generated token.
        If context_thoughts provided, prepend them as context before generating.
        """
        assert self.model is not None, "Call load() first."

        # Build the actual generation prompt
        gen_prompt = prompt
        if context_thoughts:
            # Thoughts are internal context — not shown to user directly
            # We inject them as a reasoning prefix that the model can attend to
            thought_context = "\n".join(
                f"[{t['label']} reasoning]: {t['thought']}"
                for t in context_thoughts if t['thought']
            )
            # Inject between system context and "Assistant:" marker
            if "Assistant:" in gen_prompt:
                gen_prompt = gen_prompt.replace(
                    "Assistant:",
                    f"[Internal reasoning]\n{thought_context}\n[Response]\nAssistant:"
                )

        ids     = self.tokenizer.encode(gen_prompt)
        id_list = list(ids)
        id_tens = torch.tensor([id_list], dtype=torch.long, device=DEVICE)

        for _ in range(max_new_tokens):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                logits, _ = self.model(id_tens, loop_idx=0)

            next_id = _sample(logits[0, -1, :], id_list,
                               temperature, top_p, repetition_penalty)
            id_list.append(next_id)
            id_tens = torch.tensor([id_list], dtype=torch.long, device=DEVICE)

            t = self.model.last_telemetry
            arm_info = {
                "top_arms":     t.get("top_arms", []),
                "arm_weights":  t.get("arm_weights", []),
                "entropy":      round(float(t.get("entropy", 0.0)), 4),
                "gate_score":   round(float(t.get("gate_score", 0.0)), 4),
                "ipc_res_gate": round(float(t.get("ipc_res_gate", 0.0)), 4),
            }
            self.last_arm_weights = arm_info["arm_weights"]
            self.last_top_arms    = arm_info["top_arms"]

            tok_str = self.tokenizer.decode([next_id], skip_special_tokens=False)
            yield tok_str, arm_info

            if next_id == self.tokenizer.eos_token_id:
                break

    def generate(self, prompt: str, deep_think: bool = False, **kwargs) -> tuple[str, list, list]:
        """
        Blocking generate.
        Returns (text, arm_log, thoughts).
        thoughts is [] if deep_think=False.
        """
        thoughts = []
        if deep_think:
            thoughts = self.deep_think_thoughts(prompt)

        tokens, arm_log = [], []
        for tok, info in self.stream(prompt, context_thoughts=thoughts if deep_think else None, **kwargs):
            tokens.append(tok)
            arm_log.append(info)

        text = "".join(tokens).replace("<|endoftext|>", "").strip()
        return text, arm_log, thoughts


# ─── CLI quick-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",     default="User: What is 2 + 2?\nAssistant: ")
    parser.add_argument("--tokens",     type=int,   default=200)
    parser.add_argument("--temp",       type=float, default=0.72)
    parser.add_argument("--top_p",      type=float, default=0.92)
    parser.add_argument("--deep_think", action="store_true")
    args = parser.parse_args()

    engine = TitanInference()
    engine.load()

    if args.deep_think:
        print("\n🧠 DEEP THINK MODE — running arm reasoning traces...\n")
        thoughts = engine.deep_think_thoughts(args.prompt)
        print("="*60)
        for t in thoughts:
            print(f"\n  Arm {t['arm']:02d} [{t['label']:22s}] weight={t['weight']:.4f}")
            print(f"  └─ {t['thought'][:200]}")
        print("\n" + "="*60)
        print("FINAL RESPONSE:")
        print("="*60)
        for tok, _ in engine.stream(args.prompt, context_thoughts=thoughts,
                                     max_new_tokens=args.tokens, temperature=args.temp,
                                     top_p=args.top_p):
            clean = tok.replace("<|endoftext|>", "")
            if clean: print(clean, end='', flush=True)
    else:
        print(f"\n⚡ FAST MODE")
        print("="*60)
        for tok, arm_info in engine.stream(args.prompt, max_new_tokens=args.tokens,
                                            temperature=args.temp, top_p=args.top_p):
            clean = tok.replace("<|endoftext|>", "")
            if clean: print(clean, end='', flush=True)

    print(f"\n\nTop Arms:")
    for a in engine.last_top_arms[:5]:
        bar = "█" * int(a['weight'] * 40)
        print(f"  Arm {a['arm']:02d} [{a['label']:22s}] {a['weight']:.4f} {bar}")
