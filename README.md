# thalamic-bloom

Mamba 3 MIMO reasoning engine. 150M parameters. 4 parallel cognitive arms with dynamic entropy-based routing.

Discovered and solved the **SSM Entropy Glass Ceiling**. Features Adiabatic Bloom, Trickle-Charge MoE reactivation, and Temporal Vector Gating across 4 parallel cognitive arms.

---

## Architecture

**Model:** `Mamba3MIMORLF` — `mamba3_mimo_builder.py`

| Component | Description |
|---|---|
| Embedding | `nn.Embedding(50304, 768)` — GPT-NeoX tokenizer |
| Concept Perceptron | Dual adaptive pooling (avg+max) → 16-token latent scratchpad |
| Thalamic Primer | `MambaLayer` zero-initialized and grafted before routing gate |
| Low-Rank Bridge | Bottleneck compression (768→64→768) into MIMO arms |
| MIMO Arms | 4× parallel `MambaLayer` blocks with orthogonal weight init |
| IPC Mixer | `Linear(768×4, 768×4)` cross-arm latent communication |
| Backbone | 24× sequential `MambaLayer` (d_state=16, d_conv=4, expand=2) |
| Output Head | `LayerNorm` → `Linear(768, 50304)`, weight-tied to embedding |

**Parameters:** ~150M  
**Tokenizer:** EleutherAI/gpt-neox-20b  
**SSM Core:** [mamba-ssm](https://github.com/state-spaces/mamba)

---

## Novel Mechanics

### 1. SSM Entropy Glass Ceiling

Converged Mamba SSMs collapse token probabilities to Dirac delta functions. Pre-backbone logit entropy is exactly `0.0` in float32, regardless of how much a grafted adapter's weights grow. Standard entropy-based MoE routing is fundamentally incompatible with heavily trained recurrent ODEs.

Standard LoRA/adapter grafting assumption — *"entropy will rise organically as adapter weights grow"* — does not hold for SSMs.

### 2. Primer Activation Magnitude Router

Replaced probability-space entropy with geometric routing:

```python
primer_delta = (primer_out - orig_embs).norm(dim=-1).mean()
gate_score = torch.sigmoid(primer_delta * 10.0 - 1.0)
```

Measures how much physical energy the Thalamic Primer expends warping the embedding, not how uncertain the model is about the next token. Immune to the glass ceiling because it operates in weight-space geometry, not softmax probability space.

### 3. Octopoda Trickle-Charge Patch

Prevents AdamW weight decay from eroding dormant MoE expert weights during low-entropy phases:

```python
gate_score = torch.clamp(raw_gate_score, min=0.05)
autotomic_gate = torch.clamp(raw_autotomic, min=0.05)
```

Maintains 5% basal gradient flow to all dormant arms. Eliminates the need for auxiliary load-balancing loss functions (z-loss, Switch Transformer load-balancing, Mixtral load-balancing).

### 4. Adiabatic Bloom

When all 4 MIMO arms activated simultaneously at Step 26,250, there was no catastrophic loss explosion (the standard "Non-Adiabatic" expert reactivation failure mode in dense MoE). Loss stayed in the `1.1–2.1` range. The Trickle-Charge preserved sufficient gradient warmth in dormant arms that they re-integrated smoothly into the residual stream.

### 5. Temporal Vector Gating (Phase 3j — Staged)

Upcoming upgrade replacing scalar gate with a 4D per-token router:

```python
self.domain_router = nn.Linear(d_model, 4)

route_logits = self.domain_router(primer_out.detach())   # (B, L, 4)
competitive_weights = F.softmax(route_logits, dim=-1)    # Competition
route_weights = torch.clamp(competitive_weights, min=0.05)
route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)
```

Enables each arm to specialize on different token spans within the autoregressive sequence rather than on different input domains.

---

## Training

### Phase 1 — Base Pre-training
- Dataset: OpenHermes-2.5 (streamed)
- Steps: ~43,000
- Checkpoint: `jarvis_v3.pth`

### Phase 2 — Supervised Fine-Tuning
- Dataset: GSM8K math reasoning
- Steps: ~3,600
- Goal: Hardwire `User: ... \nAssistant: <<answer=X>>` chat template
- Checkpoint: `jarvis_v3_sft.pth`

### Phase 3 — Jarvis v4 (Current)
- Start: `jarvis_v3_sft.pth`
- Dataloader: 70% OpenHermes-2.5 (live stream) + 30% Cognitive Cocktail (GSM8K, ARC, Premium Reasoning)
- Trainer: `jarvis_v4_trainer.py`
- Checkpoint: `jarvis_v4.pth` (Step 36,900+)

---

## Hardware

Developed on a single consumer GPU:
- GPU: NVIDIA (12GB VRAM)
- Power cap: 140W
- Sustained throughput: ~2,500–2,700 TPS
- VRAM usage: ~8.6–9.4 GB

---

## Files

| File | Description |
|---|---|
| `mamba3_mimo_builder.py` | Model architecture — `Mamba3MIMORLF` |
| `jarvis_v4_trainer.py` | Phase 3 training loop with dynamic LR and gradient clipping |
| `mamba3_sft_generator.py` | 70/30 Hermes+Cocktail dataloader |
| `benchmark.py` | Telemetry benchmark — Gate Score, Entropy, arm activation |
| `phase3j_vector_gating_patch.py` | Staged Phase 3j Vector Gating upgrade |
| `deepthink_full_report.txt` | Full architecture research log |
| `dashboard/` | Real-time training dashboard (HTML/JS) |

---

## Requirements

```
torch>=2.0
mamba-ssm
transformers
datasets
```

---

## Benchmark Results (Step 34,100 — Post-Bloom)

```
Gate Score:        1.0000 (All 4.00 parallel paths active)
Sequence Entropy:  56–62
Autotomic Gates:   [0.9932, 0.9932, 0.9932, 0.9932]
```

All 4 MIMO arms confirmed active. Post-Bloom domain specialization training active.

---

## License

MIT
