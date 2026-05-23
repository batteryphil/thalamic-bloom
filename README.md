# Thalamic Bloom — Mamba3 Titan 2.54B

A 2.54 billion parameter Mixture-of-Experts language model built on Mamba SSM, featuring 16 parallel MIMO reasoning arms, a sparse IPC Blackboard for inter-arm coordination, and a multi-phase training curriculum.

## Architecture

```
Embedding → Thalamic Primer → Backbone (80 Mamba layers)
  → Mid-backbone routing (layer 40) → LowRankBridge
  → 16 parallel MIMO Arms
  → Sparse IPC Blackboard (64-dim bus)
  → Route-weighted collapse → LM Head
```

### Key Components

| Component | Role |
|-----------|------|
| **Thalamic Primer** | Lightweight Mamba layer that pre-attunes the signal before the backbone |
| **16 MIMO Arms** | Parallel Mamba layers — clones in Phase 2, diverging specialists in Phase 3 |
| **Sparse IPC Blackboard** | 64-dim sparse bus for inter-arm coordination (active in Phase 1, 2, 3j) |
| **Domain Router** | Soft-MoE router reading from mid-backbone hidden state (active in Phase 3+) |
| **ConceptPerceptron** | Global context pooling injected every 6 backbone layers |

### Phase Gating

| Phase | Description | Router | Blackboard | Arms |
|-------|-------------|--------|-----------|------|
| **1** | General language pre-training | Uniform (1/16 each) | ✅ Active | Asymmetric (orthogonal init) |
| **2** | QA fine-tuning | Uniform (1/16 each) | ✅ Active | Clones (Arm 0 copied to 1–15) |
| **3** | Specialist divergence | Soft-MoE | ❌ Silent | Diverging |
| **3j** | Cross-specialist synthesis | Soft-MoE | ✅ Active | Frozen |

## Training

```bash
# Phase 1 — General language (FineWeb-Edu + OpenHermes)
python master_titan_trainer.py --phase 1

# Phase 2 — QA fine-tuning (7-dataset Enriched Cognitive Bloom mix)
python master_titan_trainer.py --phase 2

# Phase 3 — Specialist divergence
python master_titan_trainer.py --phase 3
```

## ⚠️ Read This Before Training

**See [PHASE2_INCIDENT_REPORT.md](PHASE2_INCIDENT_REPORT.md) for a full account of the architectural pitfalls we hit during development.** Key warnings:

1. **Never use a massive dense IPC mixer** (`nn.Linear(d_model*16, d_model*16)`). It creates a 1B-parameter bottleneck that the entire network adapts to — you cannot remove it later without retraining from scratch. Use the Sparse Blackboard instead.

2. **Break weight tying at phase transitions.** If Phase 1 uses `lm_head.weight = embedding.weight`, Phase 2's high LR will corrupt the embedding table through the tie. Always `clone()` to break it.

3. **`sigmoid(0) = 0.5`, not 0.0.** If you zero-init a gate parameter, using it as `sigmoid(gate)` gives you 0.5 — not silence. Use explicit `0.0` or init at `torch.tensor(-10.0)`.

4. **Arm collapse metric of 0.5 is correct in Phase 1.** The formula `(cosine_sim + 1) / 2` maps orthogonal arms to 0.5. Orthogonal init is the goal. This is not collapse.

## Requirements

```
torch >= 2.0
mamba-ssm
bitsandbytes
transformers
datasets
```

## Hardware

Developed and trained on a single RTX 3060 12GB with bitsandbytes PagedAdam8bit optimizer (optimizer moments in CPU RAM to free VRAM).
