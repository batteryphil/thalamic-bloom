# Mamba3 Titan — Phase 2 Training Crisis: Full Incident Report

**Date:** 2026-05-22  
**Model:** Mamba3 Titan 2.54B (16-arm MIMO MoE)  
**Status at report time:** Phase 1 re-running cleanly (Step ~8,000+, Loss ~6.5, descending)

---

## Executive Summary

Phase 2 training (QA/instruction fine-tuning) failed repeatedly over multiple restarts due to a **chain of three compounding architectural flaws**. None was obvious in isolation, and each fix revealed the next layer underneath. This document records every mistake, root cause, and final resolution so future developers do not repeat this.

---

## The Architecture (Context)

```
Embedding → Thalamic Primer → Backbone (80 Mamba layers, split at layer 40)
  → Mid-backbone routing → LowRankBridge → 16 parallel MIMO Arms
  → [IPC Mixer OR Blackboard] → Route-weighted collapse → LM Head
```

- **Phase 1:** General language pre-training. Arms initialized orthogonally (diverse feature subspaces).
- **Phase 2:** QA fine-tuning. Arms cloned to be identical, trained together in unison.
- **Phase 3:** Router activates. Arms diverge into 16 domain specialists (MoE).

---

## Failure Chain

### Flaw 1 — The Autotomic Gate (Inverted Logic)

**What it was:** A gate designed to prune "dead" arms by penalizing low activation norms.  
**What it actually did:** The logic was inverted — it *penalized high-performing arms* and rewarded low-activation ones. This strangled healthy arms and caused `GNorm` explosions (600+).  
**Fix:** Completely removed the Autotomic Gate logic from `mamba3_titan_builder.py`.

---

### Flaw 2 — PAD Token Loss Pollution

**Symptom:** Intermittent LM Loss spikes of 150+ at random steps.  
**Root cause:** The dataloader was padding short sequences with `pad_token_id=1`, but the `CrossEntropyLoss` was computing loss over PAD tokens, creating massive spurious gradient spikes.  
**Fix:**
```python
# In dataloader:
labels[labels == tokenizer.pad_token_id] = -100  # or use ignore_index

# In loss:
criterion = nn.CrossEntropyLoss(ignore_index=1)
```

---

### Flaw 3 — The IPC Mixer Catastrophe (Root Cause of All Phase 2 Failures)

This is the critical one. **Everything else traces back here.**

#### What the IPC Mixer was

```python
self.ipc_mixer = nn.Linear(d_model * mimo_paths, d_model * mimo_paths)
# = nn.Linear(2048 * 16, 2048 * 16) = nn.Linear(32768, 32768)
# = ~1.07 BILLION parameters
```

It concatenated the outputs of all 16 arms into a single 32,768-dim vector, passed it through a massive dense layer, then split it back out. Designed as "inter-arm cross-talk."

#### Why it killed Phase 2 — every time

**Phase 1** trained the entire backbone — arms, LM head, everything — with the IPC mixer in the forward pass. The LM head learned to decode **IPC mixer features**, not raw arm features.

When Phase 2 started:
1. We cloned all 16 arms to be identical (correct design)
2. The IPC mixer now received 16 **identical** rows instead of 16 **orthogonal** ones
3. The IPC mixer's output distribution changed completely (it was trained for diverse inputs)
4. The LM head saw garbage features → loss exploded or plateaued permanently

**Every attempted fix failed for the same reason:**

| Attempt | What We Tried | Why It Failed |
|---------|--------------|---------------|
| Stochastic Routing | Route each batch to 1 random arm | Arms diverge → can't be clones |
| Remove IPC Mixer | Amputate the 1B-param layer | LM head sees raw arm features (completely wrong feature space) → loss 185+ |
| Keep IPC Mixer, gate=0 | Force residual to zero | `sigmoid(0) = 0.5` not 0.0, mixer still corrupts → loss 180+ |
| Reinitialize LM Head (N(0,0.02)) | Fresh head, learn new features | LR too low (3e-6) → never escapes `log(vocab)` baseline = 11.2 for 30,000 steps |
| Higher LR on LM Head (3e-5) | Force fast adaptation | Catastrophic interference with arm features → GNorm 400+, loss 185+ |

#### The actual root cause (confirmed by checkpoint inspection)

```python
# This was in Phase 1 __init__:
self.lm_head.weight = self.embedding.weight  # weight tying

# Confirmed in Phase 1 checkpoint:
# lm_head.weight == embedding.weight: True
# Arm 0 vs Arm 1 cosine similarity: 0.006858 (near-perfectly orthogonal)
```

The lm_head was weight-tied to the embeddings. When Phase 2 tried to adapt the LM head via high LR, the massive gradients **simultaneously destroyed the embedding table**, breaking the model's ability to read words at all.

---

## The Correct Fix

**Start Phase 1 fresh, without the IPC Mixer, using the Blackboard instead.**

The IPC Mixer cannot be removed mid-training. The only clean solution is to never use it at all, training the entire model from scratch with a coherent architecture from step 1.

### Architecture: Blackboard vs IPC Mixer

| | IPC Mixer | Sparse IPC Blackboard |
|--|-----------|----------------------|
| Parameters | 1.07 Billion | ~200K (64-dim bus) |
| Type | Dense linear | Sparse weighted bus |
| MoE-compatible | No (mixes all experts) | Yes (gated per arm) |
| Removable mid-training | No | N/A — always present |

### Phase Gating (Final Design)

```python
# mamba3_titan_builder.py — set_phase()
bb_active = phase in ('1', '2', '3j', 'sft')
# Phase 3 = SILENT — arms must diverge freely without cross-talk
```

| Phase | Blackboard | Reason |
|-------|-----------|--------|
| 1 | ✅ Active | Arms learn shared coordination vocabulary |
| 2 | ✅ Active | Cloned arms fine-tune QA through same bus |
| 3 | ❌ Silent | Arms diverge into specialists without interference |
| 3j | ✅ Active | Specialists share synthesis context |

### Weight Tying Fix

```python
# After loading Phase 1 checkpoint for Phase 2 cold-start:
with torch.no_grad():
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
# This breaks the tie so LM head gradients cannot corrupt the embedding table
```

---

## Lessons Learned (For Future Developers)

### 1. You cannot remove major layers mid-training
If a layer (like IPC Mixer) is present during Phase 1, the entire rest of the network adapts to its output distribution. Removing it in Phase 2 changes the feature space the LM head reads, causing instant failure. **Design your final architecture first, then train.**

### 2. Weight tying is dangerous in multi-phase training
`lm_head.weight = self.embedding.weight` is common in single-phase LLMs. In multi-phase training, the high LR needed to adapt the LM head will cascade backward through the tie and destroy the embedding table. Always break the tie explicitly at phase transitions:
```python
model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
```

### 3. `sigmoid(0) = 0.5`, not 0.0
If you zero-initialize a gate parameter and compute `torch.sigmoid(gate)` expecting a "silent" multiplier, you get 0.5, not 0.0. Use:
```python
gate_value = 0.0  # explicit zero, not sigmoid(param)
```
Or use a learned gate that starts at `-inf` (maps to 0 under sigmoid):
```python
self.gate = nn.Parameter(torch.tensor(-10.0))  # sigmoid(-10) ≈ 0.000045
```

### 4. Always inspect your checkpoint before Phase 2
```python
ckpt = torch.load('phase_1.pt', map_location='cpu', weights_only=True)
# Check what's actually in there before assuming anything
for k in ckpt['model']:
    print(k, ckpt['model'][k].shape)
# Verify: are weights tied? Are unexpected layers present?
```

### 5. log(vocab_size) is the minimum loss floor for a random LM head
For vocab_size=50304: `ln(50304) ≈ 10.82`. If your Phase 2 loss plateaus at exactly ~11.0 from step 1 to step 30,000, the LM head was reinitialized randomly but the LR was too low to learn. The model is outputting uniform probability — it learned nothing.

### 6. Arm collapse metric = 0.5 is healthy in Phase 1
The formula `(cosine_similarity + 1) / 2` maps:
- `1.0` = identical clones
- `0.5` = orthogonal (perpendicular, no correlation) ← **correct for Phase 1**
- `0.0` = perfectly opposite

Orthogonal arm init (`initialize_asymmetric_arms()`) intentionally produces 0.5. This is correct and desired. If you see the metric "drop to 0.5 every 10 steps" it is because the metric is only computed every 10 steps and shows a default of 1.0 otherwise — not a real fluctuation.

### 7. A 1B-parameter dense layer is not "cross-talk" — it's a bottleneck
The IPC Mixer (`32768×32768`) is not biologically inspired inter-arm communication. It is a massive weight matrix that every gradient must flow through. Use the Sparse Blackboard instead: a 64-dim bus where arms selectively write and read, creating genuine sparse MoE-compatible communication.

---

## Current State (as of report)

- **Phase 1 re-running from step 0** with clean architecture (no IPC Mixer, Blackboard active)
- **Step ~8,000, Loss ~6.5, descending cleanly**
- **GNorm 3–9**, no explosions
- **ETA to Phase 1 complete:** ~5 hours from midnight UTC

When Phase 1 finishes:
- Phase 2 will clone arms, break the weight tie, and fine-tune with the **same** Blackboard pipeline the LM head was trained on
- No calibration shock expected
- Loss should adapt from Phase 1 baseline within the first few hundred steps

---

*Report compiled by Antigravity (Google DeepMind). Training system: RTX 3060 12GB, Linux Mint 22.3, PyTorch 2.x + bitsandbytes PagedAdam8bit.*
