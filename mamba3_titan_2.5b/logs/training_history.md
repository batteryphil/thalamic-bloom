# Training Log History & Diagnostics

This document compiles the critical log excerpts that defined the engineering challenges and breakthroughs of the Mamba3 Titan 2.54B project.

## 1. VRAM Memory Exhaustion (Adam8bit Optimizer)

When attempting to restore the Adam8bit optimizer state from a Phase 2 checkpoint, the GPU immediately crashed. The 8-bit momentum tensors for 2.54B parameters required 6.4GB of VRAM on top of the 3.19GB model and forward activations, instantly exceeding the RTX 3060's 12GB capacity.

```text
[CKPT] Resuming from step 7000
[CKPT] Could not restore optimizer state (CUDA out of memory. Tried to allocate 1024.00 MiB. 
GPU 0 has a total capacity of 11.62 GiB of which 279.12 MiB is free. Including non-PyTorch memory, 
this process has 10.86 GiB memory in use.
```

**Resolution:** We switched the optimizer to `bitsandbytes.optim.PagedAdam8bit` to allow the momentum states to be paged into host CPU RAM, dropping active VRAM to ~3.5GB. We also isolated the optimizer checkpointing into a separate `*_optim.pt` sidecar loaded with `map_location='cpu'`.

---

## 2. Telemetry Blindness (LayerNorm Masking)

Our initial attempt at the "Glass Brain" monitor captured the standard deviation (`std`) of activation states across the 16 MIMO arms. However, the `LayerNorm` at the end of each reasoning block normalized all activation variances to ~1.0, rendering the telemetry blind.

```text
step=7057  LR=1.18e-06  arm_collapse_max=1.0000
  🔴 Anchor   1.0000  ████████████████████
  🔴 Logic    1.0000  ████████████████████
  🔴 Recall   1.0000  ████████████████████
  🔴 Code     1.0000  ████████████████████
```

**Resolution:** We abandoned activation snapshots entirely and implemented a weight-based similarity metric. We extracted the flattened vectors of the initial `ssm.proj` weights from each arm and computed pairwise cosine similarity, scaling it from `[-1, 1]` to a `[0, 1]` collapse score.

---

## 3. The Corpus Callosum "Cheating" Spike

Even with Switch Transformer MoE loss added, the router refused to balance. We realized the arms were communicating through the active `bb_read`/`bb_write` Blackboard bus (Corpus Callosum) and the dense `ipc_mixer`. By sharing features, they could remain mathematically identical and satisfy the routing loss without specializing. 

When we disabled the Blackboard to computationally isolate the arms, the Language Modeling loss violently spiked from `~0.6` to `~135.0` instantly.

```text
Phase 3 | Step 09070 | LM Loss: 136.3523 | Dom Loss: 1.9043 | Gate: 0.0625 | Entropy: 0.2939 | GNorm: 109.00
Phase 3 | Step 09071 | LM Loss: 135.2376 | Dom Loss: 2.0790 | Gate: 0.0625 | Entropy: 0.2939 | GNorm: 131.00
Phase 3 | Step 09072 | LM Loss: 131.9168 | Dom Loss: 0.9811 | Gate: 0.0625 | Entropy: 0.2939 | GNorm: 66.50
```

**Resolution:** This spike proved they were cheating. By maintaining strict isolation during Phase 3, the massive gradient spikes forced the router to abandon the converged arms and push traffic into the idle arms.

---

## 4. Routing Divergence Achieved

After combining the `PagedAdam8bit` optimizer, the Switch Transformer quadratic load-balancing loss ($N \sum \mu_i^2 - 1.0$), and strict IPC isolation, the router finally began distributing traffic. The active arms count spread from 1 to 9 within hundreds of steps, and the domain balancing loss dropped exponentially.

```text
step=9404  LR=4.01e-06  DomLoss=0.2662
Active arms: 9/16  
Routing weights distribution:
  ⚫ Anchor   0.0000  
  ⚫ Logic    0.0000  
  🟢 Recall   0.2782  ███████████
  🟡 Code     0.0288  █
  🟢 Lang     0.2188  ████████
  🟢 Math     0.1013  ████
  🟡 Chat     0.0285  █
  ⚫ Fact     0.0000  
  ⚫ Reason   0.0000  
  🟢 Syn      0.1317  █████
  ⚫ Sem      0.0000  
  🟢 Ctx      0.1090  ████
  🟡 Plan     0.0223  
  ⚫ Eval     0.0000  
  🟢 Gen      0.0807  ███
  ⚫ Aux      0.0007  
```
