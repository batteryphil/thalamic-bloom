# Mamba3 Titan 2.5B

The Mamba3 Titan is a highly optimized 2.54 Billion parameter language model featuring a state-space Mamba3 backbone and 16 parallel Multiple-Input Multiple-Output (MIMO) Reasoning Arms. 

This repository contains the architecture, the multi-phase training curriculum, and the "Glass Brain" telemetry monitor used to track the real-time orthogonal divergence of the reasoning arms.

For a deeply technical breakdown of the mathematical challenges faced during training (including VRAM OOM optimizations, Soft-MoE collapse, and IPC isolation), please read the [Engineering Report](./MAMBA3_TITAN_2.5B_REPORT.md).

## 🚀 Environment Setup

This project was engineered to train on a consumer-grade RTX 3060 (12GB VRAM). 

### Dependencies
Ensure you have PyTorch 2.0+ installed with CUDA support.

```bash
pip install torch transformers datasets huggingface_hub
```

### Memory Optimization (Critical)
To train a 2.54B parameter model in 12GB of VRAM, you **must** use the `PagedAdam8bit` optimizer. This offloads the 6.4GB of optimizer momentum states into your host CPU RAM and pages them to the GPU only during the gradient update step.

```bash
pip install bitsandbytes
```

### Authentication
The dataset streaming pipeline requires a Hugging Face token.
```bash
export HF_TOKEN="your_huggingface_token"
```

---

## 🧠 Training Curriculum

The model utilizes a 4-phase curriculum to slowly build capabilities before specializing the reasoning arms:

*   **Phase 1 (`--phase 1`)**: Dense Ensemble. Pre-trains the core backbone on `fineweb-edu` and stripped `OpenHermes` chat logs. All 16 arms receive equal routing weight.
*   **Phase 2 (`--phase 2`)**: Domain Tuning. Injects specialized data (`MetaMathQA`, `CodeAlpaca`, `Wikipedia`) while keeping arms equally weighted.
*   **Phase 3 (`--phase 3`)**: Cognitive Bloom (Soft-MoE). The Domain Router activates. The IPC Blackboard is strictly isolated to prevent cross-talk cheating. The arms are mathematically forced to orthogonalize.
*   **Phase 3j (`--phase 3j`)**: Join/Alignment. The arms are frozen, the Corpus Callosum Blackboard is activated, and the model learns to synthesize the specialized domains.

### Launching the Trainer

To start training, simply run the master trainer script and provide the target phase. The script automatically handles gradient checkpointing, telemetry logging, and `PagedAdam8bit` state sidecars.

```bash
python src/master_titan_trainer.py --phase 3
```

---

## 📊 Glass Brain Monitor

The project includes a real-time HTML/JS dashboard that visually tracks the model's telemetry, including the routing weights of the 16 arms, the domain loss, the language modeling loss, and the cosine-similarity divergence metrics.

1. Ensure the trainer is running (it writes to `monitor_ui/telemetry.json` every 10 steps).
2. Open `monitor_ui/index.html` in any modern web browser.

---

## 📂 File Structure

*   **`src/`**
    *   `mamba3_titan_builder.py` — The core model architecture (Mamba3 Backbone, Router, MIMO Arms, IPC Mixer).
    *   `master_titan_trainer.py` — The training loop, dataset streamer, and checkpoint manager.
    *   `config.json` — Hyperparameter configurations.
*   **`monitor_ui/`** — The frontend dashboard files (`index.html`, `app.js`, `style.css`).
*   **`logs/`** — Historical log excerpts demonstrating systemic bottlenecks and their resolutions.
*   **`MAMBA3_TITAN_2.5B_REPORT.md`** — The comprehensive story of the project's engineering process.
