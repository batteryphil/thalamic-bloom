# Phase 4: IPC Mixer Calibration (Solving Word Salad)

With Phase 3 complete, we have successfully created a decentralized, structurally orthogonal MIMO core. The arms are specialized (e.g., Code vs Math), but the Latent IPC Mixer is currently taking these vastly different, anti-correlated signals and forcefully blending them together. 

Because the mixer hasn't learned *how* to translate these new orthogonal tensors back into the model's original language space, it produces chaotic super-positions, leading to the "Word Salad" we observed in the benchmarks.

Phase 4 is dedicated entirely to calibrating this translation layer. We will freeze the specialized arms to protect their new domains, freeze the core backbone to protect language fundamentals, and train only the bridging and mixing layers.

## The Strategy: Targeted Freezing

We will create a new dedicated training script: `jarvis_phase4_trainer.py`. This script will load the `47750` checkpoint and apply a strict gradient freeze.

### 1. Freeze the Cognitive Backbone (Protect Language)
We will freeze the `embedding`, `layers` (the 24 Mamba layers), `lm_head`, and the `ConceptPerceptron`. This ensures the model does not suffer catastrophic forgetting of basic English grammar and syntax.

### 2. Freeze the MIMO Arms (Protect Specialization)
We will freeze the `mimo_reasoning_blocks`. This ensures the arms remain mathematically orthogonal and do not collapse back into clones during the translation training.

### 3. Unfreeze the Communication Bridge (Calibrate the Mixer)
We will enable gradients **only** for:
- `ipc_mixer`: Must learn to blend the anti-correlated states into a unified vector compatible with the frozen language backbone.
- `domain_router` & `thalamic_primer`: Must remain unfrozen to continue adapting the routing weights as the mixer learns.
- `bridge`: The entrance into the MIMO loop.

By restricting gradients entirely to these layers, the optimizer is physically forced to use the `ipc_mixer` as a translation matrix to convert the specialized logic back into fluent language.

## Execution Steps
1. Copy `jarvis_v4_trainer.py` to `jarvis_phase4_trainer.py`.
2. Inject the parameter freezing logic before the optimizer instantiation.
3. Update the training loop to track `ipc_calibration_loss` rather than the `collapse_metric` (which is now fixed).
4. Run the phase 4 trainer on the 70/30 Hermes-Cocktail dataset until the loss converges and language coherence stabilizes.

## User Review Required
> [!IMPORTANT]
> This targeted freezing prevents catastrophic forgetting while forcing the mixer to learn the translation matrix. Do I have your approval to implement this plan, create the new trainer, and start Phase 4?
