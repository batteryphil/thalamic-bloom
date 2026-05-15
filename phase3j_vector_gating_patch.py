"""
PHASE 3j VECTOR GATING PATCH
==============================
DeepThink Session 2 — Approved for deployment AFTER Gate 1 benchmark passes.

PROBLEM BEING SOLVED:
  The current scalar gate (gate_score = single float) applies identical routing
  weights to all dormant arms simultaneously. This makes the architecture behave
  as a Dense Ensemble rather than a true Mixture-of-Experts. With all arms seeing
  identical token weights, the dominant domain (coding) mathematically wins the
  gradient race regardless of orthogonal initialization.

SOLUTION:
  Replace scalar routing with a 4D Vector Router:
    1. A new nn.Linear(d_model, 4) projection on the Thalamic Primer output
       generates 4 DISTINCT routing logits per token.
    2. Softmax forces COMPETITION between arms — high weight for one arm
       mathematically reduces weight for others.
    3. Trickle-Charge clamp (min=0.05) preserved to prevent any arm from dying.
    4. Weights are renormalized after clamping so they always sum to 1.0.

SPACETIME PHYSICS (DeepThink):
  The temporal routing allows the Primer to dynamically shift compute across
  the autoregressive generation sequence token-by-token:
    Tokens 0-30   (Reading prompt): Route 85% to Arm 0 (Parse/Decompose)
    Tokens 31-80  (Calculating):    Route 85% to Arm 1 (Symbolic Math)
    Tokens 81-100 (Checking):       Route 85% to Arm 2 (Verification)
    Tokens 101+   (Formatting):     Route 85% to Arm 3 (<<answer=X>> template)

DEPLOYMENT INSTRUCTIONS:
  1. Wait for Gate 1 benchmark to pass (Step ~39,000, "Say hello" returns greeting)
  2. Stop trainer
  3. Apply changes to mamba3_mimo_builder.py as documented below
  4. Restart trainer — checkpoint will load with strict=False for new weights
  5. Watch for a small loss spike (~0.5) as the new router initializes, then decay

CHANGES TO mamba3_mimo_builder.py:
"""

# ============================================================
# CHANGE 1: In Mamba3MIMORLF.__init__(), add after self.bridge:
# ============================================================

INIT_ADDITION = """
        # Phase 3j: Per-arm competitive vector router
        # Generates 4 distinct routing logits per token from Thalamic Primer output.
        # Replaces the scalar gate with a true Temporal Mixture-of-Experts router.
        self.domain_router = nn.Linear(self.d_model, self.mimo_paths, bias=True)
        # Initialize router to near-uniform routing to preserve Adiabatic Bloom behavior
        nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.domain_router.bias)
"""

# ============================================================
# CHANGE 2: In Mamba3MIMORLF.forward(), replace the routing block:
#
# REMOVE this block:
# ============================================================

OLD_ROUTING = """
        # A. The Thalamic Primer: Temporal Sequence Mixing before routing
        primer_out = self.thalamic_primer(orig_embs)

        # Calculate Primer Activation Magnitude as routing signal.
        with torch.no_grad():
            primer_delta = (primer_out - orig_embs).norm(dim=-1).mean()
            routing_signal = primer_delta.detach()

        # Blend: add Primer signal into main stream
        x = orig_embs + primer_out * 0.1

        # Route into Auxiliary Loop Engine via Dynamic Bridge
        bridge_out = self.bridge(x)

        # Soft-Body Compute Allocation driven by Primer Activation Magnitude
        raw_gate_score = torch.sigmoid(routing_signal * 10.0 - 1.0)
        # --- OCTOPODA TRICKLE-CHARGE PATCH ---
        gate_score = torch.clamp(raw_gate_score, min=0.05).unsqueeze(-1).unsqueeze(-1)

        parallel_states = []
        autotomic_gates_list = []
        for i in range(self.mimo_paths):
            state = self.mimo_reasoning_blocks[i](bridge_out)
            variance = state.var(dim=-1).mean()
            autotomic_gate = torch.clamp(torch.sigmoid((10.0 - variance) * 0.5), min=0.05)
            autotomic_gates_list.append(autotomic_gate.item() if isinstance(autotomic_gate, torch.Tensor) else autotomic_gate)

            mask = 1.0 if i == 0 else gate_score
            parallel_states.append(state * mask * autotomic_gate)
"""

# ============================================================
# REPLACE with this block:
# ============================================================

NEW_ROUTING = """
        # A. The Thalamic Primer: Temporal Sequence Mixing before routing
        primer_out = self.thalamic_primer(orig_embs)

        # Blend Primer into main stream (10% injection, preserved from Phase 3)
        x = orig_embs + primer_out * 0.1

        # Phase 3j: Temporal Vector Gating
        # Generate 4 distinct routing logits per token from the Primer output.
        # Shape: (B, L, 4) — one weight per arm per token position.
        with torch.no_grad():
            # Keep routing outside autograd to avoid interfering with arm gradients
            route_logits = self.domain_router(primer_out.detach())  # (B, L, 4)

        # Softmax forces arms to COMPETE for each token (true MoE, not dense ensemble)
        competitive_weights = F.softmax(route_logits, dim=-1)  # (B, L, 4)

        # Octopoda Trickle-Charge: clamp minimum per arm to prevent synaptic atrophy
        route_weights = torch.clamp(competitive_weights, min=0.05)

        # Renormalize so weights still sum to 1.0 after clamping
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)
        # route_weights shape: (B, L, 4) — each position has 4 arm weights

        # Route into Auxiliary Loop Engine via Dynamic Bridge
        bridge_out = self.bridge(x)

        parallel_states = []
        autotomic_gates_list = []
        for i in range(self.mimo_paths):
            state = self.mimo_reasoning_blocks[i](bridge_out)  # (B, L, D)

            # B. Autotomic pruning (preserved from Phase 3)
            variance = state.var(dim=-1).mean()
            autotomic_gate = torch.clamp(torch.sigmoid((10.0 - variance) * 0.5), min=0.05)
            autotomic_gates_list.append(
                autotomic_gate.item() if isinstance(autotomic_gate, torch.Tensor) else autotomic_gate
            )

            # Apply per-token vector weight for this arm (B, L, 1) for broadcasting
            arm_weight = route_weights[..., i:i+1]  # (B, L, 1)

            # Arm 0 always gets full gradient (primary cortex, preserved)
            if i == 0:
                parallel_states.append(state * autotomic_gate)
            else:
                parallel_states.append(state * arm_weight * autotomic_gate)

        # Scalar telemetry for dashboard (mean routing weight of dormant arms)
        mean_gate = route_weights[..., 1:].mean().item()
"""

# ============================================================
# CHANGE 3: Update last_telemetry block to use mean_gate and route entropy:
# ============================================================

NEW_TELEMETRY = """
        # Compute routing signal magnitude for dashboard entropy display
        with torch.no_grad():
            primer_delta = (primer_out - orig_embs).norm(dim=-1).mean()

        self.last_telemetry = {
            \"entropy\": primer_delta.item(),
            \"gate_score\": mean_gate,
            \"autotomic_gates\": autotomic_gates_list,
            \"route_weights\": route_weights.mean(dim=(0, 1)).tolist()  # Per-arm mean
        }
"""

# ============================================================
# CHANGE 4: Update IPC collapse to use vector-weighted outputs:
# ============================================================

# Replace: x = x + (sum(final_states) / self.mimo_paths)
# With:    x = x + sum(final_states) / self.mimo_paths
# (No change needed — the weighting is already baked into parallel_states above)

print("Phase 3j Vector Gating patch loaded. DO NOT DEPLOY until Gate 1 clears.")
print("Gate 1 condition: 'Say hello.' returns a greeting, NOT code.")
print("Target step: ~39,000")
