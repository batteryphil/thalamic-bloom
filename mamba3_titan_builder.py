import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, List

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# ─────────────────────────────────────────────────────────────────────────────
# ARM IDENTITY TABLE
# 16 named specializations — arms self-organize toward these roles via
# Phase 3j multi-domain training. Labels are for telemetry/UI only.
# ─────────────────────────────────────────────────────────────────────────────
ARM_IDENTITIES = [
    "General Language",      # 0  — always-on anchor arm
    "Symbolic Math",         # 1  — GSM8K / arithmetic
    "Logical Reasoning",     # 2  — ARC / deductive chains
    "Code Syntax",           # 3  — CodeAlpaca / programming
    "Factual Recall",        # 4  — encyclopedic knowledge
    "Summarization",         # 5  — CNN/DailyMail / compression
    "Creative Writing",      # 6  — narrative / prose
    "Instruction Following", # 7  — chat / command execution
    "Analogical Reasoning",  # 8  — metaphor / pattern matching
    "Causal Inference",      # 9  — cause-effect chains
    "Spatial Reasoning",     # 10 — geometry / layout
    "Temporal Reasoning",    # 11 — time / sequence ordering
    "Ethical Judgment",      # 12 — OO policy / harm assessment
    "Multilingual Bridge",   # 13 — cross-lingual transfer
    "Meta-Cognition",        # 14 — self-reference / uncertainty
    "Synthesis",             # 15 — multi-arm integration hub
]


class DummyMambaSSM(nn.Module):
    """Placeholder for the core Mamba state-space scan."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConceptPerceptron(nn.Module):
    """Global context pooling — compresses input into a latent scratchpad."""
    def __init__(self, d_model: int, num_tokens: int = 16, chunk_size: int = 1024) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_size = chunk_size
        self.avg_pooling = nn.AdaptiveAvgPool1d(num_tokens)
        self.max_pooling = nn.AdaptiveAvgPool1d(num_tokens)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        chunks: List[torch.Tensor] = []
        for i in range(0, L, self.chunk_size):
            chunk = x[:, i:i + self.chunk_size, :]
            chunk_t = chunk.transpose(1, 2)
            avg_p = self.avg_pooling(chunk_t).transpose(1, 2)
            max_p = self.max_pooling(chunk_t).transpose(1, 2)
            chunks.append(torch.cat([avg_p, max_p], dim=-1))
        aggregated = torch.stack(chunks, dim=0).mean(dim=0)
        return F.silu(self.proj(aggregated))


class LowRankBridge(nn.Module):
    """Bottleneck bridge before MIMO arms — compresses backbone output."""
    def __init__(self, d_model: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up   = nn.Linear(bottleneck, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


class MambaLayer(nn.Module):
    """Mamba SSM layer with pre-norm and residual."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.ssm = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.ssm = DummyMambaSSM(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        device_type = x.device.type if x.device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            x_ssm = self.ssm(self.norm(x))
        return x_ssm.to(x.dtype) + residual


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
class Mamba3Titan(nn.Module):
    """
    Mamba 3 Titan — 2.54B parameter MoE reasoning engine.
    16 parallel MIMO arms with soft routing, IPC cross-talk, and arm telemetry.

    Architecture flow:
      Embedding → Thalamic Primer → [first N/2 backbone layers]
        → Mid-backbone routing (entropy computed HERE — fixes Glass Ceiling)
        → LowRankBridge → 16 parallel MIMO arms
        → IPC mixer (arms share with each other)
        → [remaining N/2 backbone layers] + ConceptPerceptron injection
        → LM head
    """
    def __init__(self,
                 vocab_size: int = 50304,
                 d_model: int = 2048,
                 n_layers: int = 80,
                 mimo_paths: int = 16,
                 use_gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.mimo_paths  = mimo_paths
        self.n_layers    = n_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.active_phase = '1'

        # ── Input ────────────────────────────────────────────────────────────
        self.embedding      = nn.Embedding(vocab_size, d_model)
        self.cp             = ConceptPerceptron(d_model)
        self.thalamic_primer = MambaLayer(d_model)
        self.bridge         = LowRankBridge(d_model)

        # ── Backbone (80 layers, split at midpoint for routing) ──────────────
        self.layers = nn.ModuleList([MambaLayer(d_model) for _ in range(n_layers)])
        self._mid   = n_layers // 2   # layer 40 — where routing is computed

        # ── MIMO Arms (16) ───────────────────────────────────────────────────
        self.mimo_reasoning_blocks = nn.ModuleList(
            [MambaLayer(d_model) for _ in range(mimo_paths)]
        )

        # ── Routing ──────────────────────────────────────────────────────────
        # FIX: domain_router now reads from mid-backbone hidden state, not raw embeddings.
        # This is the Entropy Glass Ceiling fix from deepthink_full_report.txt.
        self.domain_router    = nn.Linear(d_model, mimo_paths, bias=True)
        self.router_temp      = nn.Parameter(torch.ones(1) * 1.0)  # learnable temperature
        nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.domain_router.bias)

        # ipc_mixer permanently removed — replaced by Sparse IPC Blackboard below.

        # ── Output ───────────────────────────────────────────────────────────
        self.cp_gate  = nn.Parameter(torch.tensor(0.01))
        self.norm_f   = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        # Weights untied to prevent Calibration Shock from cascading into embeddings

        # ── Telemetry ────────────────────────────────────────────────────────
        self.last_telemetry: dict = {
            'gate_score':    0.0,
            'entropy':       0.0,
            'arm_weights':   [1.0 / mimo_paths] * mimo_paths,
            'arm_labels':    ARM_IDENTITIES,
            'top_arms':      [],
            'arm_collapse_metric': 0.0,
            'latent_energy':       0.0,
        }

        # ── PATCH 2A: Sparse IPC Blackboard (Corpus Callosum) ────────────────
        # 64-dim bottleneck for inter-expert communication on the current token.
        # Complements the Latent Scratchpad (Hippocampus / ConceptPerceptron)
        # which handles temporal/past-context memory.
        self.bus_dim  = 64
        self.bb_write = nn.Linear(d_model, self.bus_dim, bias=False)
        self.bb_read  = nn.Linear(self.bus_dim, d_model, bias=False)
        # bb_read zero-init: Blackboard starts silent, learns gradually from Phase 1 step 1.
        # No loss spikes because bb_read output starts at zero (additive residual).
        nn.init.zeros_(self.bb_read.weight)

        # Zero-init thalamic primer output — identity pass-through at init
        if hasattr(self.thalamic_primer.ssm, 'out_proj'):
            nn.init.zeros_(self.thalamic_primer.ssm.out_proj.weight)
        elif hasattr(self.thalamic_primer.ssm, 'proj'):
            nn.init.zeros_(self.thalamic_primer.ssm.proj.weight)

    # ── Phase control ─────────────────────────────────────────────────────────
    def set_phase(self, phase: str) -> None:
        assert phase in ['1', '2', '3', '3j', 'sft'], f"Invalid phase: {phase}"
        self.active_phase = phase
        # Blackboard active in Phase 1 & 2 (arms learn to coordinate as clones).
        # SILENT in Phase 3 — arms must diverge without cross-talk to specialize.
        # Re-enabled in Phase 3j/sft for cross-arm synthesis after specialization.
        bb_active = phase in ('1', '2', '3j', 'sft')
        for p in list(self.bb_write.parameters()) + list(self.bb_read.parameters()):
            p.requires_grad_(bb_active)
                
        # ── Asymmetric Freeze (Phase 3j) ─────────────────────────────────────
        # Arms train normally in Phases 1, 2, 3. In Phase 3j, they are frozen.
        arms_active = phase in ('1', '2', '3')
        for p in self.mimo_reasoning_blocks.parameters():
            p.requires_grad_(arms_active)
            
        print(f"Titan Architecture → Phase {phase}  (Blackboard {'ACTIVE' if bb_active else 'FROZEN/SILENT'} | Arms {'TRAINING' if arms_active else 'FROZEN'})")

    # ── Asymmetric arm init ───────────────────────────────────────────────────
    def initialize_asymmetric_arms(self) -> None:
        """Orthogonal init so arms start with different feature subspaces."""
        for name, param in self.mimo_reasoning_blocks.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif param.dim() == 1:
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * 0.05)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self,
                input_ids:  torch.Tensor,
                loop_idx:   int = 0,
                domain_ids: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids:  (B, L) token ids
            loop_idx:   recursive refinement depth (0 = first pass)
            domain_ids: (B,) integer domain labels — only used in Phase 3j

        Returns:
            logits:      (B, L, vocab_size)
            domain_loss: scalar tensor
        """
        B, L = input_ids.shape
        device = input_ids.device

        # ── Embedding + Thalamic Primer ───────────────────────────────────────
        orig_embs   = self.embedding(input_ids)
        primer_out  = self.thalamic_primer(orig_embs)

        if self.active_phase in ('3', '3j', 'sft'):
            x = orig_embs + primer_out * 0.1
        else:
            x = orig_embs   # Phases 1 & 2: primer effectively zero (zero-init)

        # ConceptPerceptron scratchpad (global context anchor)
        cp_scratchpad = self.cp(x)

        # ── First half of backbone (layers 0 … mid-1) ────────────────────────
        for i, layer in enumerate(self.layers[:self._mid]):
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            # ConceptPerceptron injection every 6 layers
            if (i + 1) % 6 == 0:
                global_ctx = cp_scratchpad.mean(dim=1, keepdim=True).clone()
                x = x + (self.cp_gate * torch.tanh(global_ctx))

        # ── MID-BACKBONE ROUTING (Entropy Glass Ceiling Fix) ─────────────────
        # Routing computed from layer-40 hidden state, NOT raw embeddings.
        # This gives the router real semantic signal instead of always-zero entropy.
        mid_hidden = x  # (B, L, D) — rich semantic representation

        if self.active_phase in ('1', '2'):
            # Dense ensemble: all arms equally weighted
            route_weights = torch.ones(B, L, self.mimo_paths, device=device) / self.mimo_paths
            domain_loss   = torch.tensor(0.0, dtype=x.dtype, device=device)

        elif self.active_phase == '3':
            # Cognitive Bloom: soft routing with Switch Transformer Load-Balancing & Routing Noise.
            with torch.no_grad() if not self.training else torch.enable_grad():
                primer_delta = (mid_hidden - orig_embs).norm(dim=-1).mean()

            route_logits  = self.domain_router(mid_hidden)           # [B, L, 16]
            
            # Inject routing noise during training to force exploration of dead pathways
            if self.training:
                noise = torch.randn_like(route_logits) * 0.1
                route_logits = route_logits + noise

            temp          = torch.clamp(self.router_temp, min=0.1, max=10.0)
            route_weights = F.softmax(route_logits / temp, dim=-1)   # [B, L, 16]

            # Switch Transformer load-balancing loss: N * sum(mu_i^2) - 1.0
            # Minimized at 0.0 (perfect balance), maximized at N - 1 (full collapse)
            if self.training:
                mu = route_weights.mean(dim=(0, 1))
                load_balance_loss = self.mimo_paths * (mu ** 2).sum() - 1.0
                domain_loss = 0.15 * load_balance_loss  # stable quadratic gradient pressure
            else:
                domain_loss = torch.tensor(0.0, dtype=x.dtype, device=device)

        else:  # Phase 3j and SFT
            # Full soft-MoE: softmax competition, gradient flows through router
            route_logits = self.domain_router(mid_hidden)  # (B, L, 16)
            temp         = torch.clamp(self.router_temp, min=0.1, max=10.0)
            route_weights = F.softmax(route_logits / temp, dim=-1)

            domain_loss = torch.tensor(0.0, dtype=x.dtype, device=device)
            if domain_ids is not None:
                # Sequence-level routing supervision
                mean_logits = route_logits.mean(dim=1)  # (B, 16)
                domain_loss = F.cross_entropy(mean_logits, domain_ids)

        # ── LowRank Bridge → MIMO Arms (single pass) ──────────────────────────
        bridge_out = self.bridge(x)

        # Compute each arm exactly once
        raw_arm_outs  = []
        for i in range(self.mimo_paths):
            arm_out  = self.mimo_reasoning_blocks[i](bridge_out)
            raw_arm_outs.append(arm_out)

        # stacked_states: [B, L, d_model, 16]
        stacked_states = torch.stack(raw_arm_outs, dim=-1)

        # ipc_mixer removed — Blackboard handles inter-arm coordination below.

        # ── TEMPORAL MEMORY: Latent Scratchpad (Hippocampus) ──────────────────
        # ConceptPerceptron is injected every 6 backbone layers above — preserved.

        # ── SPATIAL MEMORY: Sparse IPC Blackboard (Corpus Callosum) ──────────
        # PATCH 2B: Silence Threshold — only arms with weight > 0.01 talk.
        comm_mask = (route_weights > 0.01).float().detach()        # [B, L, 16] — detached, no grad
        speaking_weights = route_weights * comm_mask

        # Arms compress their states to the 64-dim bus (cast for dtype safety)
        states_for_bus  = stacked_states.transpose(-1, -2)                        # [B, L, 16, d_model]
        bb_writes       = self.bb_write(states_for_bus.to(self.bb_write.weight.dtype)).to(x.dtype)  # [B, L, 16, bus_dim]

        # Gated write: dormant arms are silenced
        weighted_writes = bb_writes * speaking_weights.unsqueeze(-1)  # [B, L, 16, bus_dim]

        # Consensus blackboard — spatial agreement across active arms
        blackboard      = weighted_writes.sum(dim=-2)              # [B, L, bus_dim]

        # Gated read: broadcast shared context back ONLY to active arms
        shared_context  = self.bb_read(blackboard.to(self.bb_read.weight.dtype)).to(x.dtype)  # [B, L, d_model]
        # [B, L, d_model, 1] * [B, L, 1, 16] → [B, L, d_model, 16]
        gated_broadcast = shared_context.unsqueeze(-1) * comm_mask.unsqueeze(-2)

        # Cognitive synthesis: inject cross-talk into active arms
        stacked_states  = stacked_states + gated_broadcast         # [B, L, d_model, 16]

        # ── FINAL COLLAPSE (strict einsum — no broadcast multiply) ────────────
        # Dormant arms receive 0.5% trickle-charge via route_weights.
        # einsum maintains memory contiguity across 16-arm batch dimension.
        collapsed_mimo = torch.einsum(
            'b l d m, b l m -> b l d',
            stacked_states,
            route_weights
        )  # [B, L, d_model]
        x = x + collapsed_mimo

        # ── Second half of backbone (layers mid … end) ────────────────────────
        offset = self._mid
        for i, layer in enumerate(self.layers[self._mid:]):
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if ((offset + i + 1) % 6 == 0):
                global_ctx = cp_scratchpad.mean(dim=1, keepdim=True).clone()
                x = x + (self.cp_gate * torch.tanh(global_ctx))

        # ── Output ────────────────────────────────────────────────────────────
        x      = self.norm_f(x)
        logits = self.lm_head(x)

        # ── Telemetry (sampled to avoid overhead during training) ─────────────
        if not self.training or (self.training and torch.rand(1).item() < 0.20):
            with torch.no_grad():
                arm_w_mean = route_weights.mean(dim=(0, 1)).tolist()   # (16,)
                # Top-3 active arms by mean weight
                top_arms = sorted(
                    enumerate(arm_w_mean), key=lambda x: x[1], reverse=True
                )[:3]
                top_arm_labels = [
                    {"arm": idx, "label": ARM_IDENTITIES[idx], "weight": round(w, 4)}
                    for idx, w in top_arms
                ]
                # Entropy of routing distribution (0=collapsed, log(16)≈2.77=uniform)
                w_tensor = torch.tensor(arm_w_mean, dtype=torch.float32).clamp(min=1e-9)
                routing_entropy = float(-(w_tensor * w_tensor.log()).sum())

                self.last_telemetry.update({
                    'gate_score':   float(route_weights.mean()),
                    'entropy':      routing_entropy,
                    'arm_weights':  [round(w, 4) for w in arm_w_mean],
                    'arm_labels':   ARM_IDENTITIES,
                    'top_arms':     top_arm_labels,
                })

            # Glass Brain - Full 16x16 Pairwise Arm Divergence
            # OUTSIDE the no_grad block: inside grad-checkpoint recompute,
            # no_grad can conflict with checkpoint autograd hooks, silently
            # swallowing the computation. detach() already breaks the grad graph.
            # 1.0 = identical clones  |  0.0 = fully orthogonal
            try:
                ss = self._arm_snapshot                      # [d_model, 16] — pre-IPC mean snapshot
                arms_flat = ss                             # already [d_model, 16]
                arms_norm = F.normalize(arms_flat, dim=0)  # unit-length columns
                sim_matrix = arms_norm.T @ arms_norm       # [16, 16]
                off_diag   = ~torch.eye(16, dtype=torch.bool, device=sim_matrix.device)
                per_arm    = (sim_matrix * off_diag.float()).sum(dim=1) / 15.0
                self.last_telemetry.update({
                    'arm_collapse_metric': per_arm.mean().item(),
                    'arm_collapse_mean':   round(per_arm.mean().item(), 4),
                    'arm_collapse_max':    round(per_arm.max().item(),  4),
                    'arm_sims':            [round(v, 4) for v in per_arm.tolist()],
                    'latent_energy':       round(ss.norm(dim=-2).mean().item(), 4),
                })
            except Exception:
                pass  # never crash training over telemetry



        return logits, domain_loss
