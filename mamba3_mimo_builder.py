import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Set, List

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class PureMambaSSM(nn.Module):
    """
    Pure PyTorch Mamba S6 Selective State Space Model.
    Runs on CPU and AMD ROCm GPU without CUDA compilation.
    Matches the exact state_dict layout of official mamba_ssm.Mamba.
    """
    def __init__(self, d_model: int = 768, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = max(d_model // 16, 1)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_branch = F.silu(x_conv)

        x_dbc = self.x_proj(x_branch)
        dt, B, C = torch.split(x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = torch.clamp(F.softplus(self.dt_proj(dt)), min=1e-4, max=10.0)

        A = -torch.exp(self.A_log.float()).to(dt.dtype)
        dt_A = torch.clamp(torch.einsum('btd,dn->btdn', dt, A), min=-20.0, max=0.0)
        A_bar = torch.exp(dt_A)
        dt_B = torch.einsum('btd,btn->btdn', dt, B)

        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + dt_B[:, t] * x_branch[:, t].unsqueeze(-1)
            y_t = torch.einsum('bdn,bn->bd', h, C[:, t]) + self.D * x_branch[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class ConceptPerceptron(nn.Module):
    """Global context pooling mechanism mapping the input sequence into a condensed latent prefix."""
    def __init__(self, d_model: int, num_tokens: int = 16, chunk_size: int = 1024) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_size = chunk_size
        self.avg_pooling = nn.AdaptiveAvgPool1d(num_tokens)
        self.max_pooling = nn.AdaptiveMaxPool1d(num_tokens)
        self.proj = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        chunks: List[torch.Tensor] = []
        for i in range(0, L, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size, :]
            chunk_t = chunk.transpose(1, 2)
            avg_pool = self.avg_pooling(chunk_t).transpose(1, 2)
            max_pool = self.max_pooling(chunk_t).transpose(1, 2)
            pooled_chunk = torch.cat([avg_pool, max_pool], dim=-1)
            chunks.append(pooled_chunk)
        
        aggregated = torch.stack(chunks, dim=0).mean(dim=0)
        return F.silu(self.proj(aggregated))


class LowRankBridge(nn.Module):
    """Bottleneck compression bridge routing into auxiliary reasoning engines."""
    def __init__(self, d_model: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, d_model, bias=False)
    
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
            self.ssm = PureMambaSSM(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        device_type = x.device.type if x.device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            x_ssm = self.ssm(x_norm)
        return x_ssm.to(x.dtype) + residual


class Mamba3MIMORLF(nn.Module):
    """
    Mamba 3 MIMO architecture with Sparse IPC Blackboard (Corpus Callosum)
    and Mid-Backbone Semantic Routing.
    """
    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 768,
        n_layers: int = 24,
        mimo_paths: int = 4,
        bus_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mimo_paths = mimo_paths
        self.n_layers = n_layers
        self._mid = n_layers // 2  # layer 12 — where mid-backbone semantic routing occurs
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cp = ConceptPerceptron(d_model)
        self.thalamic_primer = MambaLayer(d_model)
        self.bridge = LowRankBridge(d_model)
        
        # Main Sequential Backbone (24 layers total, split at midpoint)
        self.layers = nn.ModuleList([MambaLayer(d_model) for _ in range(n_layers)])
        
        # MIMO Engine: 4 Parallel Reasoning Chains
        self.mimo_reasoning_blocks = nn.ModuleList([MambaLayer(d_model) for _ in range(mimo_paths)])
        
        # ── SPATIAL MEMORY: Sparse IPC Blackboard (Corpus Callosum) ──────────
        # Replaces the dense 1B-parameter IPC mixer with a 64-dim bottleneck bus
        self.bus_dim = bus_dim
        self.bb_write = nn.Linear(d_model, self.bus_dim, bias=False)
        self.bb_read = nn.Linear(self.bus_dim, d_model, bias=False)
        nn.init.zeros_(self.bb_read.weight)  # Zero-init: Blackboard starts silent, additive residual
        
        # Mid-backbone semantic router with learnable temperature
        self.domain_router = nn.Linear(d_model, mimo_paths, bias=True)
        self.router_temp = nn.Parameter(torch.ones(1) * 1.0)
        nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.domain_router.bias)
        
        # Synaptic Dam: Calibrated Gated Tanh for global context injection
        # Clamped/initialized small to prevent broadcast saturation
        self.cp_gate = nn.Parameter(torch.tensor(0.02))
        
        self.norm_f = nn.LayerNorm(d_model)
        # UNTIED LM HEAD: independent weights prevent embedding table degradation
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.last_telemetry = {
            'arm_collapse_metric': 0.0,
            'latent_energy': 0.0,
            'gate_score': 0.0,
            'primer_delta': 0.0,
            'route_weights': [1.0 / mimo_paths] * mimo_paths,
            'entropy': 0.0
        }
        
        # Zero-Init Thalamic Primer for Identity Pass-through
        if hasattr(self.thalamic_primer.ssm, 'out_proj'):
            nn.init.zeros_(self.thalamic_primer.ssm.out_proj.weight)
            
    def initialize_asymmetric_arms(self) -> None:
        """Applies orthogonal weights and breaks temporal symmetry in 1D SSM parameters."""
        for name, param in self.mimo_reasoning_blocks.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif param.dim() == 1:
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * 0.05)

    def load_legacy_checkpoint(self, checkpoint_path: str, device: torch.device) -> None:
        """
        Loads checkpoint weights with graceful adaptation:
        1. Breaks weight-tying between lm_head and embedding.
        2. Filters out deprecated ipc_mixer weights while preserving backbone/arms.
        3. Initializes Blackboard cleanly.
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else (
            ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        )
        
        # Filter out ipc_mixer keys
        filtered_sd = {k: v for k, v in sd.items() if not k.startswith('ipc_mixer.')}
        
        missing, unexpected = self.load_state_dict(filtered_sd, strict=False)
        print(f"[Loader] Loaded checkpoint. Missing: {len(missing)}, Ignored/Unexpected: {len(unexpected)}")
        
        # Explicitly break weight tie on lm_head
        with torch.no_grad():
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.clone())
            # Cap cp_gate to avoid broadcast saturation
            if self.cp_gate.data > 0.05:
                print(f"[Loader] Clamping saturated cp_gate ({self.cp_gate.data.item():.4f} -> 0.02)")
                self.cp_gate.data.fill_(0.02)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        loop_idx: int = 0,
        ablate_cp: bool = False,
        ablate_mimo: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with Mid-Backbone Semantic Routing and Sparse Blackboard.
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        orig_embs = self.embedding(input_ids)
        decay_factor = 0.7 ** loop_idx
        x = orig_embs * decay_factor
        
        # Concept Perceptron generating the condensed scratchpad
        cp_scratchpad = self.cp(x)
        
        # Thalamic Primer
        primer_out = self.thalamic_primer(orig_embs)
        x = orig_embs + primer_out * 0.1
        
        # ── First half of backbone (layers 0 ... _mid-1) ──────────────
        for i, layer in enumerate(self.layers[:self._mid]):
            x = layer(x)
            if (i + 1) % 6 == 0 and not ablate_cp:
                global_ctx = cp_scratchpad.mean(dim=1, keepdim=True)
                eff_gate = torch.clamp(self.cp_gate, max=0.05)
                x = x + (eff_gate * torch.tanh(global_ctx))
                
        # ── Mid-Backbone Semantic Routing ──────────────────────────────
        # Routing is computed from rich mid-backbone semantic features (layer 12)
        mid_hidden = x
        route_logits = self.domain_router(mid_hidden.detach() if self.training else mid_hidden)
        
        if self.training:
            # Exploration noise during training
            noise = torch.randn_like(route_logits) * 0.05
            route_logits = route_logits + noise
            
        temp = torch.clamp(self.router_temp, min=0.1, max=10.0)
        route_weights = F.softmax(route_logits / temp, dim=-1) # (B, L, 4)
        
        # Switch Transformer quadratic load balancing loss: 4 * sum(mu_i^2) - 1.0
        if self.training:
            mu = route_weights.mean(dim=(0, 1))
            self.load_balance_loss = 0.15 * (self.mimo_paths * (mu ** 2).sum() - 1.0)
        else:
            self.load_balance_loss = torch.tensor(0.0, device=device)
            
        # ── MIMO Arms Computation ──────────────────────────────────────
        if not ablate_mimo:
            bridge_out = self.bridge(x)
            raw_arm_outs = []
            for i in range(self.mimo_paths):
                arm_out = self.mimo_reasoning_blocks[i](bridge_out)
                raw_arm_outs.append(arm_out)
            
            # stacked_states: (B, L, d_model, 4)
            stacked_states = torch.stack(raw_arm_outs, dim=-1)
            
            # ── SPATIAL MEMORY: Sparse IPC Blackboard ──────────────────
            # Silence threshold: only arms with route_weight > 0.01 participate
            comm_mask = (route_weights > 0.01).float().detach()
            speaking_weights = route_weights * comm_mask
            
            # Project to bus dimension: (B, L, 4, d_model) -> (B, L, 4, bus_dim)
            states_for_bus = stacked_states.transpose(-1, -2)
            bb_writes = self.bb_write(states_for_bus.to(self.bb_write.weight.dtype)).to(x.dtype)
            weighted_writes = bb_writes * speaking_weights.unsqueeze(-1)
            
            # Consensus blackboard
            blackboard = weighted_writes.sum(dim=-2) # (B, L, bus_dim)
            
            # Broadcast back to active arms
            shared_context = self.bb_read(blackboard.to(self.bb_read.weight.dtype)).to(x.dtype) # (B, L, d_model)
            gated_broadcast = shared_context.unsqueeze(-1) * comm_mask.unsqueeze(-2)
            stacked_states = stacked_states + gated_broadcast
            
            # Collapse MIMO arms with routing weights
            collapsed_mimo = torch.einsum('b l d m, b l m -> b l d', stacked_states, route_weights)
            x = x + collapsed_mimo
            
        # ── Second half of backbone (layers _mid ... end) ─────────────
        for i, layer in enumerate(self.layers[self._mid:], start=self._mid):
            x = layer(x)
            if (i + 1) % 6 == 0 and not ablate_cp:
                global_ctx = cp_scratchpad.mean(dim=1, keepdim=True)
                eff_gate = torch.clamp(self.cp_gate, max=0.05)
                x = x + (eff_gate * torch.tanh(global_ctx))
                
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        # Telemetry
        with torch.no_grad():
            self.last_telemetry['route_weights'] = route_weights.mean(dim=(0, 1)).tolist()
            entropy = -(route_weights * torch.log(route_weights + 1e-8)).sum(dim=-1).mean().item()
            self.last_telemetry['entropy'] = entropy
            self.last_telemetry['gate_score'] = route_weights[..., 1:].mean().item()
            
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.3,
        top_k: int = 20,
        stop_sequences: Optional[Set[int]] = None,
        ablate_cp: bool = False,
        ablate_mimo: bool = False
    ) -> torch.Tensor:
        """Autoregressively generate tokens with repetition dampening."""
        self.eval()
        cur_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(cur_ids, loop_idx=0, ablate_cp=ablate_cp, ablate_mimo=ablate_mimo)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-4)
            
            # Repetition penalty on previously generated sequence
            for token_id in torch.unique(cur_ids[0]):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= 1.2
                else:
                    next_token_logits[0, token_id] *= 1.2
                    
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            cur_ids = torch.cat([cur_ids, next_token], dim=-1)
            
            if stop_sequences and next_token.item() in stop_sequences:
                break
                
        return cur_ids
