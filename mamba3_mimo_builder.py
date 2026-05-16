import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Set, List
from mamba_ssm import Mamba

class DummyMambaSSM(nn.Module):
    """
    Placeholder for the core Mamba state-space scan to allow testing without CUDA compilation.
    In production, swap this with `mamba_ssm.Mamba`.
    """
    def __init__(self, d_model: int) -> None:
        """
        Initialize the dummy Mamba SSM.
        
        Args:
            d_model (int): The dimensionality of the input and output features.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the dummy Mamba SSM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).
            
        Returns:
            torch.Tensor: Projected tensor of the same shape.
        """
        return self.proj(x)

class ConceptPerceptron(nn.Module):
    """Global context pooling mechanism mapping the input sequence into a condensed latent prefix."""
    def __init__(self, d_model: int, num_tokens: int = 16, chunk_size: int = 1024) -> None:
        """
        Initialize the Concept Perceptron.
        
        Args:
            d_model (int): Hidden size of the model.
            num_tokens (int): The number of tokens in the condensed latent prefix.
            chunk_size (int): Size of chunks to process to bypass long-context SSM saturation.
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_size = chunk_size
        self.avg_pooling = nn.AdaptiveAvgPool1d(num_tokens)
        self.max_pooling = nn.AdaptiveMaxPool1d(num_tokens)
        self.proj = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to process the input sequence and generate the concept scratchpad.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (B, L, D).
            
        Returns:
            torch.Tensor: Condensed latent prefix of shape (B, num_tokens, D).
        """
        B, L, D = x.shape
        chunks: List[torch.Tensor] = []
        
        # Handle chunked inference natively for context windows > 1024 tokens
        for i in range(0, L, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size, :]
            # Transpose for AdaptiveAvgPool1d: (B, L_chunk, D) -> (B, D, L_chunk)
            chunk_t = chunk.transpose(1, 2)
            avg_pool = self.avg_pooling(chunk_t).transpose(1, 2)
            max_pool = self.max_pooling(chunk_t).transpose(1, 2)
            
            # Suction-Cup Granular Anchoring: Concatenate global and granular features
            pooled_chunk = torch.cat([avg_pool, max_pool], dim=-1)
            chunks.append(pooled_chunk)
        
        # Aggregate chunk scratchpads
        aggregated = torch.stack(chunks, dim=0).mean(dim=0)
        return F.silu(self.proj(aggregated))

class LowRankBridge(nn.Module):
    """Bottleneck compression bridge routing into auxiliary reasoning engines."""
    def __init__(self, d_model: int, bottleneck: int = 64) -> None:
        """
        Initialize the Low-Rank Latent Bridge.
        
        Args:
            d_model (int): Full model dimension.
            bottleneck (int): Compressed dimension.
        """
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the compression bottleneck.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Reconstructed tensor.
        """
        return self.up(F.silu(self.down(x)))

class MambaLayer(nn.Module):
    """Mamba layer enforcing strict bfloat16 precision."""
    def __init__(self, d_model: int) -> None:
        """
        Initialize the Mamba Layer.
        
        Args:
            d_model (int): Hidden size of the model.
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Use real Mamba instead of Dummy
        self.ssm = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the SSM layer with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with residual connection added.
        """
        residual = x
        x_norm = self.norm(x)
        
        # Mandatory precision constraint: SSM strictly evaluated in bfloat16
        device_type = x.device.type if x.device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            x_ssm = self.ssm(x_norm)
            
        return x_ssm.to(x.dtype) + residual

class Mamba3MIMORLF(nn.Module):
    """Mamba 3 MIMO architecture with parallel latent forcing."""
    def __init__(self, vocab_size: int = 50304, d_model: int = 768, n_layers: int = 24, mimo_paths: int = 4) -> None:
        """
        Initialize the Mamba 3 MIMO model.
        
        Args:
            vocab_size (int): Size of the token vocabulary.
            d_model (int): Dimension of the model.
            n_layers (int): Number of sequential Mamba layers.
            mimo_paths (int): Number of parallel MIMO streams.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mimo_paths = mimo_paths
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cp = ConceptPerceptron(d_model)
        self.thalamic_primer = MambaLayer(d_model)
        self.bridge = LowRankBridge(d_model)
        
        # Main Sequential Backbone
        self.layers = nn.ModuleList([MambaLayer(d_model) for _ in range(n_layers)])
        
        # MIMO Engine: Parallel Latent Reasoning Chains
        self.mimo_reasoning_blocks = nn.ModuleList([MambaLayer(d_model) for _ in range(mimo_paths)])
        
        # Latent IPC (Cross-Talk)
        self.ipc_mixer = nn.Linear(d_model * mimo_paths, d_model * mimo_paths)
        
        # Synaptic Dam: Gated Tanh for global context injection
        self.cp_gate = nn.Parameter(torch.tensor(0.01))
        
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying
        
        # Phase 3j: Per-arm competitive vector router
        self.domain_router = nn.Linear(self.d_model, self.mimo_paths, bias=True)
        nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.domain_router.bias)
        
        self.last_telemetry = {
            'arm_collapse_metric': 0.0,
            'latent_energy': 0.0,
            'gate_score': 0.0,
            'primer_delta': 0.0
        }
        
        # Zero-Init Thalamic Primer for Identity Pass-through
        nn.init.zeros_(self.thalamic_primer.ssm.out_proj.weight)
        
    def initialize_asymmetric_arms(self) -> None:
        """
        Asymmetric Initialization (Decentralized Ganglionic Processing).
        Applies orthogonal weights so loops specialize in different domains.
        """
        for name, param in self.mimo_reasoning_blocks.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif param.dim() == 1:
                # Break temporal symmetry in A_log, D, dt_bias, etc.
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * 0.05)
        
    def forward(self, input_ids: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        """
        Forward pass executing parallel reasoning chains.
        
        Args:
            input_ids (torch.Tensor): Integer token IDs of shape (batch, seq_len).
            loop_idx (int): Current RLF loop index to calculate Latent Lifeline Decay.
            
        Returns:
            torch.Tensor: Output logits of shape (batch, seq_len, vocab_size).
        """
        orig_embs = self.embedding(input_ids)
        
        # Latent Lifeline Decay
        decay_factor = 0.7 ** loop_idx
        x = orig_embs * decay_factor
        
        # Concept Perceptron generating the condensed scratchpad
        cp_scratchpad = self.cp(x)
        
        # A. The Thalamic Primer: Temporal Sequence Mixing before routing
        primer_out = self.thalamic_primer(orig_embs)
        
        # --- SCALE-INVARIANT ANGULAR DEFORMATION ROUTER ---
        # Measure angular deviation instead of L2 distance to prevent saturation
        with torch.no_grad():
            primer_cos_sim = F.cosine_similarity(orig_embs, primer_out, dim=-1)
            # Invert so 0.0 = identity (no deviation), bounding max deviation up to 2.0
            primer_delta = (1.0 - primer_cos_sim).mean().detach()

        # Scale the geometric deviation (tune multiplier to 20.0 to account for smaller cosine values)
        routing_signal = primer_delta
        
        # Blend: add Primer signal into main stream
        x = orig_embs + primer_out * 0.1
        
        # Route into Auxiliary Loop Engine via Dynamic Bridge
        bridge_out = self.bridge(x)
        
        # Phase 3j: Temporal Vector Gating
        # The domain_router MUST be part of the autograd graph to learn specializations.
        # We only detach primer_out to protect the upstream backbone.
        route_logits = self.domain_router(primer_out.detach())  # (B, L, 4)

        competitive_weights = F.softmax(route_logits, dim=-1)
        
        # 1. Find the winning arm for each token
        top_indices = competitive_weights.argmax(dim=-1, keepdim=True)
        
        # 2. Create a Hard Binary Mask (1 for winner, 0 for losers)
        mask = torch.zeros_like(competitive_weights).scatter_(-1, top_indices, 1.0)
        
        # 3. Straight-Through Estimator: Forward = Hard Mask, Backward = Softmax
        hard_weights = mask - competitive_weights.detach() + competitive_weights
        
        # 4. Trickle charge maintains 0.05 minimum for dormant arms
        route_weights = torch.clamp(hard_weights, min=0.05)
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)
        
        parallel_states = []
        autotomic_gates_list = []
        for i in range(self.mimo_paths):
            state = self.mimo_reasoning_blocks[i](bridge_out)
            
            # B. Post-Compute Autotomy (Evaluate Hallucination before IPC pollution)
            variance = state.var(dim=-1).mean()
            # --- OCTOPODA TRICKLE-CHARGE PATCH ---
            # Clamp the autotomic gate to prevent total gradient death on highly variant paths
            autotomic_gate = torch.clamp(torch.sigmoid((10.0 - variance) * 0.5), min=0.05)
            autotomic_gates_list.append(autotomic_gate.item() if isinstance(autotomic_gate, torch.Tensor) else autotomic_gate)
            
            # Apply per-token vector weight for this arm (B, L, 1) for broadcasting
            arm_weight = route_weights[..., i:i+1]  # (B, L, 1)

            # Arm 0 always gets full gradient (primary cortex, preserved)
            if i == 0:
                parallel_states.append(state * autotomic_gate)
            else:
                parallel_states.append(state * arm_weight * autotomic_gate)
            
        mean_gate = route_weights[..., 1:].mean().item()
        
        # Calculate Orthogonal Regularization Loss (Repulsive Magnets)
        if self.training:
            sim_01 = F.cosine_similarity(parallel_states[0], parallel_states[1], dim=-1).mean()
            sim_02 = F.cosine_similarity(parallel_states[0], parallel_states[2], dim=-1).mean()
            sim_03 = F.cosine_similarity(parallel_states[0], parallel_states[3], dim=-1).mean()
            self.ortho_loss = (sim_01 + sim_02 + sim_03) / 3.0
        else:
            self.ortho_loss = 0.0
        
        # =====================================================================
        # DYNAMICAL SYSTEMS TELEMETRY PROBES (No Gradient Tracking)
        # =====================================================================
        # Sample ~5% of batches to save compute and keep TPS high
        if not self.training or (self.training and torch.rand(1).item() < 0.05):
            with torch.no_grad():
                # 1. LATENT COSINE SEPARATION (Arm Divergence)
                # 1.0 = Mode Collapse (Redundant). 0.0 = Orthogonal Specialization.
                arm_0, arm_1, arm_2, arm_3 = parallel_states
                
                sim_01 = F.cosine_similarity(arm_0, arm_1, dim=-1).mean()
                sim_02 = F.cosine_similarity(arm_0, arm_2, dim=-1).mean()
                sim_03 = F.cosine_similarity(arm_0, arm_3, dim=-1).mean()
                
                avg_collapse_metric = (sim_01 + sim_02 + sim_03) / 3.0
                
                # 2. RECURRENT ATTRACTOR STABILITY (Latent Energy)
                latent_energy = torch.stack(parallel_states).norm(dim=-1).mean()
                
                self.last_telemetry.update({
                    'arm_collapse_metric': avg_collapse_metric.item(),
                    'latent_energy': latent_energy.item(),
                    'primer_delta': primer_delta.item() if isinstance(primer_delta, torch.Tensor) else primer_delta
                })
        # =====================================================================
        
        # Latent IPC Cross-Talk
        ipc_in = torch.cat(parallel_states, dim=-1)
        ipc_out = self.ipc_mixer(ipc_in)
        
        # Split back to individual paths
        final_states = torch.split(ipc_out, self.d_model, dim=-1)
        
        self.last_telemetry.update({
            "entropy": routing_signal.item() if isinstance(routing_signal, torch.Tensor) else routing_signal,
            "gate_score": mean_gate,
            "autotomic_gates": autotomic_gates_list,
            "route_weights": route_weights.mean(dim=(0, 1)).tolist()  # Per-arm mean
        })
            
        # Collapse multiple latent paths back into the residual stream
        x = x + (sum(final_states) / self.mimo_paths)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Deep Injection: global context residually injected every 6 layers
            if (i + 1) % 6 == 0:
                global_ctx = cp_scratchpad.mean(dim=1, keepdim=True)
                # Synaptic Dam: Bounded injection to prevent broadcast storm
                x = x + (self.cp_gate * torch.tanh(global_ctx))
                
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 50, 
        temperature: float = 0.3, 
        top_k: int = 5, 
        stop_sequences: Optional[Set[int]] = None
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens.
        
        Args:
            input_ids (torch.Tensor): Initial token prompt.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Softmax temperature scaling parameter.
            top_k (int): Limits sampling to top k probable tokens.
            stop_sequences (Optional[Set[int]]): Set of token IDs that stop generation.
            
        Returns:
            torch.Tensor: Full generated token sequence.
        """
        self.eval()
        for _ in range(max_new_tokens): 
            logits = self.forward(input_ids, loop_idx=0)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Autoregressive State Saturation Fix: Repetition Penalty
            # Penalize tokens that have already been generated in this sequence
            for token_id in torch.unique(input_ids[0]):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= 1.2
                else:
                    next_token_logits[0, token_id] *= 1.2

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if stop_sequences and next_token.item() in stop_sequences:
                break
                
        return input_ids
