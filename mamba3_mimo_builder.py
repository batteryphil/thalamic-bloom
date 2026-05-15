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
        self.last_telemetry = {}
        
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
        
        # Calculate Primer Activation Magnitude as routing signal.
        # This measures HOW MUCH the Primer is transforming the representations.
        # Starts near 0.0 (zero-init), grows naturally as weights develop.
        # This directly breaks the SSM glass ceiling that locked entropy at 0.0000.
        with torch.no_grad():
            primer_delta = (primer_out - orig_embs).norm(dim=-1).mean()
            # Scale: delta of ~0.1 opens gate to 50%, delta of ~0.2 opens fully.
            # Use detached signal so routing heuristic stays outside autograd.
            routing_signal = primer_delta.detach()
        
        # Blend: add Primer signal into main stream
        x = orig_embs + primer_out * 0.1
        
        # Route into Auxiliary Loop Engine via Dynamic Bridge
        bridge_out = self.bridge(x)
        
        # Soft-Body Compute Allocation driven by Primer Activation Magnitude
        raw_gate_score = torch.sigmoid(routing_signal * 10.0 - 1.0)
        # --- OCTOPODA TRICKLE-CHARGE PATCH ---
        # Clamp the minimum routing weight to 0.05 (5%) to prevent Synaptic Atrophy
        gate_score = torch.clamp(raw_gate_score, min=0.05).unsqueeze(-1).unsqueeze(-1)
        
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
            
            mask = 1.0 if i == 0 else gate_score
            parallel_states.append(state * mask * autotomic_gate)
            
        # Latent IPC Cross-Talk
        ipc_in = torch.cat(parallel_states, dim=-1)
        ipc_out = self.ipc_mixer(ipc_in)
        
        # Split back to individual paths
        final_states = torch.split(ipc_out, self.d_model, dim=-1)
        
        self.last_telemetry = {
            "entropy": routing_signal.item() if isinstance(routing_signal, torch.Tensor) else routing_signal,
            "gate_score": gate_score.mean().item() if isinstance(gate_score, torch.Tensor) else gate_score,
            "autotomic_gates": autotomic_gates_list
        }
            
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
