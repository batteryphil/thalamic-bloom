import os
import time
import random
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer
from safetensors.torch import load_file
from mamba3_mimo_builder import Mamba3MIMORLF

# ROCm environment variables
os.environ['MIOPEN_USER_DB_PATH'] = '/home/phil/.gemini/antigravity/scratch/miopen_db'
os.environ['MIOPEN_CACHE_DIR'] = '/home/phil/.gemini/antigravity/scratch/miopen_db'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['AMD_SERIALIZE_KERNEL'] = '3'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

class CleanRecoveryDataset:
    def __init__(self, tokenizer, seq_len: int = 256) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        smol_dir = '/home/phil/.cache/huggingface/hub/datasets--HuggingFaceTB--smoltalk/snapshots/*/data/smol-magpie-ultra/*.parquet'
        self.smol_files = sorted(glob.glob(smol_dir))
        gsm_dir = '/home/phil/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*/main/*.parquet'
        self.gsm_files = sorted(glob.glob(gsm_dir))
        
        self.smol_df = None
        self.smol_idx = 0
        self.smol_file_idx = 0
        self.gsm_df = None
        self.gsm_idx = 0
        
        self._load_next_smol()
        self._load_gsm()
        
    def _load_next_smol(self):
        if not self.smol_files:
            return
        f = self.smol_files[self.smol_file_idx % len(self.smol_files)]
        self.smol_file_idx += 1
        self.smol_df = pd.read_parquet(f)
        self.smol_idx = 0
        
    def _load_gsm(self):
        if not self.gsm_files:
            return
        train_files = [f for f in self.gsm_files if 'train' in f]
        f = train_files[0] if train_files else self.gsm_files[0]
        self.gsm_df = pd.read_parquet(f)
        self.gsm_idx = 0
        
    def get_sample(self):
        is_smol = (random.random() < 0.70 and self.smol_df is not None) or (self.gsm_df is None)
        if is_smol and self.smol_df is not None:
            if self.smol_idx >= len(self.smol_df):
                self._load_next_smol()
            row = self.smol_df.iloc[self.smol_idx]
            self.smol_idx += 1
            
            messages = row['messages']
            user_msg = "Hello"
            asst_msg = "Hello! How can I help you today?"
            for m in messages:
                if m.get('role') == 'user':
                    user_msg = m.get('content', '')
                elif m.get('role') == 'assistant' and user_msg:
                    asst_msg = m.get('content', '')
                    break
            prompt = f"User: {user_msg}\nAssistant: "
            answer = f"{asst_msg}<|endoftext|>\n"
            return prompt, answer
        else:
            if self.gsm_idx >= len(self.gsm_df):
                self.gsm_idx = 0
            row = self.gsm_df.iloc[self.gsm_idx]
            self.gsm_idx += 1
            prompt = f"User: {row['question']}\nAssistant: "
            answer = f"{row['answer']}<|endoftext|>\n"
            return prompt, answer

    def get_batch(self, batch_size: int, device: torch.device):
        x_list, y_list = [], []
        for _ in range(batch_size):
            prompt, answer = self.get_sample()
            u_toks = self.tokenizer.encode(prompt)
            a_toks = self.tokenizer.encode(answer)
            toks = u_toks + a_toks
            tgts = [-100] * len(u_toks) + a_toks
            
            if len(toks) > self.seq_len + 1:
                toks = toks[:self.seq_len + 1]
                tgts = tgts[:self.seq_len + 1]
            else:
                pad_len = (self.seq_len + 1) - len(toks)
                toks = toks + [0] * pad_len
                tgts = tgts + [-100] * pad_len
                
            x_list.append(toks[:-1])
            y_list.append(tgts[1:])
            
        return torch.tensor(x_list, dtype=torch.long, device=device), torch.tensor(y_list, dtype=torch.long, device=device)

def load_clean_base_weights(model: Mamba3MIMORLF, model_dir: str):
    st = load_file(f'{model_dir}/model.safetensors')
    new_sd = {}
    new_sd['embedding.weight'] = st['backbone.embeddings.weight']
    new_sd['lm_head.weight'] = st.get('lm_head.weight', st['backbone.embeddings.weight']).clone()
    if 'backbone.norm_f.weight' in st:
        new_sd['norm_f.weight'] = st['backbone.norm_f.weight']

    for i in range(24):
        prefix_src = f'backbone.layers.{i}.'
        prefix_dst = f'layers.{i}.'
        new_sd[f'{prefix_dst}norm.weight'] = st[f'{prefix_src}norm.weight']
        for k in ['A_log', 'D', 'in_proj.weight', 'conv1d.weight', 'conv1d.bias', 'x_proj.weight', 'dt_proj.weight', 'dt_proj.bias', 'out_proj.weight']:
            new_sd[f'{prefix_dst}ssm.{k}'] = st[f'{prefix_src}mixer.{k}']

    for arm_idx in range(4):
        src_layer = 12 + arm_idx
        prefix_src = f'backbone.layers.{src_layer}.'
        prefix_dst = f'mimo_reasoning_blocks.{arm_idx}.'
        new_sd[f'{prefix_dst}norm.weight'] = st[f'{prefix_src}norm.weight'].clone()
        for k in ['A_log', 'D', 'in_proj.weight', 'conv1d.weight', 'conv1d.bias', 'x_proj.weight', 'dt_proj.weight', 'dt_proj.bias', 'out_proj.weight']:
            w = st[f'{prefix_src}mixer.{k}'].clone()
            if arm_idx > 0 and 'weight' in k:
                w = w + torch.randn_like(w) * 0.01
            new_sd[f'{prefix_dst}ssm.{k}'] = w

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[Init] Pristine Mamba base weights loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")

def train(max_steps: int = 250, lr: float = 3e-5, grad_accum_steps: int = 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"Training Clean Thalamic Bloom from Mamba Base on: {device}")
    print("=" * 65)
    
    model_dir = '/home/phil/.cache/huggingface/hub/models--state-spaces--mamba-130m-hf/snapshots/1e76775f628fbf1350fbe4dbb3d971ba64af25a1'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dataset = CleanRecoveryDataset(tokenizer, seq_len=256)
    
    model = Mamba3MIMORLF(vocab_size=50280, d_model=768, n_layers=24, mimo_paths=4, bus_dim=64).to(device)
    load_clean_base_weights(model, model_dir)
    
    # Train Router, Blackboard, MIMO Arms, and LM Head
    for layer in model.layers:
        for p in layer.parameters():
            p.requires_grad = False
    model.embedding.weight.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Model] Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    probe_prompts = [
        "User: Say hello.\nAssistant: ",
        "User: What is the capital of France?\nAssistant: ",
        "User: Calculate 12 * 12.\nAssistant: "
    ]
    
    model.train()
    start_time = time.time()
    smoothed_loss = None
    accum_loss = 0.0
    
    optimizer.zero_grad()
    for step in range(1, max_steps + 1):
        x, y = dataset.get_batch(batch_size=1, device=device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            lm_loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss = (lm_loss + model.load_balance_loss) / grad_accum_steps
            
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            accum_loss = 0.0
            continue
            
        total_loss.backward()
        accum_loss += lm_loss.item()
        
        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()
            
        loss_val = accum_loss / (step % grad_accum_steps if (step % grad_accum_steps != 0) else grad_accum_steps)
        smoothed_loss = loss_val if smoothed_loss is None else (0.95 * smoothed_loss + 0.05 * loss_val)
        if step % grad_accum_steps == 0:
            accum_loss = 0.0
            
        if step % 20 == 0 or step == 1:
            elapsed = time.time() - start_time
            tps = (step * 256) / max(elapsed, 1e-4)
            rw = [round(w, 3) for w in model.last_telemetry.get('route_weights', [])]
            entropy = model.last_telemetry.get('entropy', 0.0)
            print(f"Step {step:4d}/{max_steps} | Loss: {loss_val:.4f} (Smooth: {smoothed_loss:.4f}) | Bal: {model.load_balance_loss.item():.4f} | Route: {rw} | H: {entropy:.3f} | {tps:.0f} tok/s")
            
        if step % 50 == 0 or step == max_steps:
            model.eval()
            print("\n" + "=" * 50)
            print(f"[PROBE GENERATION AT STEP {step}]")
            for prompt in probe_prompts:
                inp = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
                out = model.generate(inp, max_new_tokens=25, temperature=0.7, top_k=40)
                gen = tokenizer.decode(out[0][len(inp[0]):].tolist()).strip()
                print(f"Q: {prompt.replace(chr(10), ' ')}")
                print(f"A: {repr(gen)}")
            print("=" * 50 + "\n")
            model.train()
            
    save_path = '/home/phil/.gemini/antigravity/scratch/thalamic-bloom/thalamic_bloom_150m_clean_recovered.pth'
    torch.save({
        'step': max_steps,
        'model_state_dict': model.state_dict(),
        'smoothed_loss': smoothed_loss,
        'last_telemetry': model.last_telemetry
    }, save_path)
    print(f"Clean Recovery Complete! Saved checkpoint to: {save_path}")

if __name__ == '__main__':
    train(max_steps=250, lr=3e-5, grad_accum_steps=4)
