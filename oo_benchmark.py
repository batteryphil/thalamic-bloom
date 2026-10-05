import sys
import os
import argparse
import torch
from transformers import AutoTokenizer
from mamba3_mimo_builder import Mamba3MIMORLF

os.environ['MIOPEN_USER_DB_PATH'] = '/home/phil/.gemini/antigravity/scratch/miopen_db'
os.environ['MIOPEN_CACHE_DIR'] = '/home/phil/.gemini/antigravity/scratch/miopen_db'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['AMD_SERIALIZE_KERNEL'] = '3'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark Thalamic Bloom 150M MIMO MoE")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("Loading Thalamic Bloom 150M Benchmark Suite...")
    print(f"Device: {device}")
    print("=" * 65)
    
    ckpt_path = args.checkpoint
    if not ckpt_path:
        clean_path = "/home/phil/.gemini/antigravity/scratch/thalamic-bloom/thalamic_bloom_150m_clean_recovered.pth"
        hf_path = "/home/phil/.cache/huggingface/hub/models--batteryphil--thalamic-bloom/snapshots/4ede44ff4929b948a36db42488751fec437bfd21/thalamic_bloom_150m_oo.pth"
        ckpt_path = clean_path if os.path.exists(clean_path) else hf_path
        
    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        filtered_sd = {k: v for k, v in sd.items() if not k.startswith('ipc_mixer.')}
        
        vocab_size = filtered_sd['embedding.weight'].shape[0]
        model = Mamba3MIMORLF(vocab_size=vocab_size, d_model=768, n_layers=24, mimo_paths=4, bus_dim=64).to(device)
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        print(f"Checkpoint loaded successfully! (Vocab: {vocab_size}, Missing: {len(missing)}, Ignored: {len(unexpected)})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.eval()
    tok_path = '/home/phil/.cache/huggingface/hub/models--state-spaces--mamba-130m-hf/snapshots/1e76775f628fbf1350fbe4dbb3d971ba64af25a1' if vocab_size == 50280 else 'EleutherAI/gpt-neox-20b'
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    
    def generate_and_log(prompt, max_tokens=35):
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_k=20
            )
            
        generated_tokens = output[0].tolist()[len(input_ids[0]):]
        answer = tokenizer.decode(generated_tokens).strip()
        
        print(f"PROMPT: {prompt.replace(chr(10), ' ')}")
        print(f"OUTPUT: {repr(answer)}")
        print("\n[ROUTING TELEMETRY]")
        telemetry = model.last_telemetry
        rw = [round(w, 3) for w in telemetry.get('route_weights', [])]
        print(f"Per-Arm Routing:   {rw}")
        print(f"Routing Entropy:   {telemetry.get('entropy', 0):.4f}")
        gate = telemetry.get('gate_score', 0)
        print(f"Gate Score:        {gate:.4f}")
        print("-" * 65 + "\n")

    prompts = [
        "User: Say hello.\nAssistant: ",
        "User: What is the capital of France?\nAssistant: ",
        "User: Calculate 12 * 12.\nAssistant: ",
        "User: Write a Python script to calculate the Fibonacci sequence.\nAssistant: ",
        "User: A is True. B is False. C is A AND B. What is C? Think step by step.\nAssistant: ",
        "User: Output a JSON block with three random colors, and nothing else.\nAssistant: "
    ]
    
    for prompt in prompts:
        generate_and_log(prompt)

if __name__ == "__main__":
    run_benchmark()
