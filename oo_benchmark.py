import sys
import torch
from transformers import AutoTokenizer
from mamba3_mimo_builder import Mamba3MIMORLF

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==================================================================")
    print("Loading Mamba 3 MIMO OS Telemetry Benchmark...")
    print("==================================================================")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
    model.to(device)
    
    checkpoint_path = "jarvis_v4.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.eval()
    
    stop_sequences = {0, 198, 200}
    
    def generate_and_log(prompt, max_tokens=30):
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Strict Greedy Decoding
                top_k=1,          # Strict Greedy Decoding
                stop_sequences=stop_sequences
            )
            
        generated_tokens = output[0].tolist()[len(input_ids[0]):]
        answer = tokenizer.decode(generated_tokens).strip()
        
        print(f"PROMPT: {prompt.replace('Assistant: ', '')}")
        print(f"OUTPUT: '{answer}'")
        print("\n[TELEMETRY LOGS]")
        telemetry = model.last_telemetry
        print(f"Sequence Entropy:  {telemetry.get('entropy', 0):.4f}")
        gate = telemetry.get('gate_score', 0)
        
        # Soft-Body Mass routing logic
        # Path 0 is always 1.0. Paths 1-3 are gated by gate_score.
        paths_active = 1 + (3 * gate)
        print(f"Gate Score:        {gate:.4f} (Activating {paths_active:.2f} parallel paths)")
        
        gates = telemetry.get('autotomic_gates', [])
        print(f"Autotomic Pruning: {[round(g, 4) for g in gates]}")
        print("-" * 65 + "\n")

    prompts = [
        "User: Say hello.\nAssistant: ",
        "User: Write a Python script to calculate the Fibonacci sequence.\nAssistant: ",
        "User: A is True. B is False. C is A AND B. What is C? Think step by step.\nAssistant: ",
        "User: x = 10. y = 5. x, y = y, x. What is x?\nAssistant: ",
        "User: Output a JSON block with three random colors, and nothing else.\nAssistant: "
    ]
    
    for prompt in prompts:
        generate_and_log(prompt)

if __name__ == "__main__":
    run_benchmark()
