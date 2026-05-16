import sys
import time
import torch
from transformers import AutoTokenizer
from mamba3_mimo_builder import Mamba3MIMORLF

def test_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Mamba 3 MIMO Model for Benchmarking...")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
    model.to(device)
    
    try:
        checkpoint = torch.load("jarvis_v4.pth", map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Loaded checkpoint: jarvis_v4.pth")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.eval()
    
    stop_sequences = {0, 198, 200}
    
    def generate_answer(prompt, ablation=False, max_tokens=20):
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        # Monkeypatch CP for ablation
        original_cp_forward = model.cp.forward
        if ablation:
            model.cp.forward = lambda x: torch.zeros_like(original_cp_forward(x))
            
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_k=5,
                stop_sequences=stop_sequences
            )
            
        # Restore CP
        if ablation:
            model.cp.forward = original_cp_forward
            
        generated_tokens = output[0].tolist()[len(input_ids[0]):]
        return tokenizer.decode(generated_tokens).strip()

    print("\n" + "="*65)
    print("  1. General Knowledge Test")
    print("="*65)
    
    gk_prompts = [
        ("What is the capital of France?", "Paris"),
        ("Calculate 12 * 12.", "144"),
        ("Who wrote Romeo and Juliet?", "Shakespeare")
    ]
    
    for prompt, expected in gk_prompts:
        ans = generate_answer(prompt)
        print(f"Q: {prompt}")
        print(f"Expected: {expected} | Got: '{ans}'")
        print("-" * 40)
        
    print("\n" + "="*65)
    print("  2. Needle In A Haystack (NIAH)")
    print("="*65)
    
    distractor = "The sky is blue. The grass is green. Water boils at 100 degrees Celsius. " * 50
    needle = "The secret password is 'Antigravity'. "
    distractor2 = "Apples are red. Bananas are yellow. Oranges are orange. " * 50
    niah_prompt = distractor + needle + distractor2 + "\nWhat is the secret password?"
    
    ans = generate_answer(niah_prompt, max_tokens=30)
    print(f"Prompt Length: {len(tokenizer.encode(niah_prompt))} tokens")
    print(f"Expected: Antigravity | Got: '{ans}'")
    
    print("\n" + "="*65)
    print("  3. Scratchpad Ablation Test")
    print("="*65)
    
    ablation_prompts = [
        ("A=42. B=A. C=B. What is C?", "42"),
        ("V1=99. V2=V1. V3=V2. What is V3?", "99"),
        ("x=7. y=x. z=y. What is z?", "7")
    ]
    
    normal_correct = 0
    ablated_correct = 0
    
    for prompt, expected in ablation_prompts:
        ans_normal = generate_answer(prompt, ablation=False)
        ans_ablated = generate_answer(prompt, ablation=True)
        
        print(f"Prompt: {prompt}")
        print(f"Normal Output: '{ans_normal}'")
        print(f"Ablated Output: '{ans_ablated}'")
        print("-" * 40)
        
        if expected in ans_normal:
            normal_correct += 1
        if expected in ans_ablated:
            ablated_correct += 1
            
    print(f"Normal Accuracy:  {normal_correct}/{len(ablation_prompts)}")
    print(f"Ablated Accuracy: {ablated_correct}/{len(ablation_prompts)}")
    print(f"Δ Scratchpad Contribution: {normal_correct - ablated_correct}")

if __name__ == "__main__":
    test_benchmark()
