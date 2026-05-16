import torch
from mamba3_mimo_builder import Mamba3MIMORLF
import sys
from transformers import AutoTokenizer

def evaluate(prompt: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
    model.to(device)
    
    print("Loading checkpoint...")
    try:
        checkpoint = torch.load("jarvis_v3.pth", map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print("Checkpoint not found. Proceeding with initialized weights.")
        
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # Encode prompt using the subword tokenizer
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    print(f"\n--- PROMPT ---\n{prompt}\n--------------")
    print("\nGenerating...")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=150,
            temperature=0.3,
            top_k=5,
            stop_sequences={0, 198, 200}
        )
        
    # Decode
    out_tokens = output[0].tolist()
    # The first tokens are the prompt itself
    generated_tokens = out_tokens[len(input_ids[0]):]
    
    decoded = tokenizer.decode(generated_tokens)
            
    print(f"\n--- RESPONSE ---\n{decoded}\n----------------\n")

if __name__ == "__main__":
    prompt = "def calculate_sum(a, b):\n" if len(sys.argv) < 2 else sys.argv[1]
    evaluate(prompt)
