import torch
from mamba3_mimo_builder import Mamba3MIMORLF
from transformers import AutoTokenizer

def run_baremetal_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Mamba 3 MIMO Bare-Metal Knowledge Benchmark...")
    
    model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
    model.load_state_dict(torch.load("jarvis_v4.pth", map_location=device)['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    prompts = [
        "User: Explain the 5 Organic Laws of the Operating Organism D+ Policy Engine and how they affect module actions.\nAssistant:",
        "User: Implement a module that allocates a buffer for KV Cache in the bare-metal environment.\nAssistant:",
        "User: Integrate a new inference engine using the official OO Mamba Bridge interface.\nAssistant:",
        "User: Save a new memory state to disk using the bare-metal NeuralFS.\nAssistant:",
        "User: List the commands used to evaluate and apply the Halt Policy in the OO Runtime REPL.\nAssistant:"
    ]

    print("=================================================================")
    print("  Bare-Metal (Operating Organism) Architecture Benchmark         ")
    print("=================================================================")

    for i, prompt in enumerate(prompts):
        print(f"\n[Test {i+1}]")
        print(f"PROMPT: {prompt.strip()}")
        inputs = torch.tensor([tokenizer.encode(prompt)]).to(device)
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=80, temperature=0.1, top_k=3)
        response = tokenizer.decode(out[0].tolist()[len(inputs[0]):])
        print(f"OUTPUT: '{response.strip()}'")
        print("-" * 65)

if __name__ == "__main__":
    run_baremetal_benchmark()
