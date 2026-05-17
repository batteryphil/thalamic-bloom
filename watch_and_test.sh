#!/bin/bash
# Watches phase5b.log for completion, then auto-runs the bare-metal benchmark

LOG="phase5b.log"
TEST="test_oo_baremetal.py"
RESULT="oo_benchmark_result.txt"

echo "[Watcher] Monitoring $LOG for engram burn completion..."
echo "[Watcher] $(date)"

while true; do
    if grep -q "ENGRAM BURN COMPLETE" "$LOG" 2>/dev/null; then
        echo "[Watcher] ENGRAM BURN COMPLETE detected at $(date)"
        echo "[Watcher] Running bare-metal benchmark..."
        echo ""
        # Load from the engram checkpoint
        python3 -c "
import torch
from mamba3_mimo_builder import Mamba3MIMORLF
from transformers import AutoTokenizer

device = torch.device('cuda')
model = Mamba3MIMORLF(vocab_size=50304, d_model=768, n_layers=24)
ckpt = torch.load('jarvis_v4_oo.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

prompts = [
    'User: Explain the 5 Organic Laws of the Operating Organism D+ Policy Engine and how they affect module actions.\nAssistant:',
    'User: Implement a module that allocates a buffer for KV Cache in the bare-metal environment.\nAssistant:',
    'User: Integrate a new inference engine using the official OO Mamba Bridge interface.\nAssistant:',
    'User: Save a new memory state to disk using the bare-metal NeuralFS.\nAssistant:',
    'User: List the commands used to evaluate and apply the Halt Policy in the OO Runtime REPL.\nAssistant:'
]

print('=================================================================')
print('  OO Bare-Metal Benchmark (Phase 5b Engram Checkpoint)           ')
print('=================================================================')

for i, prompt in enumerate(prompts):
    print(f'\n[Test {i+1}]')
    print(f'PROMPT: {prompt.strip()}')
    ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=100, temperature=0.05, top_k=1)
    response = tokenizer.decode(out[0].tolist()[len(ids[0]):])
    print(f'OUTPUT: {repr(response.strip()[:300])}')
    print('-' * 65)

print('\nBenchmark complete.')
" 2>&1 | tee "$RESULT"
        echo ""
        echo "[Watcher] Results saved to $RESULT"
        break
    fi
    sleep 30
done
