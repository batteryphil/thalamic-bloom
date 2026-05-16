import torch
import time
import subprocess

print("Starting artificial GPU load (Matrix Multiplications)...")
size = 10000
try:
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
except RuntimeError as e:
    print(f"OOM, reducing size...")
    size = 4096
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

start_time = time.time()
last_print = time.time()

print("Time: 0s | Heating up...")
while time.time() - start_time < 45: # 45 seconds of load
    # Heavy compute
    for _ in range(50):
         c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    if time.time() - last_print >= 5:
        try:
            temp = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                text=True
            ).strip()
            power = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader'],
                text=True
            ).strip()
            print(f"Time: {int(time.time() - start_time)}s | Temp: {temp}°C | Power: {power}")
        except Exception:
            pass
        last_print = time.time()

print("Load test complete.")
