import time
import json
import subprocess
import os

OUTPUT_FILE = "hardware.json"
POLL_INTERVAL = 2

def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            # sysfs returns temp in millidegrees Celsius
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def get_gpu_temp():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        return float(output)
    except Exception:
        return 0.0

if __name__ == "__main__":
    print("Starting Hardware Monitor Daemon...")
    while True:
        cpu_temp = get_cpu_temp()
        gpu_temp = get_gpu_temp()
        
        data = {
            "cpu_temp": cpu_temp,
            "gpu_temp": gpu_temp,
            "timestamp": time.time()
        }
        
        # Write to a temporary file then rename to ensure atomicity
        with open(OUTPUT_FILE + ".tmp", "w") as f:
            json.dump(data, f)
        os.rename(OUTPUT_FILE + ".tmp", OUTPUT_FILE)
        
        time.sleep(POLL_INTERVAL)
