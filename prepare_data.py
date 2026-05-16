import os
from datasets import load_dataset

def cache_fineweb():
    """
    Downloads and caches the FineWeb-Edu 10BT sample to the 2TB HDD.
    This prevents network bottlenecking during the high-throughput Mamba 3 training.
    """
    cache_dir = "/hdd_data/mamba_training_data/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Starting massive download of FineWeb-Edu (10BT) to {cache_dir}...")
    print("This may take a while depending on your network speed.")
    
    # Downloading without streaming=True saves it directly to the cache_dir
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train", 
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    print(f"Download complete! Cached {len(ds)} base corpus samples to the 2TB drive.")

if __name__ == "__main__":
    cache_fineweb()
