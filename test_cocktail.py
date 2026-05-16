from datasets import load_dataset

try:
    ds_gsm = load_dataset("gsm8k", "main", split="train", streaming=False)
    print("GSM8K:", len(ds_gsm))
except: print("GSM8K failed")

try:
    ds_arc = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=False)
    print("ARC:", len(ds_arc))
except: print("ARC failed")

try:
    ds_hella = load_dataset("hellaswag", split="train", streaming=False)
    print("HellaSwag:", len(ds_hella))
except: print("HellaSwag failed")

try:
    ds_wino = load_dataset("winogrande", "winogrande_xl", split="train", streaming=False)
    print("Winogrande:", len(ds_wino))
except: print("Winogrande failed")

