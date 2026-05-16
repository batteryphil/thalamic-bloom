from datasets import load_dataset
ds = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=False)
sample = next(iter(ds))
print(sample.keys())
print(sample)
