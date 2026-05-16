from datasets import load_dataset
ds = load_dataset("hellaswag", split="train", streaming=False)
sample = next(iter(ds))
print(sample.keys())
print(sample)
