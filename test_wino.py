from datasets import load_dataset
ds = load_dataset("winogrande", "winogrande_xl", split="train", streaming=False)
sample = next(iter(ds))
print(sample.keys())
print(sample)
