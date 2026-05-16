from datasets import load_dataset
ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
sample = next(iter(ds))
print(sample.keys())
print(sample['conversations'][0:2])
