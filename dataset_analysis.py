from datasets import load_dataset

ds = load_dataset("rotten_tomatoes")

print(ds)
print(ds["train"])
print(set(ds["train"]["label"]))