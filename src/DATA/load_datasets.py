from datasets import load_dataset, DatasetDict
import os
# Set your local path here
local_path = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/Datasets" 
datasets = ["Skylion007/openwebtext","roneneldan/TinyStories"]
# This will download and prepare the dataset in the specified folder
for dataset_name in datasets:
    ds = load_dataset(dataset_name)
    dataset_dir = os.path.join(local_path, dataset_name.split("/")[1])
    if "validation" in ds:
        new_ds = DatasetDict({
            "train": ds["train"],
            "val": ds["validation"]
        })
    else:
        split_ds = ds['train'].train_test_split(test_size=0.1, seed=42)
        new_ds = DatasetDict({
            "train": split_ds["train"],
            "val": split_ds["test"]
        })
    new_ds.save_to_disk(dataset_dir,max_shard_size="2GB")
    print(f"Dataset '{dataset_name}' saved with train/val splits at: {dataset_dir}")