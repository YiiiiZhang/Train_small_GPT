from datasets import load_dataset

# Set your local path here
local_path = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/Datasets" 
datasets = ["roneneldan/TinyStories", "Skylion007/openwebtext"]
# This will download and prepare the dataset in the specified folder
for dataset_name in datasets:
    dataset = load_dataset(dataset_name, cache_dir=local_path)
    print("Dataset downloaded and stored at:", local_path)