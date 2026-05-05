from datasets import load_dataset
import os
# Set your local path here
local_path = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/Datasets" 
datasets = ["Skylion007/openwebtext"]
# This will download and prepare the dataset in the specified folder
for dataset_name in datasets:
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(os.path.join(local_path, dataset_name.split("/")[1]))
    print("Dataset downloaded and stored at:", local_path)