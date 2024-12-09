'''
This code is for dataset downloading and splitting
'''
from datasets import load_dataset
import json
from sklearn.model_selection import train_test_split

# 1. Load dataset from Hugging Face Hub
ds = load_dataset("DFKI-SLT/OptimAL")

print(ds)

# Save each subset (e.g., "train", "test") as a separate JSON file
for split in ds.keys():  # ds.keys() contains subset names like "train", "test"
    ds[split].to_json(f"{split}_data.json")


# 2. Define a function to convert samples to the Rebel format
def convert_to_rebel_format(sample):
    """
    Convert a dataset sample to the Rebel triplet format.

    """
    drug_name = sample["drug_name"]
    disease_name = sample["disease_name"]
    relation = sample["Worker Answer"]
    context = sample["context"]

    # Create a triplet in Rebel format
    triplet = f"<triplet> {drug_name} <subj> {disease_name} <obj> {relation}"

    return {
        "id": str(sample["_unit_id"]),  # Unique identifier
        "title": drug_name,
        "context": context,
        "triplets": triplet
    }


# 3. Read JSON file containing raw data
with open("train_data.json", "r", encoding="utf-8") as file:
    raw_data = [json.loads(line) for line in file]

# 4. Apply the conversion function to all samples
formatted_data = [convert_to_rebel_format(sample) for sample in raw_data]

# 5. Split the formatted data into training and testing sets
train_data, test_data = train_test_split(formatted_data, test_size=0.35, random_state=42)

# 6. Save the training and testing datasets to JSON files
with open("train_rebel.json", "w", encoding="utf-8") as train_file:
    for entry in train_data:
        train_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open("test_rebel.json", "w", encoding="utf-8") as test_file:
    for entry in test_data:
        test_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
