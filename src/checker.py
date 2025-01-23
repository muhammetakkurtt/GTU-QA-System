import json
from datasets import Dataset

# Load the QA dataset from a JSON file
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Iterate over each item in the dataset
for item in qa_data:
    # Check if the answer is present in the context
    if item["context"].find(item["answer"]) == -1:
        print(f"Hatalı veri tespit edildi: {item}")

