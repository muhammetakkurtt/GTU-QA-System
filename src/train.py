import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch
import random
import os
import shutil

with open("qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# List to hold valid answers
valid_data = []

# Find start and end indices of answer text in context for each example
for d in data:
    context = d["context"]
    answer = d["answer"]
    start_idx = context.find(answer)
    
    if start_idx != -1:  # Only take examples where answer can be found in context
        end_idx = start_idx + len(answer)
        d["start_char"] = start_idx
        d["end_char"] = end_idx
        valid_data.append(d)

# Use valid_data instead of original data
data = valid_data

# Create Dataset
dataset = Dataset.from_dict({
    "context": [d["context"] for d in data],
    "question": [d["question"] for d in data],
    "answers": [{"text": [d["answer"]], "answer_start": [d["start_char"]]} for d in data]
})

# Shuffle dataset
dataset = dataset.shuffle(seed=42)

# Split into Train / Validation / Test (80/10/10)
n = len(dataset)
train_size = int(n*0.8)
val_size = int(n*0.1) 
test_size = n - train_size - val_size

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size+val_size))
test_dataset = dataset.select(range(train_size+val_size, n))

# Initialize BERT tokenizer and model for Turkish language
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def prepare_features(examples):
    """
    Tokenizes the input examples and prepares start and end positions for the answers.
    
    Args:
        examples (dict): A dictionary containing 'question', 'context', and 'answers'.
    
    Returns:
        dict: A dictionary with tokenized inputs and start/end positions.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
        answer_start = examples["answers"][i]["answer_start"][0]
        answer_text = examples["answers"][i]["text"][0]
        answer_end = answer_start + len(answer_text)
        
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find context segment
        if 1 not in sequence_ids:
            # Fallback if context not found
            start_positions.append(0)
            end_positions.append(0)
            continue
        context_start = sequence_ids.index(1)
        # Search from reverse to find context end
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If answer is outside context area, mark as no_answer
        if not (offsets[context_start][0] <= answer_start and offsets[context_end][1] >= answer_end):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Answer token indices
            start_token_idx = 0
            end_token_idx = 0
            for idx in range(context_start, context_end + 1):
                if offsets[idx][0] <= answer_start < offsets[idx][1]:
                    start_token_idx = idx
                if offsets[idx][0] < answer_end <= offsets[idx][1]:
                    end_token_idx = idx
                    break
            start_positions.append(start_token_idx)
            end_positions.append(end_token_idx)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    
    tokenized_examples.pop("offset_mapping")
    return tokenized_examples

# Tokenize datasets
train_tokenized = train_dataset.map(prepare_features, batched=True, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(prepare_features, batched=True, remove_columns=val_dataset.column_names)
test_tokenized = test_dataset.map(prepare_features, batched=True, remove_columns=test_dataset.column_names)

# Load pre-trained model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create output directory if it doesn't exist
output_dir = "./qa_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure training arguments for fine-tuning
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    learning_rate=3e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    push_to_hub=False,
    logging_dir=os.path.join(output_dir, "logs"),
    report_to=["none"]
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer
)
# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate(test_tokenized)
print("Test Results:", metrics)

# Save the model and tokenizer
model.save_pretrained("./qa_model")
tokenizer.save_pretrained("./qa_model")

print("Training completed and model saved.")
