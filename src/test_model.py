import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import re

# Load the trained model
model_path = "./qa_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Read the test data set
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    "context": [d["context"] for d in data],
    "question": [d["question"] for d in data],
    "answers": [{"text": [d["answer"]]} for d in data]
})

# Mix data set (same seed used in training)
dataset = dataset.shuffle(seed=42)

# Use the same split used in training
n = len(dataset)
train_size = int(n*0.8)
val_size = int(n*0.1)
test_size = n - train_size - val_size

# Get only the test data set
test_dataset = dataset.select(range(train_size+val_size, n))

# Prediction function
def get_prediction(question, context):
    """
    Generates an answer prediction for a given question and context using the trained model.
    
    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.
    
    Returns:
        str: The predicted answer.
    """
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)
    answer_tokens = inputs["input_ids"][0][start_idx : end_idx+1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

def normalize_answer(s):
    """
    Normalizes the answer by removing punctuation and extra spaces, and converting to lowercase.
    
    Args:
        s (str): The string to normalize.
    
    Returns:
        str: The normalized string.
    """
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)  # Clear punctuation
    s = re.sub(r"\s+", " ", s)  # Clear extra spaces
    return s

def f1_score(prediction, truth):
    """
    Calculates the F1 score between the predicted and true answers.
    
    Args:
        prediction (str): The predicted answer.
        truth (str): The true answer.
    
    Returns:
        float: The F1 score.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        # In case of empty answer f1
        return 1.0 if pred_tokens == truth_tokens else 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def exact_match_score(prediction, truth):
    """
    Checks if the predicted answer exactly matches the true answer.
    
    Args:
        prediction (str): The predicted answer.
        truth (str): The true answer.
    
    Returns:
        bool: True if the answers match exactly, False otherwise.
    """
    return normalize_answer(prediction) == normalize_answer(truth)

# Predict and calculate metrics on the test data set
em_scores = []
f1_scores = []

for example in test_dataset:
    context = example["context"]
    question = example["question"]
    truth = example["answers"]["text"][0]

    prediction = get_prediction(question, context)
    em = exact_match_score(prediction, truth)
    f1 = f1_score(prediction, truth)
    em_scores.append(em)
    f1_scores.append(f1)

em_result = np.mean(em_scores) * 100
f1_result = np.mean(f1_scores) * 100

print(f"Exact Match (EM): {em_result:.2f}")
print(f"F1 Score: {f1_result:.2f}")
