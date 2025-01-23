from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch
import json

# Load pre-trained QA model and tokenizer from local path
model_path = "../qa_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
# Initialize sentence transformer model for semantic search
embedding_model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

# Initialize FastAPI application
app = FastAPI()

# CORS settings for allowing cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for question endpoint
class QuestionRequest(BaseModel):
    question: str

# Load QA dataset from JSON file
with open('../qa_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Extract contexts and create combined texts for semantic search
questions = [item["question"] for item in dataset]
contexts = [item["context"] for item in dataset]

# Calculate question and context embeddings separately
question_embeddings = embedding_model.encode(questions, convert_to_tensor=True)
context_embeddings = embedding_model.encode(contexts, convert_to_tensor=True)

def get_top_contexts(similarities, n=2):
    """
    Get top N most similar contexts based on similarity scores
    Returns None if no context meets the minimum similarity threshold
    """
    top_k_values, top_k_idx = torch.topk(similarities, k=min(n, len(similarities)))
    
    # Filter contexts with similarity score > 0.2
    valid_contexts = []
    for score, idx in zip(top_k_values, top_k_idx):
        if score > 0.2:
            valid_contexts.append(dataset[idx.item()]["context"])
    
    if not valid_contexts:
        return None
        
    return " ".join(valid_contexts)

def get_combined_similarity(question_sim, context_sim, alpha=0.6):
    """
    Combines question and context similarities with weighted merging.
    Args:
        question_sim: Question similarity scores
        context_sim: Context similarity scores
        alpha: Weight for question similarity (range 0-1)
    """
    return alpha * question_sim + (1 - alpha) * context_sim

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Main endpoint for question answering:
    1. Find relevant contexts using semantic search
    2. Use QA model to extract answer from contexts
    """
    question = request.question.strip()
    
    # Calculate question embedding
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    
    # Calculate question and context similarities
    question_similarities = util.pytorch_cos_sim(question_embedding, question_embeddings)[0]
    context_similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)[0]
    
    # Combined similarity score
    combined_similarities = get_combined_similarity(question_similarities, context_similarities)
    
    # Choose the best context
    best_context = get_top_contexts(combined_similarities)
    
    if best_context is None:
        return {"answer": "Bu soru için uygun bir cevap bulamadım."}
    
    # Generate answer using QA model
    inputs = tokenizer(question, best_context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = qa_model(**inputs)

    # Extract answer span from model outputs
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    if end_idx < start_idx:
        return {"answer": "Anlaşılır bir cevap bulunamadı."}

    # Decode answer tokens to text
    answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {
        "question": question,
        "context": best_context,
        "answer": answer
    }