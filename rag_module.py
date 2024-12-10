# rag_module.py
from transformers import pipeline
from utils import get_top_k_chunks
import numpy as np

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def process_query(text, query, k=5):
    chunks = get_top_k_chunks(text, query, k=k)
    context = " ".join(chunks)
    
    result = qa_model(question=query, context=context)
    return {
        "answer": result["answer"],
        "confidence": float(result["score"]),
        "context": context
    }