from transformers import pipeline
from utils import get_top_k_chunks
import numpy as np

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def process_query(text, query, k=5):
    chunks = get_top_k_chunks(text, query, k=k)
    
    if not chunks:
        return {
            "answer": "No relevant information found.",
            "confidence": 0.0,
            "context": ""
        }

    context = " ".join(chunks)

    # Limit context length
    max_context_length = 512  # Adjust if necessary
    context = context[:max_context_length]

    if not context.strip():
        return {
            "answer": "Context is empty or invalid.",
            "confidence": 0.0,
            "context": ""
        }

    result = qa_model(question=query, context=context)
    return {
        "answer": result.get("answer", "No answer found."),
        "confidence": float(result.get("score", 0.0)),
        "context": context
    }