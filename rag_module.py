from transformers import pipeline
from utils import get_top_k_chunks

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def process_query(text, query):
    chunks = get_top_k_chunks(text, query, k=3)
    context = " ".join(chunks)
    
    result = qa_model(question=query, context=context)
    return result["answer"]