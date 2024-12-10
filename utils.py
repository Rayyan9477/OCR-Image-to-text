import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_text_chunks(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_top_k_chunks(text, query, k=3):
    chunks = get_text_chunks(text)
    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])
    
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    
    D, I = index.search(query_embedding, k)
    
    return [chunks[i] for i in I[0]]