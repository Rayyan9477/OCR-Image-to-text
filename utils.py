from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def get_text_chunks(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def get_top_k_chunks(text, query, k=3):
    chunks = get_text_chunks(text)
    
    if not chunks:
        return []

    # Embed the query and chunks
    query_embedding = model.encode([query])
    chunk_embeddings = model.encode(chunks)

    # Use FAISS to find the top k similar chunks
    index = faiss.IndexFlatL2(query_embedding.shape[1])
    index.add(np.array(chunk_embeddings))
    distances, indices = index.search(np.array(query_embedding), k)

    top_k_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    return top_k_chunks