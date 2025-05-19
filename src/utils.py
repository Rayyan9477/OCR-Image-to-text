import numpy as np
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_text_chunks(text, chunk_size=100, overlap=20):
    """Split text into chunks with overlap for better context preservation"""
    if not text:
        return []
        
    words = text.split()
    if not words:
        return []
        
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:  # Avoid empty chunks
            chunks.append(chunk)
    return chunks

def get_top_k_chunks(text, query, k=3):
    """Find the most relevant text chunks for a query using semantic search"""
    chunks = get_text_chunks(text)
    
    if not chunks:
        return []
    
    try:
        # Try to use FAISS and SentenceTransformer for semantic search
        import faiss
        from model_manager import get_sentence_transformer
        
        # Get the sentence transformer model from model_manager
        model = get_sentence_transformer()
        if not model:
            return fallback_keyword_search(chunks, query, k)
        
        # Embed the query and chunks
        query_embedding = model.encode([query])
        chunk_embeddings = model.encode(chunks)

        # Use FAISS to find the top k similar chunks
        index = faiss.IndexFlatL2(query_embedding.shape[1])
        index.add(np.array(chunk_embeddings))
        distances, indices = index.search(np.array(query_embedding), k)

        top_k_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
        return top_k_chunks
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        # Fallback to basic keyword search
        return fallback_keyword_search(chunks, query, k)

def fallback_keyword_search(chunks, query, k=3):
    """Fallback keyword-based search method when semantic search is not available"""
    # Clean and tokenize the query
    query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    
    # Score chunks by the number of query words they contain
    scored_chunks = []
    for chunk in chunks:
        clean_chunk = re.sub(r'[^\w\s]', '', chunk.lower())
        chunk_words = set(clean_chunk.split())
        # Calculate score based on word overlap
        score = sum(1 for word in query_words if word in chunk_words)
        scored_chunks.append((chunk, score))
    
    # Sort by score, descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k chunks
    return [chunk for chunk, score in scored_chunks[:k] if score > 0]

def clean_extracted_text(text):
    """Clean and normalize extracted text"""
    if not text:
        return ""
        
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'(\r\n|\r|\n){3,}', '\n\n', text)
    
    return text.strip()

def detect_language(text):
    """Detect the primary language of the text"""
    if not text or len(text) < 20:
        return 'en'  # Default to English for short or empty texts
        
    try:
        from langdetect import detect
        return detect(text)
    except:
        # If langdetect is not available or fails, check for common non-Latin character sets
        # Chinese
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        # Japanese
        elif re.search(r'[\u3040-\u30ff]', text):
            return 'ja'
        # Korean
        elif re.search(r'[\uac00-\ud7af]', text):
            return 'ko'
        # Arabic
        elif re.search(r'[\u0600-\u06ff]', text):
            return 'ar'
        # Russian/Cyrillic
        elif re.search(r'[\u0400-\u04ff]', text):
            return 'ru'
        # Default to English
        return 'en'