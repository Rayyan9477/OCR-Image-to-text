"""
Utility functions for the OCR application
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """
    Clean and standardize text for better processing
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
        
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up common OCR artifacts
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
    
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for processing
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    # Clean text first
    text = preprocess_text(text)
    
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    for para in paragraphs:
        # If paragraph is shorter than chunk_size, add it directly
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
            
        # Otherwise, split the paragraph into chunks
        words = para.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Add word length plus space
            word_len = len(word) + 1
            
            # If adding this word exceeds chunk size, store current chunk and start new one
            if current_length + word_len > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = min(overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_size:]
                current_length = sum(len(w) + 1 for w in current_chunk)
            
            # Add word to current chunk
            current_chunk.append(word)
            current_length += word_len
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    return chunks

def get_top_k_chunks(query: str, chunks: List[str], embeddings: Optional[np.ndarray] = None, 
                    query_embedding: Optional[np.ndarray] = None, k: int = 3) -> List[Tuple[str, float]]:
    """
    Get top k chunks most relevant to the query
    
    Args:
        query: Search query
        chunks: List of text chunks
        embeddings: Pre-computed embeddings for chunks (optional)
        query_embedding: Pre-computed embedding for query (optional)
        k: Number of top chunks to retrieve
        
    Returns:
        List of (chunk, score) tuples
    """
    if not chunks:
        return []
    
    if embeddings is None or query_embedding is None:
        # Simple keyword matching if embeddings unavailable
        return _keyword_search(query, chunks, k)
    else:
        # Use embeddings for semantic search
        return _embedding_search(query_embedding, chunks, embeddings, k)

def _keyword_search(query: str, chunks: List[str], k: int) -> List[Tuple[str, float]]:
    """Simple keyword-based search"""
    query = query.lower()
    query_words = set(re.findall(r'\w+', query))
    
    scores = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        
        # Calculate keyword matches
        chunk_words = set(re.findall(r'\w+', chunk_lower))
        word_match_count = len(query_words.intersection(chunk_words))
        
        # Direct phrase matches carry more weight
        phrase_match_score = 0
        for word in query_words:
            if len(word) > 3 and word in chunk_lower:
                phrase_match_score += 1
        
        # Combine scores
        score = word_match_count * 0.5 + phrase_match_score
        scores.append(score)
    
    # Get top k chunks
    if not scores:
        return []
        
    # Get indices of top k scores
    top_indices = np.argsort(scores)[-k:][::-1]
    
    # Create result with scores normalized to 0-1 range
    max_score = max(scores) if max(scores) > 0 else 1
    result = [(chunks[i], scores[i]/max_score) for i in top_indices if scores[i] > 0]
    
    # If we couldn't find any relevant chunks, return top chunks without score filtering
    if not result and chunks:
        return [(chunks[i], 0.1) for i in top_indices[:k]]
    
    return result

def _embedding_search(query_embedding: np.ndarray, chunks: List[str], 
                     chunk_embeddings: np.ndarray, k: int) -> List[Tuple[str, float]]:
    """Search using vector embeddings"""
    # Calculate cosine similarity
    similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
    
    # Get indices of top k similarities
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Create result tuples of (chunk, similarity)
    result = [(chunks[i], float(similarities[i])) for i in top_indices]
    
    return result

def detect_language(text: str) -> str:
    """
    Detect language of text
    
    Args:
        text: Input text
        
    Returns:
        ISO language code or 'en' if detection fails
    """
    if not text or len(text.strip()) < 10:
        return 'en'
    
    try:
        from langdetect import detect
        return detect(text)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return 'en'  # Default to English

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract common entities from text (names, dates, etc.)
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of entity types and values
    """
    entities = {
        'dates': [],
        'emails': [],
        'phones': [],
        'urls': []
    }
    
    # Extract dates (various formats)
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['dates'].extend(matches)
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['emails'] = re.findall(email_pattern, text)
    
    # Extract phone numbers
    phone_pattern = r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'
    entities['phones'] = re.findall(phone_pattern, text)
    
    # Extract URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    entities['urls'] = re.findall(url_pattern, text)
    
    return entities

def format_ocr_result(text: str, format_type: str = 'text') -> str:
    """
    Format OCR result in different formats
    
    Args:
        text: OCR text
        format_type: Format type ('text', 'markdown', 'html')
        
    Returns:
        Formatted text
    """
    if not text:
        return ""
        
    if format_type == 'text':
        return text
    elif format_type == 'markdown':
        # Convert to markdown format
        lines = text.split('\n')
        formatted = []
        
        for line in lines:
            # Make headers for lines that might be titles (all caps, short)
            if line.strip().isupper() and len(line.strip()) < 50 and len(line.strip()) > 3:
                formatted.append(f"## {line}")
            else:
                formatted.append(line)
        
        return '\n'.join(formatted)
    elif format_type == 'html':
        # Convert to simple HTML
        lines = text.split('\n')
        formatted = ['<div class="ocr-text">']
        
        for line in lines:
            if not line.strip():
                formatted.append('<br>')
            else:
                formatted.append(f'<p>{line}</p>')
        
        formatted.append('</div>')
        return '\n'.join(formatted)
    else:
        return text  # Default to plain text
