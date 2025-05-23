import logging
from utils import get_top_k_chunks
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_query(text, query, k=5):
    """
    Process a query against the extracted text using the QA model.
    Falls back to simpler methods if the model isn't available.
    """
    # Get relevant chunks
    chunks = get_top_k_chunks(text, query, k=k)
    
    if not chunks:
        return {
            "answer": "No relevant information found in the document.",
            "confidence": 0.0,
            "context": ""
        }

    # Combine chunks into context
    context = " ".join(chunks)

    # Limit context length to prevent issues with transformer models
    max_context_length = 512  # Adjust if necessary
    context = context[:max_context_length]

    if not context.strip():
        return {
            "answer": "Context is empty or invalid.",
            "confidence": 0.0,
            "context": ""
        }
    
    try:
        # Try to use the QA model from model_manager
        from model_manager import get_qa_model
        qa_model = get_qa_model()
        
        if qa_model is not None:
            result = qa_model(question=query, context=context)
            return {
                "answer": result.get("answer", "No answer found."),
                "confidence": float(result.get("score", 0.0)),
                "context": context
            }
    except Exception as e:
        logger.error(f"Error using QA model: {str(e)}")
    
    # Fallback to keyword-based answer extraction
    return fallback_answer_extraction(query, context)

def fallback_answer_extraction(query, context):
    """
    A simple fallback method when the QA model is not available.
    Uses keyword matching to find a relevant sentence.
    """
    try:
        # Split context into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Clean and lowercase the query
        query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
        
        # Score sentences by the number of query words they contain
        scored_sentences = []
        for sentence in sentences:
            clean_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
            score = sum(1 for word in query_words if word in clean_sentence)
            scored_sentences.append((sentence, score))
        
        # Sort by score, descending
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Return top sentence if it has at least one matching word
        if scored_sentences and scored_sentences[0][1] > 0:
            return {
                "answer": scored_sentences[0][0],
                "confidence": min(scored_sentences[0][1] / max(1, len(query_words)), 1.0),
                "context": context
            }
    except Exception as e:
        logger.error(f"Error in fallback answer extraction: {str(e)}")
    
    # Last resort
    return {
        "answer": "Could not generate an answer. Please check if the document contains relevant information.",
        "confidence": 0.0,
        "context": context
    }