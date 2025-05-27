"""
RAG (Retrieval-Augmented Generation) Module

Provides functionality for answering questions based on OCR text.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from ..models.model_manager import ModelManager
from ..utils.text_utils import preprocess_text, split_text_into_chunks, get_top_k_chunks
from ..config.settings import Settings

logger = logging.getLogger(__name__)

class RAGProcessor:
    """
    Question-answering system using Retrieval-Augmented Generation
    """
    
    def __init__(self, model_manager: Optional[ModelManager] = None, settings: Optional[Settings] = None):
        """
        Initialize the RAG processor
        
        Args:
            model_manager: Model manager instance (optional)
            settings: Settings instance (optional)
        """
        self.settings = settings or Settings()
        self.model_manager = model_manager or ModelManager(self.settings)
    
    def process_query(self, text: str, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a query against the extracted text
        
        Args:
            text: The text to query against
            query: The question to answer
            k: Number of top chunks to consider
            
        Returns:
            Dictionary with answer and metadata
        """
        # Check if models are available
        qa_model = self.model_manager.get_qa_model()
        sentence_transformer = self.model_manager.get_sentence_transformer()
        
        # If both models are available, use embedding-based RAG
        if qa_model and sentence_transformer:
            return self._embedding_based_qa(text, query, qa_model, sentence_transformer, k)
        
        # Fallback to simpler approach if models are not available
        return self._fallback_qa(text, query, k)
    
    def _embedding_based_qa(self, text: str, query: str, 
                           qa_model: Any, sentence_transformer: Any, k: int) -> Dict[str, Any]:
        """
        Use embeddings for retrieval and transformer model for answering
        
        Args:
            text: Source text
            query: Question to answer
            qa_model: Question-answering model
            sentence_transformer: Sentence transformer model
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Split text into chunks
            chunks = split_text_into_chunks(text)
            if not chunks:
                return {
                    "answer": "No text available to answer the question.",
                    "confidence": 0.0,
                    "chunks_used": []
                }
            
            # Encode query and chunks
            query_embedding = sentence_transformer.encode([query])[0]
            chunk_embeddings = sentence_transformer.encode(chunks)
            
            # Get top chunks
            top_chunks_with_scores = get_top_k_chunks(
                query, chunks, chunk_embeddings, query_embedding, k
            )
            
            if not top_chunks_with_scores:
                return {
                    "answer": "Couldn't find relevant information to answer the question.",
                    "confidence": 0.0,
                    "chunks_used": []
                }
            
            # Join top chunks for context
            context = " ".join([chunk for chunk, _ in top_chunks_with_scores])
            
            # Get answer from QA model
            result = qa_model(question=query, context=context)
            
            return {
                "answer": result["answer"],
                "confidence": result["score"],
                "chunks_used": [chunk for chunk, score in top_chunks_with_scores],
                "chunk_scores": [score for _, score in top_chunks_with_scores]
            }
            
        except Exception as e:
            logger.error(f"Error in embedding-based QA: {e}")
            return self._fallback_qa(text, query, k)
    
    def _fallback_qa(self, text: str, query: str, k: int) -> Dict[str, Any]:
        """
        Simple keyword-based answering when ML models are unavailable
        
        Args:
            text: Source text
            query: Question to answer
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Split text into chunks
            chunks = split_text_into_chunks(text)
            if not chunks:
                return {
                    "answer": "No text available to answer the question.",
                    "confidence": 0.0,
                    "chunks_used": []
                }
            
            # Get top chunks using keyword search
            top_chunks_with_scores = get_top_k_chunks(query, chunks, k=k)
            
            if not top_chunks_with_scores:
                return {
                    "answer": "Couldn't find relevant information to answer the question.",
                    "confidence": 0.0,
                    "chunks_used": []
                }
            
            # For simple QA, use the most relevant chunk as the answer
            best_chunk, best_score = top_chunks_with_scores[0]
            
            # Try to extract a more precise answer
            answer = self._extract_answer_from_context(query, best_chunk)
            
            return {
                "answer": answer,
                "confidence": best_score,
                "chunks_used": [chunk for chunk, _ in top_chunks_with_scores],
                "chunk_scores": [score for _, score in top_chunks_with_scores]
            }
            
        except Exception as e:
            logger.error(f"Error in fallback QA: {e}")
            return {
                "answer": f"Sorry, I couldn't answer that question due to an error: {str(e)}",
                "confidence": 0.0,
                "chunks_used": []
            }
    
    def _extract_answer_from_context(self, query: str, context: str) -> str:
        """
        Extract the most likely answer from context based on the query
        
        Args:
            query: The question
            context: Text context
            
        Returns:
            Extracted answer
        """
        import re
        
        # Clean query and convert to lowercase for matching
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', '', query)
        
        # Check for common question types
        who_match = re.search(r'\bwho\b', query)
        what_match = re.search(r'\bwhat\b', query)
        when_match = re.search(r'\bwhen\b', query)
        where_match = re.search(r'\bwhere\b', query)
        why_match = re.search(r'\bwhy\b', query)
        how_match = re.search(r'\bhow\b', query)
        
        # Extract key terms from the query (excluding common words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'by', 'to', 'for', 'with', 'about'}
        query_terms = [term for term in query.split() if term.lower() not in stop_words]
        
        # Find sentences in the context that contain query terms
        sentences = re.split(r'(?<=[.!?])\s+', context)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                relevant_sentences.append(sentence)
        
        # If no relevant sentences found, return the context
        if not relevant_sentences:
            return context
        
        # For specific question types, try to extract specific information
        if who_match:
            # Look for names in the relevant sentences
            for sentence in relevant_sentences:
                # Simple name detection - capitalized words not at start of sentence
                name_matches = re.findall(r'(?<!^)(?<![\.\!\?]\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', sentence)
                if name_matches:
                    return name_matches[0]
        
        elif when_match:
            # Look for dates or time expressions
            for sentence in relevant_sentences:
                date_matches = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b|\b\d{4}\b', sentence)
                if date_matches:
                    return date_matches[0]
        
        # Default: return the most relevant sentence
        if relevant_sentences:
            # Sort by number of query terms contained
            sentence_scores = [
                sum(1 for term in query_terms if term in sentence.lower())
                for sentence in relevant_sentences
            ]
            best_sentence = relevant_sentences[sentence_scores.index(max(sentence_scores))]
            return best_sentence.strip()
        
        # Fallback to the first part of the context
        return context.split('.')[0] + '.'
