"""
Reranking module using CrossEncoder for accurate relevance scoring.
After initial retrieval, reranking improves precision by computing exact query-document similarity.
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize CrossEncoder reranker.
        
        Args:
            model_name: HuggingFace model for reranking
                       - ms-marco-MiniLM-L-6-v2: Fast, good quality (default)
                       - ms-marco-MiniLM-L-12-v2: Better quality, slower
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        metadatas: List[Dict],
        distances: List[float],
        top_k: int = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Rerank documents using CrossEncoder.
        
        Args:
            query: Search query
            documents: List of retrieved documents
            metadatas: Metadata for each document
            distances: Original retrieval distances
            top_k: Number of top documents to return (None = all)
            
        Returns:
            Tuple of (reranked_docs, reranked_metas, rerank_scores)
        """
        if not documents:
            return [], [], []
        
        # Create query-document pairs for CrossEncoder
        pairs = [[query, doc] for doc in documents]
        
        # Get reranking scores (higher = more relevant)
        scores = self.model.predict(pairs)
        
        # Sort by scores (descending)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # Apply top_k limit if specified
        if top_k:
            sorted_indices = sorted_indices[:top_k]
        
        # Reorder all lists based on scores
        reranked_docs = [documents[i] for i in sorted_indices]
        reranked_metas = [metadatas[i] for i in sorted_indices]
        rerank_scores = [float(scores[i]) for i in sorted_indices]
        
        return reranked_docs, reranked_metas, rerank_scores
