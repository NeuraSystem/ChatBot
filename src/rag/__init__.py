"""
RAG (Retrieval Augmented Generation) Module

This module implements the RAG system for the chatbot, providing document
loading, embedding generation, vector storage, and semantic search capabilities.

Components:
- Document loading and processing
- Embedding generation using sentence-transformers
- Vector storage using ChromaDB
- Semantic search and retrieval
"""

from .document_loader import DocumentLoader
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
# CAMBIO: Importamos las clases correctas del archivo retriever.py
from .retriever import RAGRetriever, BaseRetriever
from .chat_history import ChatHistory

__all__ = [
    'DocumentLoader',
    'EmbeddingGenerator',
    'VectorStore',
    'RAGRetriever',     # CAMBIO: Exportamos el nombre correcto de la clase
    'BaseRetriever',    # CAMBIO: También exportamos la clase base para que esté disponible
    'ChatHistory'
]