# src/rag/embeddings.py

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from threading import Lock

from .logging_config import logger
from src.config import GlobalConfig as Config


class EmbeddingCache:
    """
    Caché LRU para embeddings.
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: Dict[str, np.ndarray] = {}
        self.size = 0
        self.lock = Lock()
        
    def get(self, key: str) -> Optional[np.ndarray]:
        with self.lock:
            return self.cache.get(key)
            
    def add(self, key: str, embedding: np.ndarray) -> None:
        with self.lock:
            new_size = embedding.nbytes
            if self.size + new_size > self.max_size:
                while self.size + new_size > self.max_size and self.cache:
                    oldest_key = next(iter(self.cache))
                    self.size -= self.cache[oldest_key].nbytes
                    del self.cache[oldest_key]
            self.cache[key] = embedding
            self.size += new_size
            
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.size = 0

class EmbeddingGenerator:
    """
    Generador de embeddings usando sentence-transformers.
    """
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.cache = EmbeddingCache(config.CACHE_MAX_SIZE)
        logger.info(f"Inicializado EmbeddingGenerator con modelo {config.EMBEDDING_MODEL}")
        
    # =============================================================================
    # CAMBIO: Se añade el método 'get_model' que faltaba.
    # Esto soluciona el "AttributeError: 'EmbeddingGenerator' object has no attribute 'get_model'".
    # =============================================================================
    def get_model(self) -> SentenceTransformer:
        """Retorna la instancia del modelo de SentenceTransformer."""
        return self.model

    # Renamed from generate_embedding and added .tolist()
    def embed_query(self, text: str) -> List[float]:
        try:
            # @lru_cache cannot be directly applied to methods if 'self' is part of cache key implicitly.
            # For simplicity in this step, removing lru_cache, can be added back carefully.
            # Consider caching on 'text' if 'self' parameters (like model config) don't change.
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                return cached_embedding.tolist()

            embedding = self.model.encode(text, show_progress_bar=False, normalize_embeddings=self.config.NORMALIZE_EMBEDDINGS)
            self.cache.add(text, embedding)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error en embed_query: {str(e)}")
            raise
            
    # Renamed from generate_embeddings and ensured List[List[float]]
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            # Similar caching consideration as embed_query for individual texts if needed,
            # but batch operation is usually efficient.
            embeddings_np = self.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=self.config.NORMALIZE_EMBEDDINGS
            )
            # Add to cache individually if desired, though batch results might not be cached directly by simple text key
            # For now, let's assume caching happens per text if embed_query was called, or rely on model's internal caching if any.

            # Convert list of numpy arrays to list of lists of floats
            embeddings_list = [emb.tolist() for emb in embeddings_np]
            return embeddings_list
        except Exception as e:
            logger.error(f"Error en embed_documents: {str(e)}")
            raise
            
    def clear_cache(self) -> None:
        self.cache.clear()
        logger.info("Caché de embeddings limpiada")
        
    def get_embedding_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()
        
    def check_model(self) -> bool:
        try:
            _ = self.model.encode("test")
            return True
        except Exception as e:
            logger.error(f"Error al verificar el modelo: {str(e)}")
            return False