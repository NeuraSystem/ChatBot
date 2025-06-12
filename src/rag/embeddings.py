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

    @lru_cache(maxsize=128)
    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            self.cache.add(text, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error generando embedding: {str(e)}")
            raise
            
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=32
            )
            for text, embedding in zip(texts, embeddings):
                self.cache.add(text, embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error generando embeddings: {str(e)}")
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