# src/rag/retriever.py

import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore as LangChainVectorStore

from .logging_config import logger
from src.config import GlobalConfig as Config
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

class BaseRetriever(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def get_retriever(self) -> LangChainVectorStore:
        pass

    @abstractmethod
    def clear_documents(self) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

class RAGRetriever(BaseRetriever):
    """
    Sistema de búsqueda semántica usando RAG.
    Actúa como gestor para el VectorStore de Chroma.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = EmbeddingGenerator(config)
        # =============================================================================
        # NOTA: Esta línea, que antes fallaba, ahora funcionará porque hemos
        # corregido el __init__ de VectorStore para que acepte el modelo.
        # =============================================================================
        self.vector_store_manager = VectorStore(config, self.embeddings.get_model())
        
        if not self.embeddings.check_model():
            raise RuntimeError("Error al inicializar el modelo de embeddings")
            
        logger.info("RAGRetriever inicializado y listo")
        
    def add_documents(self, documents: List[Document]) -> None:
        try:
            self.vector_store_manager.add_documents(documents)
            logger.info(f"Agregados {len(documents)} documentos al sistema")
        except Exception as e:
            logger.error(f"Error agregando documentos: {str(e)}", exc_info=True)
            raise

    def get_retriever(self) -> LangChainVectorStore:
        try:
            chroma_instance = self.vector_store_manager.get_chroma_instance()
            return chroma_instance.as_retriever()
        except Exception as e:
            logger.error(f"Error al crear el retriever: {e}", exc_info=True)
            raise

    def clear_documents(self) -> None:
        try:
            self.vector_store_manager.clear_collection()
            logger.info("Colección de documentos limpiada")
        except Exception as e:
            logger.error(f"Error limpiando documentos: {str(e)}", exc_info=True)
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        try:
            return self.vector_store_manager.get_collection_stats()
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}", exc_info=True)
            raise