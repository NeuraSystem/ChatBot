# src/rag/vector_store.py

import os
import logging
from typing import List, Dict, Optional, Any
# CAMBIO: Se importa la clase de documentos de LangChain para la coherencia
from langchain_core.documents import Document

from .logging_config import logger
from src.config import GlobalConfig as Config

# CAMBIO: Se importa Chroma de la nueva librería para usarlo como clase principal
from langchain_chroma import Chroma


class VectorStore:
    """
    Sistema de almacenamiento vectorial usando ChromaDB, integrado con LangChain.
    Esta clase ahora actúa como un wrapper delgado alrededor de la clase Chroma de LangChain.
    """
    
    def __init__(self, config: Config, embedding_function: Any):
        """
        Inicializa el vector store usando la integración de LangChain con Chroma.
        
        Args:
            config: Configuración del sistema.
            embedding_function: La función o modelo de embeddings a utilizar.
        """
        self.config = config
        self.embedding_function = embedding_function
        
        # Se inicializa la instancia de Chroma con toda la configuración necesaria.
        # Esta clase maneja la persistencia y la creación de la colección automáticamente.
        self.db = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        )
        logger.info(f"VectorStore inicializado con ChromaDB en {config.CHROMA_PERSIST_DIRECTORY}")
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Agrega documentos al vector store.
        
        Args:
            documents: Lista de documentos en formato LangChain.
        """
        try:
            # La clase Chroma de LangChain maneja la adición directamente.
            self.db.add_documents(documents)
            logger.info(f"Agregados {len(documents)} documentos al vector store")
        except Exception as e:
            logger.error(f"Error agregando documentos: {str(e)}", exc_info=True)
            raise
            
    def get_chroma_instance(self) -> Chroma:
        """Retorna la instancia de la base de datos Chroma para usarla como retriever."""
        return self.db

    def clear_collection(self) -> None:
        """
        Elimina todos los documentos de la colección.
        """
        try:
            # Obtenemos todos los IDs de la colección actual para borrarlos.
            collection_data = self.db.get()
            if collection_data and collection_data['ids']:
                self.db.delete(ids=collection_data['ids'])
                logger.info("Colección limpiada (todos los vectores eliminados).")
            else:
                logger.info("La colección ya estaba vacía.")
        except Exception as e:
            logger.error(f"Error limpiando colección: {str(e)}", exc_info=True)
            raise
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas sobre la colección."""
        try:
            # El método _collection.count() nos da el número de items.
            count = self.db._collection.count()
            return {
                "num_documents": count,
                "collection_name": self.config.CHROMA_COLLECTION_NAME,
                "persist_directory": self.config.CHROMA_PERSIST_DIRECTORY
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}", exc_info=True)
            raise