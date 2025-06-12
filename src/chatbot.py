# src/chatbot.py

import logging
from typing import List, Optional, Any # Added Any

from .rag.logging_config import logger
from .constants import MESSAGES
from src.config import GlobalConfig as Config
from .rag.chat_history import ChatHistory
from .rag.retriever import RAGRetriever, BaseRetriever
from .langgraph_service import LangGraphService
from .document_service import DocumentService

config = Config()
logger = logging.getLogger(__name__)

class Chatbot:
    """
    Chatbot principal. Orquesta los servicios de LangGraph y Documentos.
    """
    # Removed api_key and model from __init__ signature
    def __init__(self,
                 global_config: Config, # Takes global_config
                 langgraph_service: Any, # langgraph_service is now required
                 retriever: Optional[BaseRetriever] = None,
                 document_loader: Optional[Any] = None,
                 chat_history_service: Optional[Any] = None): # Service responsible for managing user-specific conversation history state (if applicable beyond LangGraph's checkpointer)
        """
        Inicializa el chatbot con configuración flexible
        """
        self.config = global_config # Store global_config
        self.langgraph_service = langgraph_service # Use provided langgraph_service
        
        if self.config.RAG_ENABLED:
            # Retriever might be optional if LGS already has it or if RAG is disabled.
            self.rag_retriever = retriever
            # ChatHistory is now managed by LangGraphService's MemorySaver or external history service
            self.chat_history = chat_history_service

            # DocumentService handles document processing logic; it's configured here if RAG is enabled.
            if document_loader:
                # DocumentService might also be better if provided directly by ServiceContainer
                self.document_service = DocumentService(self.rag_retriever, document_loader)
            else:
                logger.warning("DocumentLoader no proporcionado a Chatbot aunque RAG está habilitado.")
                self.document_service = None
        else:
            self.rag_retriever = None
            self.chat_history = None
            self.document_service = None
        
    def send_message(self, message: str, user_id: str = "default") -> str:
        """
        Envía un mensaje al chatbot y retorna la respuesta.
        """
        try:
            # CORRECCIÓN: Llamamos al método correcto del servicio.
            return self.langgraph_service.send_message(message, user_id)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error de valor o ejecución: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))
        except Exception as e:
            logger.error(f"Error inesperado procesando mensaje: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))

    def add_documents(self, documents: List[str]) -> str:
        """
        Agrega documentos al sistema RAG.
        """
        if not self.document_service:
            return "El sistema RAG no está habilitado"
            
        try:
            return self.document_service.add_documents(documents)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error de archivo o valor: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))
        except Exception as e:
            logger.error(f"Error inesperado agregando documentos: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))

    def clear_documents(self) -> str:
        """Elimina todos los documentos del sistema RAG."""
        if not self.document_service:
            return "El sistema RAG no está habilitado"
            
        try:
            return self.document_service.clear_documents()
        except RuntimeError as e:
            logger.error(f"Error de ejecución limpiando documentos: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))
        except Exception as e:
            logger.error(f"Error inesperado limpiando documentos: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))