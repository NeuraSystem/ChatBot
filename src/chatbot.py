# src/chatbot.py

import logging
from typing import List, Optional

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
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, retriever: Optional[BaseRetriever] = None):
        """
        Inicializa el chatbot con configuración flexible
        """
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        # =============================================================================
        # CAMBIO: Se usa 'ANTHROPIC_MODEL' para que coincida con el archivo de configuración.
        # =============================================================================
        self.model = model or config.ANTHROPIC_MODEL
        
        if config.RAG_ENABLED:
            self.rag_retriever = retriever or RAGRetriever(config) 
            self.chat_history = ChatHistory(config)
            self.document_service = DocumentService(self.rag_retriever)
        else:
            self.rag_retriever = None
            self.chat_history = None
            self.document_service = None
        
        self.langgraph_service = LangGraphService(
            api_key=self.api_key, 
            model=self.model, 
            chat_history=self.chat_history, 
            retriever=self.rag_retriever.get_retriever() if self.rag_retriever else None
        )
        
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