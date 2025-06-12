# src/document_service.py

from src.config import GlobalConfig as Config
from .constants import MESSAGES
from .rag.logging_config import logger

# =============================================================================
# CAMBIO: Se importa el objeto 'config' para que esté disponible en todo el archivo.
# Esto soluciona el error "NameError: name 'config' is not defined".
# =============================================================================
from src.config import config

class DocumentService:
    """
    Servicio para la gestión de documentos en el sistema RAG.
    """
    def __init__(self, rag_retriever):
        self.rag_retriever = rag_retriever

    def add_documents(self, documents):
        if not config.RAG_ENABLED:
            return "El sistema RAG no está habilitado"
        try:
            # Aquí asumimos que RAGRetriever se encarga de la lógica de carga de documentos
            self.rag_retriever.add_documents(documents)
            # El mensaje de éxito debe ser genérico o basarse en la respuesta del retriever
            return MESSAGES["DOCUMENT_UPLOADED"].format("los documentos solicitados")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error de archivo o valor agregando documentos: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))
        except Exception as e:
            logger.error(f"Error inesperado agregando documentos: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))

    def clear_documents(self):
        if not config.RAG_ENABLED:
            return "El sistema RAG no está habilitado"
        try:
            self.rag_retriever.clear_documents()
            return MESSAGES["DOCUMENTS_CLEARED"]
        except RuntimeError as e:
            logger.error(f"Error de ejecución limpiando documentos: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))
        except Exception as e:
            logger.error(f"Error inesperado limpiando documentos: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))