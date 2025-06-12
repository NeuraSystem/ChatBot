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
    def __init__(self, rag_retriever, document_loader): # Añadido document_loader
        self.rag_retriever = rag_retriever
        self.document_loader = document_loader # Almacenar document_loader

    def add_documents(self, file_paths: list[str]): # Cambiado 'documents' a 'file_paths' para claridad
        if not config.RAG_ENABLED:
            return "El sistema RAG no está habilitado"
        try:
            # Usar document_loader para cargar y procesar los archivos
            loaded_documents = self.document_loader.load_multiple_documents(file_paths)
            if not loaded_documents:
                # Esto podría ocurrir si todos los archivos fallan en cargarse o están vacíos
                logger.warn("No se cargaron documentos de las rutas proporcionadas.")
                return MESSAGES["ERROR"].format(error="No se pudieron cargar documentos de las rutas especificadas.")

            self.rag_retriever.add_documents(loaded_documents)
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