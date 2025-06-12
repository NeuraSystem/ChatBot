# src/rag/chat_history.py

import logging
from typing import List, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from .logging_config import logger
from src.config import GlobalConfig as Config


class ChatHistory:
    """
    Sistema de historial de chat integrado con LangChain.
    Su única responsabilidad es gestionar el historial de mensajes de una conversación.
    """
    
    def __init__(self, config: Config):
        """
        Inicializa el sistema de historial.
        """
        self.config = config
        # El modelo de chat ya no se usa aquí para generar respuestas, pero se mantiene
        # por si se necesita para futuras funcionalidades dentro de esta clase.
        # =============================================================================
        # CAMBIO: Se usa 'ANTHROPIC_MODEL' para que coincida con el archivo de configuración.
        # Esto soluciona el 'AttributeError'.
        # =============================================================================
        self.chat_model = ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY
        )
        self.history = ChatMessageHistory()
        logger.info("Sistema de historial de chat inicializado")
        
    def add_human_message(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Agrega un mensaje humano al historial.
        """
        try:
            message = HumanMessage(content=content, additional_kwargs=metadata or {})
            self.history.add_message(message)
            logger.info("Mensaje humano agregado al historial")
        except Exception as e:
            logger.error(f"Error agregando mensaje humano: {str(e)}")
            raise
            
    def add_ai_message(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Agrega un mensaje AI al historial.
        """
        try:
            message = AIMessage(content=content, additional_kwargs=metadata or {})
            self.history.add_message(message)
            logger.info("Mensaje AI agregado al historial")
        except Exception as e:
            logger.error(f"Error agregando mensaje AI: {str(e)}")
            raise
            
    def get_messages(self) -> List[Dict[str, Any]]:
        """Retorna el historial de mensajes como lista de diccionarios."""
        try:
            return [
                {
                    "role": msg.type,
                    "content": msg.content,
                    "metadata": msg.additional_kwargs
                }
                for msg in self.history.messages
            ]
        except Exception as e:
            logger.error(f"Error obteniendo mensajes: {str(e)}")
            raise
            
    def clear_history(self) -> None:
        """Limpia el historial de chat."""
        try:
            self.history.clear()
            logger.info("Historial limpiado")
        except Exception as e:
            logger.error(f"Error limpiando historial: {str(e)}")
            raise
