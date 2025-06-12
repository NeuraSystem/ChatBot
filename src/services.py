# src/services.py

from .chatbot import Chatbot
from .langgraph_service import LangGraphService
from .document_service import DocumentService
from .user_manager import GestorUsuarios
from .rag.retriever import RAGRetriever
from .rag.chat_history import ChatHistory
from src.config import GlobalConfig

class ServiceContainer:
    def __init__(self):
        self.config = GlobalConfig()
        self.user_manager = GestorUsuarios()

        if self.config.RAG_ENABLED:
            self.rag_retriever = RAGRetriever(self.config)
            self.chat_history = ChatHistory(self.config)
            self.document_service = DocumentService(self.rag_retriever)
            langchain_retriever = self.rag_retriever.get_retriever()
        else:
            self.rag_retriever = None
            self.chat_history = None
            self.document_service = None
            langchain_retriever = None

        # =============================================================================
        # CAMBIO: Se usa 'ANTHROPIC_MODEL' para que coincida con el archivo de configuración.
        # =============================================================================
        self.langgraph_service = LangGraphService(
            api_key=self.config.ANTHROPIC_API_KEY,
            model=self.config.ANTHROPIC_MODEL,
            chat_history=self.chat_history,
            retriever=langchain_retriever
        )

        # =============================================================================
        # CAMBIO: Se usa 'ANTHROPIC_MODEL' aquí también.
        # =============================================================================
        self.chatbot = Chatbot(
            api_key=self.config.ANTHROPIC_API_KEY,
            model=self.config.ANTHROPIC_MODEL,
            retriever=self.rag_retriever
        )