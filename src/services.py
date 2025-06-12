# src/services.py

from .chatbot import Chatbot
from .langgraph_service import LangGraphService
from .document_service import DocumentService
from .user_manager import GestorUsuarios
from .rag.retriever import RAGRetriever
from .rag.chat_history import ChatHistory
from .rag.document_loader import DocumentLoader # Import DocumentLoader
from src.config import GlobalConfig

class ServiceContainer:
    def __init__(self):
        self.config = GlobalConfig()
        self.user_manager = GestorUsuarios()

        if self.config.RAG_ENABLED:
            self.document_loader = DocumentLoader(self.config) # Instantiate DocumentLoader
            self.rag_retriever = RAGRetriever(self.config)
            self.chat_history = ChatHistory(self.config)
            # Pass document_loader to DocumentService
            self.document_service = DocumentService(self.rag_retriever, self.document_loader)
            langchain_retriever = self.rag_retriever.get_retriever()
        else:
            self.document_loader = None # Ensure it's None if RAG is disabled
            self.rag_retriever = None
            self.chat_history = None
            self.document_service = None
            langchain_retriever = None

        # LangGraphService now takes global_config
        self.langgraph_service = LangGraphService(
            global_config=self.config, # Pass the global config
            chat_history=self.chat_history, # ChatHistory might be optional if LGS uses MemorySaver exclusively
            retriever=langchain_retriever
        )

        # Chatbot also takes global_config and other services
        self.chatbot = Chatbot(
            global_config=self.config,
            langgraph_service=self.langgraph_service,
            retriever=self.rag_retriever, # RAGRetriever instance
            document_loader=self.document_loader, # DocumentLoader instance
            chat_history_service=self.chat_history # ChatHistory service instance
        )