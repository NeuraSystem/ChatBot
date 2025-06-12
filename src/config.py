# src/config.py

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import field_validator, ConfigDict


class GlobalConfig(BaseSettings):
    """Configuración global del chatbot"""
    
    # Configuración del modelo
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic") # Default to anthropic
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "claude-3-haiku-20240307")
    # Ejemplos: "claude-3-haiku-20240307" (Anthropic), "gpt-3.5-turbo" (OpenAI), "gemini-pro" (Gemini), "deepseek-chat" (DeepSeek)

    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    
    # Configuración de RAG
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    
    # Configuración de ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "chatbot_docs")
    
    # Configuración de embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    NORMALIZE_EMBEDDINGS: bool = os.getenv("NORMALIZE_EMBEDDINGS", "false").lower() == "true"
    
    # Configuración de chunking
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Configuración de documentos
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "./documents")
    ALLOWED_FILE_TYPES: str = os.getenv("ALLOWED_FILE_TYPES", "pdf,docx,txt")
    
    # Configuración de cache
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "10000"))
    
    # Configuración de memoria
    MEMORY_TYPE: str = os.getenv("MEMORY_TYPE", "in_memory")  # in_memory o persistent
    MEMORY_PERSIST_DIR: str = os.getenv("MEMORY_PERSIST_DIR", "./chat_memory")
    
    # Configuración de logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "chatbot.log")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore'  # Ignore extra fields from .env or environment
    )

    @field_validator("ALLOWED_FILE_TYPES")
    def parse_allowed_file_types(cls, v):
        return v.split(',')

    # Changed to an instance method
    # Validates the current configuration instance, typically called after all settings are loaded.
    def validate_instance(self): # Renamed to avoid conflict if validate() is a pydantic keyword
        # Validate API key based on LLM_PROVIDER
        provider = self.LLM_PROVIDER.lower()
        if provider == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError("Missing ANTHROPIC_API_KEY for LLM_PROVIDER='anthropic'")
        elif provider == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY for LLM_PROVIDER='openai'")
        elif provider == "gemini" and not self.GEMINI_API_KEY:
            raise ValueError("Missing GEMINI_API_KEY for LLM_PROVIDER='gemini'")
        elif provider == "deepseek" and not self.DEEPSEEK_API_KEY:
            raise ValueError("Missing DEEPSEEK_API_KEY for LLM_PROVIDER='deepseek'")
            
        if self.RAG_ENABLED and not os.path.exists(self.MEMORY_PERSIST_DIR):
            os.makedirs(self.MEMORY_PERSIST_DIR)
            
        return True

config = GlobalConfig()
# Call validate_instance on the global instance after it's created.
# This is for application startup validation. Tests will create their own instances.
config.validate_instance()