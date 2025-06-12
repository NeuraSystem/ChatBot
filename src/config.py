# src/config.py

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import field_validator, ConfigDict


class GlobalConfig(BaseSettings):
    """Configuración global del chatbot"""
    
    # Configuración del modelo
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    # =============================================================================
    # CAMBIO FINAL: Se revierte el nombre del campo a 'ANTHROPIC_MODEL' para que
    # coincida exactamente con la variable de entorno.
    # Esto soluciona el 'ValidationError'.
    # =============================================================================
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    
    # Configuración de RAG
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    
    # Configuración de ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "chatbot_docs")
    
    # Configuración de embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
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
        env_file_encoding="utf-8"
    )

    @field_validator("ALLOWED_FILE_TYPES")
    def parse_allowed_file_types(cls, v):
        return v.split(',')

    @staticmethod
    def validate():
        if not GlobalConfig.ANTHROPIC_API_KEY:
            raise ValueError("Missing ANTHROPIC_API_KEY in environment variables.")
            
        if GlobalConfig.RAG_ENABLED and not os.path.exists(GlobalConfig.MEMORY_PERSIST_DIR):
            os.makedirs(GlobalConfig.MEMORY_PERSIST_DIR)
            
        return True

config = GlobalConfig()