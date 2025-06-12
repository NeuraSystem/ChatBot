from pathlib import Path
from typing import List, Optional, Dict
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added import
import os

from .logging_config import logger
from ..config import GlobalConfig


class DocumentLoader:
    """
    Sistema de carga y procesamiento de documentos para RAG.
    
    Maneja diferentes tipos de documentos y los divide en chunks
    para procesamiento posterior.
    """
    
    def __init__(self, config: GlobalConfig):
        """
        Inicializa el DocumentLoader.
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.MAX_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.supported_types = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': TextLoader,
            '.html': UnstructuredFileLoader
        }
        
    def load_document(self, file_path: str) -> List[Document]:
        """
        Carga y procesa un documento.
        
        Args:
            file_path: Ruta al archivo a cargar
            
        Returns:
            Lista de Documentos procesados
            
        Raises:
            ValueError: Si el tipo de archivo no es soportado
        """
        try:
            file_path = Path(file_path)
            # Validación de path traversal
            if '..' in file_path.parts or file_path.is_absolute():
                raise ValueError("Ruta de archivo no permitida por seguridad.")
            if not file_path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_types:
                raise ValueError(f"Tipo de archivo no soportado: {file_extension}")
            # Validar contra la configuración de tipos permitidos
            if hasattr(self.config, 'ALLOWED_FILE_TYPES') and file_extension[1:] not in self.config.ALLOWED_FILE_TYPES:
                raise ValueError(f"Tipo de archivo no permitido por configuración: {file_extension}")
            loader_class = self.supported_types[file_extension]
            loader = loader_class(str(file_path))
            
            # Cargar el documento
            documents = loader.load()
            
            # Dividir en chunks
            chunks = self.split_into_chunks(documents)
            
            logger.info(f"Cargado y procesado {file_path.name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error al cargar documento {file_path}: {str(e)}")
            raise
            
    def split_into_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos en chunks usando el text splitter configurado.
        
        Args:
            documents: Lista de documentos a dividir
            
        Returns:
            Lista de documentos divididos en chunks
        """
        try:
            chunks = []
            for doc in documents:
                # Dividir el documento en chunks
                doc_chunks = self.text_splitter.split_documents([doc])
                chunks.extend(doc_chunks)
                
            logger.info(f"Dividido documento en {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error al dividir documento en chunks: {str(e)}")
            raise
            
    def load_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Carga y procesa múltiples documentos.
        
        Args:
            file_paths: Lista de rutas a los archivos
            
        Returns:
            Lista combinada de Documentos procesados
        """
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.load_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error al procesar {file_path}: {str(e)}")
                continue
                
        return all_chunks

    def validate_file(self, file_path: str) -> bool:
        """
        Valida si un archivo es soportado y existe.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            True si el archivo es válido, False en caso contrario
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
                
            if path.suffix.lower() not in self.supported_types:
                return False
                
            return True
            
        except Exception:
            return False
