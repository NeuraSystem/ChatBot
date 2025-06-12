"""
Constantes globales para el proyecto
"""

# Tipos de memoria
MEMORY_TYPES = {
    "in_memory": "Memoria en RAM",
    "persistent": "Memoria persistente"
}

# Tipos de documentos soportados
SUPPORTED_DOCUMENT_TYPES = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".txt": "Texto",
    ".md": "Markdown",
    ".html": "HTML"
}

# Estados del chatbot
CHATBOT_STATES = {
    "INIT": "Inicialización",
    "READY": "Listo",
    "PROCESSING": "Procesando",
    "ERROR": "Error"
}

# Mensajes estándar
MESSAGES = {
    "WELCOME": "Chatbot está listo! Escribe 'exit' para salir.",
    "ERROR": "Error: {error}",
    "DOCUMENT_UPLOADED": "Documento {document} cargado exitosamente",
    "DOCUMENTS_CLEARED": "Documentos limpiados exitosamente",
    "USER_REGISTERED": "Usuario {} registrado exitosamente",
    "USER_NOT_FOUND": "Usuario {} no encontrado"
}

# Configuración por defecto
DEFAULTS = {
    "MODEL": "claude-3-haiku-20240307",
    "MEMORY_TYPE": "in_memory",
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200
}
