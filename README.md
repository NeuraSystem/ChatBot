# Chatbot RAG con LangChain

Este proyecto es un chatbot modular y extensible que utiliza recuperación aumentada por generación (RAG) para responder preguntas de usuarios, integrando diversos modelos de lenguaje (LLMs) y capacidades de búsqueda semántica sobre documentos propios.

## Características principales

- **Arquitectura desacoplada y profesional**: Uso de inyección de dependencias y servicios para máxima mantenibilidad y escalabilidad.
- **Múltiples Proveedores de LLM**: Configurable para usar modelos de Anthropic, OpenAI y DeepSeek. La infraestructura para Gemini está presente pero su activación está pendiente de resolución de conflictos de dependencias.
- **Recuperación aumentada por generación (RAG)**: El bot puede buscar información relevante en documentos cargados por el usuario y usarla para enriquecer sus respuestas.
- **Persistencia**:
    - **Embeddings de Documentos**: ChromaDB se configura para persistir en disco (configurable a través de `CHROMA_PERSIST_DIRECTORY`), permitiendo que los embeddings de documentos cargados se mantengan entre sesiones.
    - **Datos de Usuario**: `UserManager` guarda el mapeo de usuarios a identificadores de historial en un archivo JSON (configurable a través de `USER_DATA_FILE`), persistiendo los perfiles de usuario.
    - **Historial de Conversación**: La persistencia del historial de mensajes depende de la configuración del `checkpointer` en `LangGraphService`. El `MemorySaver` por defecto es en memoria. Para persistencia de mensajes entre sesiones, se necesitaría configurar un `checkpointer` persistente (ej. `SqliteSaver`) para LangGraph.
- **Gestión de documentos**:
    - Soporte nativo para archivos `.txt`.
    - Para archivos `.pdf` y `.docx`, se requiere la instalación de `pypdf` y `docx2txt` respectivamente (`pip install pypdf docx2txt`). Sin estas librerías, el procesamiento de estos tipos de archivo fallará.
- **Logging centralizado**: Todos los eventos y errores relevantes quedan registrados con stack trace para fácil depuración.
- **Pruebas unitarias e integración**: Cobertura de los principales flujos y servicios, incluyendo la configuración de LLMs.

## Estructura del proyecto

```
chatbot/
├── src/
│   ├── chatbot.py            # Orquestador principal del bot
│   ├── langgraph_service.py  # Servicio para la lógica de LangGraph y selección de LLM
│   ├── document_service.py   # Servicio para gestión de documentos
│   ├── user_manager.py       # Gestión y persistencia de usuarios
│   ├── services.py           # Contenedor de servicios (inyección de dependencias)
│   ├── config.py             # Configuración global y variables de entorno
│   ├── main.py               # Entrada principal (CLI)
│   └── rag/                  # Lógica de RAG (retriever, embeddings, vector store, etc.)
├── tests/                    # Pruebas unitarias y de integración
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Esta guía
└── .env.example              # Archivo de ejemplo para variables de entorno (Crear un .env)
```

## Requisitos

- Python 3.9+
- Claves API para el proveedor LLM seleccionado.
- Dependencias listadas en `requirements.txt`.

## Configuración

1.  **Clona el repositorio y entra en la carpeta:**
    ```bash
    git clone <url-del-repo>
    cd chatbot
    ```
2.  **Crea un entorno virtual y actívalo:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    # Para soporte completo de PDF y DOCX en la carga de documentos, instala también:
    # pip install pypdf docx2txt
    ```
4.  **Configura tus variables de entorno:**
    - Crea un archivo `.env` en la raíz del proyecto. Puedes copiar de `.env.example` si existe, o crear uno nuevo.
    - Contenido mínimo de ejemplo para `.env`:

      ```env
      # --- Configuración del Modelo de Lenguaje (LLM) ---
      LLM_PROVIDER="anthropic" # Proveedores soportados: "anthropic", "openai", "deepseek". Gemini temporalmente desactivado.
      LLM_MODEL_NAME="claude-3-haiku-20240307" # Modelo específico para el proveedor
      # Ejemplos de modelos:
      # Anthropic: "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"
      # OpenAI: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"
      # DeepSeek: "deepseek-chat", "deepseek-coder"
      # Gemini: "gemini-pro" (cuando se reactive)

      # --- Claves API (solo necesitas la del proveedor seleccionado en LLM_PROVIDER) ---
      ANTHROPIC_API_KEY="tu_clave_anthropic"
      OPENAI_API_KEY="tu_clave_openai"
      # GEMINI_API_KEY="tu_clave_gemini" # Soporte pendiente de resolución de conflicto de dependencias
      DEEPSEEK_API_KEY="tu_clave_deepseek"

      # --- Configuración de RAG (Recuperación Aumentada por Generación) ---
      RAG_ENABLED="true" # "true" o "false"
      # EMBEDDING_MODEL="BAAI/bge-small-en-v1.5" # Modelo para embeddings (opcional, tiene default)
      # NORMALIZE_EMBEDDINGS="false" # Normalizar embeddings (opcional, tiene default)
      # CHROMA_PERSIST_DIRECTORY="./chroma_db" # Directorio para la base de datos vectorial persistente
      # CHROMA_COLLECTION_NAME="chatbot_docs" # Nombre de la colección en ChromaDB

      # --- Configuración de Documentos ---
      # ALLOWED_FILE_TYPES="pdf,docx,txt" # Tipos de archivo permitidos para carga

      # --- Configuración de Memoria y Persistencia ---
      USER_DATA_FILE="./usuarios.json" # Archivo para persistencia de datos de usuario (UserManager)
      # MEMORY_PERSIST_DIR="./chat_memory" # Directorio para persistencia de memoria de conversación (LangGraph MemorySaver)
                                         # Nota: La persistencia de mensajes de LangGraph depende de la configuración del Checkpointer.
                                         # El MemorySaver por defecto es en memoria. Para persistir mensajes, se requeriría
                                         # configurar LangGraphService para usar, por ejemplo, SqliteSaver con una ruta.

      # --- Otros ---
      # LOG_LEVEL="INFO"
      # LOG_FILE="chatbot.log"
      ```
    - **Nota sobre Gemini:** El soporte para Gemini (`LLM_PROVIDER="gemini"`) está implementado en el código pero actualmente desactivado en las dependencias (`requirements.txt` y tests) debido a un conflicto con la versión de la librería `protobuf`. Se podrá habilitar una vez se resuelva este conflicto.

## Uso básico (CLI)

Ejecuta el chatbot desde la terminal:

```bash
python -m src.main
```

### Comandos disponibles en la consola interactiva:

- `registro <nombre>`: Registrar un nuevo usuario
- `usuarios`: Ver lista de usuarios registrados
- `historial <nombre>`: Cambiar a la conversación de un usuario
- `new`: Iniciar una conversación temporal
- `upload <ruta_al_archivo>`: Subir un documento (soportados: .txt; .pdf y .docx con dependencias extra).
- `list_docs`: Listar documentos cargados (funcionalidad RAG).
- `clear_docs`: Limpiar documentos del sistema RAG.
- `exit`: Salir del chatbot

Simplemente escribe tu mensaje para interactuar con el bot. El sistema usará RAG si hay documentos cargados y la opción `RAG_ENABLED` es `true`.

## Ejemplo de uso

```text
> registro juan
Usuario juan registrado exitosamente.
> upload ./mis_documentos/manual_proyecto.txt
Documento ./mis_documentos/manual_proyecto.txt cargado exitosamente
> ¿Qué dice el manual sobre la configuración inicial?
Chatbot: Según el documento cargado, la configuración inicial requiere...
> clear_docs
Documentos limpiados exitosamente.
> exit
¡Hasta luego!
```

## Pruebas

Para ejecutar todas las pruebas (aseúrate de que las dependencias de desarrollo estén instaladas y el entorno configurado, aunque la mayoría de los tests usan mocks para llamadas externas):

```bash
python -m pytest
```

O para ejecutar suites específicas:

```bash
python -m unittest tests.test_config_llm
python -m unittest tests.test_user_manager_persistence
python -m pytest tests/test_chatbot.py
python -m pytest tests/test_integration.py
```
Es posible que necesites configurar variables de entorno para API keys (pueden ser valores dummy como "fake_key") para que las pruebas de configuración de `GlobalConfig` pasen la validación inicial si el proveedor por defecto en tu entorno local requiere una clave.

## Personalización y extensión

- Puedes añadir nuevos tipos de documentos o cambiar la lógica de recuperación implementando nuevas clases que hereden de `BaseRetriever`.
- La arquitectura desacoplada facilita la integración de nuevos servicios o modelos LLM.

## Notas de seguridad

- El sistema valida extensiones de archivos para la carga de documentos.
- Los logs incluyen stack trace para facilitar la depuración; revisa su contenido y nivel de detalle antes de exponerlos en producción.

## Soporte y contacto

Para dudas, sugerencias o contribuciones, abre un issue o contacta al mantenedor del repositorio.