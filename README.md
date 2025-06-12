# Chatbot RAG con LangChain y Anthropic

Este proyecto es un chatbot modular y extensible que utiliza recuperación aumentada por generación (RAG) para responder preguntas de usuarios, integrando modelos de lenguaje de Anthropic (Claude) y capacidades de búsqueda semántica sobre documentos propios.

## Características principales

- **Arquitectura desacoplada y profesional**: Uso de inyección de dependencias y servicios para máxima mantenibilidad y escalabilidad.
- **Recuperación aumentada por generación (RAG)**: El bot puede buscar información relevante en documentos cargados por el usuario y usarla para enriquecer sus respuestas.
- **Persistencia de usuarios**: Los usuarios y sus historiales se guardan automáticamente.
- **Gestión de documentos**: Permite cargar, listar y limpiar documentos para el sistema RAG.
- **Logging centralizado**: Todos los eventos y errores relevantes quedan registrados con stack trace para fácil depuración.
- **Pruebas unitarias e integración**: Cobertura de los principales flujos y servicios.

## Estructura del proyecto

```
chatbot/
├── src/
│   ├── chatbot.py            # Orquestador principal del bot
│   ├── langgraph_service.py  # Servicio para la lógica de LangGraph
│   ├── document_service.py   # Servicio para gestión de documentos
│   ├── user_manager.py       # Gestión y persistencia de usuarios
│   ├── services.py           # Contenedor de servicios (inyección de dependencias)
│   ├── main.py               # Entrada principal (CLI)
│   └── rag/                  # Lógica de RAG (retriever, embeddings, vector store, etc.)
├── tests/                    # Pruebas unitarias y de integración
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Esta guía
└── setup.py                  # Instalación como paquete (opcional)
```

## Requisitos

- Python 3.9+
- Acceso a la API de Anthropic (Claude)
- Dependencias listadas en `requirements.txt`

## Instalación

1. **Clona el repositorio y entra en la carpeta:**
   ```bash
   git clone <url-del-repo>
   cd chatbot
   ```
2. **Crea un entorno virtual y actívalo:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configura tus variables de entorno:**
   - Crea un archivo `.env` en la raíz del proyecto con al menos:
     ```env
     ANTHROPIC_API_KEY=tu_clave_de_api
     ANTHROPIC_MODEL=claude-3-haiku-20240307
     ```
   - Puedes personalizar otros parámetros en `src/config.py`.

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
- `upload <archivo>`: Subir un documento (PDF, TXT, DOCX, etc.)
- `list_docs`: Listar documentos cargados
- `clear_docs`: Limpiar documentos
- `exit`: Salir del chatbot

Simplemente escribe tu mensaje para interactuar con el bot. El sistema usará RAG si hay documentos cargados y la opción está habilitada.

## Ejemplo de uso

```text
> registro juan
Usuario juan registrado exitosamente
> upload manual.pdf
Documento manual.pdf cargado exitosamente
> ¿Qué dice el manual sobre seguridad?
Chatbot: Según el documento cargado, ...
> clear_docs
Documentos limpiados exitosamente
> exit
¡Hasta luego!
```

## Pruebas

Para ejecutar todas las pruebas:

```bash
python -m unittest discover tests
```

Esto ejecutará tanto las pruebas unitarias como las de integración.

## Personalización y extensión

- Puedes añadir nuevos tipos de documentos o cambiar la lógica de recuperación implementando nuevas clases que hereden de `BaseRetriever`.
- La arquitectura desacoplada facilita la integración de nuevos servicios o modelos.

## Notas de seguridad

- El sistema valida extensiones y rutas de archivos para evitar ataques comunes (path traversal, archivos no permitidos).
- Los logs incluyen stack trace para facilitar la depuración, pero revisa su contenido antes de exponerlos en producción.

## Soporte y contacto

Para dudas, sugerencias o contribuciones, abre un issue o contacta al mantenedor del repositorio.